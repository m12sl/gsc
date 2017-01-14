"""
Just basic CNN model for window
"""
import tensorflow as tf
from tensorflow.contrib import layers
import tensorflow.contrib as ctf


def _prelu(x):
    alpha = tf.get_variable('alpha', shape=(1,), dtype=tf.float32)
    pos = 0.5 * (x + tf.abs(x))
    neg = 0.5 * (x - tf.abs(x))
    return pos + alpha * neg


def _steam(x, channels=32):
    bs, w, h, ch = x.get_shape().as_list()
    mp_branch = layers.max_pool2d(
        inputs=x,
        kernel_size=[1, 5],
        stride=[1, 2],
        padding='SAME',
    )
    conv_branch = layers.conv2d(
        inputs=x,
        num_outputs=channels - ch,
        kernel_size=[1, 5],
        stride=[1, 2],
        padding='SAME',
        activation_fn=tf.nn.relu,
    )
    x = tf.concat(3, [mp_branch, conv_branch])
    return x


def _bottleneck(x, downsampling=False, keep_prob=1.0, rate=1, channels=32, prefix=''):
    with tf.variable_scope('btlnck_{}'.format(prefix)):
        bs, w, h, ch = x.get_shape().as_list()
        if downsampling:
            receipt = [(2, 2), (3, 1), (1, 1)]
            main_branch = layers.max_pool2d(
                inputs=x,
                kernel_size=[1, 2],
                stride=[1, 2],
                padding='SAME',
            )
            print('CHCH, ', channels, ch)
            main_branch = tf.pad(main_branch,
                                 [[0, 0], [0, 0], [0, 0], [0, channels - ch]],
                                 name='main')
        else:
            receipt = [(1, 1), (5, 1), (1, 1)]
            main_branch = x

        conv_branch = x
        for i, (ks, st) in enumerate(receipt):
            conv_branch = layers.conv2d(
                inputs=conv_branch,
                num_outputs=channels,
                kernel_size=(1, ks),
                stride=(1, st),
                rate=rate if i == 1 else 1,
                padding='SAME',
                activation_fn=_prelu if i < 2 else None,
                normalizer_fn=layers.batch_norm if i < 2 else None,
            )

        kp = tf.Variable(keep_prob, trainable=False, name='keep_prob')
        aux = tf.nn.dropout(conv_branch, kp, noise_shape=[bs, 1, 1, channels], name='aux')

        xat = _prelu(main_branch + aux)
        return xat


class ExceptionModel:
    def __init__(self, args, dev='/gpu:0'):
        with tf.device(dev):
            self.inputs = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.targets = tf.placeholder(tf.float32, [args.batch_size, 1])
            self.keep_prob = tf.placeholder(tf.float32)  # dropout control

            self.embeddings = tf.get_variable('embeddings', [args.vocab_size, args.dim])
            emb = tf.nn.embedding_lookup(self.embeddings, self.inputs)
            x = tf.expand_dims(emb, 1)

            x = _steam(x)
            print(x)
            x = _bottleneck(x, downsampling=True, prefix='downsampling_0')
            for _ in range(0, 5):
                x = _bottleneck(x, prefix='stage1_{}'.format(_))
                print(x)
            x = _bottleneck(x, downsampling=True, prefix='downsampling_1')
            print(x)
            for _ in range(0, 5):
                x = _bottleneck(x, prefix='stage2_{}'.format(2 * _))
                print(x)
                x = _bottleneck(x, rate=2 ** _, prefix='{}'.format(2 * _ + 1))
                print(x)

            global_max_pooling = tf.squeeze(tf.reduce_max(x, [2]))
            features = tf.nn.dropout(global_max_pooling, self.keep_prob)
            print(features)

            # just simple two layered mlp:
            for _ in range(2):
                features = layers.fully_connected(features, args.dim,
                                                  activation_fn=tf.nn.elu)
            logit = layers.fully_connected(features, 1)

            # just optimized binary classification loss
            costs = tf.nn.sigmoid_cross_entropy_with_logits(logit, self.targets)
            cost = tf.reduce_sum(costs)
            self.loss = cost / args.batch_size
            self.probs = tf.nn.sigmoid(logit)

            self.lr = tf.Variable(0.0, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)

            # TODO: add grad clipping
            # self.train_op = optimizer.minimize(cost, gate_gradients=tf.train.Optimizer.GATE_NONE)

            gvs = optimizer.compute_gradients(cost, gate_gradients=tf.train.Optimizer.GATE_NONE)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs)

            self.no_op = tf.no_op()  # just an opnode for validation steps

        with tf.device('/cpu:0'):
            self.saver = tf.train.Saver(tf.trainable_variables())

    def restore(self, path, session):
        session.run(tf.initialize_all_variables())
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(session, ckpt.model_checkpoint_path)

class Model:
    def __init__(self, args, dev='/gpu:0'):
        with tf.device(dev):
            self.inputs = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            self.targets = tf.placeholder(tf.float32, [args.batch_size, 1])
            self.keep_prob = tf.placeholder(tf.float32)  # dropout control

            self.embeddings = tf.get_variable('embeddings', [args.vocab_size, args.dim])
            emb = tf.nn.embedding_lookup(self.embeddings, self.inputs)

            # just add print(xxx) and run quick test if want to understand whats going on

            # There is no 1d conv... so let's expand and use 2d
            x = tf.expand_dims(emb, 1)
            print(x)

            for rate in [1, 2, 4, 8, 16, 32]:
                # let's trye gated convolutions!
                px = layers.convolution2d(x, 32, [1, 3],
                    padding='VALID',
                    rate=rate)
                gate = layers.convolution2d(x, 32, [1, 3],
                    padding='VALID',
                    rate=rate,
                    activation_fn=tf.nn.sigmoid)
                x = px * gate
                print(x)

            global_max_pooling = tf.squeeze(tf.reduce_max(x, [2]))
            features = tf.nn.dropout(global_max_pooling, self.keep_prob)
            print(features)

            # just simple two layered mlp:
            for _ in range(2):
                features = layers.fully_connected(features, args.dim,
                                                  activation_fn=tf.nn.elu)
            logit = layers.fully_connected(features, 1)

            # just optimized binary classification loss
            costs = tf.nn.sigmoid_cross_entropy_with_logits(logit, self.targets)
            cost = tf.reduce_sum(costs)
            self.loss = cost / args.batch_size
            self.probs = tf.nn.sigmoid(logit)

            self.lr = tf.Variable(0.0, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)

            # TODO: add grad clipping
            # self.train_op = optimizer.minimize(cost, gate_gradients=tf.train.Optimizer.GATE_NONE)

            gvs = optimizer.compute_gradients(cost, gate_gradients=tf.train.Optimizer.GATE_NONE)
            capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
            self.train_op = optimizer.apply_gradients(capped_gvs)

            self.no_op = tf.no_op()  # just an opnode for validation steps

        with tf.device('/cpu:0'):
            self.saver = tf.train.Saver(tf.trainable_variables())

    def restore(self, path, session):
        session.run(tf.initialize_all_variables())
        ckpt = tf.train.get_checkpoint_state(path)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(session, ckpt.model_checkpoint_path)


def quick_test():
    from argparse import Namespace

    args = Namespace(batch_size=5,
                     seq_length=128,
                     vocab_size=3,
                     dim=7)

    model = ExceptionModel(args, dev='/cpu:0')


if __name__ == "__main__":
    quick_test()