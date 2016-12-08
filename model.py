"""
Just basic CNN model for window
"""
import tensorflow as tf
from tensorflow.contrib import layers


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
            emb_4d = tf.expand_dims(emb, 1)
            print(emb_4d)
            features = []
            for fsize in [3, 5, 7, 11]:
                conv = layers.convolution2d(emb_4d, 32, [1, 3],
                                            padding='SAME', activation_fn=tf.nn.relu)

                # let's make global poolings!

                # dont's forget to squuze dat tensors
                global_avg_pooling = tf.squeeze(tf.reduce_mean(conv, [2]))
                global_max_pooling = tf.squeeze(tf.reduce_max(conv, [2]))
                features.append(global_avg_pooling)
                features.append(global_max_pooling)

            merged_features = tf.concat(1, features)

            features = tf.nn.dropout(merged_features, self.keep_prob)

            # just simple two layered mlp:
            for _ in range(2):
                features = layers.fully_connected(features, args.dim,
                                                  activation_fn=tf.nn.relu)
            logit = layers.fully_connected(features, 1)

            # just optimized binary classification loss
            costs = tf.nn.sigmoid_cross_entropy_with_logits(logit, self.targets)
            self.loss = cost = tf.reduce_mean(costs)
            self.probs = tf.nn.sigmoid(logit)

            self.lr = tf.Variable(0.0, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)

            # TODO: add grad clipping
            self.train_op = optimizer.minimize(cost, gate_gradients=tf.train.Optimizer.GATE_NONE)
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
                     seq_length=11,
                     vocab_size=3,
                     dim=7)

    model = Model(args, dev='/cpu:0')


if __name__ == "__main__":
    quick_test()