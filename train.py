import os
import argparse
import numpy as np
import tensorflow as tf
from model import Model, ExceptionModel
from tqdm import tqdm
import utils
import json
from sklearn.metrics import roc_auc_score, accuracy_score


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./data', help='Folder with data files')
    parser.add_argument('--save_dir', default='./tmp/model0', help='Path to save model')
    parser.add_argument('--vocab_size', type=int, default=4)
    parser.add_argument('--dim', default=8)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--seq_length', default=128, type=int, help='Len of chunk to classify')
    parser.add_argument('--steps', default=1000, type=int, help='Steps per epoch')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs')
    parser.add_argument('--keep_prob', default=1.0, help='Keep prob for dropout')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--seed', default=1337, type=int, help='Fixed seed, repro style')
    parser.add_argument('--model', default='basic', type=str, help='Model')

    parser.add_argument('--outer_split', default=0.2, type=float)
    parser.add_argument('--inner_split', default=0.2, type=float)

    return parser.parse_args()


class Datagen:
    def __init__(self, dataset, args):
        """Just short random window datagenerator
        Args:
            dataset - (X, y) tuple, X is list of encoded sequences, y is list of labels
        """
        self.args = args
        self.data, self.labels = dataset
        self.N = len(self.data)

        self.total = self.N // args.batch_size

    def reset(self):
        np.random.seed(self.args.seed)

    def next(self):
        idx = np.random.choice(self.N, size=self.args.batch_size)
        # now we could use larger dtype
        x = np.zeros((self.args.batch_size, self.args.seq_length), dtype=np.int32)
        y = np.zeros((self.args.batch_size, 1), dtype=np.int32)

        for line, i in enumerate(idx):
            max_len = len(self.data[i]) - self.args.seq_length
            beg = np.random.randint(0, max_len)
            x[line, :] = self.data[i][beg: beg + self.args.seq_length]
            y[line, 0] = self.labels[i]
        return x, y

    __next__ = next

    def __iter__(self):
        return self


def run_epoch(session, model, datagen, args, op, verbose=False, add_metrics=False,
              rolling_buf=-1, prefix='', keep_prob=1.0):
    if prefix == 'val':
        datagen.reset()
    avg_loss = 0.0
    counter = 0
    history = []
    steps = min(args.steps, datagen.total)
    pbar = tqdm(range(steps))
    if rolling_buf < 0:
        predictions = []
        true_values = []
    else:
        predictions = [None] * rolling_buf
        true_values = [None] * rolling_buf
    for it in pbar:
        x, y = next(datagen)
        feed = {model.inputs: x, model.targets: y, model.keep_prob: keep_prob}
        if add_metrics:
            loss, probs, _ = session.run([model.loss, model.probs, op], feed)
            if rolling_buf < 0:
                predictions.append(probs)
                true_values.append(y)
            else:
                predictions[it % rolling_buf] = probs
                true_values[it % rolling_buf] = y
        else:
            loss, _ = session.run([model.loss, op], feed)
        history.append((prefix + '_loss', loss))
        pbar.set_description("loss: {:.2f}".format(loss))
        avg_loss += loss
        counter += 1
    avg_loss /= counter
    history.append((prefix + '_avg_loss', avg_loss))
    if add_metrics:
        p = np.concatenate([_ for _ in predictions if _ is not None])
        t = np.concatenate([_ for _ in true_values if _ is not None])
        auc = roc_auc_score(t, p)
        print('{} AUC: {:.3f}'.format(prefix, auc))
        history.append((prefix + '_auc', auc))
        acc = accuracy_score(t, p > 0.5)
        print('{} ACC: {:.3f}'.format(prefix, acc))
        history.append((prefix + '_acc', acc))
    if verbose:
        return avg_loss, history
    else:
        return avg_loss



def main():
    args = _parse_args()
    try:
        os.makedirs(args.save_dir)
    except:
        pass
    history_path = os.path.join(args.save_dir, 'history.log')
    with open(history_path, 'w') as fout:
        fout.write(json.dumps(vars(args)))
        fout.write('\ntype\tloss\n')

    whole_data, whole_labels = utils.prepare_data(args.seq_length, path=args.data, seed=args.seed)

    N = int(args.outer_split * len(whole_data))
    val_outer = whole_data[:N], whole_labels[:N]
    train = whole_data[N:], whole_labels[N:]

    traingen = Datagen(train, args)
    valgen = Datagen(val_outer, args)

    print('Build model')
    if args.model == 'exception':
        model = ExceptionModel(args)
    else:
        model = Model(args)
    config = tf.ConfigProto()

    with tf.Session(config=config) as sess:
        print('init weights')
        sess.run(tf.initialize_all_variables())
        sess.run(tf.assign(model.lr, args.learning_rate))

        print('Let the train begin!')
        avg_val_loss, val_history = run_epoch(sess, model, valgen,
                                              args, model.no_op,
                                              verbose=True,
                                              add_metrics=True,
                                              prefix='val')
        print('Val before training shows {:.2f} loss'.format(avg_val_loss))

        utils.store(history_path, val_history)
        for epoch in range(args.epochs):
            lr = args.learning_rate
            lr = max(0.00001 * lr, (1.0 - epoch / args.epochs) * lr)
            sess.run(tf.assign(model.lr, lr))
            avg_train_loss, train_history = run_epoch(sess, model,
                                                      traingen, args,
                                                      model.train_op,
                                                      verbose=True,
                                                      add_metrics=True,
                                                      rolling_buf=100,
                                                      prefix='train',
                                                      keep_prob=args.keep_prob)
            avg_val_loss, val_history = run_epoch(sess, model, valgen,
                                                  args, model.no_op,
                                                  verbose=True,
                                                  add_metrics=True,
                                                  prefix='val')
            model.saver.save(sess, os.path.join(args.save_dir, "scbow.ckpt"),
                             global_step=epoch)
            utils.store(history_path, train_history)
            utils.store(history_path, val_history)


if __name__ == "__main__":
    main()