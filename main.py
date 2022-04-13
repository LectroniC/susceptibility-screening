from __future__ import print_function
import os
import numpy as np
import pickle
import tensorflow as tf
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, help='Path to the data of in-hospital mortality task',
                    default='..')
args = parser.parse_args()

np.random.seed(12345)

class TargetModel:
    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim, timesteps):

        self.x = tf.compat.v1.placeholder(tf.float32, (None, timesteps, input_dim), name='x')
        self.y = tf.compat.v1.placeholder(tf.float32, (None, output_dim), name='y')

        x = tf.unstack(self.x, timesteps, 1)
        rnn_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_dim)
        outputs, _ = tf.compat.v1.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
        d1_output = tf.compat.v1.layers.dense(outputs[-1], units=fc_dim, activation=tf.nn.relu)
        logits = tf.compat.v1.layers.dense(d1_output, units=output_dim)
        self.ybar = tf.nn.softmax(logits)

        count = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.ybar, axis=1))
        self.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        xent = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
        self.loss = tf.reduce_mean(xent)

        optimizer = tf.compat.v1.train.AdamOptimizer()
        vs = tf.compat.v1.global_variables()
        self.train_op = optimizer.minimize(self.loss, var_list=vs)

        self.saver = tf.compat.v1.train.Saver()

def run_fold(path, hidden_dim, fc_dim, timesteps, load_and_skip_train=False):

    print('Loading Data')

    ###########################
    # Input format
    # data_train.pkl: n_train x timestamps x n_feature
    # target_train.pkl: n_train x label
    # data_test.pkl: n_test x timestamps x n_feature
    # target_test.pkl: n_test x label
    ###########################

    path_string = path + '/data_train.pkl'
    fn = open(path_string, 'rb')
    data_train = pickle.load(fn)

    path_string = path + '/target_train.pkl'
    fn = open(path_string, 'rb')
    labels_train = pickle.load(fn)

    path_string = path + '/data_test.pkl'
    fn = open(path_string, 'rb')
    data_test = pickle.load(fn)

    path_string = path + '/target_test.pkl'
    fn = open(path_string, 'rb')
    labels_test = pickle.load(fn)

    input_dim = data_test.shape[2]
    output_dim = labels_test.shape[1]

    print('Construction graph')

    # Creating models
    model = TargetModel(input_dim, output_dim, hidden_dim, fc_dim, timesteps)

    print('Initializing graph')

    sess = tf.compat.v1.InteractiveSession()
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(tf.compat.v1.local_variables_initializer())
    

    def evaluate_loss_acc(sess, model, X_data, y_data):
        loss, acc = sess.run([model.loss, model.acc], feed_dict={model.x: X_data, model.y: y_data})
        print(' loss: {} acc: {}'.format(loss, acc))
        return loss, acc

    def evaluate_auc_f1(sess, model, X_data, y_data):
        y_pred = predict(sess, model, X_data)
        Y_true = np.argmax(y_data, axis=1)
        Y_pred = np.argmax(y_pred, axis=1)
        auc_score = roc_auc_score(y_data[:, 1], y_pred[:, 1], average='micro')
        f1 = f1_score(Y_true, Y_pred)
        print(' AUC: {} f1: {}'.format(auc_score, f1))
        return auc_score, f1

    def train(sess, model, X_data, y_data, X_valid=None, y_valid=None, epochs=1, load_and_skip_train=False, batch_size=64, save_name='model'):

        if load_and_skip_train:
            print('Loading saved model')
            dir = path + '/model_l1/{}'
            return model.saver.restore(sess, dir.format(save_name))

        print('Train model')
        n_sample = X_data.shape[0]
        n_batch = int((n_sample+batch_size-1) / batch_size)
        for epoch in range(epochs):
            for batch in range(n_batch):
                start = batch * batch_size
                end = min(n_sample, start + batch_size)
                sess.run(model.train_op, feed_dict={model.x: X_data[start:end], model.y: y_data[start:end]})
            print("epoch {}".format(epoch))
            
            evaluate_loss_acc(sess, model, X_data, y_data)
            evaluate_auc_f1(sess, model, X_data, y_data)

        print('Saving model')
        dir = path + '/model'
        if not os.path.exists(dir):   # compatible for python 2.
            os.makedirs(dir)
        # os.makedirs(dir, exist_ok=True) # compatible for python 3.
        fn = dir + '/{}'
        model.saver.save(sess, fn.format(save_name))

    def predict(sess, model, X_data):
        return sess.run(model.ybar, feed_dict={model.x: X_data})
    
    def write_to_report(auc,f1,precision,recall,confusion_matrix):
        with open('val_report.txt', 'a') as f:
            f.write("Train AUC_score = {}\n".format(auc))
            f.write("Train f1 = {}\n".format(f1))
            f.write("Train PS_score = {}\n".format(precision))
            f.write("Train RC_score = {}\n".format(recall))
            f.write(np.array_str(confusion_matrix))
            f.write("\n")
    
    print('Training')
    train(sess, model, data_train, labels_train, load_and_skip_train=load_and_skip_train, epochs=25, save_name='model')

    print('Evaluating on clean data')

    def evaluate_all_metrics(sess, model, input, labels, case="Train"):
        y_pred= predict(sess, model, input)
        Y_true = np.argmax(labels, axis=1)
        Y_pred = np.argmax(y_pred, axis=1)
        auc_score = roc_auc_score(labels[:, 1], y_pred[:, 1], average='micro')
        f1 = f1_score(Y_true, Y_pred)
        ps = precision_score(Y_true, Y_pred)
        rc = recall_score(Y_true, Y_pred)
        cm = confusion_matrix(Y_true, Y_pred)

        print("{} AUC_score = {}".format(case, auc_score))
        print("{} f1 = {}".format(case, f1))
        print("{} PS_score = {}".format(case, ps))
        print("{} RC_score = {}".format(case, rc))
        print(cm)
        write_to_report(auc_score,f1,ps,rc,cm)

    evaluate_all_metrics(sess, model, data_train, labels_train, case="Train")
    evaluate_all_metrics(sess, model, data_test, labels_test, case="Test")


if __name__ == '__main__':

    # The original parameters from the paper
    hidden_dim = 128
    fc_dim = 32
    timesteps = 48

    path = args.data_folder
    tf.compat.v1.set_random_seed(1)
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    run_fold(path, hidden_dim, fc_dim, timesteps)