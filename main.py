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

    @staticmethod
    def get_logits(input_x, input_dim, output_dim, hidden_dim, fc_dim, timesteps):
        with tf.compat.v1.variable_scope('model', reuse=tf.compat.v1.AUTO_REUSE):
            x = tf.unstack(input_x, timesteps, 1)
            rnn_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(hidden_dim)
            outputs, _ = tf.compat.v1.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
            d1_output = tf.compat.v1.layers.dense(outputs[-1], units=fc_dim, activation=tf.nn.relu)
            logits = tf.compat.v1.layers.dense(d1_output, units=output_dim)
        return logits

    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim, timesteps):

        self.x = tf.compat.v1.placeholder(tf.float32, (None, timesteps, input_dim), name='x')
        self.y = tf.compat.v1.placeholder(tf.float32, (None, output_dim), name='y')
        
        logits = TargetModel.get_logits(self.x, input_dim, output_dim, hidden_dim, fc_dim, timesteps)
        self.ybar = tf.nn.softmax(logits)

        count = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.ybar, axis=1))
        self.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

        xent = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=logits)
        self.loss = tf.reduce_mean(xent)

        optimizer = tf.compat.v1.train.AdamOptimizer()
        vs = tf.compat.v1.global_variables()
        self.train_op = optimizer.minimize(self.loss, var_list=vs)

        self.saver = tf.compat.v1.train.Saver()


def cw(get_logits, x, input_dim, output_dim, hidden_dim, fc_dim, timesteps, lamb=0, optimizer=None, min_prob=0):
        xshape = x.get_shape().as_list()
        noise = tf.compat.v1.get_variable('noise', xshape, tf.float32, initializer=tf.initializers.zeros)

        # ISTA
        cond1 = tf.cast(tf.greater(noise, lamb), tf.float32)
        cond2 = tf.cast(tf.less_equal(tf.abs(noise), lamb), tf.float32)
        cond3 = tf.cast(tf.less(noise, tf.negative(lamb)), tf.float32)

        assign_noise = tf.multiply(cond1,tf.subtract(noise,lamb)) + \
                        tf.multiply(cond2, tf.constant(0.0)) + \
                        tf.multiply(cond3, tf.add(noise,lamb))
        setter = tf.compat.v1.assign(noise, assign_noise)

        # Adversarial
        xadv = x + noise
        logits = get_logits(xadv, input_dim, output_dim, hidden_dim, fc_dim, timesteps)
        ybar = tf.nn.softmax(logits)

        ydim = ybar.get_shape().as_list()[1]
        y = tf.argmin(ybar, axis=1, output_type=tf.int32)

        mask = tf.one_hot(y, ydim, on_value=0.0, off_value=float('inf'))
        yt = tf.reduce_max(logits - mask, axis=1)
        yo = tf.reduce_max(logits, axis=1)

        loss0 = tf.nn.relu(yo - yt + min_prob)

        axis = list(range(1, len(xshape)))
        loss1 = tf.reduce_sum((tf.abs(xadv - x)), axis=axis)

        loss = loss0 + lamb * loss1
        train_op = optimizer.minimize(loss, var_list=[noise])

        return train_op, xadv, noise, setter

class AdversarialModel:
    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim, timesteps):
        self.x_fixed = tf.compat.v1.placeholder(tf.float32, (1, timesteps, input_dim), name='x_fixed')
        self.adv_lamb = tf.compat.v1.placeholder(tf.float32, (), name='adv_lamb')
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.02)
        self.adv_train_op, self.xadv, self.noise, self.setter = cw(TargetModel.get_logits, 
                                                                        self.x_fixed, 
                                                                        input_dim, output_dim, hidden_dim, fc_dim, timesteps,
                                                                        lamb=self.adv_lamb, 
                                                                        optimizer=optimizer)


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
    
    def write_to_report(auc,f1,precision,recall,confusion_matrix,case="Train"):
        with open('val_report.txt', 'a') as f:
            f.write("{} AUC_score = {}\n".format(case, auc))
            f.write("{} f1 = {}\n".format(case, f1))
            f.write("{} PS_score = {}\n".format(case, precision))
            f.write("{} RC_score = {}\n".format(case, recall))
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
        write_to_report(auc_score,f1,ps,rc,cm,case=case)

    evaluate_all_metrics(sess, model, data_train, labels_train, case="Train")
    evaluate_all_metrics(sess, model, data_test, labels_test, case="Test")


    print('Generating adversarial data')

    adv_model = AdversarialModel(input_dim, output_dim, hidden_dim, fc_dim, timesteps)

    def make_cw(sess, adv_model, x_data, max_epochs=10000, lamb=0.0):
        print('Making adversarials:')

        xadv = np.empty_like(x_data)
        feed_dict = {adv_model.x_fixed: x_data, adv_model.adv_lamb: lamb}

        sess.run(adv_model.noise.initializer)
        flag = 0    # break if 1
        maxp = 0    # maximum perturbation
        nonz = 0    # number of location changed
        avgp = 0    # average perturbation
        for epoch in range(max_epochs):
            # print('Epoch:', epoch)
            sess.run(adv_model.adv_train_op, feed_dict=feed_dict)
            sess.run(adv_model.setter, feed_dict=feed_dict)
            xadv = sess.run(adv_model.xadv, feed_dict=feed_dict)

            ypred = predict(sess, model, x_data)
            yadv = predict(sess, model, xadv)

            label_ypred = np.argmax(ypred, axis=1)
            label_yadv = np.argmax(yadv, axis=1)

            diff = xadv - x_data
            maxp = np.max(np.max(abs(diff), axis=0))
            nonz = np.count_nonzero(diff)
            avgp = np.sum(np.sum(abs(diff), axis=0))/nonz

            if label_yadv != label_ypred:
                print('Classification changed at Epoch {}!'.format(epoch))
                flag = 1
                print('Maximum perturbation: {:.4f}, Number of cells changed: {:d}, Average pertubation: {:.4f}'.format(maxp, nonz, avgp))
                break
        return diff, maxp, nonz, avgp, flag


    # Identify correct labeled samples
    y_pred= predict(sess, model, data_test)
    Y_true = np.argmax(labels_test, axis=1)
    Y_pred = np.argmax(y_pred, axis=1)
    idx = np.equal(Y_true, Y_pred)
    data_clean = data_test[idx]

    # seq = np.linspace(np.log(0.0005), np.log(0.05), num=40)
    seq = np.linspace(np.log(0.0005), np.log(0.015), num=25)
    lamb_list = np.exp(seq)

    # 5859 x num
    n_obs = data_clean.shape[0]
    n_lamb = len(lamb_list)
    
    MP = np.zeros((n_obs, n_lamb))
    AP = np.zeros((n_obs, n_lamb))
    NZ = np.zeros((n_obs, n_lamb))
    FL = np.zeros((n_obs, n_lamb))

    # Save all the adversarial noise
    noise_advs = []
    noise_advs_0_to_1 = []
    noise_advs_1_to_0 = []

    attack_success = []
    attack_0_to_1 = []

    MP_0_to_1 = []
    AP_0_to_1 = []
    NZ_0_to_1 = []
    FL_0_to_1 = []

    MP_1_to_0 = []
    AP_1_to_0 = []
    NZ_1_to_0 = []
    FL_1_to_0 = []

    for i in range(n_obs):
        print(i)
        # (1, 48, 19)
        x = data_clean[i:i + 1]
        noise_advs_lambs = []
        attack_success_lambs = []
        attack_0_to_1_lambs = []

        MP_lambs = []
        AP_lambs = []
        NZ_lambs = []
        FL_lambs = []
        contains_error = False
        for j in range(len(lamb_list)):
            print(j)
            lamb = lamb_list[j]
            noise_adv, maxp, nonz, avgp, flag = make_cw(sess, adv_model, x, max_epochs=100, lamb=lamb)
            
            MP[i, j] = maxp
            AP[i, j] = avgp
            NZ[i, j] = nonz
            FL[i, j] = flag
            
            array_sum = np.sum(noise_adv)
            array_has_nan = np.isnan(array_sum)
            if array_has_nan:
                print("data error")
                exit(0)
                contains_error = True
            
            noise_advs_lambs.append(noise_adv)
            attack_success_lambs.append(flag)

            MP_lambs.append(maxp)
            AP_lambs.append(avgp)
            NZ_lambs.append(nonz)
            FL_lambs.append(flag)
        
        if not contains_error:
            if Y_pred[i] == 1:
                attack_0_to_1_lambs.append(0)
                noise_advs_1_to_0.append(noise_advs_lambs)
                MP_1_to_0.append(MP_lambs)
                AP_1_to_0.append(AP_lambs)
                NZ_1_to_0.append(NZ_lambs)
                FL_1_to_0.append(FL_lambs)
            else:
                attack_0_to_1_lambs.append(1)
                noise_advs_0_to_1.append(noise_advs_lambs)
                MP_0_to_1.append(MP_lambs)
                AP_0_to_1.append(AP_lambs)
                NZ_0_to_1.append(NZ_lambs)
                FL_0_to_1.append(FL_lambs)
            
            noise_advs.append(noise_advs_lambs)
            attack_0_to_1.append(attack_0_to_1_lambs)
            attack_success.append(attack_success_lambs)

    
    fn = 'noise_advs_total.pkl'
    f = open(fn, 'wb')
    pickle.dump(np.array(noise_advs), f, protocol=2)

    fn = 'noise_advs_0to1.pkl'
    f = open(fn, 'wb')
    pickle.dump(np.array(noise_advs_0_to_1), f, protocol=2)

    fn = 'noise_advs_1to0.pkl'
    f = open(fn, 'wb')
    pickle.dump(np.array(noise_advs_1_to_0), f, protocol=2)

    fn = 'attack_0to1.pkl'
    f = open(fn, 'wb')
    pickle.dump(np.array(attack_0_to_1), f, protocol=2)

    fn = 'attack_success.pkl'
    f = open(fn, 'wb')
    pickle.dump(np.array(attack_success), f, protocol=2)

    fn = 'adv_metrics_0to1.pkl'
    f = open(fn, 'wb')
    pickle.dump([np.array(MP_0_to_1), np.array(AP_0_to_1), np.array(NZ_0_to_1), np.array(FL_0_to_1)], f, protocol=2)

    fn = 'adv_metrics_1to0.pkl'
    f = open(fn, 'wb')
    pickle.dump([np.array(MP_1_to_0), np.array(AP_1_to_0), np.array(NZ_1_to_0), np.array(FL_1_to_0)], f, protocol=2)

    fn = 'adv_metrics.pkl'
    f = open(fn, 'wb')
    pickle.dump([MP, AP, NZ, FL], f, protocol=2)
    print([MP, AP, NZ, FL])


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