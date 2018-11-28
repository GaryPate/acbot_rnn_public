
from keras.models import Model
from keras.layers import Dense, Input, Lambda, LSTM, concatenate, Dropout, GRU, BatchNormalization, LeakyReLU, ReLU, PReLU
from keras import backend as K
from keras.models import load_model
from keras import optimizers as Op
import tensorflow_probability as tfp
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, TensorBoard, History
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import math
from keras.initializers import RandomNormal, Constant
from acbot_config import PARAM_PATH, WIN_SIZE, PAA_NUM, SCORE_COV, LOSS_SHIFT, BATCH, SHUFFLE, FEATURE_NORMALIZE, COV_RNN, LOOK_FWD, EPO, FEATURE_LIST, LEARN_RATE, CELL_UNITS, DENSE_SIZE, DROPOUT, ORDER_MARKETS, HIDDEN_LAYER, NULL_RATIO
from acbot_common import paa, safe_div, pickle_dump, pickle_load, pass_args, dash_logger
from acbot_wrangle import Features
from quickplot import Quickplot
import os
import random as rn
os.environ["PYTHONHASHSEED"] = '0'
np.random.seed(42)
rn.seed(12345)
tf.set_random_seed(1234)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                              inter_op_parallelism_threads=1)

qp = Quickplot().plot_update

# Main class for generating Tensorflow model
class TrainTools:
    def __init__(self, class_ix, parameters, market=None, version=None, look_fwd=1):

        self.look_fwd = look_fwd
        self.feature_normalizer = Features().feature_normalizer
        self.rnn_model = {}
        self.market_list = ORDER_MARKETS
        self.graph = tf.Graph()
        self.parameters = parameters
        self.market = market
        self.version = version
        self.class_ix = None
        self.feat_num = len(parameters['features']['list'])
        self.graph1 = tf.Graph()
        self.graph2 = tf.Graph()
        self.loc_mvn_ = {1: [], 2: []}
        self.scl_mvn_ = {1: [], 2: []}
        self.market_map = {k: n for n, k in enumerate(self.market_list)}
        self.superset = {}
        self.mvn_tens_lb = {}
        self.mvn_tens_op = {}
        self.rn_market = None
        self.market_rn_ls = []
        self.mvn_tens = {}
        self.close_input = {}
        self.epoch_input = {}
        self.hist_save = {1: {}, 2: {}}
        with self.graph1.as_default():
            self.close_input[1] = tf.placeholder(tf.float32, shape=(None, 1))
            self.epoch_input[1] = tf.placeholder(tf.float32, shape=(None, 1))
        with self.graph2.as_default():
            self.close_input[2] = tf.placeholder(tf.float32, shape=(None, 1))
            self.epoch_input[2] = tf.placeholder(tf.float32, shape=(None, 1))
        self.model_build = None
        self.k_init = RandomNormal(seed=1337)
        self.b_init = Constant(value=0.0)

    # Creates tensor dimensions from numpy arrays
    def tensorize(self, features, labels=None, predict=False):

        npa_ = features
        labels_ = labels
        feat_normalize_ = self.parameters['features']['normalize']
        for ix in range(1, npa_.shape[2] - 1):
            npa_c = npa_[:, :, ix]
            print('Tensorize check: {}'.format(feat_normalize_ [ix-1]))
            if feat_normalize_ [ix-1] == 1:
                npa_c = self.feature_normalizer(npa_c, single=True)

            npa_[:, :, ix] = npa_c

        if not predict:
            labels_ = np.reshape(labels_, (labels_.shape[0], LOOK_FWD, 1))
            return npa_, labels_

        else:
            return npa_

    # Undersampling of dominant class label
    def under_sample(self, features, labels):

        feats_ = features
        labels_ = labels

        ix_usamp = np.where(labels_ == 0)[0]
        ix_samp = np.where(labels_ != 0)[0].tolist()

        usample_ratio = self.parameters['usample']
        ix_rn_usamp = np.random.choice(ix_usamp, size=int(labels_.shape[0]*usample_ratio)).tolist()
        ix_ = np.sort(np.array(ix_rn_usamp + ix_samp))

        return feats_[ix_], labels_[ix_]

    # Generates moving window and add as new dimension
    def create_time_sequence(self, npa, labels=None, paa_num=8, win_size=60, predict=False):

        npa_c = []
        label_ls = []

        for c_ix in range(0, npa.shape[1]):
            npa_r = []
            for i in range(0, npa.shape[0] - win_size - 1):
                back_idx = i + win_size
                x_ = npa[:, c_ix][i:back_idx]
                x_paa = paa(x_, paa_num, single=True)[0, :, 0]
                npa_r.append(x_paa)

            npa_c.append(npa_r)

        npa_ = np.array(npa_c)
        npa_ = np.moveaxis(npa_, 0, -1)

        if not predict:

            for i in range(0, npa.shape[0] - win_size - 1):
                back_idx = i + win_size
                fwd_idx = back_idx + self.look_fwd
                y_ = labels[back_idx:fwd_idx]
                label_ls.append(y_)

            return [npa_, np.array(label_ls)]

        else:
            return npa_

    # Orchestrates data transformations before model ingestion
    @pass_args
    def prepare_test_data(self, features=None, labels=None, date=None, paa_num=PAA_NUM, win_size=WIN_SIZE, predict=False):

        features_, labels_ = self.create_time_sequence(features, labels=labels, paa_num=paa_num, win_size=win_size)
        if not predict:
            features_, labels_ = self.under_sample(features_, labels_)

        features_, labels_ = self.tensorize(features_, labels_)

        return features_, labels_, date

    # Enable switching from LSTM to GRU
    def model_unit(self, paa_num, rnn, n, return_state):

        if rnn == 'gru':
            unit = GRU(paa_num, return_sequences=return_state, name='gru{}'.format(n),
                       kernel_initializer=self.k_init,
                       bias_initializer=self.b_init,
                       recurrent_dropout=0.2
                       )
        else:
            unit = LSTM(paa_num, return_sequences=return_state, name='lstm{}'.format(n),
                        kernel_initializer=self.k_init,
                        bias_initializer=self.b_init,
                        recurrent_dropout=0.2
                        )

        return unit

    # Enable switching of activation functions
    def rnn_act(self, act):
        if act == 'prelu':
            return PReLU()
        elif act == 'lrelu':
            return LeakyReLU()
        elif act == 'relu':
            return ReLU()

    # Build and compile Keras model
    def build_model(self, dense_size=DENSE_SIZE, dropout=DROPOUT, cell_units=CELL_UNITS, paa_num=PAA_NUM, hidden=HIDDEN_LAYER, rnn='gru', act1='tanh', act2='prelu'):

        self.close_input[self.class_ix] = Input(shape=(paa_num, 1), name='close_in_model')
        self.epoch_input[self.class_ix] = Input(shape=(paa_num, 1), name='epoch_in_model')

        epoch_input_ = Lambda(lambda x: x[-1, 0])(self.epoch_input[self.class_ix])
        close_input_ = Lambda(lambda x: x[-1, 0])(self.close_input[self.class_ix])
        feat_in_ls = []
        feat_gru_ls = []

        for f in range(self.feat_num):
            input_f = Input(shape=(paa_num, 1), name='feat_{}'.format(f))
            feat_in_ls.append(input_f)

        for i_, f_in in enumerate(feat_in_ls):
            return_state_ = True
            n = '{}{}'.format(i_, 1)
            if hidden == 0:
                return_state_ = False

            output = self.model_unit(paa_num, rnn, n, return_state_)(f_in)
            output = self.rnn_act(act2)(output)

            if hidden > 0:
                if hidden == 1:
                    return_state_ = False

                n = '{}{}'.format(i_, 2)
                output = self.model_unit(paa_num, rnn, n, return_state_)(output)
                output = self.rnn_act(act2)(output)

            if hidden > 1:
                n = '{}{}'.format(i_, 3)
                output = self.model_unit(paa_num, rnn, n, False)(output)
                output = self.rnn_act(act2)(output)

            feat_gru_ls.append(output)

        if len(feat_gru_ls) > 1:
            output = concatenate(feat_gru_ls)
        else:
            output = feat_gru_ls[0]

        output = Dropout(dropout)(output)
        output = BatchNormalization()(output)
        output = Dense(int(dense_size), activation='tanh', name='network1a')(output)
        output = Dropout(dropout)(output)

        output = Dense(2, activation='tanh',
                       kernel_initializer=self.k_init,
                       bias_initializer=self.b_init,
                       name='weightedAverage_output')(output)

        model = Model(inputs=[self.close_input[self.class_ix], *feat_in_ls,  self.epoch_input[self.class_ix]],
                      outputs=[output, close_input_, epoch_input_])

        adad_ = Op.Adadelta()
        model.compile(optimizer=adad_, loss=self.mvn_wrap(self.close_input[self.class_ix],
                                                          self.epoch_input[self.class_ix]), loss_weights=[1., 0.0, 0.0], metrics=['accuracy'])

        self.model_build = model


    def hot_encode(self, x):
        enc = OneHotEncoder()
        u = enc.fit_transform(x).toarray()
        return u

    # Add multi variate normal loss function to superset
    def add_mvn_superset(self, close, epoch, y_true, label_ix, market, cov_mvn=COV_RNN):

        # factor to keep epochs unique
        n = self.market_list.index(market)
        add_ls = ['1'] + ['0' for _ in range(n)]
        add_0 = int(''.join(add_ls))

        y_true_ = y_true
        y_ix = np.where(y_true_ == label_ix)[0]
        tf_close_ = close
        tf_epoch_ = epoch #* add_0
        cls_av = np.mean(tf_close_)

        scl_epoch = 9000 * add_0
        cls_epoch = cls_av / 25

        for y_ in y_ix:
            self.scl_mvn_[label_ix].append([scl_epoch, cls_epoch])
            self.loc_mvn_[label_ix].append([tf_epoch_[y_], tf_close_[y_]])

    # Creates tensor distributions under each TF graph
    def create_mvn_superset(self):

        with self.graph1.as_default():
            self.mvn_tens[1] = {1: tfp.distributions.MultivariateNormalDiag(loc=self.loc_mvn_[1], scale_diag=self.scl_mvn_[1]),
                                2: tfp.distributions.MultivariateNormalDiag(loc=self.loc_mvn_[2], scale_diag=self.scl_mvn_[2])}

        with self.graph2.as_default():
            self.mvn_tens[2] = {1: tfp.distributions.MultivariateNormalDiag(loc=self.loc_mvn_[1], scale_diag=self.scl_mvn_[1]),
                                2: tfp.distributions.MultivariateNormalDiag(loc=self.loc_mvn_[2], scale_diag=self.scl_mvn_[2])}


    # Loss function for model
    def mvn_wrap(self, x_close=tf.placeholder(tf.float32, shape=(None, PAA_NUM, 1), name='wrap_close_placer')
                  , x_epoch=tf.placeholder(tf.float32, shape=(None, PAA_NUM, 1), name='wrap_epoch_placer')):


        print(x_close.shape)
        def mvn_loss(y_true, y_pred):
            _EPSILON = K.epsilon()

            y_1_close_v = tf.reshape(tensor=x_close, shape=(-1, tf.shape(x_close)[0],  tf.shape(x_close)[1]))
            y_1_epoch_v = tf.reshape(tensor=x_epoch, shape=(-1, tf.shape(x_epoch)[0],  tf.shape(x_epoch)[1]))

            y_1_close_full_ = y_1_close_v[:, -1, 0]
            y_1_range_full_ = y_1_epoch_v[:, -1, 0]

            y_1_close_shp = K.shape(y_1_close_full_)[0]
            y_1_range_shp = K.shape(y_1_range_full_)[0]

            y_1_close_full_ = tf.reshape(y_1_close_full_, shape=(y_1_close_shp, ))
            y_1_range_full_ = tf.reshape(y_1_range_full_, shape=(y_1_range_shp, ))

            tf_pred_arr = tf.expand_dims(tf.stack([y_1_range_full_, y_1_close_full_], axis=1), 1)
            tf_pred_arr = tf.cast(tf_pred_arr, tf.float32)

            mvn_tens_p = self.mvn_tens[self.class_ix][self.label_ix]
            mvn_tens_n = self.mvn_tens[self.class_ix][self.opposing_ix]

            v_p = mvn_tens_p.prob(tf_pred_arr)
            v_n = mvn_tens_n.prob(tf_pred_arr)

            m_p = K.max(v_p, axis=1)
            m_n = K.max(v_n, axis=1)

            y_true_ = K.clip(y_true, _EPSILON, 1.0 - _EPSILON)
            y_pred_ = K.clip(y_pred, _EPSILON, 1.0 - _EPSILON)

            return K.binary_crossentropy(y_true_, y_pred_) - tf.expand_dims((tf.multiply(m_p, self.mvn_mult)), 1) + tf.expand_dims((tf.multiply(m_n, self.mvn_mult * 0.6)), 1)

        return mvn_loss

    # Expands dimension
    def fit_expander(self, npa):
        return [np.expand_dims(npa[:, :, ix], 2) for ix in range(npa.shape[2])]

    # Changes labels to 0, 1
    def binarize_set(self, ser, null_lab):

        ser_ = ser[:, 0]#.copy()
        null_ix = np.where(ser_ == null_lab)[0].tolist()

        ser_[null_ix] = 0
        ser_[ser_ != 0] = 1

        return ser_

    # One hot encoding and binarize data
    def prepare_data_for_model(self, X, y, cls):
        if cls == 1:
            y_ = self.binarize_set(y, 2)
        else:
            y_ = self.binarize_set(y, 1)

        X_ = self.fit_expander(X)
        y_ = self.hot_encode(y_.reshape(-1, 1))
        return X_, y_

    # Adds market to superset dict
    def add_to_superset(self, features, labels, date_arr, market):

        pass_dict = {**self.parameters,
                     **{'features': features,
                        'labels': labels,
                        'date': date_arr,
                        'predict': True}}

        subset = self.prepare_test_data(pass_dict)
        self.superset[market] = {'feat': subset[0], 'label': subset[1], 'date': subset[0]}


    # Determining prediction based on probabilities
    def argmax_prob(self, null1, null2, prob1, prob2):
        y_hat = []
        for n1, n2, p1, p2 in zip(null1, null2, prob1, prob2):

            if n1 > p1 and n2 > p2:
                y_hat.append(0)
            elif p1 > n1 and n2 > p2:
                y_hat.append(1)
            elif p1 < n1 and n2 < p2:
                y_hat.append(2)
            elif p1 > p2:
                y_hat.append(1)
            elif p2 > p1:
                y_hat.append(2)
            else:
                y_hat.append(0)

        return y_hat

    # Detects if batch contains a label within it other than 0
    def labelbal(self, x):
        if all(x_ == 0 for x_ in x):
            return 0
        elif any(x_ == 1 for x_ in x):
            return 1
        elif any(x_ == 2 for x_ in x):
            return 2
        else:
            return 0

    # Precalculation of indexes based on batch number. Used in fit generator.
    # Used to undersample majority class
    def index_precalc(self):
        market_pre_dict = {}
        market_rn_ls = []
        for market in self.market_list:

            # Get market and split idx length
            subset = self.superset[market]
            sp_ix = int(subset['label'].shape[0] * 0.5)

            # Calc lengths of train and val
            len_train = subset['feat'][:sp_ix].shape[0]
            len_val = subset['feat'][sp_ix:].shape[0]

            # Subdivide for number of batches
            self.steps_batch_train = math.floor(len_train/BATCH)
            self.steps_batch_val = math.floor(len_val/BATCH)

            # Create sequences of index
            ix_train = np.arange(self.steps_batch_train*BATCH)
            ix_val = np.arange(sp_ix, sp_ix + (self.steps_batch_val*BATCH))

            # Split into batches
            np_train_split = np.split(ix_train, self.steps_batch_train)
            np_val_split = np.split(ix_val, self.steps_batch_val)
            lab_train_check = np.split(np.squeeze(subset['label'])[ix_train], self.steps_batch_train)
            lab_val_check = np.split(np.squeeze(subset['label'])[ix_val], self.steps_batch_val)
            labelled_train = np.apply_along_axis(self.labelbal, 1, lab_train_check).tolist()
            labelled_val = np.apply_along_axis(self.labelbal, 1, lab_val_check).tolist()

            market_pre_dict[market] = {'train': [[t, l] for t, l in zip(np_train_split, labelled_train)],
                                       'val': [[t, l] for t, l in zip(np_val_split, labelled_val)]}

        lab_cnt1 = 0
        lab_cnt2 = 0
        null_cnt = 0
        label_limit = int(EPO/(NULL_RATIO*2))
        i = 0

        # Performs random selection based on indexs for 0, 1 and 2 labels
        while i < EPO:
            rn_market = rn.choice(self.market_list)
            rn_train = rn.choice(market_pre_dict[rn_market]['train'])
            rn_val = rn.choice(market_pre_dict[rn_market]['val'])

            is_lab1 = False
            is_lab2 = False

            if rn_train[1] and rn_val[1] == 1:
                is_lab1 = True

            elif rn_train[1] and rn_val[1] == 2:
                is_lab2 = True

            if is_lab1 and lab_cnt1 <= label_limit:
                lab_cnt1 += 1
                market_rn_ls.append([rn_market, rn_train[0], rn_val[0]])
            elif is_lab2 and lab_cnt2 <= label_limit: # or rn_val[1] == 2:
                lab_cnt2 += 1
                market_rn_ls.append([rn_market, rn_train[0], rn_val[0]])
            elif null_cnt <= (EPO - label_limit) * 1.03:
                market_rn_ls.append([rn_market, rn_train[0], rn_val[0]])
                null_cnt += 1

            i += 1

        # Shuffles final set
        rn.shuffle(market_rn_ls)

        return market_rn_ls

    # Generator function to load training data into TF model at runtime
    def batch_generator(self, validation=False):
        while True:
            for rn_ in self.market_rn_ls:
                subset = self.superset[rn_[0]]
                feats_ = subset['feat'][rn_[1]]
                labels_ = subset['label'][rn_[1]][:, 0]
                feats_v = subset['feat'][rn_[2]]
                labels_v = subset['label'][rn_[2]][:, 0]

                if validation:
                    X_train_exp, y_train_hot = self.prepare_data_for_model(feats_v, labels_v, self.class_ix)
                else:
                    X_train_exp, y_train_hot = self.prepare_data_for_model(feats_, labels_, self.class_ix)

                yield X_train_exp, [y_train_hot, X_train_exp[0][:, -1, :], X_train_exp[-1][:, -1, :]]

    # Orchestrates model building and training of Keras in it's own graph
    def fit_training_set(self, class_ix):
        tf.reset_default_graph()
        if class_ix == 1:
            graph = self.graph1
        else:
            graph = self.graph2

        # Load TF graph for each label
        with graph.as_default():
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                self.class_ix = class_ix
                training_generator = self.batch_generator()
                validation_generator = self.batch_generator(validation=True)
                sub_param = {k: self.parameters[k] for k in ('dropout', 'dense_size', 'cell_units', 'paa_num', 'hidden', 'rnn', 'act1', 'act2')}
                self.mvn_mult = self.parameters['mvn_mult']

                if self.class_ix == 1:
                    self.label_ix = 1
                    self.opposing_ix = 2

                if self.class_ix == 2:
                    self.label_ix = 2
                    self.opposing_ix = 1

                self.build_model(**sub_param)

                filepath = PARAM_PATH + "{}_model_{}.h5".format('any', self.class_ix)

                # Callbacks during model
                checkpoint_ = ModelCheckpoint(filepath, monitor='weightedAverage_output_loss', verbose=1, save_best_only=False)
                tboard_ = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True,
                                     embeddings_freq=0, embeddings_layer_names=True, embeddings_metadata=True, embeddings_data=True )
                history_ = History()
                callbacks_list = [checkpoint_, history_, tboard_]

                self.model_build.fit_generator(generator=training_generator,
                                         validation_data=next(validation_generator),
                                         epochs=EPO,
                                         use_multiprocessing=True,
                                         shuffle=SHUFFLE,
                                         callbacks=callbacks_list,
                                         steps_per_epoch=self.steps_batch_train,
                                         validation_steps=self.steps_batch_val,
                                               class_weight={0: 1, 1: 2}
                                         )

                # Save metrics to dict
                for key in ['loss', 'val_loss', 'weightedAverage_output_acc', 'val_weightedAverage_output_acc']:
                    self.hist_save[class_ix][key] = history_.history[key]

        K.clear_session()

    # Orchestrates prediction on unseen test data
    def predict_on_model(self, grid, X_test, y_test):
        y_prob_ls = {1: [], 2: []}
        y_hat_ls = {1: [], 2: []}
        for class_ix in [1, 2]:
            graph_ = tf.get_default_graph()
            with graph_.as_default():
                with tf.Session() as sess:
                    sess.run(tf.global_variables_initializer())
                    X_test_exp, y_test_hot = self.prepare_data_for_model(X_test, y_test, class_ix)
                    self.close_input = X_test_exp[0]
                    self.epoch_input = X_test_exp[-1]

                    if class_ix == 1:
                        self.label_ix = 1
                        self.opposing_ix = 2
                    else:
                        self.label_ix = 2
                        self.opposing_ix = 1

                    self.load_model_from_file(class_ix, grid=grid)
                    y_hat, y_prob = self.pred_rnn(self.rnn_model[class_ix], [*X_test_exp])
                    print('Predicted values: {}'.format(y_hat[y_hat != 0].shape[0]))
                    y_prob_ls[class_ix] = y_prob
                    y_hat_ls[class_ix] = y_hat

            K.clear_session()

        self.rnn_model[1].summary(print_fn=lambda x: dash_logger(x, self.version, self.market))
        pred = self.argmax_prob(y_prob_ls[1][:, 0],
                                y_prob_ls[2][:, 0],
                                y_prob_ls[1][:, 1],
                                y_prob_ls[2][:, 1])

        return pred, y_prob_ls


    # Load data from saved file for prediction
    def load_model_from_file(self, class_ix, market ='any', grid=''):

        path_construct = PARAM_PATH + "{}_model_{}{}.h5".format(market, class_ix, grid)
        self.rnn_model[class_ix] = load_model(path_construct,
                                              custom_objects={'mvn_loss': self.mvn_wrap()},
                                              compile=False)

