
from acbot_api import ExchangeConn
from acbot_rnn_module import TrainTools
from acbot_wrangle import Features
from acbot_common import pass_args, pickle_dump, pickle_load
import numpy as np
from sklearn.preprocessing import StandardScaler


# Base class for orchestrating model creation
class DeepRnn(object):

    def __init__(self, parameters, market_list):
        self.parameters = parameters
        self.market_list = market_list
        self.market = None
        self.indicator_dict = None
        self.ranges = None
        self.session_id = None
        self.version = 'V00'
        self.range_key = None
        self.range_point = None

    # overide for splitting data based on training or test
    def range_key_point(self):
        return None, None

    # Session Id used for saving the training/test to database
    def set_session_id(self):
        self.session_id = self.ranges['set1']
        state, _ = self.range_key_point
        if state == 'test_split':
            if 'set2' in self.ranges:
                if self.ranges['set2']:
                    self.session_id = self.ranges['set2']

        if 'version' in self.ranges:
            if self.ranges['version']:
                self.version = self.ranges['version']

    # Determines which of the ranges to split on
    def threshold_split(self):
        thresh_ = self.ranges[self.range_key]

        if self.range_key == 'train_split':
            if thresh_ < self.range_point:
                return True

        if self.range_key == 'test_split':
            if thresh_ > self.range_point:
                return True

        return False

    # Split data based on ranges and return the split sets
    def split_feature_sets(self, features_all, labels_all, date_arr):
        date_arr_ = date_arr[-features_all.shape[0]:]
        self.range_key, self.range_point = self.range_key_point
        if self.threshold_split():
            split_ix = int(labels_all.shape[0] * self.ranges[self.range_key])
            if self.range_key == 'train_split':
                split_features_ = features_all[:split_ix]
                split_labels_ = labels_all[:split_ix]
                split_date_ = date_arr_[:split_ix]
            else:
                split_features_ = features_all[split_ix:]
                split_labels_ = labels_all[split_ix:]
                split_date_ = date_arr_[split_ix:]

            return split_features_, split_labels_, split_date_

        else:
            return features_all, labels_all, date_arr_

    # Query database and update indicators. If generate set TRUE, indicators are are generated from API
    def indicators(self, session_id, generate=False):
        exc = ExchangeConn(self.market_list, parameters=self.parameters)
        if generate:
            exc.process_and_load_indicators(session_id, realtime=True)
        self.indicator_dict = exc.get_indicators_mongo('predict', session_id)

    # Iterates each market through a function passed in as argument
    def market_iterator(self, procedure):
        for market in self.market_list:
            self.market = market
            npa_5, npa_30, npa_hr, date_arr = self.indicator_dict[self.market]
            procedure(npa_5, npa_30, npa_hr, date_arr)

    # Save run arguments to a pickle for loading in elsewhere outside of Class
    def save_run_args(self, version=None, session_id=None, train_on=False, record=False, hold_out=0, testset=None,
                      mode='live', parameters={}, grid=''):

        run_args = {'train': train_on,
                    'session_id': session_id,
                    'hold_out': hold_out,
                    'version': version,
                    'testset': testset,
                    'record': record,
                    'mode': mode,
                    'parameters': parameters,
                    'grid': grid
                    }

        pickle_dump(run_args, 'run_args')

    # Perform pre-scaling of raw features. If fit set to True, scales are saved as pickle for processing test data
    def feature_prescale(self, npa, fit=True):
        feat_normalize_ = self.parameters['features']['normalize']
        feat_labels_ = self.parameters['features']['list']
        npa_ = npa.copy()
        if fit:
            fit_dict = {}
        else:
            fit_dict = pickle_load('{}_scales'.format(self.market))

        for ix in range(1, npa_.shape[1]- 1):
            lab_ = feat_labels_[ix - 1]
            if feat_normalize_[ix - 1] == 2:
                if fit:
                    scale_ser = StandardScaler()
                    scaled_ = scale_ser.fit(npa_[:, ix].reshape(-1, 1))
                    npa_[:, ix] = scaled_.transform(npa_[:, ix].reshape(-1, 1))[:, 0]
                    fit_dict[lab_] = scaled_

                else:
                    scaled_ = fit_dict[lab_]
                    npa_[:, ix] = scaled_.transform(npa_[:, ix].reshape(-1, 1))[:, 0]

        if fit:
            pickle_dump(fit_dict, '{}_scales'.format(self.market))

        return npa_


# Child class for testing on unseen data
class Test(DeepRnn):

    def __init__(self, parameters, market_list):
        super().__init__(parameters, market_list)
        self.predict_test_dict = {}

    # Main method called from class
    def test(self, ranges):
        self.ranges = ranges
        self.set_session_id()
        self.save_run_args(version=self.session_id, session_id=self.session_id)
        self.indicators(self.session_id)
        self.market_iterator(self.predict_new_set)
        self.save_test_to_db()

    @property
    def range_key_point(self):
        return 'test_split', 0

    # Orchestrates test sequence
    def predict_new_set(self, npa_5, npa_30, npa_hr, date_arr):

        pass_dict = {**self.parameters,
                     **{'npa_5': npa_5,
                        'npa_30': npa_30,
                        'npa_hr': npa_hr,
                        'market': self.market,
                        'train_on': True}}

        features_all, labels_all, ser_cca = Features().create_features(pass_dict)
        features_all = self.feature_prescale(features_all, fit=False)
        features_, labels_, date_arr_ = self.split_feature_sets(features_all, labels_all, date_arr)
        rnn_ = TrainTools(0, self.parameters, self.market, self.version)

        pass_dict = {**self.parameters,
                     **{'features': features_,
                        'labels': labels_,
                        'date': date_arr_,
                        'predict': True}}

        features, labels, _ = rnn_.prepare_test_data(pass_dict)
        y_pred, y_probs = rnn_.predict_on_model('', features, labels)
        date_trim_ = date_arr_[-len(y_pred):]
        self.predict_test_dict[self.market] = {'close': features[:, -1, 0], 'date': date_trim_,
                                               'pred': y_pred, 'labels': np.squeeze(labels_),
                                               'null1': y_probs[1][:, 0], 'prob1': y_probs[1][:, 1],
                                               'null2': y_probs[2][:, 0], 'prob2': y_probs[2][:, 1],
                                               'FT1': features_[:, 1], 'FT2': features_[:, 2], 'FT3': features_[:, 3],
                                               'FT4': features_[:, 4], 'FT5': features_[:, 5], 'FT6': features_[:, 6],
                                               'FT7': features_[:, 7]
                                               }

    # Save test data to MongoDB
    def save_test_to_db(self):
        exc = ExchangeConn(self.market_list)
        exc.write_test_set_to_mongo(self.predict_test_dict, self.version)
        exc.write_model_summary_to_mongo(self.version)


# Child class for building superset from all time series and training in Keras
class SuperTrain(DeepRnn):

    def __init__(self, parameters, market_list):
        super().__init__(parameters, market_list)
        self.train_load_dict = {}
        self.train_rnn = TrainTools(0, self.parameters)

    # Main method called from class
    def train(self, ranges, generate=False):
        self.ranges = ranges
        self.set_session_id()
        self.save_run_args(self.session_id)
        self.indicators(self.session_id, generate=generate)
        self.market_iterator(self.prepare_features_and_labels)
        self.super_train()
        self.save_train_to_db()
        print('Done Supertrain')

    @property
    def range_key_point(self):
        return 'train_split', 1

    # Orchestrates training sequence
    def prepare_features_and_labels(self, npa_5, npa_30, npa_hr, date_arr):
        pass_dict = {**self.parameters,
                     **{'npa_5': npa_5,
                        'npa_30': npa_30,
                        'npa_hr': npa_hr,
                        'market': self.market,
                        'train_on': True}
                     }

        features_all, labels_all, ser_cca = Features().create_features(pass_dict)
        features_all = self.feature_prescale(features_all)
        features_, labels_, date_arr_ = self.split_feature_sets(features_all, labels_all, date_arr)
        features_[:, -1] = self.factor_epochs(features_[:, -1], self.market)
        cov_mvn_ = self.parameters['cov_mvn']

        self.train_load_dict[self.market] = {'close': features_[:, 0], 'date': date_arr_, 'labels': labels_}
        self.train_rnn.add_mvn_superset(features_[:, 0], features_[:, -1], labels_, 1, self.market, cov_mvn_)
        self.train_rnn.add_mvn_superset(features_[:, 0], features_[:, -1], labels_, 2, self.market, cov_mvn_)
        self.train_rnn.create_mvn_superset()
        self.train_rnn.add_to_superset(features_, labels_, date_arr_, self.market)

    # Factor to keep epochs unique between all sets within supersets
    def factor_epochs(self, epoch, market):
        n = self.market_list.index(market)
        add_ls = ['1'] + ['0' for _ in range(n)]
        add_0 = int(''.join(add_ls))
        factored_epoch = epoch * add_0
        return factored_epoch

    # Perform training on model
    def super_train(self):
        self.train_rnn.market_rn_ls = self.train_rnn.index_precalc()
        for class_ix in [1, 2]:
            self.train_rnn.fit_training_set(class_ix)

    # Save training data to MongoDB
    def save_train_to_db(self):
        exc = ExchangeConn(self.market_list)
        exc.write_train_set_to_mongo(self.train_load_dict, self.train_rnn.hist_save, self.version)
