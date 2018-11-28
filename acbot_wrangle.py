import numpy as np
from numpy import inf
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from acbot_config import COLUMNS, INDICATOR_DAYS, LABEL_PAD, LABEL_THRESH, SCORE_COV, FEAT_STRATEGY, LABEL_WIN, LOOK_FWD, FEATURE_LIST, FEATURE_NORMALIZE, FEATURE_LIMIT, ICA_TIME_INT, X_SHIFT, MULTI_SMOOTH, PAA_NUM
from acbot_common import dash_logger, safe_div, pickle_dump, pickle_load, time_test, pass_args
from math import ceil
import time
from numpy.linalg import norm
import logging
from sklearn.ensemble import ExtraTreesClassifier
from dtaidistance import dtw
from sklearn.decomposition import FastICA
from quickplot import Quickplot
from sklearn.cross_decomposition import CCA
from scipy.stats import multivariate_normal as mvn
import os
import random as rn
from scipy.stats import kurtosis, skew
from scipy.special import entr
from tslearn.piecewise import PiecewiseAggregateApproximation
os.environ["PYTHONHASHSEED"] = '0'
np.random.seed(42)
rn.seed(12345)

qp = Quickplot().plot_update


# Common functions used in other classes
class CommonFunctions:

    # Calculate first two principal components of a multi dimensional array
    def pca_calc(self, npa, features=None, n_comp=2, scale=False, expvar=False):

        if features:
            npa_pc = np.copy(npa[:, features])
        else:
            npa_pc = npa

        if scale:
            scaler = StandardScaler().fit(npa_pc)
            scaled = scaler.transform(npa_pc)
        else:
            scaled = npa_pc

        pca_ = PCA(n_components=n_comp, random_state=5, svd_solver='full')
        decomposed = pca_.fit_transform(scaled)

        if n_comp == 1:
            PC1 = [p[0] for p in decomposed]
            ret = [PC1]
        if n_comp == 2:
            PC1 = [p[0] for p in decomposed]
            PC2 = [p[1] for p in decomposed]
            ret = [PC1, PC2]
        elif n_comp == 3:
            PC1 = [p[0] for p in decomposed]
            PC2 = [p[1] for p in decomposed]
            PC3 = [p[2] for p in decomposed]
            ret = [PC1, PC2, PC3]

        if expvar:
            return np.column_stack(ret), pca_.explained_variance_ratio_
        else:
            return np.column_stack(ret)

    # Checks a series and inverts based on similarity to another
    def inv_check(self, comp, close):

        comp = np.array(comp)
        idx_pos = np.where(comp >= 0)
        idx_neg = np.where(comp < 0)
        diff_close = np.diff(close, n=1, axis=-1)
        diff_close = np.pad(diff_close, (1, 0), 'constant', constant_values=(0, 0))
        sum_pos = sum(diff_close[idx_pos])
        sum_neg = sum(diff_close[idx_neg])

        if sum_pos < sum_neg:
            comp = comp * -1

        return comp.tolist()

    # Calculates Independent Component Analysis
    def ica_calc(self, npa_ica, features=None, n_comp=2, orient=False, close=None, scale=True):

        inv_check = self.inv_check

        if scale:
            scaler = StandardScaler().fit(npa_ica)
            npa_ica = scaler.transform(npa_ica)

        ica_ = FastICA(n_components=n_comp, max_iter=500, tol=1e-3)
        decompose = ica_.fit_transform(npa_ica)
        ret_list = []

        for n in range(n_comp):
            np_ica = np.array([p[n] for p in decompose])
            if orient:
                ica_inv = inv_check(np_ica, close)
                ret_list.append(ica_inv)
            else:
                ret_list.append(np_ica)

        return np.column_stack(ret_list)

    # Calculates Canonical Correlation Analysis
    def cca_calc(self, npa, labels):
        from sklearn.cross_decomposition import CCA
        close = npa[:, 0]
        cca_stack = [close]
        for ix in range(1, npa.shape[1]):
            ser = npa[:, ix]
            cca = CCA(n_components=1)
            cca_res = cca.fit(ser.reshape(-1, 1), close.reshape(-1, 1)).transform(ser.reshape(-1, 1))
            cca_stack.append(cca_res)
        return np.column_stack(cca_stack)

    # Calculated area under the curve based on a midpoint value
    def auc_calc(self, proc_list, midpoint=-0.005, reverse=False, shift=False):

        auc_list = []
        auc = 0

        for p in range(len(proc_list)):

            if reverse:
                mid_shift = np.mean(proc_list[-50:])
                if shift:
                    truemid = midpoint - ((abs(abs(midpoint) - abs(mid_shift))) * shift)
                else:
                    truemid = midpoint

                if proc_list[p] <= truemid and proc_list[p] < proc_list[p-1]:
                    auc += abs(proc_list[p])
                    auc_list.append(auc)
                elif proc_list[p] <= truemid and proc_list[p] >= proc_list[p - 1]:
                    auc_list.append(auc)
                elif proc_list[p] > truemid and proc_list[p] > proc_list[p - 1]:
                    auc = 0
                    auc_list.append(0)
                else:
                    auc = 0
                    auc_list.append(0)

            else:
                mid_shift = np.mean(proc_list[-50:])
                if shift:
                    truemid = midpoint + ((abs(abs(midpoint) - abs(mid_shift))) * shift)
                else:
                    truemid = midpoint

                if proc_list[p] >= truemid and proc_list[p] > proc_list[p - 1]:
                    auc += abs(proc_list[p])
                    auc_list.append(auc)
                elif proc_list[p] >= truemid and proc_list[p] <= proc_list[p-1]:
                    auc_list.append(auc)
                else:
                    auc = 0
                    auc_list.append(0)
        return np.array(auc_list)

    # Exponentialy weighted moving average
    def np_ewma(self, data, window):
        df = pd.DataFrame(data)
        return df.ewm(span=window).mean().values[:, 0]

    # Limits a series between two upper bounds
    def soft_limit(self, ser, bnd=2, single=False, only_low=False):

        if single:
            cmax = np.max(ser) #+ 2
            cmin = np.min(ser) #- 2

            transmax = ser[ser > bnd]
            transmin = ser[ser < -bnd]

            if only_low:
                ser[ser < -bnd] = - bnd - ((transmin + bnd) / (cmin))
            else:
                ser[ser > bnd] = bnd + ((transmax - bnd) / (cmax))
                ser[ser < -bnd] = - bnd - ((transmin + bnd) / (cmin))
            return ser

        else:
            lim_ls = []
            for s in range(ser.shape[1]):
                ccalim = ser[:, s]
                cmax = np.max(ccalim) + 2
                cmin = np.min(ccalim) - 2

                transmax = ccalim[ccalim > bnd]
                transmin = ccalim[ccalim < -bnd]

                if only_low:
                    ccalim[ccalim < -bnd] = - bnd - ((transmin + bnd) / (cmin))
                else:
                    ccalim[ccalim > bnd] = bnd + ((transmax - bnd) / (cmax))
                    ccalim[ccalim < -bnd] = - bnd - ((transmin + bnd) / (cmin))
                lim_ls.append(ccalim)
            return np.column_stack(lim_ls)


# Class for generating technical indicators from ray OHLC data
class Indicators:

    def __init__(self):
        self.comm = CommonFunctions()
        self.close_ = COLUMNS['close']
        self.high_ = COLUMNS['high']
        self.low_ = COLUMNS['low']
        self.open_ = COLUMNS['open']
        self.epoch_ = COLUMNS['epoch']
        self.vol_ = COLUMNS['vol']
        self.date_ = COLUMNS['date']
        self.macd_ = COLUMNS['macd']
        self.proc_ = COLUMNS['proc']
        self.pvo_ = COLUMNS['pvo']
        self.rsi_ = COLUMNS['rsi']
        self.stoch_ = COLUMNS['stoch']
        self.boll_ = COLUMNS['boll']

        self.ica_calc = CommonFunctions().ica_calc
        self.inv_check = CommonFunctions().inv_check
        self.cca_calc = CommonFunctions().cca_calc
        self.pca_calc = CommonFunctions().pca_calc
        self.soft_limit = CommonFunctions().soft_limit

        self.axis_lock = Features().pca_axis_lock
        self.np_ewma = CommonFunctions().np_ewma

    # Creates indicators from pandas DataFrame
    def make_indicators(self, data_source, time_int, path=''):

        if isinstance(data_source, pd.DataFrame):
            npa_raw = data_source.as_matrix()
        else:
            npa_raw = pd.DataFrame.from_csv(path + data_source, index_col=None).as_matrix()

        self.tail = 100
        length_day = INDICATOR_DAYS
        len_variable = length_day * 86400
        epoch_ = self.epoch_

        maxtime = np.max(npa_raw[:, epoch_])
        latest = npa_raw[np.where(npa_raw[:, epoch_] == maxtime)][0][epoch_]

        # Epoch time minus a day
        idx_back = latest - len_variable
        # Index of the days
        where_idx = (np.where(npa_raw[:, epoch_] > idx_back)[0])
        first_idx = max(where_idx)
        last_idx = min(where_idx) - self.tail
        range_idx = np.arange(last_idx, first_idx+1)
        npa_tail = npa_raw[range_idx]
        npa_raw = npa_tail
        npa_raw = self.add_vector_indicators_np(npa_raw)

        self.npa = np.copy(npa_raw)
        self.npa = self.trim_range(self.npa, length_day, 0)

        if time_int == 'fiveMin':
           # Generates PCA trend for 5 minute interval data
           pca_tr = self.orig_pca_trend()
           self.npa = np.hstack([self.npa, pca_tr])

        trim_tail = self.npa.shape[0] - self.tail
        self.npa = self.npa[-trim_tail:, :self.npa.shape[1]]

        return self.npa

    # Rounds to 5 decimal places
    def ica_round(self, x, base=5):
        return int(base * round(float(x) / base))

    # Trend analysis based on cumulative area under the curve algorithm
    def orig_pca_trend(self):

        POS_A = 1
        POS_B = 0.5
        NEG_A = 3
        NEG_B = 1.5

        # Generates PCA trend indicator
        features = [self.proc_, self.pvo_, self.rsi_, self.stoch_]
        pca_tr = self.comm.pca_calc(self.npa, features)
        pc1_tr, pc2_tr = pca_tr[:, 0], pca_tr[:, 1]
        pc1_tr = np.array(pc1_tr)
        pc1_tr = self.axis_lock(pc1_tr, self.npa[:, self.close_])
        pca_trend = self.np_ewma(np.array(pc1_tr), 30)

        pca_mid = (np.max(pca_trend) + np.min(pca_trend)) / 2
        pca_sd = np.std(pca_trend)
        pos_mid = pca_mid - (pca_sd * 2)
        neg_mid = pca_mid + (pca_sd * 2)

        auc_pca_pos_a = self.comm.auc_calc(proc_list=pca_trend, midpoint=pos_mid, shift=POS_A)  # <<
        auc_pca_neg_a = self.comm.auc_calc(proc_list=pca_trend, midpoint=neg_mid, reverse=True, shift=NEG_A)  # <<
        auc_pca_pos_b = self.comm.auc_calc(proc_list=pca_trend, midpoint=pos_mid, shift=POS_B)  # <<
        auc_pca_neg_b = self.comm.auc_calc(proc_list=pca_trend, midpoint=neg_mid, reverse=True, shift=NEG_B)  # <<

        pca_tr_a_pos = np.array(auc_pca_pos_a).reshape([len(auc_pca_pos_a), 1])
        pca_tr_a_neg = np.array(auc_pca_neg_a).reshape([len(auc_pca_neg_a), 1])
        pca_tr_b_pos = np.array(auc_pca_pos_b).reshape([len(auc_pca_pos_b), 1])
        pca_tr_b_neg = np.array(auc_pca_neg_b).reshape([len(auc_pca_neg_b), 1])

        return np.column_stack([pca_tr_a_pos, pca_tr_a_neg, pca_tr_b_pos, pca_tr_b_neg])

    # Trims range of time series based on max lookback time
    def trim_range(self, npa, start, end):
        np_epoch = npa[:, 5]  # .astype(int)
        maxtime = np.max((np_epoch))
        latest_epoch = npa[np.where(np_epoch == maxtime)][0][5]
        start_epoch = int(latest_epoch) - (abs(start) * 86400)
        end_epoch = int(latest_epoch) - (abs(end) * 86400)
        where_idx = np.where((np_epoch >= start_epoch) & (np_epoch <= end_epoch))

        return npa[where_idx]

    # Compiles indicators that have vectorized operations
    def add_vector_indicators_np(self, npa):

        close_ = self.close_
        vol_ = self.vol_

        newcols = []
        newcols.append(self.macd_np(npa[:, close_]))
        newcols.append(self.price_roc_np(npa[:, close_]))
        newcols.append(self.pvo_np(npa[:, vol_]))

        rsi = self.rsi_np(npa[:, close_], 14)
        newcols.append(rsi)
        newcols.append(self.stochastic_rsi(rsi))
        newcols.append(self.bolinger_bands(npa[:, close_]))

        for new in newcols:
            new_ = np.reshape(new, (new.shape[0], 1))
            npa = np.column_stack((npa, new_))

        npa[0:3] = 0

        return npa

    # Stochastic RSI
    def stochastic_rsi(self, rsi, n=14):
        rsi = pd.Series(rsi)

        def STOD(rsi, n):
            sto_rsi = ((rsi -rsi.rolling(n).min()) / (rsi.rolling(n).max() - rsi.rolling(n).min())) * 100
            return sto_rsi.fillna(method='bfill')

        return STOD(rsi, n).as_matrix()

    # Histogram RSI
    def hist_rsi(self, rsi):
        np_9 = self.np_ewma(rsi, 9)
        np_26ema = self.np_ewma(rsi, 26)
        np_12ema = self.np_ewma(rsi, 12)
        np_rsi_hist = np_12ema - np_26ema
        return np.nan_to_num(np_rsi_hist - np_9) * -1

    # Moving average Convergence Divergence
    def macd_np(self, np_wa):
        np_26ema = self.np_ewma(np_wa, 26)
        np_12ema = self.np_ewma(np_wa, 12)
        np_macd = np_12ema - np_26ema
        np.nan_to_num(np_macd)
        np_9_signal = self.np_ewma(np.nan_to_num(np_macd), 9)
        np_macd_hist = np_macd - np_9_signal

        return np.nan_to_num(np_macd_hist)

    # Relative Strength Indicator
    def rsi_np(self, np_close, window_length):

        delta = np.diff(np_close)
        delta = delta[1:]

        # Make the positive gains (up) and negative gains (down) Series
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        roll_up1 = self.np_ewma(up, window_length)
        roll_down1 = self.np_ewma(np.abs(down), window_length)

        # Calculate the RSI based on EWMA
        RS1 = roll_up1/roll_down1
        RSI1 = 100.0 - (100.0 / (1.0 + RS1))
        RSI1 = np.insert(RSI1, 0, 0)
        RSI1 = np.insert(RSI1, 1, 0)
        RSI1 = pd.Series(RSI1)
        RSI1 = RSI1.rolling(3).mean().as_matrix()

        return RSI1

    # Price volume oscillator
    def pvo_np(self, vol):
        vol_26ema = self.np_ewma(vol, 26)
        vol_12ema = self.np_ewma(vol, 12)
        np_pvo = ((vol_12ema - vol_26ema)/vol_26ema)
        np.nan_to_num(np_pvo)
        return np_pvo

    # On balance volume
    def obv_np(self, close, vol):
        diff_close = np.diff(close)
        sign_close = np.sign(diff_close)
        sign_vol = vol[1:] * sign_close
        obv = np.nan_to_num(np.cumsum(sign_vol))
        obv = np.pad(obv, (1, 0), 'constant', constant_values=(0, 0))
        obv_osc = self.np_ewma(obv, 9) - self.np_ewma(obv, 30)
        return obv_osc

    # Price Rate of Change
    def price_roc_np(self, close):
        close = pd.Series(close)

        def calc_proc(close):
            return (close[-1] - close[-14]) / close[-14]

        return close.rolling(14).apply(calc_proc).fillna(method='bfill').as_matrix()

    # Stochastic oscillator
    def stochastic_np(self, close, low, high):

        close = pd.Series(close)
        low = pd.Series(low)
        high = pd.Series(high)

        def STOD(close, low, high, n):
            STOK = ((close - pd.rolling_min(low, n)) / (pd.rolling_max(high, n) - pd.rolling_min(low, n))) * 100
            STOD = pd.rolling_mean(STOK, 5)

            return STOD.fillna(method='bfill')

        return STOD(close, low, high, 14).as_matrix()

    # Bollinger bands
    def bolinger_bands(self, close, win_size=30, n_std=2):

        ser = pd.DataFrame(close)
        rolling_mean = ser.rolling(window=win_size).mean()
        rolling_std = ser.rolling(window=win_size).std()
        upper_band = rolling_mean + (rolling_std * n_std)
        lower_band = rolling_mean - (rolling_std * n_std)
        band_width = upper_band - lower_band

        return band_width


# Class for generating features from Indicators loaded from DB
class Features:
    def __init__(self):
        self.pwl_win = 80
        self.pwl_thresh = 1
        self.close_ = COLUMNS['close']
        self.high_ = COLUMNS['high']
        self.low_ = COLUMNS['low']
        self.open_ = COLUMNS['open']
        self.epoch_ = COLUMNS['epoch']
        self.vol_ = COLUMNS['vol']
        self.date_ = COLUMNS['date']
        self.macd_ = COLUMNS['macd']
        self.proc_ = COLUMNS['proc']
        self.pvo_ = COLUMNS['pvo']
        self.rsi_ = COLUMNS['rsi']
        self.stoch_ = COLUMNS['stoch']
        self.boll_ = COLUMNS['boll']
        self.tr_a_p_ = COLUMNS['trend_a_p']
        self.tr_a_n_ = COLUMNS['trend_a_n']
        self.tr_b_p_ = COLUMNS['trend_b_p']
        self.tr_b_n_ = COLUMNS['trend_b_n']
        self.comm = CommonFunctions()

    # performs data transformation prior to generating labels
    def create_label_inflections(self, plus=2, end_w=200, thresh_m=0.01, arr_ser=None):

        close_ = self.close_
        ewma = self.comm.np_ewma
        smooth = 12

        if arr_ser is not None:
            labels = self.turning_points(arr_ser, end_w=end_w, thresh_mult=thresh_m)
        else:
            labels = self.turning_points(ewma(self.npa5[:, close_], smooth), end_w=end_w, thresh_mult=thresh_m)

        return self.minmax_trans(self.minmax_fill(labels), plus=plus)

    # Function to scale the close time series. Required for DTW
    def feature_normalizer(self, ser, single=False):

        scale_ser = StandardScaler()

        if single:
            for ix in range(ser.shape[0]):
                ser[ix, :] = scale_ser.fit_transform(ser[ix, :].reshape(-1, 1))[:, 0]
            return ser

        else:
            for ix_ in range(ser.shape[2]):
                ser_ = ser[:, :, ix_]
                for ix in range(ser_.shape[0]):
                    ser_[ix, :] = scale_ser.fit_transform(ser_[ix, :].reshape(-1, 1))[:, 0]
                ser[:, :, ix_] = ser_

            return ser

    # Calculates turning points in training series using Piecewise Linear Approximation
    def turning_points(self, ser, end_w=400, thresh_mult=0.01):

        self.max_turn = set()
        self.min_turn = set()
        self.pw_trend = 'none'

        df = pd.Series(ser)
        thresh = np.mean(df.rolling(36).mean()) * thresh_mult

        for idx in range(ser.shape[0]):

            if idx+end_w < ser.shape[0]:
                endex = end_w-1
            else:
                endex = ser.shape[0]-idx-1

            win_ser = ser[idx:idx + end_w]
            win_epo = np.arange(endex+1)
            max_win = max(win_ser)
            min_win = min(win_ser)

            if endex <= 1:
                break

            if win_ser[0] == max_win or win_ser[0] == min_win:
                self.distance_check(win_ser, idx, win_epo, endex, thresh)

        series_minmax = np.zeros_like(ser)
        series_minmax[list(self.max_turn)] = 2
        series_minmax[list(self.min_turn)] = 1

        return series_minmax

    # Calculates turning points in training series using Piecewise Linear Approximation
    def distance_check(self, win_ser, valid_idx, win_epo, endex, thresh):
        dist = self.pw_lin_rep(win_ser, win_epo, endex)
        max_idx = np.argmax(dist)
        min_idx = np.argmin(dist)

        if win_ser[0] >= win_ser[-1] and self.pw_trend != 'down':
            self.pw_trend = 'down'
        elif win_ser[0] >= win_ser[-1] and self.pw_trend == 'down':
            self.pw_trend = 'hold'

        if win_ser[0] < win_ser[-1] and self.pw_trend != 'up':
            self.pw_trend = 'up'
        elif win_ser[0] < win_ser[-1] and self.pw_trend == 'up':
            self.pw_trend = 'hold'

        if abs(dist[max_idx]) > thresh or abs(dist[min_idx]) > thresh:

            if abs(dist[max_idx]) > thresh:
                self.split_minmax(win_ser, valid_idx, max_idx, thresh)

            if abs(dist[min_idx]) > thresh:
                self.split_minmax(win_ser, valid_idx, min_idx, thresh)

        else:
            self.assign_to_minmax_list(valid_idx)


    def assign_to_minmax_list(self, valid_idx):
        if self.pw_trend == 'up':
            self.max_turn.add(valid_idx)
        elif self.pw_trend == 'down':
            self.min_turn.add(valid_idx)


    def split_minmax(self, win_ser, valid_idx, split_idx, thresh):
        win_ser_1 = win_ser[:split_idx]
        win_ser_2 = win_ser[split_idx:]
        for win_ser, idx_adjust in zip([win_ser_1, win_ser_2], [0, split_idx]):
            if win_ser.shape[0] > 24:
                win_epo = np.arange(win_ser.shape[0])
                endex = win_ser.shape[0]-1
                split_valid_idx = valid_idx + idx_adjust
                self.distance_check(win_ser, split_valid_idx, win_epo, endex, thresh)


    def minmax_trans(self, ser, plus):
        idx_diff = np.diff(ser)
        idx_1 = np.where(idx_diff > 0)
        idx_1 = idx_1
        idx_2 = np.where(idx_diff < 0)
        idx_2 = idx_2
        np_trans = np.zeros_like(ser)

        for idx in idx_1[0]:
            lb = idx - plus if idx - plus >= 0 else 0
            ub = idx + plus + 1 if idx - plus <= np_trans.shape[0] else np_trans.shape[0]
            np_trans[lb:ub] = 1

        for idx in idx_2[0]:
            lb = idx - plus if idx - plus >= 0 else 0
            ub = idx + plus + 1 if idx - plus <= np_trans.shape[0] else np_trans.shape[0]
            np_trans[lb:ub] = 2

        return np_trans


    def minmax_fill(self, ser):
        prev = np.arange(len(ser))
        prev[ser == 0] = 0
        prev = np.maximum.accumulate(prev)
        return ser[prev]

    # Calculates turning points in training series using Piecewise Linear Approximation
    def pw_lin_rep(self, win_ser, win_epo, endex):
        x1 = win_epo[0]
        xn = win_epo[endex]
        y1 = win_ser[0]
        yn = win_ser[endex]

        p1 = np.reshape(np.array([x1, y1]), (1, 2))
        p2 = np.reshape(np.array([xn, yn]), (1, 2))
        p3 = np.column_stack([win_epo, win_ser])
        dist = np.cross(p2 - p1, p1 - p3) / norm(p2 - p1)

        return dist

    def ica_loop_paa(self, paa, match):
        scaler = MinMaxScaler()
        paa_stack = []
        for ix in match.shape[1]:
            match_paa = paa.transform(scaler.fit_transform(match[ix].reshape(-1, 1))[:, 0])[0, :, 0].transpose()
            paa_stack.append(match_paa)
        return np.column_stack(paa_stack)

    # Ensures that the IC1 and IC2 features are consistently mapped and oriented correctly
    def ica_axis_lock(self, ica_1, ica_2, match_1, match_2):
        scaler = MinMaxScaler()
        paa = PiecewiseAggregateApproximation(n_segments=int(ica_1.shape[0] / 5))
        ica_1_paa = paa.transform(scaler.fit_transform(ica_1.reshape(-1, 1))[:, 0])[0, :, 0].transpose()
        ica_2_paa = paa.transform(scaler.fit_transform(ica_2.reshape(-1, 1))[:, 0])[0, :, 0].transpose()
        #match_1_paa = self.ica_loop_paa(paa, match_1)
        match_1_paa = paa.transform(scaler.fit_transform(match_1.reshape(-1, 1))[:, 0])[0, :, 0].transpose()
        match_2_paa = paa.transform(scaler.fit_transform(match_2.reshape(-1, 1))[:, 0])[0, :, 0].transpose()

        # Scores for primary matching uses DTW
        scores = np.array([
                            dtw.distance(ica_1_paa, match_1_paa, use_c=True),
                            dtw.distance(ica_1_paa * -1, match_1_paa, use_c=True),
                            dtw.distance(ica_2_paa, match_1_paa, use_c=True),
                            dtw.distance(ica_2_paa * -1, match_1_paa, use_c=True)])

        # Used only to orient the IC that is not matched to the primary
        scores_secondary = np.array([
                            dtw.distance(ica_1_paa, match_2_paa, use_c=True),
                            dtw.distance(ica_1_paa * -1, match_2_paa, use_c=True),
                            dtw.distance(ica_2_paa, match_2_paa, use_c=True),
                            dtw.distance(ica_2_paa * -1, match_2_paa, use_c=True)])

        # Determines which is IC1 and IC2 based on their DWT score and also inverts the final IC if required
        if max(scores[[0, 1]]) > max(scores[[2, 1]]):
            if np.argmax(scores[[0, 1]]) == 1:
                ica_1 *= -1
            if np.argmax(scores_secondary[[2, 3]]) == 1:
                ica_2 *= -1
            return np.column_stack([ica_2, ica_1])

        else:
            if np.argmax(scores[[2, 3]]) == 1:
                ica_2 *= -1
            if np.argmax(scores_secondary[[1, 2]]) == 1:
                ica_1 *= -1
            return np.column_stack([ica_1, ica_2])

    # Simple moving average
    def sma_dist(self, close):
        sma21 = pd.Series(close).rolling(21, min_periods=1).mean()
        return sma21.subtract(close)

    # Ensures that the PCAs are not inverted
    def pca_axis_lock(self, pca, ser):

        pc_idx_pos = np.where(pca >= 0)[0]
        pc_idx_neg = np.where(pca < 0)[0]

        diff_close = np.diff(ser)
        diff_close = np.pad(diff_close, (1, 0), 'constant', constant_values=(0, 0))

        pc_sum_pos = sum(diff_close[pc_idx_pos])
        pc_sum_neg = sum(diff_close[pc_idx_neg])

        if pc_sum_pos < pc_sum_neg:
            pca = pca * -1

        return pca

    # Determine minimum viable bounds on the time series for all 3 tme intervals (5, 30, 60)
    def define_epochs(self):
        time_start = []
        time_end = []
        epoch_5 = self.npa5[:, self.epoch_]
        epoch_30 = self.npa30[:, self.epoch_]
        epoch_hr = self.npa_hr[:, self.epoch_]
        time_start.append(np.min(epoch_hr))
        time_end.append(np.max(epoch_hr))
        time_0 = np.max(time_start)

        self.idx_5 = np.where(epoch_5 >= time_0)
        self.epoch_5 = epoch_5[self.idx_5].astype('float64')

        self.idx_30 = np.where(epoch_30 >= time_0)
        self.epoch_30 = epoch_30[self.idx_30].astype('float64')

        self.idx_hr = np.where(epoch_hr >= time_0)
        self.epoch_hr = epoch_hr[self.idx_hr].astype('float64')

    # creates 5 min interpolation for sparse time series eg: 30 min, 60 min
    def sparse_interp(self, npa, idx_npa, epoch_npa):
        col_stack = []
        for idx in range(npa.shape[1]):
            ser = npa[:, idx]
            ser_long = ser[idx_npa].astype('float64')
            col_stack.append(np.interp(self.epoch_5, epoch_npa, ser_long))

        return np.column_stack(col_stack)

    # conditional nan fix for artefacts resulting from  dimension reduction
    def nan_fix(self, arr):

        stack_list = []
        try:
            arr.shape[1]
        except:
            arr = np.reshape(arr, (arr.shape[0], 1))

        for idx in range(arr.shape[1]):
            col = arr[:, idx]

            try:
                col = col.astype('float64')
            except:
                stack_list.append(col)
                continue

            col[col == 0.0] = 0.0001
            col[col == -inf] = 0.0001
            col[col == inf] = 0.0001

            if any(np.isnan(col)):
                col = pd.Series(col)
                col = col.fillna(method='bfill').as_matrix()
                stack_list.append(col)

            else:
                stack_list.append(col)

        return np.column_stack(stack_list)

    # apply smooth to an array
    def multi_smooth(self, npa, win):
        arr_list = []

        for n in range(npa.shape[1]):
            arr_list.append(self.comm.np_ewma(npa[:, n], win))

        return np.column_stack(arr_list)

    # Combine multiple signals into one with ICA
    def combined_ica_time_int(self, series_1, series_2, series_3):
        scaler = MinMaxScaler()
        ser_stack = []
        icadict = {}

        for n, s in enumerate(range(series_1.shape[1])):

            icadict['5'] = scaler.fit_transform(series_1[:, s].reshape(-1, 1))[:, 0]
            icadict['30'] = scaler.fit_transform(series_2[:, s].reshape(-1, 1))[:, 0]
            icadict['60'] = scaler.fit_transform(series_3[:, s].reshape(-1, 1))[:, 0]

            n_ls = []
            for n in ICA_TIME_INT:
                n_ls.append(icadict[n])

            ica_comb = self.ica_calc(np.column_stack(n_ls), n_comp=1, orient=False)

            ser_scale = scaler.fit_transform(ica_comb)[:, 0]
            ser_scale_inv = scaler.fit_transform(ica_comb * -1)[:, 0]
            score_ls = []
            score_inv_ls = []

            for n in ICA_TIME_INT:
                score_ls.append(dtw.distance(ser_scale, icadict[n], use_c=True))
                score_inv_ls.append(dtw.distance(ser_scale_inv, icadict[n], use_c=True))

            appender = ica_comb[:, 0]
            if np.sum(np.array(score_ls)) > np.sum(np.array(score_inv_ls)):
                smin = -max(ica_comb[:, 0])
                smax = -min(ica_comb[:, 0])
                invscaler = MinMaxScaler(feature_range=(smin, smax))
                inv = ica_comb[:, 0] * -1
                appender = invscaler.fit_transform(inv.reshape(-1, 1))[:, 0]

            ser_stack.append(appender)
            time.sleep(1)
        ret = np.column_stack(ser_stack)
        return ret

    # Perform Canonical Corellation on ICA series
    def cca_from_ica(self, ser_ica):

        cca_ = CCA(n_components=1)
        cca_res1 = cca_.fit_transform(ser_ica[:, [1, 2, 5]], ser_ica[:, 4])[0]
        scl = StandardScaler()
        cca_res2 = scl.fit_transform(ser_ica[:, 3].reshape(-1, 1))
        ret = np.column_stack([cca_res1, cca_res2])
        return ret

    # Perform CCA on feature against PVO
    def cca_from_features(self, npa):
        cca_ = CCA(n_components=1)
        cca_res1 = cca_.fit_transform(npa, self.sub_series[:, 4])[0]
        return self.comm.np_ewma(cca_res1[:, 0], 5)

    # Add feature to processing dict
    def feature_add(self, tag, feature_function):
        if tag in self.feature_param['list']:
            self.feature_dict[tag] = feature_function

    # differnce between 1st and 2nd CCCA series
    def feature_cca_difference(self):
        diff_ = np.array(self.ser_cca[:, 0] - self.ser_cca[:, 1])
        return np.squeeze(diff_)

    # Calculates On Balance Volume moving average convergence
    def feature_obv_difference(self):
        obv_ = Indicators().obv_np(self.ser_ica[:, 0], self.ser_ica[:, 6])
        scl = StandardScaler()
        obv_scl_ = scl.fit_transform(obv_.reshape(-1, 1))
        return self.comm.np_ewma(obv_scl_, 9) - self.comm.np_ewma(obv_scl_, 30)

    # Assembles features into a dict
    def assemble_features(self):
        self.features_all.append(self.sub_series[:, 0])
        features_ = self.feature_param['list']
        limit_ = self.feature_param['limit']

        for feat_lab, limit_bool in zip(features_, limit_):
            feat_arr = self.feature_dict[feat_lab]
            if limit_bool:
                feat_arr = self.comm.soft_limit(feat_arr.reshape(-1, 1), bnd=limit_bool)[:, 0].tolist()
            else:
                feat_arr = feat_arr.tolist()
            self.features_all.append(feat_arr)
        self.features_all.append(self.sub_series[:, -1])

    # Moving average
    def feature_moving_average(self):
        scl = StandardScaler()
        arr = self.comm.np_ewma(self.sub_series[:, 0], 9) - self.comm.np_ewma(self.sub_series[:, 0], 30)
        return scl.fit_transform(arr.reshape(-1, 1))

    # Rolling entrpy based on window
    def feature_entropy(self, win=50):
        ser_ = pd.Series(self.sub_series[:, 0])
        def roll_entropy(x_):
            x_entr = entr(x_)
            x_entr[x_entr == -inf] = 0
            return x_entr.sum() / np.log(2)

        return ser_.rolling(window=win, min_periods=win).apply(lambda x: roll_entropy(x), raw=False).fillna(method='bfill')

    # Rolling skew value based on window
    def feature_skew(self, win=50):
        ser_ = pd.Series(self.sub_series[:, 0])

        def roll_skew(x_):
            return skew(x_)

        return ser_.rolling(window=win, min_periods=win).apply(lambda x: roll_skew(x), raw=False).fillna(method='bfill').values

    # Rollings kurtosis value based on window
    def feature_kurtosis(self, win=50):
        ser_ = pd.Series(self.sub_series[:, 0])

        def roll_kurtosis(x_):
            return kurtosis(x_)

        return ser_.rolling(window=win, min_periods=win).apply(lambda x: roll_kurtosis(x), raw=False).fillna(method='bfill')

    # AUC trend data
    def feature_auctrend(self, n):
        idx_ = [self.tr_a_p_, self.tr_b_p_, self.tr_a_n_, self.tr_b_n_]
        return self.npa5[:, idx_[n]][self.idx_5]

    # Orchestrate creation of features from technical indicators
    @pass_args
    def create_features(self, npa_5=None, npa_30=None, npa_hr=None, market=None, hold_out=0, features=None, train_on=False):

        print(self.close_)
        close_ = self.close_
        macd_ = self.macd_
        proc_ = self.proc_
        stochrsi_ = self.stoch_
        rsi_ = self.rsi_
        boll_ = self.boll_
        pvo_ = self.pvo_
        vol_ = self.vol_
        epoch_ = self.epoch_

        self.feature_param = features
        self.market = market
        self.npa5 = npa_5
        self.npa30 = npa_30
        self.npa_hr = npa_hr
        self.define_epochs()
        self.feature_dict = {}
        self.features_all = []
        multi_smooth = self.multi_smooth

        feature_columns = [close_, stochrsi_, macd_, boll_, proc_, pvo_, vol_, epoch_]
        series = self.npa5[:, feature_columns]
        self.sub_series = series[self.idx_5]

        # Subset of indexes to smooth
        smooth_idx = [1, 2, 3, 4, 5, 6]
        self.sub_series[:, smooth_idx] = multi_smooth(self.sub_series[:, smooth_idx], MULTI_SMOOTH)

        # Creating interpolation betweensparse points
        ser_interp_30 = self.sparse_interp(npa_30, idx_npa=self.idx_30, epoch_npa=self.epoch_30)[:, feature_columns]
        ser_interp_hr = self.sparse_interp(npa_hr, idx_npa=self.idx_hr, epoch_npa=self.epoch_hr)[:, feature_columns]

        self.ser_ica = self.combined_ica_time_int(self.sub_series, ser_interp_30, ser_interp_hr)
        self.ser_cca = self.cca_from_ica(self.ser_ica)

        # Function sets up features to add into a dict
        self.feature_add('OBV_DIFF', self.feature_obv_difference()  )
        self.feature_add('CCA1', self.ser_cca[:, 0])
        self.feature_add('CCA2', self.ser_cca[:, 1])
        self.feature_add('MA', self.feature_moving_average() )
        self.feature_add('CCA_DIFF', self.feature_cca_difference())
        self.feature_add('ENTR', self.feature_entropy() )
        self.feature_add('SKEW', self.feature_skew())
        self.feature_add('KURTOSIS', self.feature_kurtosis())
        self.feature_add('CCA_FT', self.cca_from_features(np.column_stack([self.feature_skew(), self.feature_obv_difference()])))
        self.feature_add('CCA_FT_1', self.cca_from_features(np.column_stack([self.feature_obv_difference()])))
        self.feature_add('CCA_FT_2', self.cca_from_features(np.column_stack([self.feature_moving_average()])))
        self.feature_add('CCA_SKEW', self.cca_from_features(np.column_stack([self.feature_skew()])))
        self.feature_add('CCA_KURT', self.cca_from_features(np.column_stack([self.feature_kurtosis()])))
        self.feature_add('CCA_KURTSKEW', self.cca_from_features(np.column_stack([self.feature_skew(), self.feature_kurtosis()])))
        self.feature_add('CCA_AUC_P', self.cca_from_features(np.column_stack([self.feature_auctrend(0), self.feature_auctrend(1)])))
        self.feature_add('CCA_AUC_N', self.cca_from_features(np.column_stack([self.feature_auctrend(2), self.feature_auctrend(3)])))
        self.feature_add('CCA_AUC_A',self.cca_from_features(np.column_stack([self.feature_auctrend(0), self.feature_auctrend(2)])))
        self.feature_add('CCA_AUC_B',self.cca_from_features(np.column_stack([self.feature_auctrend(1), self.feature_auctrend(3)])))

        self.assemble_features()

        if train_on:
            ser_sub = self.ser_cca[:, 0] - self.ser_cca[:, 1]
            ser_cross_zero = np.zeros_like(self.ser_cca[:, 0])
            ser_cross_zero[np.where(ser_sub > 0)[0]] = 1
            ser_cross_ix_b = np.where(np.diff(ser_cross_zero) == 1)[0]
            ser_cross_ix_s = np.where(np.diff(ser_cross_zero) == -1)[0]
            ser_cross = np.zeros_like(self.ser_cca[:, 0])
            ser_cross[ser_cross_ix_b] = 1
            ser_cross[ser_cross_ix_s] = 2
            labels = self.label_generate(self.comm.np_ewma(self.sub_series[:, 0], 5), ser_cross, hold_out)
        else:
            labels = None

        return np.column_stack(self.features_all), labels, self.ser_cca



    def split_labels(self, labels):
        # Split labels into min/max
        lab_1 = np.where(labels == 1)[0]
        lab_2 = np.where(labels == 2)[0]
        split_1 = np.split(lab_1, np.where(np.diff(lab_1) != 1)[0] + 1)
        split_2 = np.split(lab_2, np.where(np.diff(lab_2) != 1)[0] + 1)

        return lab_1, lab_2, split_1, split_2

    # Function for removing training labels that don't meet requirements for ideal labels
    def label_validator(self, win_adapt, thresh_adapt, pad_adapt, gain, price, hold_out, cross):

            # Recursive function for checking labels effectiveness and removing problem ones
            labels_full = self.get_labels(win_adapt, thresh_adapt, ceil(pad_adapt), arr_ser=price)
            labels_for_kf = labels_full[:int((labels_full.shape[0])*(1-hold_out))]

            lab_1, lab_2, split_1, split_2 = self.split_labels(labels_for_kf)

            try:
                # Manually removes labels that are generated on long flat periods
                lab_1_mid = [int(np.median(x)) for x in split_1]
                lab_2_mid = [int(np.median(x)) for x in split_2]
                ix_delt = 0
                lazy_list = []
            except:
                return 'lost', None

            for ix in range(max([len(lab_1_mid), len(lab_2_mid)])):
                # While loop to determine the correct index for a min/max label pair
                try:
                    while not lab_1_mid[ix + ix_delt] < lab_2_mid[ix] < lab_1_mid[ix + 1 + ix_delt]:
                        # While not a low - high pairing in the index, shift the index
                        if lab_1_mid[ix + 1] < lab_2_mid[ix] < lab_1_mid[ix + 2]:
                            ix_delt += 1
                        elif lab_1_mid[ix - 1] < lab_2_mid[ix] < lab_1_mid[ix]:
                            ix_delt -= 1
                        elif lab_1_mid[ix + 2] < lab_2_mid[ix] < lab_1_mid[ix + 3]:
                            ix_delt += 2
                        elif lab_1_mid[ix - 2] < lab_2_mid[ix] < lab_1_mid[ix - 1]:
                            ix_delt -= 2
                        else:
                            break

                except Exception as e:
                    continue

                # Evaluating if the price increase is a legitimate label pair
                # If not, save label pair for deletion
                price_min = price[lab_1_mid[ix + ix_delt]]
                price_max = price[lab_2_mid[ix]]
                if price_max/price_min > gain:
                    pass
                else:
                    lazy_list.append([lab_1_mid[ix + ix_delt], lab_2_mid[ix]])

            # Remove the lazy labels
            unlazy_1 = []
            for sp in split_1:
                lazy_sp1 = np.array([l[0] for l in lazy_list])
                intst = np.in1d(sp, lazy_sp1)
                if not any(intst):
                    unlazy_1.append(sp)

            unlazy_2 = []
            for sp in split_2:
                lazy_sp2 = np.array([l[1] for l in lazy_list])
                intst = np.in1d(sp, lazy_sp2)
                if not any(intst):
                    unlazy_2.append(sp)

            labels_for_kf = np.zeros_like(labels_for_kf)
            if unlazy_1:
                labels_for_kf[np.concatenate(unlazy_1, axis=0)] = 1
            if unlazy_2:
                labels_for_kf[np.concatenate(unlazy_2, axis=0)] = 2

            round_two = True
            if round_two:
                lab_1, lab_2, split_1, split_2 = self.split_labels(labels_for_kf)

                try:
                    lab_1_mid = [int(np.median(x)) for x in split_1]
                    lab_2_mid = [int(np.median(x)) for x in split_2]
                    ix_delt = 0
                    lazy_list = []
                except:
                    return 'lost', None


                for ix in range(max([len(lab_1_mid), len(lab_2_mid)])):
                    # While loop to determine the correct index for a min/max label pair
                    try:
                        while not lab_1_mid[ix + ix_delt] < lab_2_mid[ix] < lab_1_mid[ix + 1 + ix_delt]:
                            # While not a low - high pairing in the index, shift the index
                            if lab_1_mid[ix + 1] < lab_2_mid[ix] < lab_1_mid[ix + 2]:
                                ix_delt += 1
                            elif lab_1_mid[ix - 1] < lab_2_mid[ix] < lab_1_mid[ix]:
                                ix_delt -= 1
                            elif lab_1_mid[ix + 2] < lab_2_mid[ix] < lab_1_mid[ix + 3]:
                                ix_delt += 2
                            elif lab_1_mid[ix - 2] < lab_2_mid[ix] < lab_1_mid[ix - 1]:
                                ix_delt -= 2
                            else:
                                break

                    except Exception as e:
                        continue

                    # Evaluate if sell and the next buy are at same level to see if can be removed
                    price_min = price[lab_1_mid[ix + ix_delt + 1]]
                    price_max = price[lab_2_mid[ix]]

                    if price_max < price_min:
                        pass

                    elif split_1[ix + ix_delt + 1][0] - split_2[ix][-1] < 40:
                        minlen_ln = True

                    else:
                        minlen_ln = False

                    #minlen_sh = minlen_ln = True
                    try:
                        if price_max / price_min < 1.035 and minlen_ln:
                            lazy_list.append([lab_1_mid[ix + ix_delt + 1], lab_2_mid[ix]])
                    except:
                        pass

                # Remove the lazy labels
                unlazy_1 = []
                for sp in split_1:
                    lazy_sp1 = np.array([l[0] for l in lazy_list])
                    intst = np.in1d(sp, lazy_sp1)
                    if not any(intst):
                        unlazy_1.append(sp)

                unlazy_2 = []
                for sp in split_2:
                    lazy_sp2 = np.array([l[1] for l in lazy_list])
                    intst = np.in1d(sp, lazy_sp2)
                    if not any(intst):
                        unlazy_2.append(sp)

                labels_for_kf = np.zeros_like(labels_for_kf)
                if unlazy_1:
                    labels_for_kf[np.concatenate(unlazy_1, axis=0)] = 1
                if unlazy_2:
                    labels_for_kf[np.concatenate(unlazy_2, axis=0)] = 2

                # trim ends for label artefacts close to edge
                labels_for_kf[:50] = 0
                labels_for_kf[-50:] = 0

                # Shift towards feature crossover
                cross_shift = X_SHIFT
                if cross_shift:

                    lab_1, lab_2, split_1, split_2 = self.split_labels(labels_for_kf)
                    labels_zero = np.zeros_like(labels_for_kf)

                    for split in split_1:
                        overlap_found = False
                        overlap_fwd = False
                        overlap_bak = False
                        overlap_delta = 1
                        overlap_idx_ls = []

                        while not overlap_found:
                            #try:
                            overlap_delta += 1
                            split_shift_fwd = split + overlap_delta
                            split_shift_bak = split - overlap_delta

                            try:
                                cross_match_fwd = [c == 1 for c in cross[split_shift_fwd]]
                                cross_match_bak = [c == 1 for c in cross[split_shift_bak]]
                                past_alt_fwd = [a == 2 for a in cross[split_shift_fwd]]
                                past_alt_bak = [a == 2 for a in cross[split_shift_bak]]
                            except:
                                break

                            if any(cross_match_fwd) and not overlap_fwd:
                                overlap_idx_ls.append(split_shift_fwd)
                                overlap_fwd = True
                            elif any(past_alt_fwd) and not overlap_fwd:
                                overlap_idx_ls.append(False)
                                overlap_fwd = True

                            if any(cross_match_bak) and not overlap_bak:
                                overlap_idx_ls.append(split_shift_bak)
                                overlap_bak = True
                            elif any(past_alt_bak) and not overlap_bak:
                                overlap_idx_ls.append(False)
                                overlap_bak = True

                            if overlap_fwd and overlap_bak:

                                if all([o for o in overlap_idx_ls if o is False]):
                                    pass
                                elif overlap_idx_ls[1] is False:
                                    labels_zero[overlap_idx_ls[0]] = 1
                                elif overlap_idx_ls[0] is False:
                                    labels_zero[overlap_idx_ls[1]] = 1
                                else:
                                    m1 = np.mean(price[overlap_idx_ls[0]])
                                    m2 = np.mean(price[overlap_idx_ls[1]])
                                    min_idx = overlap_idx_ls[np.argmin([m1, m2])]
                                    labels_zero[min_idx] = 1

                                overlap_found = True

                    for split in split_2:
                        overlap_found = False
                        overlap_fwd = False
                        overlap_bak = False
                        overlap_delta = 1
                        overlap_idx_ls = []

                        while not overlap_found:
                            overlap_delta += 1
                            split_shift_fwd = split + overlap_delta
                            split_shift_bak = split - overlap_delta

                            try:
                                cross_match_fwd = [c == 2 for c in cross[split_shift_fwd]]
                                cross_match_bak = [c == 2 for c in cross[split_shift_bak]]
                                past_alt_fwd = [a == 1 for a in cross[split_shift_fwd]]
                                past_alt_bak = [a == 1 for a in cross[split_shift_bak]]
                            except:
                                break

                            if any(cross_match_fwd) and not overlap_fwd:
                                overlap_idx_ls.append(split_shift_fwd)
                                overlap_fwd = True
                            elif any(past_alt_fwd) and not overlap_fwd:
                                overlap_idx_ls.append(False)
                                overlap_fwd = True

                            if any(cross_match_bak) and not overlap_bak:
                                overlap_idx_ls.append(split_shift_bak)
                                overlap_bak = True
                            elif any(past_alt_bak) and not overlap_bak:
                                overlap_idx_ls.append(False)
                                overlap_bak = True

                            if overlap_fwd and overlap_bak:
                                mean_ls = [np.mean(price[ix]) if ix is not False else 0 for ix in overlap_idx_ls]
                                max_idx = overlap_idx_ls[np.argmax(mean_ls)]
                                labels_zero[max_idx] = 2

                                if all([o for o in overlap_idx_ls if o is False]):
                                    pass
                                elif overlap_idx_ls[1] is False:
                                    labels_zero[overlap_idx_ls[0]] = 2
                                elif overlap_idx_ls[0] is False:
                                    labels_zero[overlap_idx_ls[1]] = 2
                                else:
                                    m1 = np.mean(price[overlap_idx_ls[0]])
                                    m2 = np.mean(price[overlap_idx_ls[1]])
                                    min_idx = overlap_idx_ls[np.argmax([m1, m2])]
                                    labels_zero[min_idx] = 2

                                overlap_found = True

                    labels_for_kf = labels_zero

                # If more than 2 kfolds loose labels then break
                null_fold = 0
                for sp in np.array_split(labels_for_kf, 3):
                    if sp[sp == 1].shape[0] == 0 or sp[sp == 2].shape[0] == 0:
                        null_fold += 1
                        if null_fold > 1:
                            logging.info('Labels: break - labels missing from 2 kfolds')
                            print('got lost')
                            return 'lost', None

                # Checks for labels that are too close and loop again
                lab_0 = np.where(labels_for_kf == 0)[0]
                gaps = np.split(lab_0, np.where(np.diff(lab_0) != 1)[0] + 1)
                gap_sizes = [g.shape[0] for g in gaps][:-1]
                if any(g < 15 for g in gap_sizes):
                    logging.info('Labels: too close')
                    return 'close', labels_for_kf
                logging.info('Labels: Good')

            return 'good', labels_for_kf

    # Main recursive function of creating and validating labels
    def label_generate(self, price, cross, hold_out):

        win_adapt = LABEL_WIN
        pad_adapt = LABEL_PAD
        thresh_adapt = LABEL_THRESH
        iter = 0
        gain = 1.02
        label_condition, labels_new = self.label_validator(win_adapt, thresh_adapt, pad_adapt, gain, price, hold_out, cross)
        while label_condition != 'good':
            labels_hold = labels_new
            win_adapt += 5
            label_condition, labels_new = self.label_validator(win_adapt, thresh_adapt, pad_adapt, gain, price, hold_out, cross)

            if label_condition == 'lost':
                gain -= 0.0025

            iter += 1
            if iter > 15:
                break
        labels_last = labels_new if labels_new is not None else labels_hold

        return labels_last
