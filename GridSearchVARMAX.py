import math
import warnings
from warnings import catch_warnings

import numpy as np
import pandas as pd
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.seasonal import STL
import numpy.fft as fft

dta = pd.read_stata("lutkepohl2.dta")
dta.index = dta.qtr
dta.index.freq = dta.index.inferred_freq
endog = dta.loc['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]
exog = endog['dln_consump']


def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


def sarima_forecast(history, config, start, end):
    order, trend = config

    # define model
    model = VARMAX(history, order=order, trend=trend, enforce_invertibility=False, enforce_stationarity=False)
    # fit model
    model_fit = model.fit(maxiter=10000, disp=False)
    # make one step forecast
    yhat = model_fit.predict(start, end)
    return yhat[0]


def measure_rmse(actual, predicted):
    return math.sqrt(mean_squared_error(actual, predicted))


def difference_trend(diff, data, trend):
    for i in range(len(trend)):
        diff[i].append(data[i] - trend[i])


def top_n_indexes(lst, n):
    # Sort the list with indexes and take the top N
    return [i[0] for i in sorted(enumerate(lst), key=lambda x: x[1], reverse=True)[:n]]


class GridSearchVARMAX:
    def walk_forward_validation(self, data, n_test, cfg):
        predictions = list()
        # split dataset
        train, test = train_test_split(data, n_test)
        # seed history with training dataset
        history = [x for x in train]
        # step over each time-step in the test set
        for i in range(len(test)):
            # fit model and make forecast for history
            yhat = sarima_forecast(history, cfg, len(history), len(history))
            # store forecast in list of predictions
            predictions.append(yhat)
            # add actual observation to history for the next loop
            history.append(test[i])
        # estimate prediction error
        error = measure_rmse(test, predictions)
        return error

    def score_model(self, data, n_test, cfg, debug=False):
        error = None
        # convert config to a key
        key = cfg
        # show all warnings and fail on exception if debugging
        if debug:
            result = self.walk_forward_validation(data, n_test, cfg)
        else:
            # one failure during model validation suggests an unstable config
            try:
                # never show warnings when grid searching, too noisy
                with catch_warnings():
                    error = self.walk_forward_validation(data, n_test, cfg)

            except:
                error = None

        # check for an interesting result
        if error is not None:
            print(' > Model[%s] %.3f' % (key, error))
        return key, error

    def grid_search(self, data, cfg_list, n_test):
        scores = [self.score_model(data, n_test, cfg) for cfg in cfg_list]

        print(scores)

        # remove empty results
        scores = [r for r in scores if r[1] is not None]
        # sort configs by error, asc
        scores.sort(key=lambda tup: tup[1])
        return scores

    def sarima_configs(self):
        models = list()
        # define config lists
        p_params = [1, 2, 3, 4, 5]
        q_params = [1, 2, 3, 4, 5]
        t_params = ['n', 'c', 't', 'ct']
        # create config instances
        for p in p_params:
            for t in t_params:
                cfg = [(p, 0), t]
                models.append(cfg)

        for q in q_params:
            for t in t_params:
                cfg = [(0, q), t]
                models.append(cfg)

        return models


def fourier_difference(arr):
    fourier = []

    for set in arr:
        print(set)
        yf = fft.fft(arr[set])
        top_yf = top_n_indexes(yf, len(arr[set]) // 16)
        yf_clean = yf.copy()
        yf_clean[np.abs(yf) < np.abs(yf[top_yf[-1]])] = 0
        yf_clean[len(arr[set]) // 2:] = 0
        y_clean = fft.ifft(yf_clean)
        y_real = [float(i.real) for i in y_clean]
        fourier.append(y_real)

    return fourier


def STL_difference(arr):
    differenced = []

    for i in range(len(arr[0])):
        differenced.append([])

    for set in arr:
        stl = STL(set, period=len(set))
        res = stl.fit()
        difference_trend(differenced, set, res.trend)

    return differenced


if __name__ == '__main__':
    gs = GridSearchVARMAX()

    csv = pd.read_csv("data.csv")

    dta = pd.read_stata("lutkepohl2.dta")
    dta.index = dta.qtr
    dta.index.freq = dta.index.inferred_freq
    endog = dta.loc['1960-04-01':'1978-10-01', ['dln_inv', 'dln_inc', 'dln_consump']]

    #fourier = fourier_difference(csv)

    differenced = STL_difference(csv.values)

    #print(np.array(differenced).tolist())

    # data split
    n_test = 100  # model configs
    cfg_list = gs.sarima_configs()
    # grid search
    scores = gs.grid_search(np.array(differenced).tolist(), cfg_list, n_test)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)
