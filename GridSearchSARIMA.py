# grid search sarima hyperparameters for daily female dataset
import math
import warnings
from math import sqrt
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed
from warnings import catch_warnings
from warnings import filterwarnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from pandas import read_csv


def train_test_split(data, n_test):
    return data[:-n_test], data[-n_test:]


def measure_rmse(actual, predicted):
    return sqrt(mean_squared_error(actual, predicted))


def sarima_forecast(history, config, start, end):
    order, sorder, trend = config

    # define model
    model = SARIMAX(history, order=order, seasonal_order=sorder, trend=trend, enforce_stationarity=False,
                    enforce_invertibility=False)
    # fit model
    model_fit = model.fit(disp=False)
    # make one step forecast
    yhat = model_fit.predict(start, end)
    return yhat[0]


class GridSearchSARIMA:
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
                    warnings.filterwarnings("ignore")
                    error = self.walk_forward_validation(data, n_test, cfg)

            except:
                error = None

        # check for an interesting result
        if error is not None:
            print(' > Model[%s] %.3f' % (key, error))
        return key, error

    def grid_search(self, data, cfg_list, n_test, parallel=True):
        scores = None
        if parallel:
            # execute configs in parallel
            executor = Parallel(n_jobs=cpu_count(), backend='multiprocessing')
            tasks = (delayed(self.score_model)(data, n_test, cfg) for cfg in cfg_list)
            scores = executor(tasks)
        else:
            scores = [self.score_model(data, n_test, cfg) for cfg in cfg_list]

        # remove empty results
        scores = [r for r in scores if r[1] is not None]
        # sort configs by error, asc
        scores.sort(key=lambda tup: tup[1])
        return scores

    def sarima_configs(self, seasonal=[0]):
        models = list()
        # define config lists
        p_params = [0, 1]
        d_params = [0]
        q_params = [0, 1]
        t_params = ['n', 'c', 't', 'ct']
        P_params = [0, 1]
        D_params = [0]
        Q_params = [0, 1]
        m_params = seasonal
        # create config instances
        for p in p_params:
            for d in d_params:
                for q in q_params:
                    for t in t_params:
                        for P in P_params:
                            for D in D_params:
                                for Q in Q_params:
                                    for m in m_params:
                                        cfg = [(p, d, q), (P, D, Q, m), t]
                                        models.append(cfg)
        return models


if __name__ == '__main__':
    gs = GridSearchSARIMA()

    series = read_csv('daily-total-female-births.csv', header=0, index_col=0)
    data = series.values

    # data split
    n_test = 165 # model configs
    cfg_list = gs.sarima_configs(seasonal=[0, 2])
    # grid search
    scores = gs.grid_search(data, cfg_list, n_test)
    print('done')
    # list top 3 configs
    for cfg, error in scores[:3]:
        print(cfg, error)