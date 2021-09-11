import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path

import random

from scipy import stats as sp
from pylab import rcParams
from matplotlib import rc
from tqdm.notebook import tqdm

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from pathlib import Path

from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.algorithms.so_pso import PSO
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.optimize import minimize
from pymoo.model.problem import Problem
from pymoo.algorithms.so_cmaes import CMAES
from pymoo.algorithms.so_de import DE
from pymoo.algorithms.so_pattern_search import PatternSearch
import sys
# %%


"""## Helpers"""


def regress(T):
    x = T.index.values
    y = T.values
    model = sm.OLS(y, sm.add_constant(x)).fit()

    out = {}
    if len(model.params) > 1:
        out = {
            'intercept': model.params[0],
            'slope': model.params[1],
            'length': x[-1] - x[0],
            'error': np.sqrt(model.ssr)
        }
    else:
        out = {
            'intercept': model.params[0],
            'slope': 9999,
            'length': x[-1] - x[0],
            'error': np.sqrt(model.ssr)
        }

    return out


def sliding_window(T, max_error=100, df=True):
    rows = []
    anchor = 0
    while anchor < T.size:
        i = 2
        line = regress(T[anchor: anchor + i])
        previous_line = line
        while line['error'] < max_error and (anchor + i) < T.size:
            i += 1
            previous_line = line
            line = regress(T[anchor: anchor + i])
        if previous_line['slope'] != 9999:
            rows.append([previous_line['slope'], previous_line['length']])
        anchor += i
    if df:
        return pd.DataFrame(rows, columns=['Slope', 'Length'])
    return rows


def create_sequences(input_data: pd.DataFrame, target_column, sequence_length, to_numpy=True):
    sequences = []
    data_size = len(input_data)

    for i in range(data_size - sequence_length):

        sequence = input_data[i: i + sequence_length]
        label_postion = i + sequence_length
        if isinstance(target_column, list):
            label = input_data.iloc[label_postion][target_column].values
        else:
            label = input_data.iloc[label_postion][target_column]

        if to_numpy:
            v = (torch.Tensor(sequence.to_numpy()), torch.tensor(label).float())
        else:
            v = (sequence, label)

        sequences.append(v)

    return sequences


def abline(slope, intercept, length):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array()
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--')


def plot_trends(trends, y_0):
    axes = plt.gca()
    v_vals, y_vals = [], []
    trends = trends.iloc()

    x_start = axes.get_xlim()[0]
    x_finish = trends[0]['Length']

    for trend in trends:
        x_vals = np.array(x_start, x_finish)

        y_vals = intercept + trend['Slope'] * x_vals
        intercept = y_vals[-1]
        x_start = trend['Length'] + 1
        x_finish = x_start

    plt.plot(x_vals, y_vals, '--')


# %%
"""# Data Preprocessing"""

# %%
"""# Models

## MLP
"""


class MlpModel(nn.Module):
    def __init__(self, n_features=8, n_hidden=2, hidden_sizes=[7, 14], dropout_rate=0.25):
        super().__init__()
        self.hidden = nn.ModuleList()
        self.n_features = n_features
        current_dim = n_features

        for i in range(n_hidden):
            size = hidden_sizes[i]
            self.hidden.append(nn.Linear(current_dim, size))
            current_dim = size
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden.append(nn.Linear(current_dim, 2))

    def forward(self, x):
        x = x.view(-1, self.n_features)
        for layer in self.hidden[:-1]:
            x = torch.relu(layer(x))
        x = self.dropout(x)
        return self.hidden[-1](x)


"""## LSTM"""


class LstmModel(nn.Module):
    def __init__(self, n_features, n_hidden=5, n_layers=2, dropout=0.2):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_features = n_features

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            batch_first=True,
            num_layers=n_layers,
            dropout=dropout
        )

        # output prediction of our model
        # takes number of hidden units as the size fir the input
        self.regressor = nn.Linear(n_hidden, 2)  # second number is number of features to output (will be 2)

    def forward(self, x):
        x = x.view(-1, 4, self.n_features)
        self.lstm.flatten_parameters()

        _, (hidden, _) = self.lstm(x)
        out = hidden[-1]

        return self.regressor(out)


# %%
"""# Training"""


def get_data(filename, column_name='Close'):
    df = pd.read_csv("data/" + filename + ".csv", parse_dates=['Date'])
    df.head()

    rows = []
    index = []

    for _, row in df.iterrows():
        try:
            x = float(row[column_name])
            row_data = {
                column_name: x
            }
            index.append((row.Date - df.Date[0]).days)
            rows.append(row_data)
        except:
            print("Not a float")

    features_df = pd.DataFrame(rows, index=index)
    features_df.head()
    # %%
    # sax = features_df[column_name].plot()
    # plt.show()
    # %%

    # CONVERT TO TREND SEQUENCES
    features_df = sliding_window(features_df[column_name])
    features_df.head()

    train_size = int(len(features_df) * .6)
    val_size = int(len(features_df) * .2)
    test_size = int(len(features_df) * .2)

    train_df, val_df, test_df = features_df[:train_size], features_df[
                                                          train_size + 1: train_size + 1 + val_size], features_df[
                                                                                                      train_size + val_size + 1:]
    train_df.shape, val_df.shape, test_df.shape

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_df)

    train_df = pd.DataFrame(
        train_df,  # scaler.transform(train_df),
        index=train_df.index,
        columns=train_df.columns
    )

    train_df.head()

    val_df = pd.DataFrame(
        val_df,  # scaler.transform(val_df),
        index=val_df.index,
        columns=val_df.columns
    )

    val_df.head()

    test_df = pd.DataFrame(
        test_df,  # scaler.transform(test_df),
        index=test_df.index,
        columns=test_df.columns
    )

    test_df.head()
    # %%
    SEQUENCE_LENGTH = 4
    target = ['Slope', 'Length']
    train_sequences = create_sequences(train_df, target, SEQUENCE_LENGTH)
    val_sequences = create_sequences(val_df, target, SEQUENCE_LENGTH)
    test_sequences = create_sequences(test_df, target, SEQUENCE_LENGTH)

    trainset = torch.utils.data.DataLoader(train_sequences, batch_size=64, shuffle=False)
    valset = torch.utils.data.DataLoader(val_sequences, batch_size=64, shuffle=False)
    testset = torch.utils.data.DataLoader(test_sequences, batch_size=64, shuffle=False)

    return trainset, valset, testset


def build_model(params):
    if params['model'] == models[0]:
        model = LstmModel(2, n_hidden=params['n_hidden'], n_layers=params['hidden_1'], dropout=params['dropout'])
    elif params['model'] == models[1]:
        sizes = [];

        if params.get('hidden_1') > 0:
            sizes.append(params.get('hidden_1'))
            if params.get('hidden_2') > 0:
                sizes.append(params.get('hidden_2'))
                if params.get('hidden_3') > 0:
                    sizes.append(params.get('hidden_3'))
                    if params.get('hidden_4') > 0:
                        sizes.append(params.get('hidden_4'))
                        if params.get('hidden_5') > 0:
                            sizes.append(params.get('hidden_5'))

        model = MlpModel(n_hidden=params['n_hidden'], hidden_sizes=sizes, dropout_rate=params['dropout'])

    return model


"""## Train Method"""


def train(params, model_only=False):

    if isinstance(params, list):
        params = params_list_to_dict(params)

    if 'num_epochs' not in globals():
        num_epochs = 200

    start_time = time.time()

    learning_rate = 0.01
    optimizer_name = 'adam'

    if (params.get('num_epochs') != None):
        num_epochs = params.get('num_epochs')

    if (params.get('learning_rate') != None):
        learning_rate = params.get('learning_rate')

    model = build_model(params)
    model.to(device)

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        for data in trainset:
            X, y = data
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()  # adjusts weights

    if model_only:
        return model

    actual, predicted = [], []
    with torch.no_grad():
        for data in valset:
            X, y = data
            X, y = X.to(device), y.to(device)

            for idx, i in enumerate(y):
                actual.append(i)

            for idx, i in enumerate(model(X)):
                predicted.append(i)

    actual, predicted = torch.stack(actual), torch.stack(predicted)
    end_time = time.time()
    mse = F.mse_loss(actual, predicted).cpu().item()

    if 'tracker' in globals():
        if isinstance(tracker, PerformanceTracker):
            tracker.add_row(start_time, end_time, mse, model, params)
    print(str(mse))
    return mse


"""## Test Method"""


def test(model):
    model.to(device)
    actual, predicted = [], []

    with torch.no_grad():
        for data in testset:
            X, y = data
            X, y = X.to(device), y.to(device)

            for idx, i in enumerate(y):
                actual.append(i.cpu().numpy())

            for idx, i in enumerate(model(X)):
                predicted.append(i.cpu().numpy())

    actual = np.stack(actual)
    predicted = np.stack(predicted)

    return actual, predicted, np.square(np.subtract(actual, predicted)).mean()


# %%
"""# Combined Hyper-Parameter and Algorithm Selection

## Parameter Grid
"""

models = ['lstm', 'mlp']
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n_hidden = [1, 2, 3, 4, 5]

optimizers = ['sgd']
num_epochs = [25, 50, 75]
'''
	1.	model [lstm, mlp]
	2.	learning_rate 0-4
	3.	dropout 0-9
	4.	n_hidden 1-5
	5.	hidden_1 10-200
	6.	hidden_2
	7.	hidden_3
	8.	hidden_4
	9.	hidden_5
'''
lower_bounds = [0, 0, 0, 1, 4, 4, 4, 4, 4]
upper_bounds = [
    len(models) - 1,
    len(learning_rates) - 1,
    len(dropout_rates) - 1,
    5, 100, 100, 100, 100, 100,
    # len(optimizers) - 1,
    # len(num_epochs) - 1
]

hidden_1 = np.arange(10, 201)
hidden_2 = np.arange(10, 201)
hidden_3 = np.arange(10, 201)
hidden_4 = np.arange(10, 201)
hidden_5 = np.arange(10, 201)

cnn_n_layers = [1, 2, 3]
lstm_n_layers = [1, 2, 3]
kernal_1 = [1, 2, 3]
kernal_2 = [1, 2, 3]
kernal_3 = [1, 2, 3]

params = {
    'model': models,
    'learning_rate': learning_rates,
    'dropout': dropout_rates,
    'optimizer': optimizers,
    'num_epochs': num_epochs,
    'n_hidden': n_hidden,
    'hidden_1': hidden_1,
    'hidden_2': hidden_2,
    'hidden_3': hidden_3,
    'hidden_4': hidden_4,
    'hidden_5': hidden_5,
}


def params_list_to_dict(params):
    print("Params in:", params)
    if isinstance(params, dict):
        return params

    if isinstance(params, np.ndarray):
        params = params.astype(int)

    new_params = {}

    new_params['model'] = models[params[0]] if len(params) > 0 else models[0]
    new_params['learning_rate'] = learning_rates[params[1]] if len(params) > 1 else learning_rates[1]
    new_params['dropout'] = dropout_rates[params[2]] if len(params) > 2 else dropout_rates[0]
    new_params['n_hidden'] = params[3] if len(params) > 3 else 1
    new_params['hidden_1'] = params[4] if len(params) > 4 else 20
    new_params['hidden_2'] = params[5] if len(params) > 5 else 0
    new_params['hidden_3'] = params[6] if len(params) > 6 else 0
    new_params['hidden_4'] = params[7] if len(params) > 7 else 0
    new_params['hidden_5'] = params[8] if len(params) > 8 else 0

    return new_params


# %%

class PerformanceTracker:

    def __init__(self, experiment_name=None, pop_size=1):

        self.start_time = time.time()

        if experiment_name != None:
            self.experiment_name = experiment_name
        else:
            self.experiment_name = "Trial at" + time.ctime()

        self.rows = []
        self.results = []
        self.models = []
        self.params = []
        self.pop_size = pop_size
        self.buffer_earliest_start = time.time() + 9999
        self.buffer_latest_finish = time.time()
        self.buffer_best_mse = 9999
        self.buffer_best_model = None
        self.buffer_best_params = None
        self.iteration = 0
        self.function_evaluations = 0

        self.best_model = None
        self.best_model_params = None
        self.best_mse = 9999

        self.predicted = None

    def add_row(self, start_time, end_time, val_mse, model=None, params=None):

        self.function_evaluations += 1

        row = {
            'function_evaluations': self.function_evaluations,
            'iteration': self.iteration + 1,
            'start_time': start_time,
            'end_time': end_time,
            'val_mse': val_mse
        }

        self.rows.append(row)
        self.buffer_earliest_start = start_time if start_time < self.buffer_earliest_start else self.buffer_earliest_start
        self.buffer_latest_finish = end_time if end_time > self.buffer_latest_finish else self.buffer_latest_finish

        if val_mse < self.best_mse:
            self.best_mse = val_mse
            self.best_model = model
            self.best_model_params = params

        if val_mse < self.buffer_best_mse:
            self.buffer_best_mse = val_mse
            if model != None:
                self.buffer_best_model = model
            if params != None:
                self.buffer_best_params = params

        if self.function_evaluations % self.pop_size == 0:
            self.iteration += 1
            self.results.append({
                'iteration': self.iteration,
                'start_time': self.buffer_earliest_start,
                'end_time': self.buffer_latest_finish,
                'val_mse': self.buffer_best_mse
            })
            self.params.append(self.buffer_best_params)
            self.models.append(self.buffer_best_model)
            self.buffer_earliest_start = time.time() + 9999
            self.buffer_latest_finish = time.time()
            self.buffer_best_mse = 9999
            self.buffer_best_model = None
            self.buffer_best_params = None

    @property
    def test_best_model(self):

        predicted_s = []
        predicted_l = []
        mse_arr = []
        for i in range(5):
            model = train(self.best_model_params, model_only=True)
            self.actual, predicted, mse = test(model)
            predicted_s.append(predicted[:, 0])
            predicted_l.append(predicted[:, 1])
            mse_arr.append(mse)

        predicted_s, predicted_l = np.stack(predicted_s), np.stack(predicted_l)

        avg_s, avg_l = predicted_s.mean(axis = 0), predicted_l.mean(axis=0)
        self.best_model_test_mse = np.array(mse_arr).mean()

        self.test_results = pd.DataFrame(data={
            'actual_slope': self.actual[:, 0],
            'actual_length': self.actual[:, 1],
            'predicted_slope': avg_s,
            'predicted_length': avg_l
        })

        rmse = lambda actual, predicted: np.sqrt(np.square(np.subtract(actual, predicted)).mean())

        self.slope_rmse = rmse(self.actual[:, 0], avg_s)
        self.duration_rmse = rmse(self.actual[:, 1], avg_l)

        return self.test_results

    def summary(self):
        total_time = self.rows[-1]['end_time'] - self.rows[0]['start_time']
        idx = ['best model', 'val mse', 'test mse', 'total time', 'function calls', 'iterations']
        vals = [str(self.best_model_params), self.best_mse, self.best_model_test_mse, total_time,
                self.function_evaluations, self.iteration]
        return pd.DataFrame(vals, index=idx, columns=['Value'])

    def SDA(self):
        idx = ['Slope RMSE', 'Duration RMSE', 'Average RMSE']
        vals = [
            round(self.slope_rmse, 3),
            round(self.duration_rmse, 3),
            round(np.sqrt(self.best_model_test_mse), 3)
        ]
        return pd.DataFrame(vals, index=idx, columns=['Value'])

    def get_results(self, panda=True):
        if panda:
            return pd.DataFrame(self.results)
        else:
            return self.results

    def get_rows(self, panda=True):
        if panda:
            return pd.DataFrame(self.rows)
        else:
            return self.rows

    def get_params(self):
        return pd.DataFrame(self.params)

    def export(self):

        pd.set_option('display.float_format', '{:.2f}'.format)
        self.get_results().to_csv(self.__csv_name('Results'))
        self.get_rows().to_csv(self.__csv_name('Rows'))
        self.get_params().to_csv(self.__csv_name('Params'))
        self.test_best_model.to_csv(self.__csv_name('Predictions'))
        self.summary().to_csv(self.__csv_name('Summary'))
        self.SDA().to_csv(self.__csv_name('SDA'))

    def __csv_name(self, name):
        p = "Experiment Results (GPUO)/" + self.experiment_name
        Path(p).mkdir(parents=True, exist_ok=True)
        return p + "/" + name + ".csv"


# %%

def discrete_random_search(f, lower_bounds, upper_bounds, iterations=10):
    D = len(lower_bounds)
    best_f = 9999.0
    best_x = [None] * D

    for i in range(iterations):

        new_x = [random.randint(lower_bounds[d], upper_bounds[d]) for d in range(D)]
        new_f = f(new_x)

        if new_f < best_f:
            best_f = new_f
            best_x = new_x

    return {'best_x': best_x, 'best_f': best_f}


class CASH(Problem):

    def __init__(self):
        super().__init__(n_var=len(lower_bounds), n_obj=1, xl=lower_bounds, xu=upper_bounds, type_var=int)

    def _evaluate(self, x, out, *args, **kwargs):
        print(x.shape[0])
        fs = []
        for i in range(x.shape[0]):
            fs.append(train(params_list_to_dict(x[i])))
        out["F"] = np.array(fs)


## GLOBALS
cuda_off = True
dataset_name = "STX40"
column_name = 'Close'
trainset = None
valset = None
testset = None
tracker = None
device = None
num_epochs = 200


def main():
    global trainset, valset, testset, tracker, device, num_epochs

    datasets = ['NYSE', 'NASDAQ', 'STX40', 'BARC']
    algos = ['random', 'ga', 'ps', 'de']
    budget = 360

    if len(sys.argv) >= 2:
        cuda_off = False if sys.argv[1] == 'cuda_on' else True

    if len(sys.argv) >= 3:
        datasets = sys.argv[2].split(',')

    if len(sys.argv) >= 4:
        nums = sys.argv[3].split(',')
        budget = int(nums[0])
        num_epochs = int(nums[1])



    print(cuda_off, budget, num_epochs, datasets)
    return 0

    if cuda_off:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")


    iterations = 1

    for d in datasets:
        trainset, valset, testset = get_data(d)
        for a in algos:
            for i in range(iterations):

                tracker_name = a + " " + d + " " + str(i)


                if a == 'random':
                    pop_size = 1
                    iters = math.floor(budget / pop_size)

                    tracker = PerformanceTracker(tracker_name, pop_size=pop_size)
                    discrete_random_search(train, lower_bounds, upper_bounds, iters)
                    tracker.export()
                if a == 'ga':
                    pop_size = 9
                    iters = math.floor(budget / pop_size)

                    tracker = PerformanceTracker(tracker_name, pop_size=pop_size)
                    ga = GA(
                        pop_size=pop_size,
                        sampling=get_sampling("int_random"),
                        crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
                        mutation=get_mutation("int_pm", eta=3.0),
                        eliminate_duplicates=True,
                    )
                    res = minimize(
                        CASH(),
                        ga,
                        termination=('n_gen', iters),
                        seed=1,
                        save_history=True
                    )
                    print(f"Best solution found: \nX = {res.X}\nF = {res.F}\nCV= {res.CV}")
                    tracker.export()
                if a == 'ps':
                    pop_size = 45
                    iters = math.floor(budget / pop_size)

                    tracker = PerformanceTracker(tracker_name, pop_size=pop_size)
                    ps = PatternSearch(
                        sampling=get_sampling("int_random"),
                        eliminate_duplicates=True,
                    )
                    res = minimize(CASH(),
                                   ps,
                                   ('n_iter', iters),
                                   seed=1,
                                   verbose=False)
                    tracker.export()
                if a == 'de':
                    pop_size = 5
                    iters = math.floor(budget / pop_size)

                    tracker = PerformanceTracker(tracker_name, pop_size=pop_size)
                    de = DE(
                        pop_size=pop_size,
                        sampling=get_sampling("int_random"),
                        eliminate_duplicates=True,
                    )
                    res = minimize(
                        CASH(),
                        de,
                        termination=('n_gen', iters),
                        seed=1,
                        save_history=True
                    )

                    print("Best solution found: %s" % res.X)
                    print("Function value: %s" % res.F)
                    print("Constraint violation: %s" % res.CV)
                    tracker.export()


if __name__ == '__main__':
    main()