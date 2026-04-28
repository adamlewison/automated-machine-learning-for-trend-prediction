"""
Algorithm Comparison Experiment
================================
Compares Random Search, Pattern Search, and Differential Evolution for the
CASH (Combined Algorithm Selection and Hyperparameter optimisation) problem
on NYSE, NASDAQ, and STX40 datasets.

Usage:
    python algorithm_comparison.py [dataset,...] [budget,num_epochs]

Examples:
    python algorithm_comparison.py
    python algorithm_comparison.py NYSE
    python algorithm_comparison.py NYSE,NASDAQ 540,50
"""

import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.preprocessing import MinMaxScaler

from pymoo.algorithms.soo.nongrad.de import DE
from pymoo.algorithms.soo.nongrad.pattern import PatternSearch
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from pymoo.core.problem import Problem


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def regress(T):
    x = T.index.values
    y = T.values
    model = sm.OLS(y, sm.add_constant(x)).fit()

    if len(model.params) > 1:
        return {
            'intercept': model.params[0],
            'slope': model.params[1],
            'length': x[-1] - x[0],
            'error': np.sqrt(model.ssr),
        }
    return {
        'intercept': model.params[0],
        'slope': 9999,
        'length': x[-1] - x[0],
        'error': np.sqrt(model.ssr),
    }


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
        label_position = i + sequence_length
        if isinstance(target_column, list):
            label = input_data.iloc[label_position][target_column].values
        else:
            label = input_data.iloc[label_position][target_column]
        if to_numpy:
            v = (torch.Tensor(sequence.to_numpy()), torch.tensor(label).float())
        else:
            v = (sequence, label)
        sequences.append(v)
    return sequences


def get_data(filename, column_name='Close'):
    df = pd.read_csv(Path('data') / (filename + '.csv'), parse_dates=['Date'])

    rows, index = [], []
    for _, row in df.iterrows():
        try:
            x = float(row[column_name])
            index.append((row.Date - df.Date[0]).days)
            rows.append({column_name: x})
        except (ValueError, TypeError):
            pass

    features_df = pd.DataFrame(rows, index=index)
    features_df = sliding_window(features_df[column_name])

    train_size = int(len(features_df) * .6)
    val_size = int(len(features_df) * .2)

    train_df = features_df[:train_size]
    val_df = features_df[train_size + 1: train_size + 1 + val_size]
    test_df = features_df[train_size + val_size + 1:]

    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(train_df)
    train_df = pd.DataFrame(train_df, index=train_df.index, columns=train_df.columns)
    val_df = pd.DataFrame(val_df, index=val_df.index, columns=val_df.columns)
    test_df = pd.DataFrame(test_df, index=test_df.index, columns=test_df.columns)

    SEQUENCE_LENGTH = 4
    target = ['Slope', 'Length']
    trainset = DataLoader(create_sequences(train_df, target, SEQUENCE_LENGTH), batch_size=64, shuffle=False)
    valset = DataLoader(create_sequences(val_df, target, SEQUENCE_LENGTH), batch_size=64, shuffle=False)
    testset = DataLoader(create_sequences(test_df, target, SEQUENCE_LENGTH), batch_size=64, shuffle=False)

    return trainset, valset, testset


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class MlpModel(nn.Module):
    def __init__(self, n_features=8, n_hidden=2, hidden_sizes=None, dropout_rate=0.25):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [7, 14]
        self.hidden = nn.ModuleList()
        self.n_features = n_features
        current_dim = n_features
        for size in hidden_sizes[:n_hidden]:
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
            dropout=dropout,
        )
        self.regressor = nn.Linear(n_hidden, 2)

    def forward(self, x):
        x = x.view(-1, 4, self.n_features)
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(x)
        return self.regressor(hidden[-1])


# ---------------------------------------------------------------------------
# Parameter space
# ---------------------------------------------------------------------------

models = ['lstm', 'mlp']
learning_rates = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
n_hidden_options = [1, 2, 3, 4, 5]

lower_bounds = [0, 0, 0, 1, 2, 2, 2, 2, 2, 2]
upper_bounds = [
    len(models) - 1,
    len(learning_rates) - 1,
    len(dropout_rates) - 1,
    5, 50, 50, 50, 50, 50, 5,
]


def params_list_to_dict(params):
    if isinstance(params, dict):
        return params
    if isinstance(params, np.ndarray):
        params = params.astype(int)
    return {
        'model': models[params[0]] if len(params) > 0 else models[0],
        'learning_rate': learning_rates[params[1]] if len(params) > 1 else learning_rates[1],
        'dropout': dropout_rates[params[2]] if len(params) > 2 else dropout_rates[0],
        'n_hidden': params[3] if len(params) > 3 else 1,
        'hidden_1': params[4] if len(params) > 4 else 20,
        'hidden_2': params[5] if len(params) > 5 else 0,
        'hidden_3': params[6] if len(params) > 6 else 0,
        'hidden_4': params[7] if len(params) > 7 else 0,
        'hidden_5': params[8] if len(params) > 8 else 0,
        'lstm_layers': params[9] if len(params) > 9 else 2,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

# Module-level globals set by main() before running optimisation
cuda_off = True
dataset_name = "STX40"
column_name = 'Close'
trainset = None
valset = None
testset = None
tracker = None
device = None
num_epochs = 50
experiment_start_time = time.ctime()


def build_model(params):
    if params['model'] == models[0]:
        return LstmModel(2, n_hidden=params['n_hidden'], n_layers=params['lstm_layers'], dropout=params['dropout'])
    sizes = [params[f'hidden_{i}'] for i in range(1, 6) if params.get(f'hidden_{i}', 0) > 0]
    return MlpModel(n_hidden=params['n_hidden'], hidden_sizes=sizes, dropout_rate=params['dropout'])


def train(params, model_only=False):
    global num_epochs
    if isinstance(params, list):
        params = params_list_to_dict(params)

    if 'num_epochs' not in globals():
        num_epochs = 200

    start_time = time.time()
    learning_rate = params.get('learning_rate', 0.01)
    if params.get('num_epochs') is not None:
        num_epochs = params['num_epochs']

    model = build_model(params)
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for _ in range(num_epochs):
        for X, y in trainset:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

    if model_only:
        return model

    actual, predicted = [], []
    with torch.no_grad():
        for X, y in valset:
            X, y = X.to(device), y.to(device)
            actual.extend(y)
            predicted.extend(model(X))

    actual, predicted = torch.stack(actual), torch.stack(predicted)
    end_time = time.time()
    mse = F.mse_loss(actual, predicted).cpu().item()

    if isinstance(tracker, PerformanceTracker):
        tracker.add_row(start_time, end_time, mse, model, params)
    return mse


def test(model):
    model.to(device)
    actual, predicted = [], []
    with torch.no_grad():
        for X, y in testset:
            X, y = X.to(device), y.to(device)
            actual.extend(i.cpu().numpy() for i in y)
            predicted.extend(i.cpu().numpy() for i in model(X))
    actual = np.stack(actual)
    predicted = np.stack(predicted)
    return actual, predicted, np.square(np.subtract(actual, predicted)).mean()


# ---------------------------------------------------------------------------
# CASH problem definition
# ---------------------------------------------------------------------------

class CASH(Problem):
    def __init__(self):
        super().__init__(n_var=len(lower_bounds), n_obj=1, xl=lower_bounds, xu=upper_bounds)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.array([train(params_list_to_dict(x[i])) for i in range(x.shape[0])])


def discrete_random_search(f, lb, ub, iterations=10):
    D = len(lb)
    best_f = float('inf')
    best_x = [None] * D
    for _ in range(iterations):
        new_x = [random.randint(lb[d], ub[d]) for d in range(D)]
        new_f = f(new_x)
        if new_f < best_f:
            best_f = new_f
            best_x = new_x
    return {'best_x': best_x, 'best_f': best_f}


# ---------------------------------------------------------------------------
# Performance tracking
# ---------------------------------------------------------------------------

class PerformanceTracker:

    def __init__(self, experiment_name=None, pop_size=1):
        self.start_time = time.time()
        self.experiment_name = experiment_name or f"Trial at {time.ctime()}"
        self.rows = []
        self.results = []
        self.models = []
        self.params = []
        self.pop_size = pop_size
        self.buffer_earliest_start = time.time() + 9999
        self.buffer_latest_finish = time.time()
        self.buffer_best_mse = float('inf')
        self.buffer_best_model = None
        self.buffer_best_params = None
        self.iteration = 0
        self.function_evaluations = 0
        self.best_model = None
        self.best_model_params = None
        self.best_mse = float('inf')
        self.predicted = None

    def add_row(self, start_time, end_time, val_mse, model=None, params=None):
        self.function_evaluations += 1
        self.rows.append({
            'function_evaluations': self.function_evaluations,
            'iteration': self.iteration + 1,
            'start_time': start_time,
            'end_time': end_time,
            'val_mse': val_mse,
        })
        self.buffer_earliest_start = min(start_time, self.buffer_earliest_start)
        self.buffer_latest_finish = max(end_time, self.buffer_latest_finish)

        if val_mse < self.best_mse:
            self.best_mse = val_mse
            self.best_model = model
            self.best_model_params = params

        if val_mse < self.buffer_best_mse:
            self.buffer_best_mse = val_mse
            if model is not None:
                self.buffer_best_model = model
            if params is not None:
                self.buffer_best_params = params

        if self.function_evaluations % self.pop_size == 0:
            self.iteration += 1
            self.results.append({
                'iteration': self.iteration,
                'start_time': self.buffer_earliest_start,
                'end_time': self.buffer_latest_finish,
                'val_mse': self.buffer_best_mse,
            })
            if self.buffer_best_params is not None:
                self.params.append(self.buffer_best_params)
            if self.buffer_best_model is not None:
                self.models.append(self.buffer_best_model)
            self.buffer_earliest_start = time.time() + 9999
            self.buffer_latest_finish = time.time()
            self.buffer_best_mse = float('inf')
            self.buffer_best_model = None
            self.buffer_best_params = None

    @property
    def test_best_model(self):
        predicted_s, predicted_l, mse_arr = [], [], []
        for _ in range(5):
            model = train(self.best_model_params, model_only=True)
            self.actual, predicted, mse = test(model)
            predicted_s.append(predicted[:, 0])
            predicted_l.append(predicted[:, 1])
            mse_arr.append(mse)

        predicted_s, predicted_l = np.stack(predicted_s), np.stack(predicted_l)
        avg_s, avg_l = predicted_s.mean(axis=0), predicted_l.mean(axis=0)
        self.best_model_test_mse = np.array(mse_arr).mean()

        self.test_results = pd.DataFrame({
            'actual_slope': self.actual[:, 0],
            'actual_length': self.actual[:, 1],
            'predicted_slope': avg_s,
            'predicted_length': avg_l,
        })

        rmse = lambda a, p: np.sqrt(np.square(np.subtract(a, p)).mean())
        self.slope_rmse = rmse(self.actual[:, 0], avg_s)
        self.duration_rmse = rmse(self.actual[:, 1], avg_l)
        return self.test_results

    def summary(self):
        total_time = self.rows[-1]['end_time'] - self.rows[0]['start_time']
        idx = ['best model', 'val mse', 'test mse', 'total time', 'function calls', 'iterations']
        vals = [str(self.best_model_params), self.best_mse, self.best_model_test_mse,
                total_time, self.function_evaluations, self.iteration]
        return pd.DataFrame(vals, index=idx, columns=['Value'])

    def SDA(self):
        idx = ['Slope RMSE', 'Duration RMSE', 'Average RMSE']
        vals = [round(self.slope_rmse, 3), round(self.duration_rmse, 3),
                round(np.sqrt(self.best_model_test_mse), 3)]
        return pd.DataFrame(vals, index=idx, columns=['Value'])

    def get_results(self):
        return pd.DataFrame(self.results)

    def get_rows(self):
        return pd.DataFrame(self.rows)

    def get_params(self):
        return pd.DataFrame(self.params)

    def export(self):
        pd.set_option('display.float_format', '{:.2f}'.format)
        self.get_results().to_csv(self._csv_name('Results'))
        self.get_rows().to_csv(self._csv_name('Rows'))
        self.get_params().to_csv(self._csv_name('Params'))
        self.test_best_model.to_csv(self._csv_name('Predictions'))
        self.summary().to_csv(self._csv_name('Summary'))
        self.SDA().to_csv(self._csv_name('SDA'))

    def _csv_name(self, name):
        p = Path(f"Experiment Results ({experiment_start_time})") / self.experiment_name
        p.mkdir(parents=True, exist_ok=True)
        return str(p / f"{name}.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global device, num_epochs, trainset, valset, testset, tracker

    datasets = ['NYSE', 'NASDAQ', 'STX40']
    algos = ['random', 'ps', 'de']
    budget = 540

    if len(sys.argv) >= 2 and sys.argv[1] != 'all':
        datasets = sys.argv[1].split(',')

    if len(sys.argv) >= 3:
        budget, num_epochs = (int(v) for v in sys.argv[2].split(','))

    print(f"Datasets: {datasets}  budget: {budget}  num_epochs: {num_epochs}")

    device = torch.device("cpu") if cuda_off else torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    for d in datasets:
        trainset, valset, testset = get_data(d)
        for a in algos:
            print(d, a)
            tracker_name = f"{a} {d}"
            if a == 'random':
                pop_size = 1
                iters = math.floor(budget / pop_size)
                tracker = PerformanceTracker(tracker_name, pop_size=pop_size)
                discrete_random_search(train, lower_bounds, upper_bounds, iters)
                tracker.export()
            elif a == 'ps':
                pop_size = 45
                iters = math.floor(budget / pop_size)
                tracker = PerformanceTracker(tracker_name, pop_size=pop_size)
                ps = PatternSearch(sampling=IntegerRandomSampling(), eliminate_duplicates=True)
                minimize(CASH(), ps, ('n_iter', iters), seed=1, verbose=False)
                tracker.export()
            elif a == 'de':
                pop_size = 20
                iters = math.floor(budget / pop_size)
                tracker = PerformanceTracker(tracker_name, pop_size=pop_size)
                de = DE(
                    pop_size=pop_size,
                    F=0.9,
                    CR=0.45,
                    sampling=IntegerRandomSampling(),
                    eliminate_duplicates=True,
                )
                res = minimize(CASH(), de, ('n_gen', iters), seed=1, save_history=True)
                print(f"Best solution found: X={res.X}  F={res.F}  CV={res.CV}")
                tracker.export()


if __name__ == '__main__':
    main()
