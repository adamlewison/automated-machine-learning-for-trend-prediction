"""Tests for core data processing and model components."""

import math
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# helpers.py — TrendLine and StraightLine
# ---------------------------------------------------------------------------

class TestTrendLine:
    def setup_method(self):
        from helpers import TrendLine
        self.TrendLine = TrendLine

    def test_attributes(self):
        tl = self.TrendLine(slope=2.5, length=10)
        assert tl.slope == 2.5
        assert tl.length == 10

    def test_repr_rounds_slope(self):
        tl = self.TrendLine(slope=1.23456, length=5)
        assert "1.23" in repr(tl)

    def test_str(self):
        tl = self.TrendLine(slope=1.0, length=4)
        assert "1.0" in str(tl)
        assert "4" in str(tl)


# ---------------------------------------------------------------------------
# exp.py — sliding_window and create_sequences (pure logic, no GPU)
# ---------------------------------------------------------------------------

class TestSlidingWindow:
    def setup_method(self):
        from exp import sliding_window
        self.sliding_window = sliding_window

    def _make_series(self, n=50, freq=1):
        idx = pd.RangeIndex(start=0, stop=n * freq, step=freq)
        return pd.Series(np.linspace(0, 10, n), index=idx)

    def test_returns_dataframe(self):
        s = self._make_series()
        result = self.sliding_window(s)
        assert isinstance(result, pd.DataFrame)

    def test_columns(self):
        s = self._make_series()
        result = self.sliding_window(s)
        assert list(result.columns) == ['Slope', 'Length']

    def test_nonempty(self):
        s = self._make_series()
        result = self.sliding_window(s)
        assert len(result) > 0

    def test_returns_list_when_df_false(self):
        s = self._make_series()
        result = self.sliding_window(s, df=False)
        assert isinstance(result, list)

    def test_noisy_series(self):
        rng = np.random.default_rng(42)
        idx = pd.RangeIndex(100)
        s = pd.Series(rng.normal(0, 1, 100), index=idx)
        result = self.sliding_window(s)
        assert len(result) > 0


class TestCreateSequences:
    def setup_method(self):
        from exp import create_sequences
        self.create_sequences = create_sequences

    def _make_df(self, n=20):
        return pd.DataFrame({'Slope': np.random.randn(n), 'Length': np.random.rand(n) * 10})

    def test_length(self):
        df = self._make_df(20)
        seqs = self.create_sequences(df, ['Slope', 'Length'], sequence_length=4)
        assert len(seqs) == 20 - 4

    def test_tuple_structure(self):
        import torch
        df = self._make_df(10)
        seqs = self.create_sequences(df, ['Slope', 'Length'], sequence_length=4)
        x, y = seqs[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_sequence_shape(self):
        import torch
        df = self._make_df(10)
        seqs = self.create_sequences(df, ['Slope', 'Length'], sequence_length=4)
        x, _ = seqs[0]
        assert x.shape == (4, 2)


# ---------------------------------------------------------------------------
# exp.py — params_list_to_dict
# ---------------------------------------------------------------------------

class TestParamsListToDict:
    def setup_method(self):
        from exp import params_list_to_dict, models, learning_rates, dropout_rates
        self.fn = params_list_to_dict
        self.models = models
        self.learning_rates = learning_rates
        self.dropout_rates = dropout_rates

    def test_passthrough_dict(self):
        d = {'model': 'lstm', 'learning_rate': 0.01}
        assert self.fn(d) is d

    def test_list_input(self):
        params = [0, 1, 2, 3, 20, 0, 0, 0, 0]
        result = self.fn(params)
        assert result['model'] == self.models[0]
        assert result['learning_rate'] == self.learning_rates[1]
        assert result['dropout'] == self.dropout_rates[2]
        assert result['n_hidden'] == 3

    def test_numpy_input(self):
        params = np.array([1, 0, 0, 2, 15, 0, 0, 0, 0])
        result = self.fn(params)
        assert result['model'] == self.models[1]

    def test_defaults_for_short_list(self):
        result = self.fn([0])
        assert result['model'] == self.models[0]
        assert result['learning_rate'] == self.learning_rates[1]


# ---------------------------------------------------------------------------
# exp.py — MlpModel / LstmModel (forward pass, CPU only)
# ---------------------------------------------------------------------------

class TestMlpModel:
    def setup_method(self):
        import torch
        from exp import MlpModel
        self.torch = torch
        self.MlpModel = MlpModel

    def test_forward_shape(self):
        model = self.MlpModel(n_features=2, n_hidden=2, hidden_sizes=[8, 4])
        x = self.torch.randn(16, 4, 2)  # batch=16, seq=4, features=2
        out = model(x)
        assert out.shape == (16, 2)

    def test_single_hidden_layer(self):
        model = self.MlpModel(n_features=2, n_hidden=1, hidden_sizes=[8])
        x = self.torch.randn(4, 4, 2)
        assert model(x).shape == (4, 2)


class TestLstmModel:
    def setup_method(self):
        import torch
        from exp import LstmModel
        self.torch = torch
        self.LstmModel = LstmModel

    def test_forward_shape(self):
        model = self.LstmModel(n_features=2, n_hidden=8, n_layers=2)
        x = self.torch.randn(16, 4, 2)
        out = model(x)
        assert out.shape == (16, 2)

    def test_single_layer(self):
        model = self.LstmModel(n_features=2, n_hidden=4, n_layers=1, dropout=0.0)
        x = self.torch.randn(8, 4, 2)
        assert model(x).shape == (8, 2)
