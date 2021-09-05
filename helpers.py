import math

import numpy as np
import pandas as pd
from scipy import stats as sp
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.regressionplots import abline_plot

class TrendLine:

    def __init__(self, slope, length):
        self.length = length
        self.slope = slope

    def __str__(self):
        return f"<{self.length}, {self.slope}>"

    def __repr__(self):
        return f"<{self.length}, {round(self.slope,2)}>"

class StraightLine:

    def __init__(self, slope = 0, intercept = 0,**kwargs):
        for key, value in kwargs.items():
            print(key,value)

        self.slope = slope
        self.intercept = intercept

    @classmethod
    def regress(cls, T):
        motif = pd.DataFrame(data = {'close': T.values, 'days': (T.index - T.index[0]).days.values})
        line = cls()
        x = (T.index - T.index[0]).days.values
        x = sm.add_constant(x)
        y = T.values
        model = sm.OLS(y, x).fit()
        """
        ax = motif.plot(x='days', y='close', kind='scatter')
        abline_plot(model_results= model, ax=ax)
        plt.show()
        """
        line.intercept, line.slope, line.error = model.params[0], np.rad2deg(math.atan(model.params[1])), math.sqrt(model.ssr)
        return line

    @classmethod
    def regress1(cls, T):
        line = cls()
        x = (T.index - T.index[0]).days.values
        y = T.values
        line.slope, line.intercept, line.r_value, line.p_value, line.error = sp.linregress(x, y)
        return line

    def __str__(self):
        return f" <{self.intercept}, {self.slope}>"