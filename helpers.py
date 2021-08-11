from scipy import stats as sp

class TrendLine:

    def __init__(self, slope, length):
        self.length = length
        self.slope = slope

    def __str__(self):
        return f" <{self.length}, {self.slope}>"

class StraightLine:

    def __init__(self, slope = 0, intercept = 0,**kwargs):
        for key, value in kwargs.items():
            print(key,value)

        self.slope = slope
        self.intercept = intercept

    @classmethod
    def regress(cls, T):
        line = cls()
        x = (T.index - T.index[0]).days.values
        y = T.values
        line.slope, line.intercept, line.r_value, line.p_value, line.error = sp.linregress(x, y)
        return line

    def __str__(self):
        return f" <{self.intercept}, {self.slope}>"