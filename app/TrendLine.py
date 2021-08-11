class TrendLine:
    def __init__(self, slope, intercept, error):
        self.slope = slope
        self.intercept = intercept
        self.error = error

    def __str__(self):
        return "y = " + self.slope + "x + " + self.intercept

    def setLength(self, length):
        self.length = length
