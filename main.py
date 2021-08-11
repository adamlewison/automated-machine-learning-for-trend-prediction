import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import linear_model,metrics
from scipy import stats as sp
from helpers import TrendLine,StraightLine

def timeseries(csv_file_name, value_col, index_col = 'Date', parse_dates = True):
    return pd.read_csv(Path().joinpath('data', csv_file_name + '.csv'), index_col= index_col, parse_dates = parse_dates)[value_col]

def sliding_window(T, max_error = 100):
    arr = []
    anchor = 0
    while anchor < T.size:
        i = 2
        line = StraightLine.regress(T[anchor : anchor + i])
        previous_line = None
        while line.error < max_error:
            i += 1
            previous_line = line
            line = StraightLine.regress(T[anchor: anchor + i])
        arr.append(TrendLine(previous_line.slope, (i - 1)))
        anchor += i
    print("Line Segments Constructed.")
    return arr

#%%
jse = timeseries('jse-test', 'Close')
T = jse[0:5]


#%%

def main():
    jse = timeseries('jse-test', 'Close')
    T = jse[0:4]

    print('dont end')
    segs = sliding_window(jse)
    print(segs)

if __name__ == '__main__':
    main()


