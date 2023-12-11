import numpy as np
import pandas as pd

def standardize_numeric(series: pd.Series, use_log: bool = False) -> pd.Series:
    # write code here that optionally takes the log of in the input series, then standardizes it
    if use_log:
        series = np.log(series)
    series = (series - np.mean(series)) / np.std(series)
    return series

