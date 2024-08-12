import pandas as pd
import numpy as np


def add_noise(ret_df: pd.DataFrame, cols: list, num_noise: int,
              std_mult) -> pd.DataFrame:
    ret_df = ret_df.copy(deep=True)

    for col in cols:
        mean, std = 0, std_mult * ret_df.std()[col]
        noise = np.random.normal(mean, std, num_noise)
        idx = np.random.randint(0, ret_df.shape[0] - num_noise)
        ret_df.iloc[idx:idx + num_noise][col] += noise
    return ret_df
