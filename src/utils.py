import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
import json
from src.portfolio_selection_ga import optimize_markowitz


def get_combinations():
    rg = [item / 100 for item in list(range(0, 101, 20))]
    risk_aver_list = rg
    min_ret_percentile_list = rg
    max_var_perc_list = rg
    window_list = [24, 52]
    step_list = [1, 4]

    grid = []
    for risk_aver, min_ret_percentile, max_var_perc, window, step in product(
            risk_aver_list, min_ret_percentile_list, max_var_perc_list,
            window_list, step_list):
        grid.append({
            "risk_aver": risk_aver,
            "min_ret_percentile": min_ret_percentile,
            "max_var_perc": max_var_perc,
            "window": window,
            "step": step
        })
    return grid


def get_returns_df():
    data = pd.read_excel('../data/base_dados.xlsx',
                         index_col="Date").pct_change().dropna()
    data.index = data.index.astype(str)
    return data


def add_noise(ret_df: pd.DataFrame, num_cols: int, num_noise: int,
              std_mult) -> pd.DataFrame:
    ret_df = ret_df.copy(deep=True)
    cols = np.random.choice(ret_df.columns, num_cols, replace=False).tolist()
    print(cols)
    for col in cols:
        mean, std = 0, std_mult * ret_df.std()[col]
        noise = np.random.normal(mean, std, num_noise)
        idx = np.random.randint(0, ret_df.shape[0] - num_noise)
        ret_df.iloc[idx:idx + num_noise][col] += noise
    return ret_df


def find_weights(data: pd.DataFrame, parameters: list) -> pd.DataFrame:
    risk_aver, min_ret_percentile, max_var_perc, window, step = parameters.values(
    )
    df = pd.DataFrame(range(window + 1, data.shape[0], step), columns=["idx"])
    df.idx = df.idx - 1
    df["data"] = df.idx.apply(lambda x: data.iloc[x - window:x])
    df["Date"] = df.idx.apply(lambda x: data.index[x])
    df["weights"] = df.data.apply(
        lambda x: optimize_markowitz(x,
                                     risk_aver=risk_aver,
                                     min_ret_percentile=min_ret_percentile,
                                     max_var_perc=max_var_perc))
    df[data.columns] = pd.DataFrame(df.weights.tolist(), index=df.index)
    df = pd.merge(pd.Series(data.index[window:]), df, on="Date", how='left')
    df.fillna(method='ffill', inplace=True)
    df["portfolio_return"] = df.apply(lambda x: x["weights"] @ data.iloc[x["idx"]], axis=1)
    df.drop(["idx", "data"], axis=1, inplace=True)
    return df


def train(data: pd.DataFrame, grid: list):
    results = []
    num_ast = data.shape[1]

    for prms in tqdm(grid):
        weights_df = pd.DataFrame(columns=data.columns)

        for i in tqdm(range(prms["window"], len(data), prms["step"])):
            hist_data = data.iloc[i - prms["window"]:i].copy()
            best_individual = optimize_markowitz(hist_data, prms["risk_aver"],
                                                 prms["min_ret_percentile"],
                                                 prms["max_var_perc"])
            weights_dict = pd.Series(best_individual,
                                     index=hist_data.columns).to_dict()
            weights_dict["Date"] = data.iloc[i - prms["window"]:i +
                                             1].index.max()
            weights_df = weights_df.append(weights_dict, ignore_index=True)

        weights_df = weights_df.set_index("Date")
        dates_index = data.index[data.index > data.iloc[prms["window"] -
                                                        1:].index.min()]
        weights_df = weights_df.reindex(index=dates_index, method='ffill')

        results.append({
            "weights":
            weights_df.copy(),
            "params":
            prms,
            "portfolio_returns":
            (weights_df *
             data[data.index > data.iloc[prms["window"] -
                                         1:].index.min()]).sum(axis=1)
        })

    for res in results:
        for col in res["weights"].keys():
            for k, v in res["weights"][col].items():
                res["weights"][col] = {
                    str(k): v
                    for k, v in res["weights"][col].items()
                }
                res["portfolio_returns"] = {
                    str(k): v
                    for k, v in res["portfolio_returns"].items()
                }

    with open('../data/results.json', 'w') as f:
        json.dump(results, f, default=str)

    return results


def read_train():
    with open('../data/results.json') as f:
        results = json.load(f)
    for item in results:
        returns_df = pd.Series(item["portfolio_returns"])
        item["portfolio_returns"] = returns_df
        item["portfolio_returns"].index = pd.to_datetime(
            item["portfolio_returns"].index)

        weights_df = pd.DataFrame(item["weights"])
        weights_df.index = pd.to_datetime(weights_df.index)
        item["weights"] = weights_df
    return results
