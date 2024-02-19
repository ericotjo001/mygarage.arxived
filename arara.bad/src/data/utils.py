import argparse
import numpy as np
import pandas as pd

# Basic packaging
def package_df(df, config):
    # df is a dataframe. See data/toyexample.py get_toy_data.

    Y = df[config["DV"]].to_numpy() # (n,1)
    X = df[ [config["IV"]] + list(config["NUMERICAL_FEATURES"]) 
        + list(config["TOKEN_FEATURES"])].to_numpy() # (n, 1+_features)

    data_package = {'Y':Y, 'X':X, 'config': config}
    return data_package

def compute_means(df, DV, IV, idx2IV):
    # idx2IV: like ["placebo", "drugX"] or [0,1]
    means = {}
    df_ = df[[DV, IV]]

    for group_ in idx2IV:
        m_ = np.mean(df_[df_[IV]==group_][DV]) 
        means[group_] = m_
    return means