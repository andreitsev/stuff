import numpy as np
import os
import sys
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def pairwise_boxplot(
        df: pd.DataFrame, feature_x: str, feature_y: str, bins: int = 10,
        quantile_lower: float = None, quantile_upper: float = 0.9, title: str = None,
        save_path: str = None
):
    """
     Рисует boxplot для переменной feature_y в зависимости от значений переменной feature_x
    """

    tmp_df = df.copy()

    plt.figure(figsize=(14, 4))
    if title is not None:
        plt.title(title, fontsize=15);
    # разбиваем значение фичи по бинам ----------------------------------------
    a = pd.cut(tmp_df[feature_x], bins=bins).value_counts().sort_index().keys()
    tmp_dict = dict(zip(a, range(len(a))))
    re_tmp_dict = {x: y for y, x in tmp_dict.items()}
    tmp_df[feature_x + '_cut'] = tmp_df[feature_x].map(tmp_dict)
    unq_vals = tmp_df[feature_x + '_cut'].unique()
    # --------------------------------------------------------------------------
    sns.boxplot(x=feature_x + '_cut', y=feature_y, data=tmp_df);
    plt.xticks(np.arange(len([val for val in unq_vals if not np.isnan(val)])),
               sorted([re_tmp_dict[val] for val in unq_vals if not np.isnan(val)]),
               rotation=30)
    plt.ylim(tmp_df[feature_y].quantile(quantile_lower) if quantile_lower is not None else -100,
             tmp_df[feature_y].quantile(quantile_upper));
    if save_path is not None:
        plt.savefig(save_path);
    plt.show();

