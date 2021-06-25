import pandas as pd
import numpy as np
from typing import List, Dict, Callable, Union

def split_dataframe(
        df: pd.DataFrame,
        target_col: Union[str, np.ndarray] = None,
        time_col: str = None,
        proportion_splits: List[float] = [0.64, 0.18, 0.18],
        n_splits: int = None,
        split_type: str = 'random',
        seed: int = 42,
) -> Dict:
    """
    Сплитит датафрейм df на n_splits частей (или на len(proroprtion_splits) частей)
    Args:
        df: датафрейм, который нужно сплитить
        target_col: вектор с таргетами или столбец из df
        time_col: стобец в df со временем (главное, чтобы по нему можно было сортировать)
        proportion_splits: датафреймы таких пропорций нужно получить на выходе
        n_splits: на сколько датафреймов сплитить (нужен либо этот аргумент, либо proportion_splits)
        split_type:
            'stratified' - сплитить по значениям в target_col
            'time_split' - сплитить по дате (колонка time_col)
            'random_split' - сплитить случайным образом
    Returns:
        список из n_splits (или len(proportion_splits)) таплов: (pd.DataFrame, np.ndarray) (если target_col не None)
            или список pd.DataFrame'ов
    """

    tmp_df = df.reset_index(drop=True).copy()
    len_df = len(df)
    assert split_type in ['random', 'stratified', 'time_split'], \
        'split_type может быть "time_split", "random" или "stratified"'
    if n_splits is not None:
        assert n_splits > 1, 'n_splits должен быть > 1'
    if proportion_splits is not None:
        assert sum([prop >= 0 for prop in proportion_splits]) == len(proportion_splits), \
            'элементы в proportion_splits должны быть неотрицательным'
        assert abs(sum(proportion_splits) - 1) <= 1e-2, \
            'proportion_splits должны суммироваться в 1'

    if split_type == 'stratified' and target_col is None:
        raise ValueError("при split_type='stratified' должна быть задана target_col")

    if split_type == 'time_split' and time_col is None:
        raise ValueError("при split_type='time_split' должна быть задана time_col")

    # SEED ---------------
    np.random.seed(seed)
    # ---------------------

    # записываем значения таргета в отдельную переменную ------------------------------------------
    if target_col is not None:
        if isinstance(target_col, str):
            if target_col in tmp_df.columns:
                target_vals = tmp_df[target_col].values
            else:
                raise ValueError("target_col нету в df.columns")
        else:
            target_vals = target_col
    # ---------------------------------------------------------------------------------------------

    # Random split ----------------------------------------------------------------------------------------------------
    if split_type == 'random':
        if proportion_splits is not None:
            split_positions = np.cumsum([int(prop * len_df) for prop in proportion_splits])[:-1]
        splits_idxs = np.array_split(
            ary=np.random.permutation(len_df),
            indices_or_sections=split_positions if proportion_splits is not None else n_splits,
        )
        if target_col is not None:
            resulting_list = [(tmp_df.loc[current_idxs], target_vals[current_idxs]) for current_idxs in splits_idxs]
        else:
            resulting_list = [tmp_df.loc[current_idxs] for current_idxs in splits_idxs]

    # Stratified split ------------------------------------------------------------------------------------------------
    elif split_type == 'stratified':

        df_indexes_dict = {fold_number: [] for fold_number in
                           range(len(proportion_splits) if proportion_splits is not None else n_splits)}

        # множество уникальных значений тартега для того, чтобы по нему бить датафрейм
        unq_target_vals = np.unique(target_vals)
        for i, curr_unq_val in enumerate(unq_target_vals):
            curr_unq_vals_target_idxs = np.where(target_vals == curr_unq_val)[0]
            len_curr_unq_target = len(curr_unq_vals_target_idxs)

            if proportion_splits is not None:
                split_positions = np.cumsum([int(prop * len_curr_unq_target) for prop in proportion_splits])[:-1]
            splits_idxs = np.array_split(
                ary=np.random.permutation(curr_unq_vals_target_idxs),
                indices_or_sections=split_positions if proportion_splits is not None else n_splits,
            )

            [df_indexes_dict[fold_number].extend(curr_target_fold_idxs.tolist()) for
             fold_number, curr_target_fold_idxs in enumerate(splits_idxs)];

        resulting_list = [(tmp_df.loc[current_idxs], target_vals[current_idxs]) for current_idxs in
                          df_indexes_dict.values()]

    # Time split -----------------------------------------------------------------------------------------------------
    else:
        tmp_df.sort_values(by=[time_col], ascending=True, inplace=True)
        if proportion_splits is not None:
            split_positions = np.cumsum([int(prop * len_df) for prop in proportion_splits])[:-1]
        splits_idxs = np.array_split(
            ary=np.arange(len_df),
            indices_or_sections=split_positions if proportion_splits is not None else n_splits,
        )

        if target_col is not None:
            resulting_list = [(tmp_df.loc[current_idxs], target_vals[current_idxs]) for current_idxs in splits_idxs]
        else:
            resulting_list = [tmp_df.loc[current_idxs] for current_idxs in splits_idxs]

    return resulting_list
