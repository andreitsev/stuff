from typing import Union, List, Dict, Tuple
from collections import Counter

import pandas as pd
import numpy as np
from tqdm import tqdm

def df_unq_and_na(
        df: pd.DataFrame,
        show_unq_vals_threshold: int = 20,
        verbose: bool = True,
        key_val_gap: int = 50,
        columns: list = None
) -> None:

    """
    Показывает метаинформацию по датафрейму df - число строк, число нанов по каждой колонке, уникальные значения
    в каждой колонке
    Args:
        df: pd.DataFrame, для которого сделать диагностику
        show_unq_vals_threshold: сколько уникальных значений переменной показывать
        verbose: визуализировать ли цикл по колонкам
        key_val_gap: ширина '-' между уникальным значением в колонке и её числом встречаний
    """

    print(f'Размер датафрейма: {df.shape[0]:,} x {df.shape[1]:,}')
    cols = df.columns
    if columns is not None:
        cols = columns
    for col in tqdm(cols) if verbose else cols:
        print('=' * 100)
        print(col, end='\n' * 2)
        # Если в колоке дата - сортируем по убыванию даты
        is_datetime_type = str(df[col].dtype).find('datetime') != -1
        unq_vals_dict = df[col].value_counts().sort_index(ascending=False).to_dict() if is_datetime_type else df[
            col].value_counts().to_dict()
        unq_vals_percents_dict = df[col].value_counts(1).sort_index(ascending=False).to_dict() if is_datetime_type else \
        df[col].value_counts(1).to_dict()
        print(' ' * 2, f'Число уникальных значений: {len(unq_vals_dict):,}')
        nans = df[col].isnull().sum()
        print(' ' * 2, f"Число NaN's: {nans:,} ({round(nans * 100 / len(df), 1)}%)")
        print(' ' * 2, 'Уникальные значения:', end='\n' * 2)
        n_ = 0
        for (key1, val1), (key2, val2) in zip(unq_vals_dict.items(), unq_vals_percents_dict.items()):
            print(' ' * 6, key1, '-' * (key_val_gap - len(f"{val1:,} ({round(val2 * 100, 2)}%)") - len(str(key1))),
                  f"{val1:,} ({round(val2 * 100, 2)}%)")
            n_ += 1
            if n_ == show_unq_vals_threshold:
                break
    return


def split_dataframe(
        df: pd.DataFrame,
        target_col: Union[str, np.ndarray, pd.Series] = None,
        time_col: str = None,
        proportion_splits: List[float] = [0.64, 0.18, 0.18],
        n_splits: int = None,
        split_type: str = 'stratified',
        seed: int = 42,
        return_target_statistics: bool=False,
) -> Union[Tuple[List], List]:
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

    if return_target_statistics:
        assert target_col is not None, 'target_col должна быть определена при return_target_statistics=True'

    tmp_df = df.reset_index(drop=True).copy()
    len_df = len(df)
    # Сортируем датафрейм по времени -------------------------------------------------
    if split_type == 'time_split':
        assert time_col in tmp_df.columns, f'time_col нету в df.columns!'
        tmp_df = tmp_df.sort_values(by=[time_col], ascending=True).reset_index(drop=True)
    # ---------------------------------------------------------------------------------

    # SEED ---------------
    np.random.seed(seed)
    # ---------------------

    # записываем значения таргета в отдельную переменную ------------------------------------------
    if target_col is not None:
        if isinstance(target_col, str):
            if target_col in tmp_df.columns:
                target_vals = tmp_df[target_col].values
                tmp_df = tmp_df.drop([target_col], axis=1)
            else:
                raise ValueError("target_col нету в df.columns")
        else:
            target_vals = target_col.values if isinstance(target_col, pd.core.series.Series) else target_col
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


    if return_target_statistics:
        split_results_str = ''
        for i, (df_, target_) in enumerate(resulting_list):
            n_obj = len(target_)
            split_results_str += f'dataset{i+1} size: {n_obj:,}\n'
            unq_target_vals_ = dict(Counter(target_))
            try:
                unq_target_vals_ = sorted(unq_target_vals_.items(), key=lambda x: x[0])
            except:
                unq_target_vals_ = unq_target_vals_.items()
            for target_val, counts_ in unq_target_vals_:
                split_results_str += ' ' * 4 + f'{target_val} - {counts_:,} ({round(100 * counts_ / n_obj, 2)}%)\n'
            split_results_str += '\n'

        return resulting_list, split_results_str

    else:
        return resulting_list