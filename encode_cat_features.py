import numpy as np
import pandas as pd

from typing import List, Union
from category_encoders import OneHotEncoder, OrdinalEncoder, TargetEncoder


def encode_cat_features(
        df: pd.DataFrame,
        target_col: Union[str, np.ndarray] = None,
        cat_columns: List[str] = None,
        encoding_type: str = 'ohe',
) -> pd.DataFrame:
    """
    Возращает датафрейм с кодированными категориальными фичами

    Args:
        df: датафрейм, кэт фичи которого нужно закодировать
        cat_columns: список кэт фичей
        target_col: на основе какой фичи делать target_encoding
        encode_cat_features:
            - ohe
            - le
            - target_enc
    Returns:
        датафрейм с закодированными кэт фичами
    """

    if cat_columns is None:
        cat_columns = [col for col in df.columns if df[col].dtype == 'object' or str(df[col].dtype) == 'category']

    tmp_df = df[cat_columns]

    intersect_columns = set(cat_columns) & set(df.columns)
    if len(cat_columns) != len(intersect_columns):
        print('Не все фичи из cat_columns есть в df!')

    assert encoding_type in ['ohe', 'le', 'target_enc'], \
        "encoding_type может быть одним из ['ohe', 'le', 'target_enc']"

    if encoding_type == 'target_enc':
        assert target_col is not None, 'target_col не может быть None!'
        if isinstance(target_col, str):
            assert target_col in df.columns, 'target_col должна быть в df'
            target_values = df[target_col].values
        else:
            assert len(target_col) == len(df), 'len(target_col) != len(df)'
            target_values = np.array(target_col)

    if encoding_type == 'ohe':
        enc = OneHotEncoder(use_cat_names=True)
        enc.fit(tmp_df)
        tmp_df = enc.transform(tmp_df)
    elif encoding_type == 'le':
        enc = OrdinalEncoder()
        enc.fit(tmp_df)
        tmp_df = enc.transform(tmp_df)
    elif encoding_type == 'target_enc':
        enc = TargetEncoder()
        enc.fit(X=tmp_df, y=target_values)
        tmp_df = enc.transform(tmp_df)
    else:
        raise (ValueError, 'Неизвестный энкодинг')

    return tmp_df
