from copy import deepcopy
import pandas as pd
import statsmodels.formula.api as smf

def multiple_correlation_coefficient(df):
    """
    :params df - датафрейм с переменными, для которых надо посчитать многомерный корелляционный коэффициент. Коэффициент 
    под считывается только для числовых переменных, но для подсчёта использует так же категориальные (object) переменные.
    """
    numerical_cols = [col for col in df.columns if df[col].dtype != 'object']
    rsquare_dict = dict(zip(numerical_cols, [0 for i in range(len(numerical_cols))]))
    for col in numerical_cols:
        new_cols = list(deepcopy(df.columns))
        new_cols.remove(col)
        formula = col + ' ~ ' + ' + '.join([restcol if df[restcol].dtype != 'object' else 'C(' + restcol + ')'  for restcol in new_cols])
        model = smf.ols(formula, data=df).fit()
        rsquare_dict[col] = model.rsquared_adj**(0.5)
    return pd.Series(rsquare_dict)
