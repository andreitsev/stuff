import pandas as pd
import statsmodels.formula.api as smf
import itertools

def adl_regression(dataframe, target_variable, lags_dict):
    """
    param dataframe: df содержащий необходимые переменные
    param target_variable: (str) название зависимой переменной
    param lags_dict: (dict) словарь: ключи - названия (str) колонок в dataframe, которые будут 
    использоваться для регрессии, значения (list) - списки лагов для каждой из переменных
    """
    dat = dataframe
    all_length = list(itertools.chain.from_iterable([[len(dat['{}'.format(varname)].values[lag:]) for lag in lags_dict[varname]] for varname in lags_dict]))         
    data_list = []
    min_length = min(all_length)
    for i in range(min_length):
        data_list.append(list(itertools.chain.from_iterable([[dat['{}'.format(varname)].values[lag:][i] for lag in lags_dict[varname]] for varname in lags_dict])))
        data_list[i].append(dat[target_variable].values[i])
    data_for_regression = pd.DataFrame(data_list, columns=list(itertools.chain.from_iterable([['{}_{}'.format(varname, lag) for lag in lags_dict[varname]] for varname in lags_dict])) + ['target'])
    formula = ' '.join(['{}'.format(varname) + ' + ' for varname in data_for_regression.columns[:-1]])[:-3]
    model = smf.ols('target ~ ' + formula, data=data_for_regression).fit()
    return model
