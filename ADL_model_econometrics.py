import pandas as pd
import statsmodels.formula.api as smf
import itertools

def lags_df(dataframe, lags_dict, target_variable):
    """
    param dataframe: df содержащий необходимые переменные
    param target_variable: (str) название зависимой переменной
    param lags_dict: (dict) словарь: ключи - названия (str) колонок в dataframe, которые будут 
    использоваться для регрессии, значения (list) - списки лагов для каждой из переменных
    """
    dat_dict = {}
    max_lag = 0
    for varname in lags_dict:
        m_l = np.max(lags_dict[varname])
        if m_l > max_lag:
            max_lag = m_l
    target_variable_array = dataframe[target_variable].values[max_lag+1:]
    dat_dict.update({'target': target_variable_array})
    target_len = len(target_variable_array)
    colnames_list = ['target']
    for varname in lags_dict:
        variable_len = len(dataframe[varname])
        for lag in lags_dict[varname]:
            dat_dict.update({varname + '_{}'.format(lag): dataframe[varname].values[max_lag+1-lag : variable_len-lag]})
            colnames_list.append('{}_{}'.format(varname, lag))
    return pd.DataFrame(dat_dict, columns=colnames_list)
    

def adl_regression(dataframe, target_variable, lags_dict=None,
                   cov_type='nonrobust', model_type='gls'):
    """
    param dataframe: df содержащий необходимые переменные
    param target_variable: (str) название зависимой переменной
    param lags_dict: (dict) словарь: ключи - названия (str) колонок в dataframe, которые будут 
    использоваться для регрессии, значения (list) - списки лагов для каждой из переменных
    param cov_type: ['nonrobust', 'HC0', 'HC1', 'HC2', 'HC3']
    param model_type: ['ols', 'gls']
    """
    if lags_dict is not None:
        dat_dict = {}
        max_lag = 0
        for varname in lags_dict:
            m_l = np.max(lags_dict[varname])
            if m_l > max_lag:
                max_lag = m_l
        target_variable_array = dataframe[target_variable].values[max_lag+1:]
        dat_dict.update({'target': target_variable_array})
        target_len = len(target_variable_array)
        colnames_list = ['target']
        for varname in lags_dict:
            variable_len = len(dataframe[varname])
            for lag in lags_dict[varname]:
                dat_dict.update({varname + '_{}'.format(lag): dataframe[varname].values[max_lag+1-lag : variable_len-lag]})
                colnames_list.append('{}_{}'.format(varname, lag))
        data_for_regression = pd.DataFrame(dat_dict, columns=colnames_list)
    else:
        data_for_regression = dataframe
    formula = ' '.join(['{}'.format(varname) + ' + ' for varname in data_for_regression.columns[:-1]])[:-3]
    if model_type == 'ols':
        model = smf.ols('target ~ ' + formula, data=data_for_regression).fit(cov_type=cov_type)
    else:
        model = smf.gls('target ~ ' + formula, data=data_for_regression).fit()
    return model