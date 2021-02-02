import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import ShuffleSplit
import inspect


def make_cv_metrics(
        folds_data: pd.DataFrame,
        folds_y: np.ndarray,
        searched_param: str,
        param_space: np.ndarray,
        fixed_params: dict,
        folds_indexes_dict: dict=None,
        cat_features: list = None,
        model: callable = lgb,
        outer_test_data: pd.DataFrame=None,
        outer_test_y: np.ndarray=None,
        metric_func: callable = roc_auc_score,
        seed: int = 42,
        n_folds: int = 5,
        verbose: bool = True
):

    """
    Считает метрики на кросс-валидации для модели model на данных folds_data и folds_y, перебирая значения параметра
    searched_param из множества param_space. Остальные параметры зафиксированы в fixed_params.

    Args:
        folds_data: Матрица объект-признак, которую режут на фолды
        folds_y: Вектор с таргетами
        searched_param: название параметры, для которого считать метрики на кросс-валидации
        param_space: множество значений параметра searched_param, которые перебрать
        fixed_params: остальные параметры модели model
        folds_indexes_dict: словарь с индексами для фолдов вида:
            {
                "fold_1": {
                    "train_indexes": np.array([...]),
                    "test_indexes": np.array([...])
                },
                "fold_2": {
                    "train_indexes": np.array([...]),
                    "test_indexes": np.array([...])
                },
                ...
            }
            Если этот словарь определён, то не использовать ShuffleSplit
        cat_features: список с категориальными фичами
        model: что-то у чего есть методы .fit и .predict (пока что реализована только возможность model = lgb)
        outer_test_data: Если не None, то это матрица объект-признак, для которой посчитать метрику
            (это должны быть объекты, не входящие в folds_data)
        outer_test_y: Если не None, то вектор таргетов, (это должны быть объекты, не входящие в folds_data)
        metric_func: функция подсчёта метрики (должны быть аргументы y_true и y_pred/y_score)
        seed: random_seed
        n_folds: число фолдов
        verbose: отслеживать ли процесс подсчёта скоров

    Returns:
        Словарь с метриками на валидации для значений параметра searched_param из множество param_space
    """

    if folds_indexes_dict is not None:
        for key, val in folds_indexes_dict.items():
            if 'train_indexes' not in val or 'test_indexes' not in val:
                raise ValueError("folds_indexes_dict[<fold_i>] должен содержать ключи 'train_indexes' и 'test_indexes'")

    metrics_dict = {}
    for param in tqdm(param_space):

        shuff_split = ShuffleSplit(n_splits=n_folds, test_size=0.3, random_state=seed)

        metrics_dict[param] = {'train_folds': [], 'test_folds': [], 'outer_test': None}
        fixed_params[searched_param] = param

        if verbose:
            print(f"{'=' * 4}{searched_param}: {param}{'=' * 40}", end='\n')

        for (tr_idxs, te_idxs) in tqdm(
                shuff_split.split(X=folds_data, y=folds_y),
                total=n_folds
        ) if folds_indexes_dict is None else tqdm(folds_indexes_dict.items(), total=len(folds_indexes_dict)):

            if folds_indexes_dict is None:
                tr_data, tr_target = folds_data.loc[tr_idxs], folds_y[tr_idxs]
                te_data, te_target = folds_data.loc[te_idxs], folds_y[te_idxs]
            else:
                tr_data, tr_target = folds_data.loc[te_idxs["train_indexes"]], folds_y[te_idxs["train_indexes"]]
                te_data, te_target = folds_data.loc[te_idxs["test_indexes"]], folds_y[te_idxs["test_indexes"]]

            if str(model).find('lightgbm') != -1:
                current_fold_model = lgb.train(
                    params=fixed_params,
                    train_set=lgb.Dataset(
                        data=tr_data,
                        label=tr_target
                    ),
                    categorical_feature=cat_features,
                    verbose_eval=10
                )
            else:
                raise ValueError('Пока что реализована только возможность обучения lgb')

            y_pred_or_score = 'y_score' if 'y_score' in inspect.signature(metric_func).parameters else \
                'y_pred' if 'y_pred' in inspect.signature(metric_func).parameters else \
                    -1
            assert y_pred_or_score != -1, 'Нет ни y_pred, ни y_score!'

            # метрика на фолдах обучения -----------------------------------------------------------------
            metric_params = {
                'y_true': tr_target,
                y_pred_or_score: current_fold_model.predict_proba(tr_data)[:, 1] if 'predict_proba' in \
                                                                                    dict(inspect.getmembers(
                                                                                        current_fold_model,
                                                                                        predicate=inspect.ismethod)) else \
                    current_fold_model.predict(tr_data)
            }

            metrics_dict[param]['train_folds'].append(metric_func(**metric_params))

            # метрика на отложенном фолде--------------------------------------------------------
            metric_params = {
                'y_true': te_target,
                y_pred_or_score: current_fold_model.predict_proba(te_data)[:, 1] if 'predict_proba' in \
                                                                                    dict(inspect.getmembers(
                                                                                        current_fold_model,
                                                                                        predicate=inspect.ismethod)) else \
                    current_fold_model.predict(te_data)
            }

            metrics_dict[param]['test_folds'].append(metric_func(**metric_params))

        if outer_test_data is not None and outer_test_y is not None:
            # метрика на отложенном тесте --------------------------------------------------------------------------
            metric_params = {
                'y_true': outer_test_y,
                y_pred_or_score: current_fold_model.predict_proba(outer_test_data)[:, 1] if 'predict_proba' in \
                                                                                      dict(inspect.getmembers(
                                                                                          current_fold_model,
                                                                                          predicate=inspect.ismethod)) else \
                    current_fold_model.predict(outer_test_data)
            }

            metrics_dict[param]['outer_test'] = metric_func(**metric_params)

        if verbose:
            print(
                f"    train_folds_metrics: {metrics_dict[param]['train_folds']} (mean: {np.mean(metrics_dict[param]['train_folds'])})")
            print(
                f"    test_folds_metrics: {metrics_dict[param]['test_folds']} (mean: {np.mean(metrics_dict[param]['test_folds'])})")
            if outer_test_data is not None and outer_test_y is not None:
                print(f"    outer_test_metric: {metrics_dict[param]['outer_test']}")

    return metrics_dict


"""
Функцию make_cv_metrics удобно использовать совместно с функцией plot_cv_metrics.
"""


def plot_cv_metrics(
        metrics: dict,
        param_name: str,
        xaxis_log: bool = False,
        yaxis_log: bool = False,
        outer_test: bool = False,
        figsize: object = (8, 8),
        show_variance: str = 'std',
        show_by_folds_points: bool = False,
):
    """
    Args:
        metrics: словарь с метриками на фолдах для обучения, теста и отложенного теста
            вид словаря: (результат работы функции make_cv_metrics)
                        {
                            param_val1: {
                                "train_folds": [val1, val2, ..., val_Nfolds],
                                "test_folds": [val1, val2, ..., val_Nfolds],
                                "outer_test": val
                            },

                            param_val2: {
                                "train_folds": [val1, val2, ..., val_Nfolds],
                                "test_folds": [val1, val2, ..., val_Nfolds],
                                "outer_test": val
                            },

                            ...

                        }
        param_name: будет использовано в названии графика
        xaxis_log: если True, то будет лог шкала по иксу
        yaxis_log: если True, то будет лог шкала по игреку
        outer_test: если True, то будет нарисована метрика на отложенном тесте (matrics["param_val<...>"]["outer_test"])
        figsize: plt.figure(figsize=???)
        show_variance: нужно ли показывать разброс значений на фолдах ('minmax', 'std' или None)
        show_by_folds_points: если True, то рисует метрики для каждого фолда в виде scatterplot'а

    Returns:
        График, показывающий метрики на валидации и отложенном тесте
    """


    assert show_variance in ['minmax', 'std', None], 'show_variance может быть "minmax" или "std" или None'

    plt.figure(figsize=figsize)
    plt.title(param_name, fontsize=15)
    tmp_train = []
    tmp_test = []
    tmp_outer_test = []
    for param, vals_dict in sorted(metrics.items(), key=lambda x: x[0]):
        if show_by_folds_points:
            for i, (tr_test_key, vals_list) in enumerate(vals_dict.items()):
                if tr_test_key == "outer_test":
                    continue
                plt.scatter([param] * len(vals_list), vals_list,
                            color=np.array(['cyan', 'pink', 'green', 'brown'])[i], s=15);
        tmp_train.append([
            param,
            np.mean(vals_dict['train_folds']),
            np.std(vals_dict['train_folds']),
            np.min(vals_dict['train_folds']),
            np.max(vals_dict['train_folds'])
        ])
        tmp_test.append([
            param,
            np.mean(vals_dict['test_folds']),
            np.std(vals_dict['test_folds']),
            np.min(vals_dict['test_folds']),
            np.max(vals_dict['test_folds'])
        ])
        if outer_test:
            tmp_outer_test.append([param, vals_dict['outer_test']])

    tmp_train = np.array(tmp_train)
    tmp_test = np.array(tmp_test)
    tmp_outer_test = np.array(tmp_outer_test)

    plt.scatter(tmp_train[:, 0], tmp_train[:, 1], color='blue', s=30);
    plt.plot(tmp_train[:, 0], tmp_train[:, 1], color='blue', label='train_folds');
    plt.scatter(tmp_test[:, 0], tmp_test[:, 1], color='red', s=30);
    plt.plot(tmp_test[:, 0], tmp_test[:, 1], color='red', label='test_folds');

    if show_variance is not None:
        if show_variance == 'std':
            plt.fill_between(tmp_train[:, 0], tmp_train[:, 1] + tmp_train[:, 2], tmp_train[:, 1] - tmp_train[:, 2],
                             color='blue', alpha=0.5)
            plt.fill_between(tmp_test[:, 0], tmp_test[:, 1] + tmp_test[:, 2], tmp_test[:, 1] - tmp_test[:, 2],
                             color='red', alpha=0.5)
        else:
            plt.fill_between(tmp_train[:, 0], tmp_train[:, 3], tmp_train[:, 4],
                             color='blue', alpha=0.5)
            plt.fill_between(tmp_test[:, 0], tmp_test[:, 3], tmp_test[:, 4],
                             color='red', alpha=0.5)

    if outer_test:
        plt.plot(tmp_outer_test[:, 0], tmp_outer_test[:, 1], color='black', linewidth=2, label='outer_test');
        plt.scatter(tmp_outer_test[:, 0], tmp_outer_test[:, 1], color='black', s=30);

    if xaxis_log:
        plt.xscale('log')
    if yaxis_log:
        plt.yscale('log')
    plt.legend(fontsize=14);
