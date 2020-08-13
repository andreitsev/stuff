import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def partial_plot(
        df: pd.DataFrame,
        feature_of_interest: str,
        model: object,
        feature_domain: list = [-2.5, 2.5],
        feature_domain_type: str = 'percent',
        n_predictions: int = 100,
        plot_scatter: bool = False,
        figure: object = None,
        save_plot_path: str = None,
):
    """
    Рисует значения предсказаний модели при изменении значения feature_of_interest на величины,
    указанные в feature_domain

    Args:
        df - объект, для которого делать предсказания df.shape = (1, len(features))
        feature_of_interest - какой признак изменять
        model - объект, у которого есть метод .predict
        feature_domain - на какие величины изменять feature_of_interest
        feature_domain_type - может быть 'absolute' или 'percent'.
            Если 'absolute', то там написаны границы вида
                    [base_value - feature_domain[0], base_value + feature_domain[1]],
            Если 'percent', nо границы вида:
                [base_value - feature_domain[0]*base_value, base_value + feature_domain[1]*base_value],
        n_predictions - сколько предсказаний делать,
        plot_scatter - рисовать ли scatter_plot или только line_plot
        figure - plt.figure(figsize=(16, 8))
    Returns:
        График предсказаний в зависимости от значения фичи
    """

    tmp_df = df.copy()

    features_variation, predictions_change = [], []
    base_value, base_prediction = tmp_df[feature_of_interest].values[0], model.predict(tmp_df)[0]

    if feature_domain is None:
        feature_domain = np.linspace(-0.2*base_value, 0.2*base_value, n_predictions)
    else:
        assert feature_domain_type in ['absolute', 'percent'], 'feature_domain_type должен быть "absolute" или "percent"'
        if feature_domain_type == 'absolute':
            feature_domain = np.linspace(feature_domain[0], feature_domain[1], n_predictions)
        else:
            feature_domain = np.linspace(feature_domain[0]*base_value, feature_domain[1]*base_value, n_predictions)

    for val in feature_domain:
        tmp_df[feature_of_interest] = base_value + val
        current_prediction = model.predict(tmp_df)[0]

        features_variation.append(tmp_df[feature_of_interest].values[0])
        predictions_change.append(current_prediction)

    if figure is not None:
        figure
    else:
        plt.figure(figsize=(16, 8));
    plt.title(f'Предсказания модели при различных значениях переменной "{feature_of_interest}"', fontsize=15);
    plt.plot(features_variation, predictions_change);
    if plot_scatter:
        plt.scatter(features_variation, predictions_change, s=10);
    plt.text(base_value, base_prediction, f'{base_value}', size=10);
    plt.scatter(base_value, base_prediction, color='red', s=50);
    if save_plot_path is not None:
        plt.savefig(save_plot_path);
    plt.show();
