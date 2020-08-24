import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def partial_plot(
        df: pd.DataFrame,
        feature_of_interest: str,
        model: object,
        category_feature_values_dict: dict=None,
        feature_domain: list = [-2.5, 2.5],
        feature_domain_type: str = 'percent',
        n_predictions: int = 100,
        plot_scatter: bool = False,
        figure: object = None,
        save_plot_path: str = None,
        title: str=None,
):
    """
    Рисует значения предсказаний модели при изменении значения feature_of_interest на величины,
    указанные в feature_domain

    Args:
        df - объект, для которого делать предсказания df.shape = (1, len(features))
        feature_of_interest - какой признак изменять
        category_feature_values_dict - признак по которому разделять прогнозы на
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

    colors = ['blue', 'red', 'green', 'orange', 'pink', 'yellow', 'purple', 'black', 'brown', 'grey'] + \
        list(matplotlib.colors.get_named_colors_mapping().values())

    tmp_df = df.copy()

    features_variation, predictions_change = [], []
    base_value, base_prediction = tmp_df[feature_of_interest].values[0], model.predict(tmp_df)[0]

    if feature_domain is None:
        feature_domain = np.linspace(-0.2 * base_value, 0.2 * base_value, n_predictions)
    else:
        assert feature_domain_type in ['absolute', 'percent'], \
            'feature_domain_type должен быть "absolute" или "percent"'
        if feature_domain_type == 'absolute':
            feature_domain = np.linspace(feature_domain[0], feature_domain[1], n_predictions)
        else:
            feature_domain = np.linspace(feature_domain[0] * base_value, feature_domain[1] * base_value, n_predictions)

    # Если определён словарь category_feature_values_dict, то рисуем partial plot'ы для каждого значения из
    # category_feature_values_dict ------------------------------------------------------------------------------------
    if category_feature_values_dict is not None:
        cat_name = list(category_feature_values_dict.keys())[0]

        if figure is not None:
            figure
        else:
            plt.figure(figsize=(16, 8));

        for i, cat_val in enumerate(category_feature_values_dict[cat_name]):

            tmp_df = df.copy()
            features_variation, predictions_change = [], []
            tmp_df[cat_name] = cat_val
            base_value, base_prediction = tmp_df[feature_of_interest].values[0], model.predict(tmp_df)[0]

            for val in feature_domain:
                tmp_df[feature_of_interest] = base_value + val
                tmp_df[cat_name] = cat_val
                current_prediction = model.predict(tmp_df)[0]

                features_variation.append(tmp_df[feature_of_interest].values[0])
                predictions_change.append(current_prediction)

            if title is None:
                plt.title(f'Предсказания модели при различных значениях переменной "{feature_of_interest}"',
                          fontsize=15);
            else:
                plt.title(title, fontsize=15);

            plt.plot(features_variation, predictions_change, color=colors[i], label=f'{cat_name}: {cat_val}');
            if plot_scatter:
                plt.scatter(features_variation, predictions_change, color=colors[i], s=10);

            plt.text(base_value, base_prediction, f'{base_value}', size=10);
            plt.scatter(base_value, base_prediction, color='red', s=50);

        plt.legend(fontsize=13);
        if save_plot_path is not None:
            plt.savefig(save_plot_path);
        plt.show();
    #------------------------------------------------------------------------------------------------------------------

    else:
        for val in feature_domain:
            tmp_df[feature_of_interest] = base_value + val
            current_prediction = model.predict(tmp_df)[0]

            features_variation.append(tmp_df[feature_of_interest].values[0])
            predictions_change.append(current_prediction)

        if figure is not None:
            figure
        else:
            plt.figure(figsize=(16, 8));

        if title is None:
            plt.title(f'Предсказания модели при различных значениях переменной "{feature_of_interest}"', fontsize=15);
        else:
            plt.title(title, fontsize=15);
        plt.plot(features_variation, predictions_change);
        if plot_scatter:
            plt.scatter(features_variation, predictions_change, s=10);
        plt.text(base_value, base_prediction, f'{base_value}', size=10);
        plt.scatter(base_value, base_prediction, color='red', s=50);
        if save_plot_path is not None:
            plt.savefig(save_plot_path);
        plt.show();
