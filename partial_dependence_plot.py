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



def group_partial_plot(
        df: pd.DataFrame,
        feature_of_interest: str,
        model: object,
        feature_range_type: str = 'absolute',
        feature_change_range: np.array = np.linspace(-1.5, 1.5, 21),
        scatter_params: dict = None,
        save_path: str = None,
        title: str = None,
        figsize: tuple = (16, 8),
        type_plot: str = 'absolute',
        threshold_feature: float = 1,
        threshold_forecast: float = 1,
        bins: int = 70,
        show_points: bool = True,
        show_agg: str = None,
        feature_diffs_values: list = None,
        feature_diffs_values_bins: int = 50,
        feature_diffs_values_drop_zeros: bool = True,
        feature_diffs_values_scaling_factor: int = 3,
        feature_diffs_values_params: dict = None,
        mean_forecast_value: float=None,
        dont_show_outer_historical_values: bool=False,
        xlim: tuple = None,
        ylim: tuple = None,
):
    """
    Рисует влияние изменения фичи feature_of_interest на прогноз модели

    Args:
        feature_range_type: может быть "absolute" или "percent"
        type_plot: может быть может быть "absolute" или "percent"
        threshold_feature: нужно только для type_plot = 'percent'
        threshold_forecast: нужно только для type_plot = 'percent'
        feature_diffs_values_drop_zeros: выкидывать ли из рисования нулевые значения feature_diffs_values
        feature_diffs_values_scaling_factor: определяем высоту барплота плотности значений в истории
        mean_forecast_value: отображает в легенде среднее предсказание
        dont_show_outer_historical_values: если True, то ограничивает ось икс только теми значениями изменений, которые
            имели место быть в истории (в feature_diffs_values)
    """

    assert type_plot in ['absolute', 'percent'], 'type_plot может быть "absolute" или "percent"'
    assert feature_range_type in ['absolute', 'percent'], 'feature_range_type может быть "absolute" или "percent"'

    tmp_df = df.copy()
    # изначальные значения фичи
    initial_feature_values = tmp_df[feature_of_interest].copy().values
    # изначальные прогнозы
    initial_forecasts = model.predict(tmp_df)

    feature_values_diff, forecasts_diff = np.array([]), np.array([])
    for change in feature_change_range:
        if feature_range_type == 'absolute':
            # новое значение фичи
            new_feature_values = initial_feature_values + change
        elif feature_range_type == 'percent':
            new_feature_values = initial_feature_values * (1 + change)

        tmp_df[feature_of_interest] = new_feature_values
        new_forecast = model.predict(tmp_df)

        if type_plot == 'absolute':
            f_diff = new_feature_values - initial_feature_values
            forecast_d = new_forecast - initial_forecasts
        elif type_plot == 'percent':
            nonzero_idxs = [i for i in range(len(initial_feature_values)) if
                            abs(initial_feature_values[i]) > threshold_feature and
                            abs(initial_forecasts[i]) > threshold_forecast]
            f_diff = (new_feature_values[nonzero_idxs] - initial_feature_values[nonzero_idxs]) / initial_feature_values[
                nonzero_idxs]
            forecast_d = (new_forecast[nonzero_idxs] - initial_forecasts[nonzero_idxs]) / initial_forecasts[
                nonzero_idxs]

        forecasts_diff = np.append(forecasts_diff, forecast_d)
        feature_values_diff = np.append(feature_values_diff, f_diff)

    forecasts_diff = forecasts_diff[np.argsort(feature_values_diff)]
    feature_values_diff = np.sort(feature_values_diff)

    plt.figure(figsize=figsize);
    if title is not None:
        plt.title(title, fontsize=15);
    if show_points:
        plt.scatter(feature_values_diff, forecasts_diff,
                    s=scatter_params.get('s', 5),
                    color=scatter_params.get('color', 'blue'),
                    alpha=scatter_params.get('alpha', 1));

    if show_agg is not None:
        mean_change_df = pd.DataFrame({'forecasts_diff': forecasts_diff, "feature_values_diff": feature_values_diff})
        mean_change_df['feature_values_diff_binarized'] = pd.cut(mean_change_df['feature_values_diff'],
                                                                 bins=bins).apply(lambda x: x.right)

        mean_change_df = mean_change_df.groupby(['feature_values_diff_binarized'])[
            'forecasts_diff'].mean().reset_index().dropna() \
            .sort_values(['feature_values_diff_binarized'], ascending=True)

        plt.plot([val for val in mean_change_df['feature_values_diff_binarized'].values],
                 mean_change_df['forecasts_diff'].values, color='red',
        label=f'{show_agg}' if mean_forecast_value is None else f"{show_agg} \n mean predict: {round(mean_forecast_value, 2)}");
        plt.scatter([val for val in mean_change_df['feature_values_diff_binarized'].values],
                    mean_change_df['forecasts_diff'].values, s=10, color='red');
        plt.legend(fontsize=15);

    # Рисуем какие были изменения цены в истории ------------------------------------------------------------------
    if feature_diffs_values is not None:
        feature_diffs_values = np.array(feature_diffs_values)
        if feature_diffs_values_drop_zeros:
            feature_diffs_values = feature_diffs_values[feature_diffs_values != 0]

        tmp = pd.cut(
            pd.Series(feature_diffs_values).dropna(), bins=feature_diffs_values_bins
        ).apply(lambda x: x.right)

        vals, counts = np.unique(tmp, return_counts=True)

        counts = counts[np.argsort(vals)]
        vals = np.sort(vals)

        if 'color' not in feature_diffs_values_params:
            feature_diffs_values_params['color'] = 'black'
        if 'alpha' not in feature_diffs_values_params:
            feature_diffs_values_params['alpha'] = 1

        # автоматическое определение width в plt.bar ---------------------------------------------------------------
        normalize_by_factor = max(abs(forecasts_diff)) if show_points else max(abs(mean_change_df['forecasts_diff'].values))
        if 'width' not in feature_diffs_values_params:
            feature_diffs_values_params['width'] = np.mean(np.diff(vals))
            plt.bar(vals,
                    (counts / counts.sum()) * feature_diffs_values_scaling_factor * normalize_by_factor,
                    **(feature_diffs_values_params or {}))
        else:
            plt.bar(vals,
                    (counts / counts.sum()) * feature_diffs_values_scaling_factor * normalize_by_factor,
                    **(feature_diffs_values_params or {}))
        # -----------------------------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------------------------------
    if dont_show_outer_historical_values:
        plt.xlim(np.min(vals) - np.std(vals), np.max(vals) + np.std(vals))
    else:
        if xlim is not None:
            plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(f'Абсолютное изменение {feature_of_interest}', fontsize=15);
    if save_path is not None:
        plt.savefig(save_path)
    plt.show();
