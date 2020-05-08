import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def pca_explanation_plot(loadings_matrix: np.array, feature_names: np.array, component_number: int=0) -> plt.barh:
    """
    Args:
        loadings_matrix: матрица нагрузок - из неё достаём выражение компонент через изначальные признаки
        feature_names: название фичей
        component_number: какую компоненту визуализировать
    Return:
        Рисуем барплот
    """
    assert component_number < len(loadings_matrix), 'Матрица нагрузок содержит меньшее число компонент!'
    
    sorted_features = np.array(feature_names)[np.argsort(loadings_matrix[component_number])[::-1]]
    sorted_loadings = np.sort(loadings_matrix[component_number])[::-1]
    
    if len(feature_names) > 20:
        sorted_features = np.array(sorted_features[:10].tolist() + sorted_features[-10:].tolist())
        sorted_loadings = np.array(sorted_loadings[:10].tolist() + sorted_loadings[-10:].tolist())
    
    plt.figure(figsize=(8, 10))
    plt.title(f'Главная компонента №{component_number + 1}', fontsize=15);
    plt.barh(sorted_features, sorted_loadings);
    plt.yticks(rotation=0, fontsize=15);
