import numpy as np
from sklearn.neighbors import NearestNeighbors

def space_compactness_profile(X, y, metric='euclidean'):
    """
    http://www.machinelearning.ru/wiki/index.php?title=Профиль_компактности
    param X: матрица объект-признак, для которой хотим посчитать профиль компактности
    param y: вектор таргетов для этих объектов
    param metric: ['euclidean', 'cosine', ...]
    """
    neigh = NearestNeighbors(n_neighbors=X.shape[0], metric=metric)
    neigh.fit(X, y)
    distances, indexes = neigh.kneighbors(X)
    misclassification_fraction = []
    for neighbour_number in range(1, X.shape[0]):
        misclassification_fraction.append((y != y[indexes[:, neighbour_number]]).mean())
    return np.array(misclassification_fraction)
    
