import numpy as np
import matplotlib
%matplotlib inline
import matplotlib.pyplot as plt

def plot_2d_separator(classifier, X, y, fill_between=True):
    """
    classifier: обученный классификатор у которого есть метод .predict_proba()
    X: матрица объект-признак (n x 2)
    y: вектор таргетов
    fill_between: (True/False) если True - зарисовывает области 
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    from_x1 = np.min(x1) - 0.1*np.std(x1)
    to_x1 = np.max(x1) + 0.1*np.std(x1)
    from_x2 = np.min(x2) - 0.1*np.std(x2)
    to_x2 = np.max(x2) + 0.1*np.std(x2)
    xx, yy = np.mgrid[from_x1:to_x1:.01, 
                      from_x2:to_x2:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = classifier.predict_proba(grid)[:, 1].reshape(xx.shape)


    f, ax = plt.subplots(figsize=(10, 8))
    if fill_between:
        contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                          vmin=0, vmax=1)
        ax_c = f.colorbar(contour)
        ax_c.set_label("$P(y = 1)$")
        ax_c.set_ticks([0, .25, .5, .75, 1])
    else:
        ax.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.6)

    ax.scatter(X[:,0], X[:, 1], c=y, s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(from_x1, to_x1), ylim=(from_x2, to_x2),
           xlabel="$X_1$", ylabel="$X_2$")
    
    