from typing import Callable, List, Dict, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_recall_curve, roc_auc_score, auc, precision_score,
    recall_score, f1_score, roc_curve, ndcg_score
)

def compute_binary_metrics(
    model: Callable=None,
    train_data: pd.DataFrame=None,
    train_target: np.ndarray=None,
    train_predictions: np.ndarray=None,
    test_data: pd.DataFrame=None,
    test_target: np.ndarray=None,
    test_predictions: np.ndarray=None,
    val_data: pd.DataFrame=None,
    val_target: np.ndarray=None,
    val_predictions: np.ndarray=None,
    show_roc_and_pr_curves: bool=True,
    save_plots_path: str=None,
    curr_feature_columns: List=None,
    ax=None,
    title: str=None,
) -> str:
    """
    Возвращает строку с подсчитанными метриками и рисует ROC и PR кривые
    """

    if model is None:
        assert sum([
            train_predictions is not None and train_target is not None,
            test_predictions is not None and test_target is not None,
            val_predictions is not None and val_target is not None
        ]) > 0, 'Если model не подаётся, то хотя бы одна из *_predictions, *_target пар должна быть не None!'

    else:
        assert sum([
            train_data is not None and train_target is not None,
            test_data is not None and test_target is not None,
            val_data is not None and val_target is not None
        ]) > 0, 'Хотя бы одна из *_data, *_target пар должна быть не None!'

    if curr_feature_columns is None and model is not None:
        curr_feature_columns = model.feature_name_ if 'feature_name_' in model.__dir__() else \
                            model.feature_name()
    if train_target is not None:
        if train_predictions is not None:
            y_pred_tr = train_predictions
        else:
            y_pred_tr = model.predict_proba(train_data[curr_feature_columns])[:, 1] \
                        if 'predict_proba' in model.__dir__() \
                        else model.predict(train_data[curr_feature_columns])
        y_true_tr = train_target
    if test_target is not None:
        if test_predictions is not None:
            y_pred_te = test_predictions
        else:
            y_pred_te = model.predict_proba(test_data[curr_feature_columns])[:, 1] \
                        if 'predict_proba' in model.__dir__() \
                        else model.predict(test_data[curr_feature_columns])
        y_true_te = test_target
    if val_target is not None:
        if val_predictions is not None:
            y_pred_val = val_predictions
        else:
            y_pred_val = model.predict_proba(val_data[curr_feature_columns])[:, 1] \
                            if 'predict_proba' in model.__dir__() \
                            else model.predict(val_data[curr_feature_columns])
        y_true_val = val_target
    metrics_str = ''

    if train_target is not None:
        metrics_str += f'Число объектов на обучении: {train_target.shape[0]:,}\n'
    if val_target is not None:
        metrics_str += f'Число объектов на валидации: {val_target.shape[0]:,}\n'
    if test_target is not None:
        metrics_str += f'Число объектов на тесте: {test_target.shape[0]:,}\n'
    metrics_str += '\n'

    if train_target is not None:
        tmp_metric = round(accuracy_score(
                     y_true=y_true_tr,
                     y_pred=(y_pred_tr > 0.5).astype(int)
                    ), 3)
        metrics_str += f'Accuracy на обучении: {tmp_metric}\n'
    if test_target is not None:
        tmp_metric = round(accuracy_score(
                     y_true=y_true_te,
                     y_pred=(y_pred_te > 0.5).astype(int)
                    ), 3)
        metrics_str += f'Accuracy на тесте: {tmp_metric}\n'
    if val_target is not None:
        tmp_metric = round(accuracy_score(
                     y_true=y_true_val,
                     y_pred=(y_pred_val > 0.5).astype(int)
                    ), 3)
        metrics_str += f'Accuracy на валидации: {tmp_metric}\n'
    for threshold in np.linspace(0.1, 1, 10):
        metric_f1_str = f'порог {round(threshold, 2)} ->\t'
        if train_target is not None:
            f1_tr = round(f1_score(
                     y_true=y_true_tr,
                     y_pred=(y_pred_tr > threshold).astype(int)
                 ), 3)
            metric_f1_str += f'F1 на обучении: {f1_tr}\t'
        if test_target is not None:
            f1_te = round(f1_score(
                     y_true=y_true_te,
                     y_pred=(y_pred_te > threshold).astype(int)
                 ), 3)
            metric_f1_str += f'F1 на тесте: {f1_te}\t'
        if val_target is not None:
            f1_val = round(f1_score(
                     y_true=y_true_val,
                     y_pred=(y_pred_val > threshold).astype(int)
                 ), 3)
            metric_f1_str += f'F1 на валидации: {f1_val}'
        metrics_str += ' '*4 + metric_f1_str + '\n'
    if train_target is not None:
        roc_auc_tr = round(roc_auc_score(
                 y_true=y_true_tr,
                 y_score=y_pred_tr
             ), 3)
        metrics_str += f'AUC-ROC на обучении: {roc_auc_tr}\n'
    if test_target is not None:
        roc_auc_te = round(roc_auc_score(
                 y_true=y_true_te,
                 y_score=y_pred_te
             ), 3)
        metrics_str += f'AUC-ROC на тесте: {roc_auc_te}\n'
    if val_target is not None:
        roc_auc_val = round(roc_auc_score(
             y_true=y_true_val,
             y_score=y_pred_val
         ), 3)
        metrics_str += f'AUC-ROC на валидации: {roc_auc_val}\n'

    # nDCG ---------------------------------------------------------------------------------
    if train_target is not None:
        ndcg_tr = round(ndcg_score(
            y_true=y_true_tr.reshape(1, -1),
            y_score=y_pred_tr.reshape(1, -1)
        ), 3)
        metrics_str += f'nDCG на обучении: {ndcg_tr}\n'
    if test_target is not None:
        ndcg_te = round(ndcg_score(
            y_true=y_true_te.reshape(1, -1),
            y_score=y_pred_te.reshape(1, -1)
        ), 3)
        metrics_str += f'nDCG на тесте: {ndcg_te}\n'
    if val_target is not None:
        ndcg_val = round(ndcg_score(
            y_true=y_true_val.reshape(1, -1),
            y_score=y_pred_val.reshape(1, -1)
        ), 3)
        metrics_str += f'nDCG на валидации: {ndcg_val}\n'
    # ------------------------------------------------------------------------------------------

    if train_target is not None:
        fpr_tr, tpr_tr, _ = roc_curve(
                y_true=y_true_tr,
                y_score=y_pred_tr
            )
        pr_train, rec_train, _ = precision_recall_curve(y_true=y_true_tr,
                        probas_pred=y_pred_tr)
    if test_target is not None:
        fpr_te, tpr_te, _ = roc_curve(
                y_true=y_true_te,
                y_score=y_pred_te
            )
        pr_test, rec_test, _ = precision_recall_curve(y_true=y_true_te,
                        probas_pred=y_pred_te)
    if val_target is not None:
        fpr_val, tpr_val, _ = roc_curve(
            y_true=y_true_val,
            y_score=y_pred_val
        )
        pr_val, rec_val, _ = precision_recall_curve(y_true=y_true_val,
                        probas_pred=y_pred_val)
    if train_target is not None:
        auc_pr_tr = round(auc(
                 x=rec_train,
                 y=pr_train
             ), 3)
        metrics_str += f'AUC-PR на обучении: {auc_pr_tr}\n'
    if test_target is not None:
        auc_pr_te = round(auc(
                 x=rec_test,
                 y=pr_test
             ), 3)
        metrics_str += f'AUC-PR на тесте: {auc_pr_te}\n'
    if val_target is not None:
        auc_pr_val = round(auc(
             x=rec_val,
             y=pr_val
         ), 3)
        metrics_str += f'AUC-PR на валидации: {auc_pr_val}\n'

    if show_roc_and_pr_curves:
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
        if title is not None:
            fig.suptitle(title, fontsize=17);
        ax[0].set_title('ROC', fontsize=15);
        if train_target is not None:
            ax[0].plot(fpr_tr, tpr_tr, color='blue', linewidth=1, label=f'train: {roc_auc_tr}');
        if test_target is not None:
            ax[0].plot(fpr_te, tpr_te, color='red', linewidth=1, label=f'test: {roc_auc_te}');
        if val_target is not None:
            ax[0].plot(fpr_val, tpr_val, color='orange', linewidth=1, label=f'val: {roc_auc_val}');
        ax[0].plot([0, 1], [0, 1], color='black', linewidth=0.5, linestyle='--');
        ax[0].legend(fontsize=15);
        ax[0].set_xlabel(f'FPR', fontsize=15);
        ax[0].set_ylabel(f"TPR", fontsize=15);
        ax[1].set_title('PR', fontsize=15);
        if train_target is not None:
            ax[1].plot(rec_train, pr_train, color='blue', linewidth=1, label=f'train: {auc_pr_tr}');
        if test_target is not None:
            ax[1].plot(rec_test, pr_test, color='red', linewidth=1, label=f'test: {auc_pr_te}');
        if val_target is not None:
            ax[1].plot(rec_val, pr_val, color='orange', linewidth=1, label=f'val: {auc_pr_val}');
        ax[1].legend(fontsize=15);
        ax[1].set_xlabel(f'Recall', fontsize=15);
        ax[1].set_ylabel(f"Precision", fontsize=15);
        if save_plots_path is not None:
            plt.savefig(save_plots_path, bbox_inches='tight')
        plt.show();
    return metrics_str