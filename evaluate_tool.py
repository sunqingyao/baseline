# -*- coding: utf-8 -*-
"""评估指标（自包含，等价于 MMMV.evaluate_tool 中的常用指标）。

采用 CAFA 常用的多标签蛋白功能预测指标：
    - F_max / P_max / R_max / t_max：扫描阈值取最优的（微平均）精确率/召回率/F1
    - AUC：微平均 ROC-AUC
    - AUPR：微平均 PR 曲线下面积
    - MCC：马修斯相关系数
    - micro_score：给定阈值下的微平均 P/R/F1
所有指标均在 (蛋白, GO 术语) 对上做微平均（pair-centric / micro）。
"""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, matthews_corrcoef

_EPS = 1e-10


def micro_score(preds, trues, threshold=0.5):
    """微平均 P/R/F1（给定阈值）。返回 P, R, F1, 预测阳性数, 真实阳性数。"""
    preds_b = (preds >= threshold).astype(np.float64)
    tp = float(np.sum(preds_b * trues))
    fp = float(np.sum(preds_b * (1.0 - trues)))
    fn = float(np.sum((1.0 - preds_b) * trues))
    p = tp / (tp + fp + _EPS)
    r = tp / (tp + fn + _EPS)
    f = 2.0 * p * r / (p + r + _EPS)
    return p, r, f, tp + fp, tp + fn


def compute_performance(preds, trues):
    """扫描阈值，返回 F_max, P_max, R_max, t_max 及该阈值下的二值预测。"""
    best_f = best_p = best_r = best_t = 0.0
    best_preds = None
    for t in np.arange(0.0, 1.0, 0.01):
        preds_b = (preds >= t).astype(np.float64)
        tp = float(np.sum(preds_b * trues))
        fp = float(np.sum(preds_b * (1.0 - trues)))
        fn = float(np.sum((1.0 - preds_b) * trues))
        p = tp / (tp + fp + _EPS)
        r = tp / (tp + fn + _EPS)
        f = 2.0 * p * r / (p + r + _EPS)
        if f > best_f:
            best_f, best_p, best_r, best_t = f, p, r, t
            best_preds = preds_b
    return best_f, best_p, best_r, best_t, best_preds


def compute_roc(preds, trues):
    """微平均 ROC-AUC。某类别全负/全正时退回 0.5。"""
    try:
        return roc_auc_score(trues.ravel(), preds.ravel())
    except ValueError:
        return 0.5


def cacul_aupr(preds, trues):
    """微平均 PR-AUC。无正样本时返回 0。"""
    if float(trues.sum()) == 0.0:
        return 0.0
    try:
        return average_precision_score(trues.ravel(), preds.ravel())
    except ValueError:
        return 0.0


def compute_mcc(preds_bin, trues):
    """马修斯相关系数（preds_bin 为已二值化的预测）。"""
    return float(matthews_corrcoef(trues.ravel(), preds_bin.ravel()))


def evaluate_result(preds, trues):
    """汇总测试指标：f_max, p_max, r_max, t_max, auc, aupr, mcc。"""
    f_max, p_max, r_max, t_max, preds_bin = compute_performance(preds, trues)
    auc = compute_roc(preds, trues)
    aupr = cacul_aupr(preds, trues)
    mcc = compute_mcc(preds_bin, trues) if preds_bin is not None else 0.0
    return f_max, p_max, r_max, t_max, auc, aupr, mcc
