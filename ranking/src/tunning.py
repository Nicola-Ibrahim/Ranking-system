import math

import numpy as np
import pandas as pd
import pandas as p
from scipy.spatial.distance import squareform

from . import settings


def entropy_weights_method(goal_time: int, decision_matrix: pd.DataFrame = None) -> pd.DataFrame:
    """Tunning the criteria weights using entropy method for ranking system

    Args:
        goal_time (int): the desired time span
        decision_matrix (pd.DataFrame, optional): spaces combinations details. Defaults to None.

    Returns:
        pd.DataFrame: weights dataframe
    """
    dmatrix = pd.read_pickle(settings.DECISION_MATRIX_PATH) if decision_matrix is None else decision_matrix

    # Start the algorithm
    rows, cols = dmatrix.shape
    k = 1.0 / math.log(rows)

    lnf = [[None] * cols for i in range(rows)]

    for i in range(0, rows):
        for j in range(0, cols):
            if dmatrix.iloc[i][j] == 0:
                lnfij = 0.0
            else:
                p = dmatrix.iloc[i][j] / dmatrix.iloc[:, j].sum()
                lnfij = math.log(p) * p * (-k)
            lnf[i][j] = lnfij
    lnf = pd.DataFrame(lnf, index=decision_matrix.index, columns=decision_matrix.columns)

    e = 1 - lnf.sum(axis=0)
    weights = [[None] * 1 for i in range(cols)]
    for j in range(0, cols):
        weightsj = e[j] / sum(e)
        weights[j] = weightsj

    weights = pd.DataFrame(weights)
    weights = weights.round(5)
    weights.index = dmatrix.columns
    weights.columns = ["weight"]
    return weights


def make_eval_matrix(arr: list) -> np.ndarray:
    """Transform the input to appropriate format

    Args:
        arr (list): the array with the weight with respect to the right order (w = w1, w2, w3, w4...)

    Returns:
        np.ndarray: Transformed the w array to get the right format of the A matrix
    """

    X = (squareform(arr)).astype(float)
    row, col = np.diag_indices(X.shape[0])
    X[row, col] = np.ones(X.shape[0])
    for i in range(0, len(row)):
        for j in range(0, len(col)):
            if j < i:
                X[i, j] = 1 / X[i, j]
    A = np.asarray(X)
    return A


def AHP_weights_method(sub_weights: list) -> list:
    eval_matrix = make_eval_matrix(sub_weights)
    eval_matrix_len = len(eval_matrix)
    sums = np.array(pd.DataFrame(eval_matrix).sum())

    ln_rgmm = np.log(eval_matrix)
    rgmm_sum = np.exp(ln_rgmm.sum(axis=1) / eval_matrix_len)
    rggm = rgmm_sum / rgmm_sum.sum()

    errors = np.zeros(eval_matrix.shape)
    size = errors.shape[1]
    # for i in range(0, size):
    #     for j in range(0, size):
    #         errors[i, j] = np.log(eval_matrix[i, j] * rggm[j] / rggm[i]) ** 2

    # print(errors)

    # errors_sum = errors.sum(axis=0)
    # error_calc = np.sqrt(errors_sum / (size - 1))
    # rggm_cosh = rggm * np.cosh(error_calc)
    # rggm_cosh_sum = np.sum(rggm_cosh)
    # rggm_final = rggm_cosh / rggm_cosh_sum
    # rggm_matmul = np.matmul(sums, rggm)

    # plus_minus = rggm * np.sinh(error_calc) / rggm_cosh_sum
    # cr0 = (rggm_matmul - eval_matrix_len) / (
    #     (2.7699 * eval_matrix_len - 4.3513) - eval_matrix_len
    # )
    eig_val = np.linalg.eig(eval_matrix)[0].max()
    eig_vec = np.linalg.eig(eval_matrix)[1][:, 0]
    priorities = np.round(np.real(eig_vec / eig_vec.sum()), 3)  # weights
    print(f"{priorities = }")
    cr = np.round(
        np.real((eig_val - eval_matrix_len) / ((2.7699 * eval_matrix_len - 4.3513) - eval_matrix_len)),
        3,
    )
    evt = np.real(eval_matrix * size / eig_val)

    # for i in range(0, size):
    #     for j in range(0, size):
    #         evt[i, j] = evt[i, j] * rggm_final[j]

    # pi_pi = np.zeros(eval_matrix.shape)
    # for i in range(0, size):
    #     for j in range(0, size):
    #         pi_pi[i, j] = rggm[j] / rggm[i]

    # pi_pi_A = pi_pi * eval_matrix
    # pi_pi_A2 = np.zeros(eval_matrix.shape)
    # for i in range(0, size):
    #     for j in range(0, size):
    #         if pi_pi_A[i, j] > 1 / 9 and pi_pi_A[i, j] < 9:
    #             if pi_pi_A[i, j] > 1:
    #                 pi_pi_A2[i, j] = eval_matrix[i, j] * pi_pi[i, j]
    #             else:
    #                 pi_pi_A2[i, j] = 1 / (eval_matrix[i, j] * pi_pi[i, j])
    #         else:
    #             pi_pi_A2[i, j] = 0
    # Consistency_ratio = list(pi_pi_A2[np.triu_indices(eval_matrix_len, k=1)])
    # std = np.array(pd.DataFrame(evt).std(1))

    return p, cr, rggm, evt
