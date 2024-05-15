#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# This script is for implementing Transfer Elastic Net proposed in the paper "Transfer Elastic Net for Developing Epigenetic Clocks for the Japanese Population."
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International Public License.
# See the LICENSE file in the root of this repository for more details.
#
# Authors: Yui tomo, Ryo nakaki
# Date: 2024/05/15


import numpy as np


def softThresholding(z, gamma1, gamma2, b):
    if b >= 0:
        if -gamma1 <= z <= gamma2:
            return 0
        elif gamma2 + b <= z <= gamma1 + b:
            return b
        elif gamma2 <= z <= gamma2 + b:
            return z - gamma2 * np.sign(b)
        else:
            return z - gamma1 * np.sign(z)
    else:  # b <= 0
        if -gamma2 <= z <= gamma1:
            return 0
        elif -gamma1 + b <= z <= -gamma2 + b:
            return b
        elif -gamma2 + b <= z <= -gamma2:
            return z - gamma2 * np.sign(b)
        else:
            return z - gamma1 * np.sign(z)


def estimateTransferElasticNet(
    X,
    y,
    lambda_,
    alpha,
    rho,
    beta_tilde,
    beta_init=None,
    skip_intercept=True,
    max_iter=1000,
    tol=1e-4,
):
    """
    Estimate linear regression coefficients through Transfer Elastic Net.

    Args:
    X: numpy array (n_samples, n_features)
        Input data matrix.
    y: numpy array (n_samples,)
        Target vector.
    lambda_: float
        Tuning parameter that controls the intensity of reguralization.
    alpha: float
        Tuning parameter that balances the penalties to shrikage estimates to 0 and beta_tilde.
        Should be between 0 and 1.
    rho: float
        Tuning parameter that balances the l1 and l2 penalties.
        Should be between 0 and 1.
    beta_tilde: numpy array (n_features,)
        Initial estimates in source domain.
    beta_init: numpy array (n_features,), optional (default=None)
        Initial values of regression coefficients in the optimization algorithm.
        If not provided, a zero vector is used.
    skip_intercept: bool, optional (default=True)
        Flag to indicate whether to regularize the intercept term.
    max_iter: int, optional (default=1000)
        Maximum number of iterations.
    tol: float, optional (default=1e-4)
        Tolerance for convergence.

    Returns:
    beta: numpy array (n_features,)
        Estimated regression coefficients.
    """
    n, p = X.shape
    if beta_init is None:
        beta = np.zeros(p)
    else:
        beta = beta_init.copy()
    norm_cols_X = np.sum(X ** 2, axis=0) / n

    for iteration in range(max_iter):
        beta_old = beta.copy()

        for j in range(p):
            _lambda_ = lambda_
            if j == 0:
                if skip_intercept:
                    _lambda_ = 0
                else:
                    pass

            X_j = X[:, j]
            X_without_j = np.delete(X, j, axis=1)
            beta_without_j = np.delete(beta, j)
            lin_pred_without_j = np.dot(X_without_j, beta_without_j)

            z = np.dot(X_j.T, (y - lin_pred_without_j)) / n
            gamma1 = _lambda_ * rho
            gamma2 = _lambda_ * rho * (2 * alpha - 1)
            beta[j] = (
                softThresholding(z, gamma1, gamma2, beta_tilde[j])
                + 2 * _lambda_ * (1 - alpha) * (1 - rho) * beta_tilde[j]
            ) / (norm_cols_X[j] + 2 * _lambda_ * (1 - rho))

        norm = np.linalg.norm(beta - beta_old)
        if norm < tol:
            break

    return beta


if __name__ == "__main__":
    np.random.seed(100)

    # initial estimation using elastic net
    beta = np.array([0, 1, -1, 0.5, -0.5, 0, 0])
    p = len(beta)
    n_source = 500
    X = np.random.randn(n_source, p)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = beta @ X.T
    lambda_ = 0.5
    rho = 0.5
    alpha = 1
    beta_tilde = estimateTransferElasticNet(
        X, y, lambda_, alpha, rho, beta, max_iter=1000, tol=1e-4
    )
    print(beta_tilde)

    # generate data
    n_target = 10
    X = np.random.randn(n_target, p)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y = beta @ X.T

    # tuning parameter
    lambda_ = 0.8

    # Elastic Net
    rho = 0.5
    alpha = 1
    res = estimateTransferElasticNet(
        X, y, lambda_, alpha, rho, beta_tilde, max_iter=1000, tol=1e-4
    )
    print(res)

    # Transfer Elastic Net
    rho = 0.5
    alpha = 0.5
    res = estimateTransferElasticNet(
        X, y, lambda_, alpha, rho, beta_tilde, max_iter=1000, tol=1e-4
    )
    print(res)

    # only transfer regularization term
    rho = 0.5
    alpha = 0
    res = estimateTransferElasticNet(
        X, y, lambda_, alpha, rho, beta_tilde, max_iter=1000, tol=1e-4
    )
    print(res)

    # Lasso
    rho = 1
    alpha = 1
    res = estimateTransferElasticNet(
        X, y, lambda_, alpha, rho, beta_tilde, max_iter=1000, tol=1e-4
    )
    print(res)

    # Transfer Lasso
    rho = 1
    alpha = 0.5
    res = estimateTransferElasticNet(
        X, y, lambda_, alpha, rho, beta_tilde, max_iter=1000, tol=1e-4
    )
    print(res)
