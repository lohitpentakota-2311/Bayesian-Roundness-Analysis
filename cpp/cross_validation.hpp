#ifndef CROSS_VALIDATION_HPP
#define CROSS_VALIDATION_HPP

#include <Eigen/Dense>
#include <vector>

struct CVResult {
    double lambda_opt;
    double lambda_opt1se;

    Eigen::VectorXd rmse_cv;
    Eigen::VectorXd rmse_se;
    Eigen::VectorXd rmse_cvtrain;

    Eigen::MatrixXd rmse_folds_all;
    Eigen::MatrixXd rmse_train_all;
};

/**
 * Group K-fold cross-validation for Ridge regression.
 *
 * This is a C++ replication of the MATLAB function:
 *
 *     find_lambda_cv.m
 *
 * Groups are kept together during cross-validation to avoid
 * data leakage between training and validation sets.
 *
 * @param X             Feature matrix (n x p)
 * @param y             Target vector (n)
 * @param K             Number of folds
 * @param PP_train      Grouping matrix/vector
 * @param lambda_grid   Candidate ridge regularization parameters
 * @param seed          Random seed
 *
 * @return CVResult containing optimal lambda, 1-SE lambda,
 *         CV RMSE values and fold-level results.
 */
CVResult find_lambda_cv(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    int K,
    const Eigen::MatrixXd& PP_train,
    const Eigen::VectorXd& lambda_grid,
    unsigned int seed
);

#endif