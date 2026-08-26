#ifndef RIDGE_REGRESSION_HPP
#define RIDGE_REGRESSION_HPP

#include <Eigen/Dense>

#include "cross_validation.hpp"

struct RidgeResult
{
    Eigen::VectorXd beta;

    double lambda_opt;

    Eigen::VectorXd y_pred_test;
    Eigen::VectorXd y_pred_train;

    double rmse_test;
    double rmse_train;

    Eigen::MatrixXd Cov_beta;
    Eigen::VectorXd VarBeta;

    double sigma2;
    double df;

    Eigen::MatrixXd pred_CI_mean;
    Eigen::MatrixXd pred_CI_pred;

    Eigen::MatrixXd RMSE_folds;
    Eigen::MatrixXd trainRMSE_folds;

    Eigen::MatrixXd beta_grid;

    Eigen::VectorXd stability;
};

RidgeResult ridgeModelCI(
    const Eigen::MatrixXd& Xtrain,
    const Eigen::VectorXd& ytrain,
    const Eigen::MatrixXd& Xtest,
    const Eigen::VectorXd& ytest,
    const Eigen::VectorXd& lambda_grid,
    int K,
    const Eigen::MatrixXd& PP_train,
    unsigned int seed,
    double OptimalLambda = -1.0);

#endif