#include "ridge_regression.hpp"

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>

// Approximation of the inverse normal CDF.
// Used to obtain t-critical values when the degrees of freedom
// are reasonably large.
namespace
{
    double normalQuantile(double p)
    {
        // Acklam approximation

        const double a1 = -3.969683028665376e+01;
        const double a2 = 2.209460984245205e+02;
        const double a3 = -2.759285104469687e+02;
        const double a4 = 1.383577518672690e+02;
        const double a5 = -3.066479806614716e+01;
        const double a6 = 2.506628277459239e+00;

        const double b1 = -5.447609879822406e+01;
        const double b2 = 1.615858368580409e+02;
        const double b3 = -1.556989798598866e+02;
        const double b4 = 6.680131188771972e+01;
        const double b5 = -1.328068155288572e+01;

        const double c1 = -7.784894002430293e-03;
        const double c2 = -3.223964580411365e-01;
        const double c3 = -2.400758277161838e+00;
        const double c4 = -2.549732539343734e+00;
        const double c5 = 4.374664141464968e+00;
        const double c6 = 2.938163982698783e+00;

        const double d1 = 7.784695709041462e-03;
        const double d2 = 3.224671290700398e-01;
        const double d3 = 2.445134137142996e+00;
        const double d4 = 3.754408661907416e+00;

        const double plow = 0.02425;
        const double phigh = 1.0 - plow;

        if (p < plow)
        {
            double q =
                std::sqrt(-2.0 * std::log(p));

            return
                (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
        }

        if (p > phigh)
        {
            double q =
                std::sqrt(-2.0 * std::log(1.0 - p));

            return -
                (((((c1*q+c2)*q+c3)*q+c4)*q+c5)*q+c6) /
                ((((d1*q+d2)*q+d3)*q+d4)*q+1.0);
        }

        double q = p - 0.5;
        double r = q * q;

        return
            (((((a1*r+a2)*r+a3)*r+a4)*r+a5)*r+a6)*q /
            (((((b1*r+b2)*r+b3)*r+b4)*r+b5)*r+1.0);
    }

    // Approximate Student-t critical value.
    //
    // For moderate/large df this converges rapidly to
    // the normal quantile. A correction is used for finite df.
    double tCritical(double probability, double df)
    {
        if (df <= 0.0)
            df = 1.0;

        double z =
            normalQuantile(probability);

        double z2 = z * z;
        double z3 = z2 * z;
        double z5 = z3 * z2;
        double z7 = z5 * z2;

        double t =
            z
            + (z3 + z) / (4.0 * df)
            + (5.0*z5 + 16.0*z3 + 3.0*z)
              / (96.0 * df * df)
            + (3.0*z7 + 19.0*z5 + 17.0*z3 - 15.0*z)
              / (384.0 * df * df * df);

        return t;
    }

    Eigen::MatrixXd normalizeData(
        const Eigen::MatrixXd& X,
        Eigen::VectorXd& mean,
        Eigen::VectorXd& scale)
    {
        const int n = X.rows();
        const int p = X.cols();

        mean =
            X.colwise().mean();

        scale.resize(p);

        for (int j = 0; j < p; ++j)
        {
            double variance =
                (X.col(j).array() - mean(j))
                .square()
                .sum()
                / static_cast<double>(n - 1);

            scale(j) =
                std::sqrt(variance);

            if (scale(j) == 0.0)
                scale(j) = 1.0;
        }

        Eigen::MatrixXd result =
            X.rowwise()
            - mean.transpose();

        result.array().rowwise()
            /= scale.transpose().array();

        return result;
    }
}

RidgeResult ridgeModelCI(
    const Eigen::MatrixXd& Xtrain,
    const Eigen::VectorXd& ytrain,
    const Eigen::MatrixXd& Xtest,
    const Eigen::VectorXd& ytest,
    const Eigen::VectorXd& lambda_grid,
    int K,
    const Eigen::MatrixXd& PP_train,
    unsigned int seed,
    double OptimalLambda)
{
    const int n =
        Xtrain.rows();

    const int n_test =
        Xtest.rows();

    if (Xtrain.cols() != Xtest.cols())
        throw std::invalid_argument(
            "Xtrain and Xtest must have the same number of columns.");

    if (Xtrain.rows() != ytrain.size())
        throw std::invalid_argument(
            "Xtrain and ytrain dimensions do not match.");

    if (Xtest.rows() != ytest.size())
        throw std::invalid_argument(
            "Xtest and ytest dimensions do not match.");

    RidgeResult result;

    // ----------------------------------------------------------
    // Step 1: Lambda selection
    // ----------------------------------------------------------

    if (OptimalLambda < 0.0)
    {
        CVResult cv =
            find_lambda_cv(
                Xtrain,
                ytrain,
                K,
                PP_train,
                lambda_grid,
                seed);

        result.lambda_opt =
            cv.lambda_opt;

        result.RMSE_folds =
            cv.rmse_folds_all;

        result.trainRMSE_folds =
            cv.rmse_train_all;
    }
    else
    {
        result.lambda_opt =
            OptimalLambda;

        result.RMSE_folds =
            Eigen::MatrixXd();

        result.trainRMSE_folds =
            Eigen::MatrixXd();
    }

    // ----------------------------------------------------------
    // Step 2: Normalize predictors
    // ----------------------------------------------------------

    Eigen::VectorXd mu;
    Eigen::VectorXd sigma;

    Eigen::MatrixXd XtrainNorm =
        normalizeData(
            Xtrain,
            mu,
            sigma);

    Eigen::MatrixXd XtestNorm =
        Xtest.rowwise()
        - mu.transpose();

    XtestNorm.array().rowwise()
        /= sigma.transpose().array();

    // Add intercept.
    Eigen::MatrixXd XtrainFinal(
        n,
        Xtrain.cols() + 1);

    Eigen::MatrixXd XtestFinal(
        n_test,
        Xtest.cols() + 1);

    XtrainFinal.col(0).setOnes();
    XtestFinal.col(0).setOnes();

    XtrainFinal.rightCols(
        Xtrain.cols()) =
        XtrainNorm;

    XtestFinal.rightCols(
        Xtest.cols()) =
        XtestNorm;

    // ----------------------------------------------------------
    // Step 3: Ridge regression
    // ----------------------------------------------------------

    const int p =
        XtrainFinal.cols();

    Eigen::MatrixXd I =
        Eigen::MatrixXd::Identity(p, p);

    // Intercept is not penalized.
    I(0, 0) = 0.0;

    Eigen::MatrixXd XtX =
        XtrainFinal.transpose()
        * XtrainFinal;

    Eigen::MatrixXd XtXReg =
        XtX +
        result.lambda_opt * I;

    // Do NOT explicitly calculate inv() where possible.
    Eigen::LDLT<Eigen::MatrixXd> solver(
        XtXReg);

    Eigen::VectorXd beta =
        solver.solve(
            XtrainFinal.transpose()
            * ytrain);

    result.beta =
        beta;

    // ----------------------------------------------------------
    // Step 3b: Coefficients across lambda grid
    // ----------------------------------------------------------

    const int numLambda =
        lambda_grid.size();

    result.beta_grid =
        Eigen::MatrixXd(
            p,
            numLambda);

    for (int i = 0; i < numLambda; ++i)
    {
        double lambda =
            lambda_grid(i);

        Eigen::MatrixXd XtXRegTmp =
            XtX + lambda * I;

        Eigen::LDLT<Eigen::MatrixXd> tmpSolver(
            XtXRegTmp);

        result.beta_grid.col(i) =
            tmpSolver.solve(
                XtrainFinal.transpose()
                * ytrain);
    }

    // ----------------------------------------------------------
    // Stability
    // ----------------------------------------------------------

    result.stability =
        Eigen::VectorXd(p - 1);

    for (int j = 1; j < p; ++j)
    {
        double meanAbs =
            result.beta_grid
                .row(j)
                .cwiseAbs()
                .mean();

        double maxAbs =
            result.beta_grid
                .row(j)
                .cwiseAbs()
                .maxCoeff();

        if (maxAbs == 0.0)
            result.stability(j - 1) = 0.0;
        else
            result.stability(j - 1) =
                meanAbs / maxAbs;
    }

    // ----------------------------------------------------------
    // Step 4: Predictions
    // ----------------------------------------------------------

    result.y_pred_train =
        XtrainFinal * beta;

    result.y_pred_test =
        XtestFinal * beta;

    result.rmse_train =
        std::sqrt(
            (ytrain - result.y_pred_train)
                .squaredNorm()
            / static_cast<double>(n));

    result.rmse_test =
        std::sqrt(
            (ytest - result.y_pred_test)
                .squaredNorm()
            / static_cast<double>(n_test));

    // ----------------------------------------------------------
    // Step 5: Residual variance and degrees of freedom
    // ----------------------------------------------------------

    Eigen::VectorXd residual =
        ytrain -
        result.y_pred_train;

    // Ridge hat matrix:
    //
    // H = X * inv(X'X + lambda I) * X'

    Eigen::MatrixXd XtXInv =
        solver.solve(
            Eigen::MatrixXd::Identity(p, p));

    Eigen::MatrixXd H =
        XtrainFinal *
        XtXInv *
        XtrainFinal.transpose();

    result.df =
        H.trace();

    result.sigma2 =
        residual.squaredNorm()
        / (static_cast<double>(n)
           - result.df);

    // ----------------------------------------------------------
    // Step 6: Covariance of ridge estimator
    // ----------------------------------------------------------

    result.Cov_beta =
        result.sigma2 *
        XtXInv *
        XtX *
        XtXInv;

    result.VarBeta =
        result.Cov_beta.diagonal();

    // ----------------------------------------------------------
    // Step 7: Prediction intervals
    // ----------------------------------------------------------

    const double alpha =
        0.05;

    double tcrit =
        tCritical(
            1.0 - alpha / 2.0,
            std::max(
                1.0,
                static_cast<double>(n) - result.df));

    result.pred_CI_mean.resize(
        n_test,
        2);

    result.pred_CI_pred.resize(
        n_test,
        2);

    for (int i = 0; i < n_test; ++i)
    {
        Eigen::VectorXd x =
            XtestFinal.row(i).transpose();

        double varMean =
            (x.transpose()
             * result.Cov_beta
             * x)(0, 0);

        if (varMean < 0.0)
            varMean = 0.0;

        double seMean =
            std::sqrt(varMean);

        double sePred =
            std::sqrt(
                varMean +
                result.sigma2);

        double prediction =
            result.y_pred_test(i);

        result.pred_CI_mean(i, 0) =
            prediction -
            tcrit * seMean;

        result.pred_CI_mean(i, 1) =
            prediction +
            tcrit * seMean;

        result.pred_CI_pred(i, 0) =
            prediction -
            tcrit * sePred;

        result.pred_CI_pred(i, 1) =
            prediction +
            tcrit * sePred;
    }

    return result;
}