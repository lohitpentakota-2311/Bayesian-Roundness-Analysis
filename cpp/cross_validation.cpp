#include "cross_validation.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

/**
 * Normalize a matrix column-wise.
 *
 * Equivalent in spirit to MATLAB:
 *
 *     [Xnorm, mu, sigma] = normalize(X,1)
 *
 * The normalization is calculated ONLY from the training data.
 */
struct NormalizationResult {
    Eigen::MatrixXd Xnorm;
    Eigen::RowVectorXd mean;
    Eigen::RowVectorXd stddev;
};

NormalizationResult normalizeColumns(const Eigen::MatrixXd& X)
{
    NormalizationResult result;

    const int n = X.rows();
    const int p = X.cols();

    result.mean = X.colwise().mean();
    result.stddev.resize(p);

    for (int j = 0; j < p; ++j) {

        double sum_sq = 0.0;

        for (int i = 0; i < n; ++i) {
            double diff = X(i, j) - result.mean(j);
            sum_sq += diff * diff;
        }

        // MATLAB normalize uses a standard deviation.
        // Use n-1 for sample standard deviation.
        double sd = 0.0;

        if (n > 1) {
            sd = std::sqrt(sum_sq / static_cast<double>(n - 1));
        }

        // Avoid division by zero for constant variables.
        if (sd == 0.0 || !std::isfinite(sd)) {
            sd = 1.0;
        }

        result.stddev(j) = sd;
    }

    result.Xnorm.resize(n, p);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < p; ++j) {
            result.Xnorm(i, j) =
                (X(i, j) - result.mean(j)) / result.stddev(j);
        }
    }

    return result;
}


/**
 * Determine whether the first column is an intercept column.
 *
 * MATLAB code:
 *
 *     if all(Xtr1(:,1) == 1)
 */
bool hasIntercept(const Eigen::MatrixXd& X)
{
    if (X.cols() == 0) {
        return false;
    }

    for (int i = 0; i < X.rows(); ++i) {
        if (X(i, 0) != 1.0) {
            return false;
        }
    }

    return true;
}


/**
 * Extract selected rows from a matrix.
 */
Eigen::MatrixXd selectRows(
    const Eigen::MatrixXd& X,
    const std::vector<int>& indices)
{
    Eigen::MatrixXd result(
        static_cast<int>(indices.size()),
        X.cols());

    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        result.row(i) = X.row(indices[i]);
    }

    return result;
}


/**
 * Extract selected elements from a vector.
 */
Eigen::VectorXd selectRows(
    const Eigen::VectorXd& x,
    const std::vector<int>& indices)
{
    Eigen::VectorXd result(
        static_cast<int>(indices.size()));

    for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        result(i) = x(indices[i]);
    }

    return result;
}


/**
 * Create a group identifier for every observation.
 *
 * MATLAB:
 *
 *     [~,~,groupID] = unique(PP_sub,'rows');
 *
 * Here each unique row of PP_train is treated as a group.
 */
std::vector<int> createGroupIDs(
    const Eigen::MatrixXd& PP_train)
{
    const int n = PP_train.rows();

    std::vector<int> groupID(n, -1);

    std::vector<std::vector<double>> groups;

    for (int i = 0; i < n; ++i) {

        std::vector<double> row(
            PP_train.cols());

        for (int j = 0; j < PP_train.cols(); ++j) {
            row[j] = PP_train(i, j);
        }

        int foundGroup = -1;

        for (int g = 0;
             g < static_cast<int>(groups.size());
             ++g) {

            if (groups[g] == row) {
                foundGroup = g;
                break;
            }
        }

        if (foundGroup == -1) {
            groups.push_back(row);
            foundGroup =
                static_cast<int>(groups.size()) - 1;
        }

        groupID[i] = foundGroup;
    }

    return groupID;
}

} // anonymous namespace


CVResult find_lambda_cv(
    const Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    int K,
    const Eigen::MatrixXd& PP_train,
    const Eigen::VectorXd& lambda_grid,
    unsigned int seed)
{
    // -------------------------------------------------------------
    // Input validation
    // -------------------------------------------------------------

    const int n = X.rows();
    const int p = X.cols();
    const int nLambda = lambda_grid.size();

    if (n == 0 || p == 0) {
        throw std::invalid_argument(
            "X must not be empty.");
    }

    if (y.size() != n) {
        throw std::invalid_argument(
            "X and y have incompatible dimensions.");
    }

    if (PP_train.rows() != n) {
        throw std::invalid_argument(
            "PP_train must have the same number of rows as X.");
    }

    if (K < 2) {
        throw std::invalid_argument(
            "K must be at least 2.");
    }

    if (nLambda == 0) {
        throw std::invalid_argument(
            "lambda_grid must not be empty.");
    }

    // -------------------------------------------------------------
    // Step 1: Create groups
    //
    // MATLAB:
    //
    // PP_sub = PP_train;
    // [~,~,groupID] = unique(PP_sub,'rows');
    // uniqueGroups = unique(groupID);
    // -------------------------------------------------------------

    std::vector<int> groupID =
        createGroupIDs(PP_train);

    int numberOfGroups = 0;

    for (int id : groupID) {
        numberOfGroups =
            std::max(numberOfGroups, id + 1);
    }

    if (numberOfGroups < K) {

        throw std::invalid_argument(
            "Number of unique groups is smaller than K.");
    }

    // -------------------------------------------------------------
    // Step 2: Randomized K-fold assignment
    //
    // MATLAB:
    //
    // rng(seed)
    // cv = cvpartition(length(uniqueGroups),'KFold',K);
    //
    // We reproduce the important behavior: groups are randomly
    // assigned to K folds.
    // -------------------------------------------------------------

    std::vector<int> groupOrder(numberOfGroups);

    std::iota(
        groupOrder.begin(),
        groupOrder.end(),
        0);

    std::mt19937 generator(seed);

    std::shuffle(
        groupOrder.begin(),
        groupOrder.end(),
        generator);

    std::vector<int> groupFold(numberOfGroups);

    for (int i = 0;
         i < numberOfGroups;
         ++i) {

        groupFold[groupOrder[i]] =
            i % K;
    }

    // -------------------------------------------------------------
    // Step 3: Allocate output arrays
    // -------------------------------------------------------------

    Eigen::MatrixXd rmse_folds_all =
        Eigen::MatrixXd::Zero(
            nLambda,
            K);

    Eigen::MatrixXd rmse_train_all =
        Eigen::MatrixXd::Zero(
            nLambda,
            K);

    // -------------------------------------------------------------
    // Step 4: Loop over lambda values
    // -------------------------------------------------------------

    for (int l = 0;
         l < nLambda;
         ++l) {

        const double lambda =
            lambda_grid(l);

        // ---------------------------------------------------------
        // Group K-fold loop
        // ---------------------------------------------------------

        for (int fold = 0;
             fold < K;
             ++fold) {

            std::vector<int> trainIndices;
            std::vector<int> validationIndices;

            // -----------------------------------------------------
            // Assign observations according to their group.
            // -----------------------------------------------------

            for (int i = 0; i < n; ++i) {

                if (groupFold[groupID[i]] == fold) {
                    validationIndices.push_back(i);
                }
                else {
                    trainIndices.push_back(i);
                }
            }

            if (trainIndices.empty() ||
                validationIndices.empty()) {

                throw std::runtime_error(
                    "Empty training or validation fold.");
            }

            // -----------------------------------------------------
            // Split data
            // -----------------------------------------------------

            Eigen::MatrixXd Xtr1 =
                selectRows(X, trainIndices);

            Eigen::VectorXd ytr =
                selectRows(y, trainIndices);

            Eigen::MatrixXd Xval1 =
                selectRows(X, validationIndices);

            Eigen::VectorXd yval =
                selectRows(y, validationIndices);

            Eigen::MatrixXd Xtr;
            Eigen::MatrixXd Xval;

            // -----------------------------------------------------
            // Normalize using TRAINING fold only
            //
            // This is important to avoid data leakage.
            // -----------------------------------------------------

            if (hasIntercept(Xtr1)) {

                // MATLAB:
                //
                // Xtr_rest = Xtr1(:,2:end);
                // [Xtr_norm,xTrain_c,xTrain_s] =
                //      normalize(Xtr_rest,1);
                //
                // Xtr = [ones(...),Xtr_norm]
                //

                Eigen::MatrixXd Xtr_rest =
                    Xtr1.rightCols(Xtr1.cols() - 1);

                NormalizationResult norm =
                    normalizeColumns(Xtr_rest);

                Xtr.resize(
                    Xtr1.rows(),
                    Xtr1.cols());

                Xtr.col(0).setOnes();

                Xtr.rightCols(
                    Xtr.cols() - 1) =
                    norm.Xnorm;

                // Apply training normalization
                // to validation data.
                Xval = Xval1;

                for (int i = 0;
                     i < Xval.rows();
                     ++i) {

                    for (int j = 1;
                         j < Xval.cols();
                         ++j) {

                        Xval(i, j) =
                            (Xval1(i, j) -
                             norm.mean(j - 1))
                            / norm.stddev(j - 1);
                    }
                }
            }
            else {

                // MATLAB:
                //
                // [Xtr,xTrain_c,xTrain_s] =
                //      normalize(Xtr1,1);
                //
                // Xval = (Xval1-xTrain_c)./xTrain_s;
                //
                // Xtr = [ones(...),Xtr];
                //

                NormalizationResult norm =
                    normalizeColumns(Xtr1);

                Xtr.resize(
                    Xtr1.rows(),
                    Xtr1.cols() + 1);

                Xtr.col(0).setOnes();

                Xtr.rightCols(
                    Xtr1.cols()) =
                    norm.Xnorm;

                Xval.resize(
                    Xval1.rows(),
                    Xval1.cols() + 1);

                Xval.col(0).setOnes();

                for (int i = 0;
                     i < Xval1.rows();
                     ++i) {

                    for (int j = 0;
                         j < Xval1.cols();
                         ++j) {

                        Xval(i, j + 1) =
                            (Xval1(i, j) -
                             norm.mean(j))
                            / norm.stddev(j);
                    }
                }
            }

            // -----------------------------------------------------
            // Ridge regression
            //
            // MATLAB:
            //
            // p = size(Xtr,2);
            // I = eye(p);
            // I(1,1) = 0;
            //
            // beta =
            // (Xtr'*Xtr + lambda*I)
            //     \ (Xtr'*ytr);
            //
            // -----------------------------------------------------

            const int pRidge =
                Xtr.cols();

            Eigen::MatrixXd I =
                Eigen::MatrixXd::Identity(
                    pRidge,
                    pRidge);

            // Intercept is not penalized.
            I(0, 0) = 0.0;

            Eigen::MatrixXd XtX =
                Xtr.transpose() * Xtr;

            Eigen::VectorXd Xty =
                Xtr.transpose() * ytr;

            Eigen::MatrixXd ridgeMatrix =
                XtX + lambda * I;

            // Equivalent to MATLAB backslash:
            //
            // A \ b
            //
            // LDLT is appropriate for this symmetric
            // positive-semidefinite/regularized system.

            Eigen::VectorXd beta =
                ridgeMatrix.ldlt().solve(Xty);

            // -----------------------------------------------------
            // Predictions
            // -----------------------------------------------------

            Eigen::VectorXd y_pred =
                Xval * beta;

            Eigen::VectorXd y_pred_train =
                Xtr * beta;

            // -----------------------------------------------------
            // Validation RMSE
            //
            // sqrt(mean((yval-y_pred).^2))
            // -----------------------------------------------------

            double validationMSE =
                (yval - y_pred).squaredNorm()
                / static_cast<double>(yval.size());

            double validationRMSE =
                std::sqrt(validationMSE);

            // -----------------------------------------------------
            // Training RMSE
            // -----------------------------------------------------

            double trainingMSE =
                (ytr - y_pred_train).squaredNorm()
                / static_cast<double>(ytr.size());

            double trainingRMSE =
                std::sqrt(trainingMSE);

            rmse_folds_all(l, fold) =
                validationRMSE;

            rmse_train_all(l, fold) =
                trainingRMSE;
        }
    }

    // -------------------------------------------------------------
    // Step 5: Model selection
    //
    // MATLAB:
    //
    // rmse_cv = mean(rmse_folds_all,2);
    // rmse_se = std(rmse_folds_all,0,2)/sqrt(K);
    // -------------------------------------------------------------

    Eigen::VectorXd rmse_cv =
        Eigen::VectorXd::Zero(nLambda);

    Eigen::VectorXd rmse_se =
        Eigen::VectorXd::Zero(nLambda);

    Eigen::VectorXd rmse_cvtrain =
        Eigen::VectorXd::Zero(nLambda);

    for (int l = 0;
         l < nLambda;
         ++l) {

        double meanRMSE =
            rmse_folds_all.row(l).mean();

        rmse_cv(l) =
            meanRMSE;

        // MATLAB std(...,0,2) uses N-1 denominator.
        double variance = 0.0;

        if (K > 1) {

            for (int fold = 0;
                 fold < K;
                 ++fold) {

                double diff =
                    rmse_folds_all(l, fold)
                    - meanRMSE;

                variance +=
                    diff * diff;
            }

            variance /=
                static_cast<double>(K - 1);
        }

        double standardDeviation =
            std::sqrt(variance);

        rmse_se(l) =
            standardDeviation /
            std::sqrt(static_cast<double>(K));

        rmse_cvtrain(l) =
            rmse_train_all.row(l).mean();
    }

    // -------------------------------------------------------------
    // Find lambda with minimum CV RMSE
    //
    // MATLAB:
    //
    // [rmse_min_val,idx_min] = min(rmse_cv);
    // -------------------------------------------------------------

    int idx_min = 0;

    double rmse_min_val =
        rmse_cv(0);

    for (int l = 1;
         l < nLambda;
         ++l) {

        if (rmse_cv(l) <
            rmse_min_val) {

            rmse_min_val =
                rmse_cv(l);

            idx_min = l;
        }
    }

    // -------------------------------------------------------------
    // Step 6: 1-SE rule
    //
    // MATLAB:
    //
    // threshold_1se =
    //     rmse_min_val + rmse_se(idx_min);
    //
    // lambda_1se_idx =
    //     find(rmse_cv <= threshold_1se,1,'first');
    //
    // -------------------------------------------------------------

    double threshold_1se =
        rmse_min_val +
        rmse_se(idx_min);

    int lambda_1se_idx = 0;

    for (int l = 0;
         l < nLambda;
         ++l) {

        if (rmse_cv(l) <=
            threshold_1se) {

            lambda_1se_idx = l;
            break;
        }
    }

    // -------------------------------------------------------------
    // Step 7: Build output
    // -------------------------------------------------------------

    CVResult result;

    result.lambda_opt =
        lambda_grid(idx_min);

    result.lambda_opt1se =
        lambda_grid(lambda_1se_idx);

    result.rmse_cv =
        rmse_cv;

    result.rmse_se =
        rmse_se;

    result.rmse_cvtrain =
        rmse_cvtrain;

    result.rmse_folds_all =
        rmse_folds_all;

    result.rmse_train_all =
        rmse_train_all;

    return result;
}