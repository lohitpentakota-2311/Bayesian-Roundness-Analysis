#ifndef HYBRID_ERROR_HPP
#define HYBRID_ERROR_HPP

#include <Eigen/Dense>

/**
 * Computes the normalized absolute hybrid error.
 *
 * MATLAB equivalent:
 *
 * computeHybridrror(yTrue, yPred, epsilon)
 *
 * custom_error =
 * mean(abs(yTrue-yPred) ./ (abs(yTrue)+epsilon))
 *
 * @param yTrue   True/observed values
 * @param yPred   Predicted values
 * @param epsilon Small constant to avoid division by zero.
 *                Default = 3.0
 *
 * @return Dimensionless normalized error.
 */
double computeHybridError(
    const Eigen::VectorXd& yTrue,
    const Eigen::VectorXd& yPred,
    double epsilon = 3.0
);

#endif