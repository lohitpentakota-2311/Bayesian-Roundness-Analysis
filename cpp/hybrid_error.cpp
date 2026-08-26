#include "hybrid_error.hpp"

#include <cmath>
#include <stdexcept>

double computeHybridError(
    const Eigen::VectorXd& yTrue,
    const Eigen::VectorXd& yPred,
    double epsilon)
{
    if (yTrue.size() != yPred.size()) {
        throw std::invalid_argument(
            "yTrue and yPred must have the same size."
        );
    }

    if (yTrue.size() == 0) {
        throw std::invalid_argument(
            "yTrue and yPred must not be empty."
        );
    }

    if (epsilon < 0.0) {
        throw std::invalid_argument(
            "epsilon must be non-negative."
        );
    }

    double errorSum = 0.0;

    for (Eigen::Index i = 0; i < yTrue.size(); ++i) {

        errorSum +=
            std::abs(yTrue(i) - yPred(i))
            /
            (std::abs(yTrue(i)) + epsilon);
    }

    return errorSum /
           static_cast<double>(yTrue.size());
}