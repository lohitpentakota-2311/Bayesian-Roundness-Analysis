#include <iostream>
#include <Eigen/Dense>

#include "cluster_correlation.hpp"
#include "cross_validation.hpp"
#include "ridge_regression.hpp"
#include "hybrid_error.hpp"

int main()
{
    std::cout << "=====================================\n";
    std::cout << "Bayesian Roundness Analysis - C++ Test\n";
    std::cout << "=====================================\n\n";

    // -------------------------------------------------
    // Test 1: Hybrid error
    // MATLAB:
    // computeHybridrror(yTrue, yPred, epsilon)
    // -------------------------------------------------

    Eigen::VectorXd yTrue(4);
    Eigen::VectorXd yPred(4);

    yTrue << 10.0, 20.0, 30.0, 40.0;
    yPred << 11.0, 18.0, 33.0, 39.0;

    double error = computeHybridError(yTrue, yPred, 3.0);

    std::cout << "Hybrid error: " << error << "\n\n";

    // -------------------------------------------------
    // Test 2: Correlation clustering
    // -------------------------------------------------

    Eigen::MatrixXd corr(4, 4);

    corr <<
        1.0, 0.9, 0.2, 0.1,
        0.9, 1.0, 0.3, 0.1,
        0.2, 0.3, 1.0, 0.8,
        0.1, 0.1, 0.8, 1.0;

    std::vector<std::string> labels = {
        "Variable1",
        "Variable2",
        "Variable3",
        "Variable4"
    };

    auto clusters = clusterCorrelation(corr, labels, 0.7);

    std::cout << "Number of clusters: "
              << clusters.clusterIndices.size() << "\n\n";

    // -------------------------------------------------
    // Finished
    // -------------------------------------------------

    std::cout << "C++ modules linked successfully.\n";
    std::cout << "All basic tests completed.\n";

    return 0;
}
