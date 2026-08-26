#include "cluster_correlation.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

ClusterResult clusterCorrelation(
    const Eigen::MatrixXd& corrMatrix,
    const std::vector<std::string>& varLabels,
    double threshold)
{
    const int n = static_cast<int>(corrMatrix.rows());

    if (corrMatrix.rows() != corrMatrix.cols()) {
        throw std::invalid_argument(
            "Correlation matrix must be square.");
    }

    if (static_cast<int>(varLabels.size()) != n) {
        throw std::invalid_argument(
            "Number of variable labels must match matrix size.");
    }

    if (threshold <= 0.0 || threshold > 1.0) {
        throw std::invalid_argument(
            "Threshold must satisfy 0 < threshold <= 1.");
    }

    // MATLAB:
    // LinkMatrix = abs(CorrMatrix);
    Eigen::MatrixXd linkMatrix = corrMatrix.cwiseAbs();

    // Keep track of the original indices.
    std::vector<int> actualIndices(n);

    for (int i = 0; i < n; ++i) {
        actualIndices[i] = i;
    }

    std::vector<std::string> actualLabels = varLabels;

    ClusterResult result;

    while (linkMatrix.rows() > 0) {

        // Find variables correlated with the first remaining variable.
        std::vector<int> correlatedIndices;

        for (int j = 0; j < linkMatrix.cols(); ++j) {
            if (linkMatrix(0, j) > threshold) {
                correlatedIndices.push_back(j);
            }
        }

        if (correlatedIndices.empty()) {
            throw std::runtime_error(
                "Variable autocorrelation is lower than the threshold: " +
                actualLabels[0]);
        }

        // Create current cluster.
        std::vector<int> clusterIndices;
        std::vector<std::string> clusterLabels;

        for (int idx : correlatedIndices) {
            clusterIndices.push_back(actualIndices[idx]);
            clusterLabels.push_back(actualLabels[idx]);
        }

        result.clusterIndices.push_back(clusterIndices);
        result.clusterLabels.push_back(clusterLabels);

        // Remove clustered variables.
        std::vector<int> remainingIndices;
        std::vector<std::string> remainingLabels;

        for (int i = 0; i < static_cast<int>(actualIndices.size()); ++i) {

            bool remove = false;

            for (int idx : correlatedIndices) {
                if (i == idx) {
                    remove = true;
                    break;
                }
            }

            if (!remove) {
                remainingIndices.push_back(actualIndices[i]);
                remainingLabels.push_back(actualLabels[i]);
            }
        }

        // If nothing remains, we're done.
        if (remainingIndices.empty()) {
            break;
        }

        // Construct reduced correlation/link matrix.
        Eigen::MatrixXd newLinkMatrix(
            remainingIndices.size(),
            remainingIndices.size());

        int newRow = 0;

        for (int oldRow : remainingIndices) {

            int oldRowPosition = -1;

            // Find position of oldRow in actualIndices.
            for (int i = 0; i < static_cast<int>(actualIndices.size()); ++i) {
                if (actualIndices[i] == oldRow) {
                    oldRowPosition = i;
                    break;
                }
            }

            int newCol = 0;

            for (int oldCol : remainingIndices) {

                int oldColPosition = -1;

                for (int i = 0;
                     i < static_cast<int>(actualIndices.size());
                     ++i) {

                    if (actualIndices[i] == oldCol) {
                        oldColPosition = i;
                        break;
                    }
                }

                newLinkMatrix(newRow, newCol) =
                    linkMatrix(oldRowPosition, oldColPosition);

                ++newCol;
            }

            ++newRow;
        }

        linkMatrix = newLinkMatrix;
        actualIndices = remainingIndices;
        actualLabels = remainingLabels;
    }

    // -------------------------------------------------------------
    // Reorder correlation matrix by clusters
    // -------------------------------------------------------------

    std::vector<int> reorderedIndices;

    for (const auto& cluster : result.clusterIndices) {
        for (int index : cluster) {
            reorderedIndices.push_back(index);
        }
    }

    result.reorderedLabels.reserve(reorderedIndices.size());

    result.reorderedCorrMatrix.resize(
        reorderedIndices.size(),
        reorderedIndices.size());

    for (int i = 0; i < static_cast<int>(reorderedIndices.size()); ++i) {

        result.reorderedLabels.push_back(
            varLabels[reorderedIndices[i]]);

        for (int j = 0;
             j < static_cast<int>(reorderedIndices.size());
             ++j) {

            result.reorderedCorrMatrix(i, j) =
                corrMatrix(
                    reorderedIndices[i],
                    reorderedIndices[j]);
        }
    }

    return result;
}