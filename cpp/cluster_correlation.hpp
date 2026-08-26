#ifndef CLUSTER_CORRELATION_HPP
#define CLUSTER_CORRELATION_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>

struct ClusterResult {
    std::vector<std::vector<int>> clusterIndices;
    std::vector<std::vector<std::string>> clusterLabels;
    Eigen::MatrixXd reorderedCorrMatrix;
    std::vector<std::string> reorderedLabels;
};

ClusterResult clusterCorrelation(
    const Eigen::MatrixXd& corrMatrix,
    const std::vector<std::string>& varLabels,
    double threshold
);

#endif

