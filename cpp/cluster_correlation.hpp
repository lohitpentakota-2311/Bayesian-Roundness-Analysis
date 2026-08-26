#ifndef CLUSTER_CORRELATION_HPP
#define CLUSTER_CORRELATION_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>

struct ClusterCorrelationResult
{
    std::vector<std::vector<int>> clusterIndices;
    std::vector<std::vector<std::string>> clusterLabels;
    Eigen::MatrixXd reorderedCorrelationMatrix;
    std::vector<std::string> reorderedLabels;
};

ClusterCorrelationResult clusterCorrelation(
    const Eigen::MatrixXd& corrMatrix,
    const std::vector<std::string>& varLabels,
    double threshold
);

#endif