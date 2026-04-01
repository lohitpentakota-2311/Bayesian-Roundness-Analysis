function [ClusterInd, ClustersLabel, ReorderedCorrMatrix, ReorderedLabels] = clusterCorrelation(CorrMatrix,VarLabels,Threshold)
% CLUSTERCORRELATION
% -------------------------------------------------------------------------
% Groups variables based on correlation exceeding a given threshold.
% Produces clusters, cluster labels, and optionally a reordered correlation matrix.
%
% INPUTS:
%   CorrMatrix - Square correlation matrix (n x n)
%   VarLabels  - Cell array of variable names (1 x n)
%   Threshold  - Correlation threshold for clustering (0 < Threshold <= 1)
%
% OUTPUTS:
%   ClusterInd         - Cell array of indices for each cluster
%   ClustersLabel      - Cell array of variable labels for each cluster
%   ReorderedCorrMatrix- Correlation matrix reordered by cluster
%   ReorderedLabels    - Variable labels reordered according to clusters
% -------------------------------------------------------------------------
% Take absolute value of correlations: treat positive and negative equally
LinkMatrix = abs(CorrMatrix);
% Initialize indices and labels
ActualIndices = 1:size(CorrMatrix,1);
ActualLabels = VarLabels;
ActualCluster = 0;
FinishFlag = false;
while ~FinishFlag
    % Find variables correlated above threshold with the first variable
    ICorr = find(LinkMatrix(1,:)>Threshold);
    if isempty(ICorr)   % at least each variabile should be correlated with itself!
        error('this variabile autocorrelation is lower than the threshold!! var:%s \n',ActualLabels(1))
    end
    ActualCluster = ActualCluster+1;
    ClusterInd{ActualCluster} = ActualIndices(ICorr);
    ClustersLabel{ActualCluster} = ActualLabels(ICorr);
    % removes the variables allocated to a cluster:
    LinkMatrix(ICorr,:) = []; LinkMatrix(:,ICorr) = [];
    ActualIndices(ICorr) = []; ActualLabels(ICorr) = [];
    % Stop if no variables left
    FinishFlag = isempty(LinkMatrix);
end
%% --- Reorder correlation matrix by clusters ---
ReorderedIndices = [ClusterInd{:}];  % concatenate all cluster indices
ReorderedCorrMatrix = CorrMatrix(ReorderedIndices, ReorderedIndices);
% Optional: reordered variable labels
ReorderedLabels = VarLabels(ReorderedIndices);
end