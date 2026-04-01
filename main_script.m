%% Bayesian Models for Roundness Prediction
% Authors: Lohit Kumar, Marco Leonesio, Giacomo Biancho
% Date: 2026-03-20
% ---------------------------------------------------------
% ---------------------------------------------------------
% This script performs Bayesian modeling and ridge regression to
% predict roundness, including:
% - Data preprocessing & leakage prevention
% - Bayesian Lasso and MixConjugate models
% - Minimal models via correlation clustering
% - Hyperparameter tuning (lambda, PIP, correlation cutoff)
% - Learning curves, prediction intervals, and residual analysis
%% 1. Environment Setup
clc; clear; close all;
seed = 1;
rng(seed, 'twister');
% Add utils folder to path for helper functions (findBestLambda, etc.)
if isfolder('utils')
    addpath(genpath('utils'));
end
% Global Settings
FlagforParamterIdentification = 0; % Set to 1 to run hyperparameter optimization
epsilon = 2.8;                     % Threshold for hybrid error calculation
%% 2. Load Dataset
% Using relative pathing so it works on any machine
data_path = fullfile('data', 'InputData.mat');
if ~exist(data_path, 'file')
    error(['Dataset not found at: ', data_path, ...
        '\nPlease ensure InputData.mat is in the /data/ folder.']);
end
load(data_path);
%% 3. Data Preprocessing & Leakage Prevention
tic
% Hold-out 14% for testing
cv1 = cvpartition(size(x,1), 'HoldOut', 0.14);
InittrainIdx = training(cv1);
testIdx      = test(cv1);

% Prevent Data Leakage: Identify repeated rows in process parameters
ProcessParameters = x(:, end-3:end);
isLeaking = ismember(ProcessParameters(testIdx,:), ProcessParameters(InittrainIdx,:), 'rows');

% Move leaking samples from test -> train
testGlobalIDs    = find(testIdx);
leakingGlobalIDs = testGlobalIDs(isLeaking);
testIdx(leakingGlobalIDs)      = false;
InittrainIdx(leakingGlobalIDs) = true;

% Prepare Final Sets
xTrain_Valid = x(InittrainIdx, :);
yTrain_Valid = y(InittrainIdx);
PP_train     = ProcessParameters(InittrainIdx, :);
xTest        = x(testIdx, :);
yTest        = y(testIdx);

% Normalize Features
[xTrain_ValidNorm, xTrain_ValidNorm_c, xTrain_ValidNorm_v] = normalize(xTrain_Valid);
xTestNorm = (xTest - xTrain_ValidNorm_c) ./ xTrain_ValidNorm_v;
p = size(xTestNorm,2);
%% 4. Bayesian Lasso Model
fprintf('Running: Bayesian Lasso Model...\n');
if isequal(FlagforParamterIdentification,'on')
    K=5;
    bestlambda = findBestLambda(x, y, InittrainIdx, testIdx, ProcessParameters, K, seed);
end
% Define Bayesian Lasso prior
PriorMdlLasso = bayeslm(p, 'ModelType', 'Lasso', 'VarNames', FeatureNames);
PriorMdlLasso.Lambda = bestlambda;
% Estimate Bayesian Lasso using MCMC
rng(seed);
[EstMdlLasso, ~] = estimate(PriorMdlLasso, xTrain_ValidNorm, yTrain_Valid, ...
    'BurnIn', 5000, 'NumDraws', 35000);
% Compute convergence metrics to check MCMC stability
[SummaryTable, ~, ~, ~] =computeConvergence(EstMdlLasso);
disp(SummaryTable);
% Forecast predictions and covariance for train and test sets
[yPredTrainLasso, YFCovTrainLasso] = forecast(EstMdlLasso, xTrain_ValidNorm);
[ypredTestLasso, YFCovTestLasso]   = forecast(EstMdlLasso, xTestNorm);
% Compute standard RMSE and custom hybrid error
rmse_train_Lasso = rmse(yPredTrainLasso, yTrain_Valid);
rmse_test_Lasso  = rmse(ypredTestLasso, yTest);
custom_error_train_Lasso = computeHybridrror(yTrain_Valid, yPredTrainLasso, epsilon);
custom_error_test_Lasso  = computeHybridrror(yTest, ypredTestLasso, epsilon);

% 4. Plot Lasso Results - Training
figure('Color', 'w','Position',[10 10 650 600]);  % white background
plot(yPredTrainLasso, yTrain_Valid, 'ko', 'MarkerFaceColor', 'k', ...
    'MarkerSize', 6, 'LineWidth', 1.2); % Scatter predicted vs observed
hold on;
xlabel('Predicted Roundness [$\mu$m]', 'FontSize', 16, 'Interpreter', 'latex');
ylabel('Measured Roundness [$\mu$m]', 'FontSize', 16, 'Interpreter', 'latex');
lims = [0, max([yTrain_Valid(:); yPredTrainLasso(:)])];
axis([lims(1), lims(2), lims(1), lims(2)]);
plot(lims, lims, 'k', 'LineWidth', 2);
h = findobj(gca,'Type','Line');
set(h(1), 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5); % Format identity line
% Legend showing hybrid error and RMSE for reviewer clarity
legend({...
    ['$\mathrm{Hybrid Error}: ', num2str(round(custom_error_train_Lasso,3)*100, '%.3f'), '\%$, ','$\mathrm{RMSE}: ', num2str(rmse_train_Lasso, '%.3f'), '\,\mu m$'],'Diagonal Reference Line'}, ...
    'FontSize', 16, 'Location', 'best', 'Interpreter', 'latex');
title('Bayesian LASSO','FontSize', 16,'Interpreter', 'latex');
% Refine axes limits and ticks
lims = [0, ceil(max([yPredTrainLasso; yTrain_Valid])) + 2];
xlim(lims); ylim(lims);
yticks_auto = get(gca, 'YTick'); xticks(yticks_auto);
set(gca, 'FontSize', 17, 'LineWidth', 1.2, 'TickDir', 'out', 'Box', 'off');

% Plot Lasso Results - Testing
figure('Color', 'w','Position',[10 10 650 600]);  % white background
plot(ypredTestLasso,yTest, 'ko', 'MarkerFaceColor', 'k', ...
    'MarkerSize', 6, 'LineWidth', 1.2); % Scatter predicted vs observed
hold on;
% Axis labels with LaTeX formatting
xlabel('Predicted Roundness [$\mu$m]', 'FontSize', 16, 'Interpreter', 'latex');
ylabel('Measured Roundness [$\mu$m]', 'FontSize', 16, 'Interpreter', 'latex');
% Diagonal reference line (y=x)
lims = [0, max([yTest(:); ypredTestLasso(:)])];
axis([lims(1), lims(2), lims(1), lims(2)]);
plot(lims, lims, 'k', 'LineWidth', 2);
h = findobj(gca,'Type','Line');
set(h(1), 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5); % format identity line
% Legend for reviewer to check accuracy metrics
legend({...
    ['$\mathrm{Hybrid Error}: ', num2str(round(custom_error_test_Lasso,3)*100, '%.3f'), '\%$, ','$\mathrm{RMSE}: ', num2str(rmse_test_Lasso, '%.3f'), '\,\mu m$'],'Diagonal Reference Line'}, ...
    'FontSize', 16, 'Location', 'best', 'Interpreter', 'latex');
title('Bayesian LASSO','FontSize', 16,'Interpreter', 'latex');
% Refine axes limits and ticks
lims = [0, ceil(max([ypredTestLasso;yTest]))];
xlim(lims); ylim(lims);
yticks_auto = get(gca, 'YTick'); xticks(yticks_auto);
set(gca, 'FontSize', 17, 'LineWidth', 1.2, 'TickDir', 'out', 'Box', 'off');

% Training Prediction with Error Bars
figure('Color','w','Position',[10 10 850 550]);
x1 = (1:length(yPredTrainLasso))';
err_train = norminv(0.975) * sqrt(diag(YFCovTrainLasso)); % 95% CI
errorbar(x1, yPredTrainLasso, err_train, 'o', ...
    'Color','k', ...
    'MarkerEdgeColor', [0 0.5 0], ...
    'MarkerFaceColor', [0 0.8 0], ...
    'MarkerSize',7, 'LineWidth',2, 'CapSize',8);
hold on;
scatter((1:length(yTrain_Valid))',yTrain_Valid,'xr',LineWidth=2); % measured points
legend({'Prediction (mean $\pm$ 95\% CI)', 'Measured data'}, ...
    'Location','best','Interpreter','latex','FontSize',17);
xlabel('Post-Inspection Measurements', 'FontSize',17,'Interpreter','latex');
ylabel('Roundness Indicator [$\mu$m]', 'FontSize',17,'Interpreter','latex');
title('Bayesian LASSO','FontSize', 16,'Interpreter', 'latex');
axis tight;
set(gca,'FontSize',17,'LineWidth',1.2,'TickDir','out','Box','off');

% 7. Testing Prediction with Error Bars
figure('Color','w','Position',[10 10 850 550]);
x1 = (1:length(ypredTestLasso))';
err_test = norminv(0.975) * sqrt(diag(YFCovTestLasso)); % 95% CI
errorbar(x1, ypredTestLasso, err_test, 'o', ...
    'Color','k', ...
    'MarkerEdgeColor', [0 0.5 0], ...
    'MarkerFaceColor', [0 0.8 0], ...
    'MarkerSize',7, 'LineWidth',2, 'CapSize',8);
hold on;
scatter((1:length(yTest))',yTest,'xr',LineWidth=2); % measured points
legend({'Prediction (mean $\pm$ 95\% CI)', 'Measured data'}, ...
    'Location','best','Interpreter','latex','FontSize',17);
xlabel('Post-Inspection Measurements', 'FontSize',17,'Interpreter','latex');
ylabel('Roundness Indicator [$\mu$m]', 'FontSize',17,'Interpreter','latex');
title('Bayesian LASSO','FontSize', 16,'Interpreter', 'latex');
axis tight;
set(gca,'FontSize',17,'LineWidth',1.2,'TickDir','out','Box','off');

%% 4d. Residuals Analysis (Training & Testing)
Trainresiduals = (yTrain_Valid - yPredTrainLasso);
Testresiduals  = (yTest - ypredTestLasso);

% Train residuals normal probability plot
figure('Color', 'w','Position',[10 10 650 600]);
h = probplot(Trainresiduals);
set(h(1), 'Marker', 'o', 'MarkerSize', 8, 'LineWidth', 1.2, 'Color', [0 0 0]); % data points
set(h(2), 'Color', [0 0.2 0.6], 'LineWidth', 3); % normal fit
xlabel('Residuals from Training [$\mu m$]', 'FontSize', 17, 'Interpreter', 'latex');
ylabel('Probability', 'FontSize', 17, 'Interpreter', 'latex');
legend([h(1), h(2)], {'Data', 'Normal'}, 'FontSize', 17, 'Location', 'best');
title('Bayesian LASSO','FontSize', 16,'Interpreter', 'latex');
set(gca, 'FontSize', 17, 'LineWidth', 1.2, 'TickDir', 'out', 'Box', 'off');

% Test residuals normal probability plot
figure('Color', 'w','Position',[10 10 650 600]);
h = probplot(Testresiduals);
set(h(1), 'Marker', 'o', 'MarkerSize', 8, 'LineWidth', 1.2, 'Color', [0 0 0]); % data points
set(h(2), 'Color', [0 0.2 0.6], 'LineWidth', 3); % normal fit
xlabel('Residuals from Testing [$\mu$m]', 'FontSize', 17, 'Interpreter', 'latex');
ylabel('Probability', 'FontSize', 17, 'Interpreter', 'latex');
legend([h(1), h(2)], {'Data', 'Normal'}, 'FontSize', 17, 'Location', 'best');
title('Bayesian LASSO','FontSize', 16,'Interpreter', 'latex');
set(gca, 'FontSize', 17, 'LineWidth', 1.2, 'TickDir', 'out', 'Box', 'off');
%% 5. Bayesian Conjugate Model (MixConjugate)
fprintf('Running: Bayesian Mix Conjugate SSVS Model...\n');
% Hyperparameter optimization for V1, V2
if isequal(FlagforParamterIdentification,'on')
    [bestV1, bestV2] = findBestV1V2(x, y, InittrainIdx, ProcessParameters, logspace(-7, -2, 12), logspace(1, 1.7, 12), K, seed);
end
% Define prior model
V = [bestV1*ones(p+1,1), bestV2*ones(p+1,1)];
PriorMdl = bayeslm(p,'ModelType','MixConjugate','V',V,'VarNames',FeatureNames);
PriorMdl.Probability = 0.5; % Initial inclusion probability
% Estimate posterior
rng(seed);
[EstMdl, summary] = estimate(PriorMdl, xTrain_ValidNorm, yTrain_Valid, ...
    'Display',false,'BurnIn',5000,'NumDraws',50000);
% Convergence diagnostics
[SummaryTable, ~, ~, ~] =computeConvergence(EstMdl);
disp(SummaryTable);
% Simulate posterior draws for coefficient analysis
rng(seed);
[BetaSim, sigma2Sim, RegimeSim] = simulate(PriorMdl, xTrain_ValidNorm ...
    , yTrain_Valid ,'BurnIn', 5000,'NumDraws', 50000);
% Remove intercept for variable selection summary
RegimeNoInt = RegimeSim(2:end,1:end);
VarSelection= summary.Regime(2:end-1);
% Forecast predictions
[yPredTrain, YFCovTrain] = forecast(EstMdl, xTrain_ValidNorm);
[ypredtest, YFCovTest] = forecast(EstMdl, xTestNorm);

rmse_test  = rmse(ypredtest, yTest);
custom_error_test  = computeHybridrror(yTest, ypredtest, epsilon);

% Plot Conjugate Model - Testing
figure('Color', 'w','Position',[10 10 650 600]);  % white background
plot(ypredtest,yTest, 'ko', 'MarkerFaceColor', 'k', ...
    'MarkerSize', 6, 'LineWidth', 1.2); % Scatter predicted vs observed
hold on;
xlabel('Predicted Roundness [$\mu$m]', 'FontSize', 16, 'Interpreter', 'latex');
ylabel('Measured Roundness [$\mu$m]', 'FontSize', 16, 'Interpreter', 'latex');
% Diagonal reference line (y=x)
lims = [0, max([yTest(:); ypredtest(:)])];
axis([lims(1), lims(2), lims(1), lims(2)]);
plot(lims, lims, 'k', 'LineWidth', 2);
h = findobj(gca,'Type','Line');
set(h(1), 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5); % format identity line
legend({...
    ['$\mathrm{Hybrid Error}: ', num2str(round(custom_error_test,3)*100, '%.3f'), '\%$, ','$\mathrm{RMSE}: ', num2str(rmse_test, '%.3f'), '\,\mu m$'],'Diagonal Reference Line'}, ...
    'FontSize', 16, 'Location', 'best', 'Interpreter', 'latex');
title('Bayesian Mix Conj SSVS','FontSize', 16,'Interpreter', 'latex');
lims = [0, ceil(max([ypredTestLasso;yTest]))];
xlim(lims); ylim(lims);
yticks_auto = get(gca, 'YTick'); xticks(yticks_auto);
set(gca, 'FontSize', 17, 'LineWidth', 1.2, 'TickDir', 'out', 'Box', 'off');

% Testing Prediction with Error Bars
figure('Color','w','Position',[10 10 850 550]);
x1 = (1:length(ypredTestLasso))';
err_test = norminv(0.975) * sqrt(diag(YFCovTestLasso)); % 95% CI
errorbar(x1, ypredTestLasso, err_test, 'o', ...
    'Color','k', ...
    'MarkerEdgeColor', [0 0.5 0], ...
    'MarkerFaceColor', [0 0.8 0], ...
    'MarkerSize',7, 'LineWidth',2, 'CapSize',8);
hold on;
scatter((1:length(yTest))',yTest,'xr',LineWidth=2); % measured points
legend({'Prediction (mean $\pm$ 95\% CI)', 'Measured data'}, ...
    'Location','best','Interpreter','latex','FontSize',17);
xlabel('Post-Inspection Measurements', 'FontSize',17,'Interpreter','latex');
ylabel('Roundness Indicator [$\mu$m]', 'FontSize',17,'Interpreter','latex');
title('Bayesian Mix Conj SSVS','FontSize', 16,'Interpreter', 'latex');
axis tight;
set(gca,'FontSize',17,'LineWidth',1.2,'TickDir','out','Box','off');
%% 6. Minimal Models: CorrCutOff Sensitivity Analysis
fprintf('Running: Identification for Minimal Models...\n');
CorrGrid = [0.05:0.1:0.95 1];         % grid of correlation cutoffs
PIP_Grid  = 0.1:0.1:0.5;            % grid of PIP thresholds for top predictor selection
K = 5;                                % number of folds for cross-validation
figureflag = 'off';                    % option to suppress figures
nGrid = numel(CorrGrid);              % number of correlation thresholds
lambda_grid = logspace(-6,6,50);      % ridge regularization grid

% Initialize storage variables for results across correlation thresholds
cumclusterCell = cell(nGrid,1);           % clusters per CorrCutOff
IdsRep = cell(nGrid,1);                   % indices of selected predictors
NoClusters = zeros(nGrid,1);              % number of clusters per CorrCutOff
Cumrmsefolds_test = zeros(nGrid,1);       % mean test RMSE (folds)
Cumstdfolds_test = zeros(nGrid,1);        % std test RMSE (folds)
Cumrmsefolds_train = zeros(nGrid,1);      % mean training RMSE (folds)
Cumstdfolds_train = zeros(nGrid,1);       % std training RMSE (folds)
Cumrmse_train = zeros(nGrid,1);           % training RMSE per CorrCutOff
Cumrmse_test = zeros(nGrid,1);            % test RMSE per CorrCutOff
Cumy_pred_test = cell(nGrid,1);           % test predictions
Cumy_pred_train = cell(nGrid,1);          % training predictions
CumT_sorted = cell(nGrid,1);              % sorted cluster info table
CumRMSE_folds_all = cell(nGrid,1);        % all fold RMSE (validation)
CumtrainRMSE_folds_all = cell(nGrid,1);   % all fold RMSE (training)
Cumbeta = cell(nGrid,1);                  % regression coefficients
SignalNames = cell(nGrid,1);              % selected signal names
CumCov_beta = cell(nGrid,1);              % covariance of beta estimates
CumVarBeta = cell(nGrid,1);               % variance of beta estimates
Cumpred_CI_mean = cell(nGrid,1);          % prediction interval mean
Cumpred_CI_pred = cell(nGrid,1);          % prediction interval prediction
CumReorderedLabels = cell(nGrid,1);       % reordered variable labels (for clusters)
CumReorderedCorrMatrix = cell(nGrid,1);   % reordered correlation matrices
Cumlambda_opt = zeros(nGrid,1);           % optimal lambda per CorrCutOff
StoreBestPIP = zeros(nGrid,1);            % best PIP threshold selected
Cumdf = zeros(nGrid,1);                   % degrees of freedom for ridge fit
% Storage for PIP optimization results
nPIP = length(PIP_Grid);
RMSE_test_PIP  = nan(nGrid,nPIP);
RMSE_train_PIP = nan(nGrid,nPIP);
NumFeatures_PIP = nan(nGrid,nPIP);
% ============================================================
% Main Loop: iterate over correlation thresholds
% ============================================================
for g = 1:nGrid
    fprintf('\nRunning CorrCutOff = %.3f\n', CorrGrid(g));
    CorrCutOff = CorrGrid(g);

    if CorrCutOff == 1
        % If cutoff = 1, all features are treated independently (no clustering)
        topPredIdx = 1:140;
        NoClusters(g) = 140;
        IdsRep{g} = 1:140;
    else
        % Cluster predictors with correlation > CorrCutOff
        [ClusterInd,ClustersLabel, ReorderedCorrMatrix, ReorderedLabels] = ...
            clusterCorrelation(corr(x),FeatureNames,CorrCutOff);
        % Store cluster info
        cumclusterCell{g} = ClusterInd;
        CumReorderedCorrMatrix{g} = ReorderedCorrMatrix;
        CumReorderedLabels{g} = ReorderedLabels;
        NoClusters(g) = numel(ClusterInd);
        % --- Build cluster-level metrics ---
        ClusterIndices = ClusterInd;
        numClusters = numel(ClusterIndices);
        clusterID = zeros(numel(VarSelection),1);
        for c = 1:numClusters
            idx = ClusterIndices{c};
            clusterID(idx) = c;
        end
        % Compute cluster-level PIP statistics
        clusterPIP_sum  = accumarray(clusterID, VarSelection, [numClusters 1], @sum, 0);
        clusterPIP_mean = accumarray(clusterID, VarSelection, [numClusters 1], @mean, 0);
        clusterPIP_max  = accumarray(clusterID, VarSelection, [numClusters 1], @max, 0);
        clusterSize     = cellfun(@numel, ClusterIndices(:));
        % Group PIP (probability at least one predictor included in cluster)
        clusterPIP_group = zeros(numClusters,1);
        clusterPIP_model = zeros(numClusters,1);
        for c = 1:numClusters
            idx = ClusterIndices{c};
            clusterPIP_group(c) = 1 - prod(1 - VarSelection(idx));
            clusterPIP_model(c) = mean(sum(RegimeNoInt(idx,:), 1) > 0);
        end
        % --- Select top predictor from each cluster ---
        topPredIdx = zeros(numClusters,1);
        topPredPIP = zeros(numClusters,1);
        ClusterFeatureNames = cell(numClusters,1);
        TopPredNames = cell(numClusters,1);
        for c = 1:numClusters
            idx = ClusterIndices{c};
            ClusterFeatureNames{c} = strjoin(FeatureNames(idx), ', ');
            [topPredPIP(c), rel] = max(VarSelection(idx));
            topPredIdx(c) = idx(rel);
            TopPredNames{c} = FeatureNames(idx(rel));
        end
        % Create cluster summary table and sort by importance
        T_Import = table((1:numClusters).', clusterSize(:), ClusterFeatureNames, ...
            topPredPIP, topPredIdx, TopPredNames, clusterPIP_model, ...
            'VariableNames', {'clusterID','No. of Features','FeatureNames','TopPredPIP','TopPredictorID','Rep.PIP','PIP Model'});
        T_sorted = sortrows(T_Import, 'PIP Model', 'descend');
        CumT_sorted{g} = T_sorted;
        % --- Nested PIP optimization: select subset of top predictors ---
        best_RMSE_PIP = inf;
        best_indices_for_g = T_sorted.TopPredictorID; % fallback
        for p = 1:length(PIP_Grid)
            current_PIP = PIP_Grid(p);
            temp_indices = T_sorted.TopPredictorID(T_sorted.("PIP Model") > current_PIP);
            if isempty(temp_indices), continue; end
            % Mini cross-validation to evaluate this subset
            [~, lambda_optPIP, ~, ~, ~,~,~,~,~,~,~,~,RMSE_folds_allPIP, trainRMSE_folds_allPIP] = ...
                ridge_model_CI(xTrain_Valid(:, temp_indices), yTrain_Valid, ...
                xTest(:, temp_indices),yTest, lambda_grid, 5, PP_train, seed);
            % Select lambda minimizing mean RMSE across folds
            [~, bestLambdaIdxPIP] = min(mean(RMSE_folds_allPIP, 2));
            rmse_test_temp  = mean(RMSE_folds_allPIP(bestLambdaIdxPIP, :));
            rmse_train_temp = mean(trainRMSE_folds_allPIP(bestLambdaIdxPIP,:));
            % Store results
            RMSE_test_PIP(g,p)  = rmse_test_temp;
            RMSE_train_PIP(g,p) = rmse_train_temp;
            NumFeatures_PIP(g,p) = length(temp_indices);
            if rmse_test_temp < best_RMSE_PIP
                best_RMSE_PIP = rmse_test_temp;
                best_indices_for_g = temp_indices;
                best_PIP = current_PIP;
            end
        end
        IdsRep{g} = best_indices_for_g;
        StoreBestPIP(g) = best_PIP;
    end
    % --- Fit Ridge regression using selected predictors ---
    Xtrain = xTrain_Valid(:,  IdsRep{g});
    SignalNames{g} = FeatureNames(IdsRep{g})';
    Xtest  = xTest(:,  IdsRep{g});
    [beta, lambda_opt, y_pred_test, y_pred_train, rmse_test, rmse_train, ...
        Cov_beta, VarBeta, sigma2, df, ...
        pred_CI_mean, pred_CI_pred, RMSE_folds_all, trainRMSE_folds_all] = ...
        ridge_model_CI(Xtrain, yTrain_Valid, Xtest,yTest, lambda_grid, K, PP_train, seed);
    % Store results
    Cumbeta{g} = beta;
    CumRMSE_folds_all{g} = RMSE_folds_all;
    CumtrainRMSE_folds_all{g} = trainRMSE_folds_all;
    CumCov_beta{g} = Cov_beta;
    CumVarBeta{g} = VarBeta;
    Cumdf(g) = df;
    Cumpred_CI_mean{g} = pred_CI_mean;
    Cumpred_CI_pred{g} = pred_CI_pred;
    % Fold-wise RMSE statistics
    [~, bestLambdaIdx] = min(mean(RMSE_folds_all, 2));
    best_folds_test  = RMSE_folds_all(bestLambdaIdx, :);
    best_folds_train = trainRMSE_folds_all(bestLambdaIdx, :);
    Cumrmsefolds_test(g)  = mean(best_folds_test);
    Cumstdfolds_test(g)   = std(best_folds_test);
    Cumrmsefolds_train(g) = mean(best_folds_train);
    Cumstdfolds_train(g)  = std(best_folds_train);
    % Store RMSE and predictions
    Cumrmse_train(g) = rmse_train;
    Cumrmse_test(g)  = rmse_test;
    Cumy_pred_test{g} = y_pred_test;
    Cumy_pred_train{g} = y_pred_train;
    Cumlambda_opt(g) = lambda_opt;
end
%% ============================================================
% Post-analysis: Model selection and visualization
% % error bars
% --- Step 1: Calculate the Key Thresholds ---
% Find absolute minimum of Test RMSE
[minRMSE, CorrminIdx] = min(Cumrmsefolds_test);
bestThreshold_Min = CorrGrid(CorrminIdx);
% Calculate the 1-SE limit
% K is the number of folds (e.g., 10)
se_at_min = Cumstdfolds_test(CorrminIdx) / sqrt(K);
threshold_limit = minRMSE + se_at_min;
% Find the highest threshold (simplest model) within 1-SE of the minimum
candidate_idx = find(Cumrmsefolds_test <= threshold_limit);
bestThreshold_1SE = max(CorrGrid(candidate_idx));
% --- Step 2: Plotting ---
figure('Color', 'w'); hold on;
yyaxis left
valColor = [0 0.4470 0.7410];      % validation RMSE color (blue)
trainColor = [0.8500 0.3250 0.0980]; % training RMSE color
plot(CorrGrid, Cumrmsefolds_test, 'o-', ...
    'LineWidth',3,'Color',valColor,...
    'DisplayName','$\mathrm{Validation~RMSE \pm SE}$');
plot(CorrGrid, Cumrmsefolds_train, '^-','LineWidth',3,'MarkerSize',8,...
    'Color',trainColor,'DisplayName','Training RMSE');
hold on;
% Testing Error Bars
errorbar(CorrGrid, Cumrmsefolds_test, Cumstdfolds_test, 'k.', 'CapSize', 6, 'HandleVisibility','off',LineWidth=1.2);
% --- Step 3: Add Highlight Lines ---
yLim = ylim; % Get current y-axis limits for the lines
% Vertical line for Minimum Validation RMSE
line([bestThreshold_Min bestThreshold_Min], yLim, 'Color', [0.46 0.67 0.18], ...
    'LineStyle', '--', 'LineWidth', 2);
ylabel('RMSE','Interpreter','latex','FontSize',17);% % Make axis color match RMSE curve
ax = gca;
ax.YColor = valColor;
yyaxis right
clusterColor = [0.4940 0.1840 0.5560]; % purple
plot(CorrGrid, NoClusters,'s-','LineWidth',2,'MarkerSize',7,...
    'Color',clusterColor,...
    'DisplayName','Number of clusters');
ylabel('Number of Clusters','Interpreter','latex','FontSize',17);
ax.YColor = clusterColor;
xlabel('Correlation Threshold','Interpreter','latex','FontSize',17);
title('RMSE vs Correlation Threshold','Interpreter','latex','FontSize',17);
legend('Location','best','Interpreter','latex','FontSize',17);
set(gca, 'FontSize', 17, 'LineWidth', 1.2, 'TickDir', 'out', 'Box', 'off');
axis tight
xlabel('Correlation Threshold','Interpreter','latex','FontSize',17);
title('Model Selection','Interpreter','latex','FontSize',17);
legend('Location', 'best');
grid on;
hold off;
%% ============================================================
% Heatmap of RMSE across CorrCutOff and PIP thresholds
%% ============================================================
figure('Color', 'w','Units','inches','Position',[1 1 6 5]);
% Heatmap of validation RMSE
imagesc(PIP_Grid, CorrGrid, RMSE_test_PIP(1:end-1,:));  % exclude CorrCutOff=1 for clarity
colormap(parula); % suitable for publication
cb = colorbar;
ylabel(cb, 'CV RMSE','Interpreter','latex','FontSize',12,'FontWeight','bold');
xlabel('Group PIP Threshold','FontSize',12,'FontWeight','bold');
ylabel('Correlation Threshold','FontSize',12,'FontWeight','bold');
title('Performance Landscape: RMSE vs. Hyperparameters','FontSize',14);
set(gca, 'XTick', PIP_Grid, 'YTick', CorrGrid);
% Highlight optimal point in heatmap
ValidRMSE = RMSE_test_PIP(1:end-1,:);
[minVal, minIdx] = min(ValidRMSE(:));
[bestRow, bestCol] = ind2sub(size(ValidRMSE), minIdx);
hold on;
plot(PIP_Grid(bestCol), CorrGrid(bestRow), 'rp', 'MarkerSize',12, ...
    'MarkerFaceColor','r','DisplayName','Optimal Point');
legend('Location','northeast','FontSize',10);
grid off;
%% ============================================================
% Predicted vs Measured Plot with RMSE and Custom Error
%% ============================================================
ypredtest = Cumy_pred_test{CorrminIdx};  % predictions at optimal CorrCutOff
figure('Color', 'w','Position',[10 10 650 600]);
plot(ypredtest,yTest, 'ko', 'MarkerFaceColor','k','MarkerSize',6,'LineWidth',1.2);
hold on;
% Compute errors
rmse_val = rmse(ypredtest,yTest);                     % standard RMSE
custom_error = computeHybridrror(yTest, ypredtest, epsilon); % normalized error
% Identity line y=x
lims = [0, max([yTest(:); ypredtest(:)])];
plot(lims, lims, 'k','LineWidth',2);
xlabel('Predicted Roundness [$\mu$m]','FontSize',16,'Interpreter','latex');
ylabel('Measured Roundness [$\mu$m]','FontSize',16,'Interpreter','latex');
% Legend with RMSE and custom error
legend({...
    ['$\mathrm{Hyb-Error}: ', num2str(round(custom_error,3)*100, '%.3f'), '\%$, ','$\mathrm{RMSE}: ', num2str(rmse_val, '%.3f'), '\,\mu m$'],...
    'Diagonal Reference Line'},...
    'FontSize',16,'Location','best','Interpreter','latex');
lims = [0, ceil(max([ypredtest;yTest])) + 1];
xlim(lims); ylim(lims);
yticks_auto = get(gca,'YTick'); xticks(yticks_auto);
set(gca,'FontSize',17,'LineWidth',1.2,'TickDir','out','Box','off');
%% ============================================================
% Ridge Validation Curve (RMSE vs Lambda)
%% ============================================================
figure('Color','w'); hold on;
valRMSE   = CumRMSE_folds_all{CorrminIdx};
trainRMSE = CumtrainRMSE_folds_all{CorrminIdx};
meanVal = mean(valRMSE,2);
semVal  = std(valRMSE,0,2) ./ sqrt(size(valRMSE,2));
meanTrain = mean(trainRMSE,2);
% Optimal lambda
[~, idxMin] = min(meanVal);
lambda_min = lambda_grid(idxMin);
% Plot validation and training curves
errorbar(lambda_grid, meanVal, semVal, '-o','LineWidth',1.8,'MarkerSize',6,'CapSize',8,...
    'DisplayName','$\mathrm{Validation\ RMSE}$');
plot(lambda_grid, meanTrain, '-s','LineWidth',1.8,'MarkerSize',6,'DisplayName','$\mathrm{Training\ RMSE}$');
xline(lambda_min,'--k','LineWidth',1.6,'DisplayName','$\lambda_{\min}$','Interpreter','latex');
set(gca,'XScale','log','XDir','reverse','FontSize',12,'LineWidth',1.2,'TickLabelInterpreter','latex');
xlabel('$\lambda$ (Ridge regularization)','Interpreter','latex','FontSize',17);
ylabel('$\mathrm{RMSE}$','Interpreter','latex','FontSize',17);
legend('Interpreter','latex','Location','best','FontSize',17);
box on;
%% ============================================================
% Prediction Intervals for Ridge Model
%% ============================================================
y_pred_lower = Cumpred_CI_pred{CorrminIdx}(:,1);
y_pred_upper = Cumpred_CI_pred{CorrminIdx}(:,2);
x1 = (1:length(ypredtest))';
% Plot predictions with asymmetric error bars (95% PI)
figure('Color','w','Position',[10 10 850 550]);
err_low  = ypredtest - y_pred_lower;
err_high = y_pred_upper - ypredtest;
errorbar(x1, ypredtest, err_low, err_high, 'o', ...
    'Color','k', 'MarkerEdgeColor',[0 0.5 0], 'MarkerFaceColor',[0 0.8 0], ...
    'MarkerSize',7,'LineWidth',2,'CapSize',8);
hold on;
% Overlay measured data
scatter((1:length(yTest))', yTest, 'xr', 'LineWidth',2.5);
legend({'Ridge prediction (95\% PI)','Observed data'}, 'Location','best','Interpreter','latex','FontSize',17);
xlabel('Post-Inspection Measurements','FontSize',17,'Interpreter','latex');
ylabel('Roundness Indicator [$\mu$m]','FontSize',17,'Interpreter','latex');
axis tight;
set(gca,'FontSize',17,'LineWidth',1.2,'TickDir','out','Box','off');
CI_width_minimal = y_pred_upper - y_pred_lower;  % Full CI width
mean_CI_minimal = mean(CI_width_minimal);
median_CI_minimal = median(CI_width_minimal);
%% ============================================================
% Beta Coefficients with 95% Confidence Intervals
%% ============================================================
final_beta   = Cumbeta{CorrminIdx};
final_VarBeta = CumVarBeta{CorrminIdx};
SE_beta = sqrt(final_VarBeta);
n = size(xTrain_Valid,1);
df_vecnulmdl = Cumdf(CorrminIdx);
alpha = 0.05;
tcrit = tinv(1 - alpha/2, n-df_vecnulmdl); % t critical value
beta_opt = final_beta;
CI_lower = beta_opt - tcrit * SE_beta;
CI_upper = beta_opt + tcrit * SE_beta;
CI_95 = [CI_lower, CI_upper];
PredictorNames = ['Intercept'; SignalNames{CorrminIdx}];
CI_Table = table(PredictorNames, beta_opt, SE_beta, CI_lower, CI_upper, ...
    'VariableNames', {'Predictor','Beta','StdError','CI_Lower','CI_Upper'});
fprintf('Features Contribution from the Ridge Regression\n');
disp(CI_Table);
elapsedTime = toc;   % stop timer and get time (seconds)
disp(elapsedTime)
%% ============================================================
fprintf('Running: Data Efficiency Analysis (i.e., Learning Curve)...\n');
% Learning Curve
rng(seed)
K= 10;
X_optimal = xTrain_Valid(:,IdsRep{CorrminIdx});
X_optimal = [ones(size(X_optimal,1),1), X_optimal];
% % using from the previous analysis (active features and optimal lambda at that step)
lambda_opt= Cumlambda_opt(CorrminIdx);
trainFractions = 0.05:0.05:1.0;
nLevels = numel(trainFractions);
lc_train = zeros(nLevels, 1);
lc_valid = zeros(nLevels, 1);
lc_train_std = zeros(nLevels,1);
lc_valid_std = zeros(nLevels,1);
for s = 1:nLevels
    frac = trainFractions(s);
    nSub = round(frac * size(X_optimal,1));
    subsetIdx = randsample(size(X_optimal,1), nSub);
    xSub = X_optimal(subsetIdx,:);
    ySub = yTrain_Valid(subsetIdx);
    PP_sub = PP_train(subsetIdx,:);
    [~,~,groupID] = unique(PP_sub, 'rows');
    uniqueGroups = unique(groupID);
    cvLC  = cvpartition(length(uniqueGroups), 'KFold', K);
    % foldErrT = 0; foldErrV = 0;
    foldErrT = zeros(K,1);
    foldErrV = zeros(K,1);
    for f = 1:K
        trainMask = ismember(groupID, find(training(cvLC, f)));
        validMask = ismember(groupID, find(test(cvLC, f)));
        xTrain = xSub(trainMask,:);
        yTrain = ySub(trainMask);
        xValid = xSub(validMask,:);
        yValid = ySub(validMask);
        % ---- Normalize using TRAIN fold only ----
        % Check if the first column is all ones (intercept)
        if all(xTrain(:,1) == 1)
            % Do not normalize the first column
            Xtr_rest =  xTrain(:,2:end);                % exclude first column
            [Xtr_norm, xTrain_c, xTrain_s] = normalize(Xtr_rest, 1);  % normalize remaining columns
            Xtr = [ones(size(Xtr_norm,1),1), Xtr_norm];               % add back intercept
            Xval1 = xValid(:,2:end);                                              % start with original Xval
            Xval1 = (Xval1 - xTrain_c) ./ xTrain_s;  % normalize remaining columns
            Xval = [ones(size(Xval1,1),1), Xval1];
        else
            % Normalize all columns
            [Xtr, xTrain_c, xTrain_s] = normalize(Xtr1, 1);
            Xval = (Xval1 - xTrain_c) ./ xTrain_s;
        end
        % ---- Ridge Regression ----
        p = size(Xtr,2);
        I = eye(p);
        I(1,1) = 0;    % do NOT penalize intercept
        beta = (Xtr' * Xtr + lambda_opt * I) \ (Xtr' * yTrain);
        ypred = Xval * beta;
        ypred_train = Xtr * beta;
        foldErrT(f) =  sqrt(mean((yTrain - ypred_train).^2));
        foldErrV(f) =  sqrt(mean((yValid - ypred).^2));
    end
    % Mean
    lc_train(s) = mean(foldErrT);
    lc_valid(s) = mean(foldErrV);
    % Standard deviation
    lc_train_std(s) = std(foldErrT);
    lc_valid_std(s) = std(foldErrV);
end
% plot
% ---- Plot Learning Curve ----
nSamples = round(trainFractions * size(X_optimal,1));
figure
hold on
plot(nSamples, lc_train, '-o', ...
    'LineWidth',1.5,'MarkerSize',6);
% Validation curve
errorbar(nSamples, lc_valid, lc_valid_std, '-s', ...
    'LineWidth',1.5,'MarkerSize',6)
% ---- Highlight final model ----
plot(nSamples(end), lc_valid(end), 'ro', ...
    'MarkerSize',10,'LineWidth',2)
% ---- Highlight best validation point ----
[~,idx_best] = min(lc_valid);
plot(nSamples(idx_best), lc_valid(idx_best), 'ks', ...
    'MarkerSize',10,'LineWidth',2)
xlabel('Number of Training Samples', 'Interpreter','latex', 'FontSize',17)
ylabel('RMSE', 'Interpreter','latex', 'FontSize',17)
title('Learning Curve', 'Interpreter','latex', 'FontSize',17)
legend('Training','Validation', ...
    'Final Model','Best Validation','Location','best', 'Interpreter','latex', 'FontSize',17)
xlim([0 max(nSamples)])
set(gca, 'FontSize', 17, 'LineWidth', 1.2, 'TickDir', 'out', 'Box', 'off');
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
