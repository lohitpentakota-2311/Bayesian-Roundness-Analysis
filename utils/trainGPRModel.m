function results = trainGPRModel(xTrain_Valid, yTrain_Valid, xTest, yTest, PP_train, KernalName, seed, FlagforParamterIdentification,KernelParamters,Sigma)
% trainGPRModel
% ------------------------------------------------------------
% Trains a Gaussian Process Regression (GPR) model with
% Bayesian optimization and generates evaluation plots.
%
% INPUTS:
%   xTrain_Valid  - Training features
%   yTrain_Valid  - Training targets
%   xTest         - Test features
%   yTest         - Test targets
%   PP_train      - Grouping variable for CV
%   KernalName    - Kernel function name
%
% OUTPUT:
%   results       - Struct with model, predictions, metrics
%
% ------------------------------------------------------------
rng(seed)
%% Hyperparameter definition
optimVars = [
    optimizableVariable('KernelScale',[0.1, 500],'Transform','log')
    optimizableVariable('Sigma',[0.1, 4],'Transform','log')
    ];

%% Grouped cross-validation
[~,~,groupID] = unique(PP_train, 'rows');
c = cvpartition(groupID, 'KFold', 10);

%% Optimization options
hpOpts = struct( ...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'MaxObjectiveEvaluations',200, ...
    'CVPartition', c);

%% Train model
if isequal(FlagforParamterIdentification,'on')
    gprMdl = fitrgp(xTrain_Valid, yTrain_Valid, ...
        'KernelFunction', KernalName, ...
        'Standardize', true, ...
        'OptimizeHyperparameters', optimVars, ...
        'HyperparameterOptimizationOptions', hpOpts);
else
    mu = mean(xTrain_Valid);
    sigma = std(xTrain_Valid);
    % Standardize training and test data
    xTrain_Valid = (xTrain_Valid - mu) ./ sigma;
    xTest  = (xTest - mu) ./ sigma;
    rng(seed)
    gprMdl = fitrgp(xTrain_Valid, yTrain_Valid, ...
        'KernelFunction',KernalName, ...
        'Standardize', false, ...
        'KernelParameters', KernelParamters, ...  % fixed
        'Sigma', Sigma);
end
%% Predictions
[yPredTrain, ~, yciTrain] = predict(gprMdl, xTrain_Valid);
[ypredtest, ~, yciTest] = predict(gprMdl, xTest);

%% Metrics
epsilon = 2.8;

rmse_train = rmse(yPredTrain, yTrain_Valid);
rmse_test  = rmse(ypredtest, yTest);

custom_train = computeMEHEError(yTrain_Valid, yPredTrain, epsilon);
custom_test  = computeMEHEError(yTest, ypredtest, epsilon);

%% =======================
%% TRAINING PLOT
figure('Color', 'w','Position',[10 10 650 600]);
plot(yPredTrain, yTrain_Valid, 'ko', 'MarkerFaceColor','k', ...
    'MarkerSize',6,'LineWidth',1.2);
hold on;

lims = [0, max([yTrain_Valid(:); yPredTrain(:)])];
axis([lims lims]);

plot(lims, lims, 'k','LineWidth',2);

xlabel('Predicted Roundness [$\mu$m]','Interpreter','latex','FontSize',16);
ylabel('Measured Roundness [$\mu$m]','Interpreter','latex','FontSize',16);
title('GPR Predictions','FontSize',16);
legend({...
    ['$\mathrm{HybridError}: ', num2str(round(custom_train,3)*100,'%.3f'), '\%$, ', ...
    '$\mathrm{RMSE}: ', num2str(rmse_train,'%.3f'), '\,\mu m$'], ...
    'Diagonal Reference Line'}, ...
    'Interpreter','latex','FontSize',16,'Location','best');

lims = [0, ceil(max([yPredTrain; yTrain_Valid])) + 2];
xlim(lims); ylim(lims);

set(gca,'FontSize',17,'LineWidth',1.2,'TickDir','out','Box','off');

%% =======================
%% TEST PLOT
figure('Color', 'w','Position',[10 10 650 600]);
plot(ypredtest, yTest, 'ko','MarkerFaceColor','k', ...
    'MarkerSize',6,'LineWidth',1.2);
hold on;

lims = [0, max([yTest(:); ypredtest(:)])];
axis([lims lims]);

plot(lims, lims, 'k','LineWidth',2);

xlabel('Predicted Roundness [$\mu$m]','Interpreter','latex','FontSize',16);
ylabel('Measured Roundness [$\mu$m]','Interpreter','latex','FontSize',16);
title('GPR Predictions','FontSize',16);
legend({...
    ['$\mathrm{HybridError}: ', num2str(round(custom_test,3)*100,'%.3f'), '\%$, ', ...
    '$\mathrm{RMSE}: ', num2str(rmse_test,'%.3f'), '\,\mu m$'], ...
    'Diagonal Reference Line'}, ...
    'Interpreter','latex','FontSize',16,'Location','best');

lims = [0, ceil(max([ypredtest; yTest])) + 1];
xlim(lims); ylim(lims);

set(gca,'FontSize',17,'LineWidth',1.2,'TickDir','out','Box','off');

%% =======================
%% ERROR BAR PLOT (TEST)
err = (yciTest(:,2) - yciTest(:,1)) / 2;

figure('Color','w','Position',[10 10 850 550]);
x1 = (1:length(ypredtest))';

errorbar(x1, ypredtest, err, 'o', ...
    'Color','k', ...
    'MarkerEdgeColor',[0 0.5 0], ...
    'MarkerFaceColor',[0 0.8 0], ...
    'MarkerSize',7,'LineWidth',2,'CapSize',8);
hold on;

scatter(x1, yTest, 'xr','LineWidth',3);

legend({'Predictions (mean $\pm$ 95\% CI)', 'Measured data'}, ...
    'Interpreter','latex','FontSize',22,'Location','best');
title('GPR Predictions','FontSize',16);
xlabel('Post-Inspection Measurements','Interpreter','latex','FontSize',22);
ylabel('Roundness Indicator [$\mu$m]','Interpreter','latex','FontSize',22);

axis tight;
set(gca,'FontSize',22,'LineWidth',1.2,'TickDir','out','Box','off');

%% =======================
%% Store outputs
results.model = gprMdl;

results.train.predictions = yPredTrain;
results.train.rmse = rmse_train;
results.train.custom_error = custom_train;
results.train.ci = yciTrain;

results.test.predictions = ypredtest;
results.test.rmse = rmse_test;
results.test.custom_error = custom_test;
results.test.ci = yciTest;

end