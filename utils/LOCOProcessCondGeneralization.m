%% LOCO: Leave-One-Condition-Out Robustness Analysis
fprintf('Running: LOGO Robustness Analysis...\n');

% Identify unique process conditions in the FULL training set
[uniqueConditions, ~, condGroupID] = unique(PP_train, 'rows');

% % % Unique Conditions
nConditions = size(uniqueConditions, 1);

% % % % Storing
LOCO_rmse_Lasso = zeros(nConditions, 1);
LOCO_rmse_Conj  = zeros(nConditions, 1);
LOCO_coverage_Lasso = zeros(nConditions, 1);
LOCO_coverage_Conj  = zeros(nConditions, 1);


rng(seed);

for k = 1:nConditions

    % Split: leave condition k out
    locoTestMask  = (condGroupID == k);

    locoTrainMask = ~locoTestMask;

    if sum(locoTestMask) < 2
        LOCO_rmse_Lasso(k) = NaN;
        LOCO_rmse_Conj(k)  = NaN;
        LOCO_coverage_Lasso(k) = NaN;
        LOCO_coverage_Conj(k)  = NaN;
        continue;
    end  % skip tiny groups

    xLOCO_train = xTrain_Valid(locoTrainMask, :);

    yLOCO_train = yTrain_Valid(locoTrainMask);

    xLOCO_test  = xTrain_Valid(locoTestMask, :);

    yLOCO_test  = yTrain_Valid(locoTestMask);

    % Normalize using LOCO train fold only
    [xLOCO_trainNorm, mu_loco, sig_loco] = normalize(xLOCO_train);

    sig_loco(sig_loco == 0) = 1;

    xLOCO_testNorm = (xLOCO_test - mu_loco) ./ sig_loco;

    % -- Lasso --
    PriorTmp = bayeslm(size(xLOCO_trainNorm,2), 'ModelType','Lasso','VarNames',FeatureNames);

    PriorTmp.Lambda = bestlambda;

    [EstTmpLasso, ~] = estimate(PriorTmp, xLOCO_trainNorm, yLOCO_train, ...
        'BurnIn',1000,'NumDraws',5000);

    [yhatLasso, CovHatLasso] = forecast(EstTmpLasso, xLOCO_testNorm);

    % % % train
    [yhattrainLasso, CovHattrainLasso] = forecast(EstTmpLasso, xLOCO_trainNorm);

    PriorMdl = bayeslm(size(xLOCO_trainNorm,2),'ModelType','MixConjugate','V',V,'VarNames',FeatureNames);

    [EstTmp, ~] = estimate(PriorMdl, xLOCO_trainNorm, yLOCO_train, ...
        'Display',false,'BurnIn',1000,'NumDraws',5000);

    [yhat, CovHat] = forecast(EstTmp, xLOCO_testNorm);

    [yhattrain, CovHattrain] = forecast(EstTmp, xLOCO_trainNorm);

    % % Errors
    LOCO_rmse_Lasso(k) = rmse(yhatLasso, yLOCO_test);

    LOCO_rmse_trainLasso(k) = rmse(yhattrainLasso, yLOCO_train);

    LOCO_rmse_Conj(k) = rmse(yhat, yLOCO_test);

    LOCO_trainrmse_Conj(k) = rmse(yhattrain, yLOCO_train);

    % % Confidence Intervals (test)
    lbLasso = yhatLasso - norminv(0.975)*sqrt(diag(CovHatLasso));

    ubLasso = yhatLasso + norminv(0.975)*sqrt(diag(CovHatLasso));
    
    % % train
    lbLassotrain = yhattrainLasso - norminv(0.975)*sqrt(diag(CovHattrainLasso));

    ubLassotrain = yhattrainLasso + norminv(0.975)*sqrt(diag(CovHattrainLasso));

    % % % % Mix-Conj SSVS (test)
    lb = yhat - norminv(0.975)*sqrt(diag(CovHat));

    ub = yhat + norminv(0.975)*sqrt(diag(CovHat));
    
    % % % % Mix-Conj SSVS (train)
    lbtrain = yhattrain - norminv(0.975)*sqrt(diag(CovHattrain));

    ubtrain = yhattrain + norminv(0.975)*sqrt(diag(CovHattrain));

    % LOCO_coverage_Lasso(k) = mean((yLOCO_test >= lbLasso) & (yLOCO_test <= ubLasso)) * 100;

    LOCO_coverage_Conj(k) = mean((yLOCO_test >= lb) & (yLOCO_test <= ub)) * 100;

    LOCO_coverage_trainConj(k) = mean((yLOCO_train >= lbtrain) & (yLOCO_train <= ubtrain)) * 100;

    % % % % % % % % % % 
    LOCO_coverage_Lasso(k) = mean((yLOCO_test >= lbLasso) & (yLOCO_test <= ubLasso)) * 100;

    LOCO_coverage_trainlasso(k) = mean((yLOCO_train >= lbLassotrain) & (yLOCO_train <= ubLassotrain)) * 100;
end

% % % % % % % conjugate 
% % % % % Remove conditions that were skipped
% % % % valid = ~isnan(LOCO_rmse_Conj) & ~isnan(LOCO_coverage_Conj);
% % % % % 
% % % % % figure('Color','w','Position',[10 10 750 500]);
% % % % % bar(find(valid),  LOCO_rmse_Conj(valid), 'FaceColor',[0.2 0.4 0.8],'EdgeColor','k');
% % % % % hold on;
% % % % % yline(rmse_test, 'r--', 'LineWidth', 2, ...
% % % % %     'DisplayName', 'Test RMSE (Unseen Process Conditions)');
% % % % % title({'Bayesian (Mix-Conj SSVS)', ...
% % % % %     'LOCO RMSE on Training Set'}, ...
% % % % %     'FontSize',16,'Interpreter','latex')
% % % % % xlabel('Left-Out Process Condition ID', ...
% % % % %     'FontSize',17, ...
% % % % %     'Interpreter','latex')
% % % % % ylabel('Prediction RMSE [$\mu$m]', ...
% % % % %     'FontSize',17, ...
% % % % %     'Interpreter','latex')
% % % % % legend('LOCO RMSE', 'Test Set RMSE', 'Location','best', 'FontSize',14);
% % % % % set(gca, 'FontSize', 17, 'LineWidth', 1.2, 'TickDir', 'out', 'Box', 'off');
% % % % 
% % % % rmse_vals = LOCO_rmse_Conj(valid);
% % % % 
% % % % condID = find(valid);
% % % % 
% % % % cov_vals  = LOCO_coverage_Conj(valid);
% % % % 
% % % % figure('Color','w','Position',[100 100 850 450]);
% % % % 
% % % % yyaxis left
% % % % 
% % % % h1 = bar(condID, rmse_vals, ...
% % % %     'FaceColor',[0.2 0.4 0.8], 'EdgeColor','k');
% % % % 
% % % % ylabel('RMSE on Held-Out Process Condition ($\mu$m)', 'FontSize',17,'Interpreter','latex');
% % % % 
% % % % hold on;
% % % % 
% % % % h2 =yline(rmse_test, 'r--', 'LineWidth', 2);
% % % % 
% % % % yyaxis right
% % % % 
% % % % h3 = plot(condID, cov_vals, 'o-', ...
% % % %     'LineWidth', 2, 'MarkerSize', 6, ...
% % % %     'Color',[0.1 0.6 0.2]);
% % % % 
% % % % ylabel('Coverage (\%)', 'FontSize',17,'Interpreter','latex');
% % % % 
% % % % 
% % % % h4 = yline(95, '--', 'Color',[0.2 0.2 0.2], 'LineWidth', 1.5);
% % % % 
% % % % ylim([0 110]);
% % % % 
% % % % xlabel('Condition ID', 'FontSize',17,'Interpreter','latex');
% % % % 
% % % % title('LOCO Robustness and Calibration Across Training Process Conditions', ...
% % % %     'FontSize',17,'Interpreter','latex');
% % % % 
% % % % legend([h1 h2 h3 h4], ...
% % % %     {'RMSE (Mix-Conj SSVS)', ...
% % % %      'Test Set RMSE', ...
% % % %      'Coverage (\%)', ...
% % % %      '95\% nominal'}, ...
% % % %     'Location','best','Interpreter','latex');
% % % % ax = gca;
% % % % ax.YAxis(1).Color = 'k';             % left axis black (RMSE)
% % % % ax.YAxis(2).Color = [0.1 0.6 0.2];  % right axis green (Coverage)
% % % % 
% % % % set(gca,'FontSize',17,'LineWidth',1.2,'TickDir','out');
% % % % box off;
% % % % 
% % % % % fprintf('Mean LOCO RMSE (Lasso): %.3f um\n', mean( LOCO_rmse_Conj(valid)));
% % % % % fprintf('Mean LOCO Coverage (Lasso): %.1f%%\n', mean( LOCO_coverage_Conj(valid)));
% % % % 
% % % % % % % % 
% % % % 
% % % % % Remove conditions that were skipped
% % % % validLasso = ~isnan(LOCO_rmse_Lasso) & ~isnan( LOCO_coverage_Lasso);
% % % % % 
% % % % % figure('Color','w','Position',[10 10 750 500]);
% % % % % bar(find(valid),  LOCO_rmse_Conj(valid), 'FaceColor',[0.2 0.4 0.8],'EdgeColor','k');
% % % % % hold on;
% % % % % yline(rmse_test, 'r--', 'LineWidth', 2, ...
% % % % %     'DisplayName', 'Test RMSE (Unseen Process Conditions)');
% % % % % title({'Bayesian (Mix-Conj SSVS)', ...
% % % % %     'LOCO RMSE on Training Set'}, ...
% % % % %     'FontSize',16,'Interpreter','latex')
% % % % % xlabel('Left-Out Process Condition ID', ...
% % % % %     'FontSize',17, ...
% % % % %     'Interpreter','latex')
% % % % % ylabel('Prediction RMSE [$\mu$m]', ...
% % % % %     'FontSize',17, ...
% % % % %     'Interpreter','latex')
% % % % % legend('LOCO RMSE', 'Test Set RMSE', 'Location','best', 'FontSize',14);
% % % % % set(gca, 'FontSize', 17, 'LineWidth', 1.2, 'TickDir', 'out', 'Box', 'off');
% % % % 
% % % % rmse_valsLasso = LOCO_rmse_Lasso(validLasso);
% % % % 
% % % % condIDLasso = find(validLasso);
% % % % 
% % % % cov_valsLasso  = LOCO_coverage_lasso(validLasso);
% % % % 
% % % % figure('Color','w','Position',[100 100 850 450]);
% % % % 
% % % % yyaxis left
% % % % 
% % % % h1 = bar(condIDLasso, rmse_valsLasso, ...
% % % %     'FaceColor',[0.2 0.4 0.8], 'EdgeColor','k');
% % % % 
% % % % ylabel('RMSE on Held-Out Process Condition ($\mu$m)', 'FontSize',17,'Interpreter','latex');
% % % % 
% % % % hold on;
% % % % 
% % % % h2 =yline(rmse_test_Lasso, 'r--', 'LineWidth', 2);
% % % % 
% % % % yyaxis right
% % % % 
% % % % h3 = plot(condIDLasso, cov_valsLasso, 'o-', ...
% % % %     'LineWidth', 2, 'MarkerSize', 6, ...
% % % %     'Color',[0.1 0.6 0.2]);
% % % % 
% % % % ylabel('Coverage (\%)', 'FontSize',17,'Interpreter','latex');
% % % % 
% % % % 
% % % % h4 = yline(95, '--', 'Color',[0.2 0.2 0.2], 'LineWidth', 1.5);
% % % % 
% % % % ylim([0 110]);
% % % % 
% % % % xlabel('Condition ID', 'FontSize',17,'Interpreter','latex');
% % % % 
% % % % title('LOCO Robustness and Calibration Across Training Process Conditions', ...
% % % %     'FontSize',17,'Interpreter','latex');
% % % % 
% % % % legend([h1 h2 h3 h4], ...
% % % %     {'RMSE (Mix-Conj SSVS)', ...
% % % %      'Test Set RMSE', ...
% % % %      'Coverage (\%)', ...
% % % %      '95\% nominal'}, ...
% % % %     'Location','best','Interpreter','latex');
% % % % ax = gca;
% % % % ax.YAxis(1).Color = 'k';             % left axis black (RMSE)
% % % % ax.YAxis(2).Color = [0.1 0.6 0.2];  % right axis green (Coverage)
% % % % 
% % % % set(gca,'FontSize',17,'LineWidth',1.2,'TickDir','out');
% % % % box off;

% % % % % % % % % % % % % % % % % % % % % % % % % % % % 
valid = ~isnan(LOCO_rmse_Conj) & ~isnan(LOCO_rmse_Lasso);

condID = find(valid);

rmse_conj  = LOCO_rmse_Conj(valid);
rmse_lasso = LOCO_rmse_Lasso(valid);

cov_conj   = LOCO_coverage_Conj(valid);
cov_lasso  = LOCO_coverage_Lasso(valid);

figure('Color','w','Position',[100 100 900 650]);

% =========================
% TOP: RMSE
% =========================
subplot(2,1,1)

plot(condID, rmse_conj, '-o', 'LineWidth', 2); hold on;

plot(condID, rmse_lasso, '-s', 'LineWidth', 2);

yline(rmse_test_Lasso, 'r--', 'LineWidth', 1.5);

yline(rmse_test, 'b--', 'LineWidth', 1.5);


ylabel('RMSE ($\mu$m)', 'FontSize',20,'Interpreter','latex');

title('Leave-one-Group-out Robustness','FontSize',20,'Interpreter','latex');

legend({'Mix-Conj SSVS RMSE','LASSO RMSE','LASSO: Test Set RMSE','Mix-Conj SSVS: Test Set RMSE'}, ...
    'Location','best','Interpreter','latex');

% grid on;

set(gca,'FontSize',20,'LineWidth',1.2,'TickDir','out');

% =========================
% BOTTOM: COVERAGE
% =========================
subplot(2,1,2)

plot(condID, cov_conj, '--o', 'LineWidth', 2); hold on;

plot(condID, cov_lasso, '--s', 'LineWidth', 2);

yline(95, 'k--', 'LineWidth', 1.5);

ylabel('PI Coverage (\%)', 'FontSize',20,'Interpreter','latex');

xlabel('Left-Out Process Condition ID', 'FontSize',20,'Interpreter','latex');

legend({'Mix-Conj SSVS Coverage','LASSO Coverage','95$\%$ nominal'}, ...
    'Location','best','Interpreter','latex');

ylim([30 105])                          % not [0 110] — cuts dead whitespace
% yticks([50 75 95 100])                  % tick at 95 makes nominal line obvious

% grid on;

set(gca,'FontSize',20,'LineWidth',1.2,'TickDir','out');

% % % values used for the paper
% RMSE summary
fprintf('Mix-Conj SSVS: Mean RMSE = %.3f, Std = %.3f\n', mean(rmse_conj), std(rmse_conj));
fprintf('LASSO: Mean RMSE = %.3f, Std = %.3f\n', mean(rmse_lasso), std(rmse_lasso));
fprintf('Conditions where SSVS < LASSO: %d/%d\n', sum(rmse_conj < rmse_lasso), numel(rmse_conj));

% Coverage summary
fprintf('Mix-Conj SSVS: Mean Coverage = %.1f%%\n', mean(cov_conj));
fprintf('LASSO: Mean Coverage = %.1f%%\n', mean(cov_lasso));
fprintf('SSVS conditions below 95%%: %d\n', sum(cov_conj < 95));
fprintf('LASSO conditions below 95%%: %d\n', sum(cov_lasso < 95));
fprintf('LASSO min coverage: %.1f%%\n', min(cov_lasso));
fprintf('SSVS min coverage: %.1f%%\n', min(cov_conj));