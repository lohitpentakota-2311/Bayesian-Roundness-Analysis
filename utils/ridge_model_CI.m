function [beta, lambda_opt, y_pred_test,y_pred_train, rmse_test,rmse_train, ...
    Cov_beta, VarBeta, sigma2, df, ...
    pred_CI_mean, pred_CI_pred, RMSE_folds, trainRMSE_folds, beta_grid, stability] = ...
    ridge_model_CI(Xtrain, ytrain, Xtest, ytest, lambda_grid, K, PP_train, seed, OptimalLambda)
% RIDGE_MODEL_CI
% -------------------------------------------------------------------------
% Performs ridge regression with cross-validated lambda, computes predictions,
% training and test RMSE, covariance of coefficients, and 95% prediction intervals.
%
% INPUTS:
%   Xtrain      - Training feature matrix (n x p)
%   ytrain      - Training response vector (n x 1)
%   Xtest       - Test feature matrix (n_test x p)
%   ytest       - Test response vector (n_test x 1)
%   lambda_grid - Vector of candidate ridge regularization parameters
%   K           - Number of folds for cross-validation
%   PP_train    - Grouping variable to avoid leakage during CV
%   seed        - Random seed for reproducibility
%
% OUTPUTS:
%   beta            - Ridge regression coefficients
%   lambda_opt      - Optimal lambda from CV
%   y_pred_test     - Predictions on test set
%   y_pred_train    - Predictions on training set
%   rmse_test       - RMSE on test set
%   rmse_train      - RMSE on training set
%   Cov_beta        - Covariance matrix of ridge coefficients
%   VarBeta         - Variance of each coefficient (diagonal of Cov_beta)
%   sigma2          - Residual variance estimate
%   df              - Effective degrees of freedom (trace of ridge hat matrix)
%   pred_CI_mean    - 95% confidence interval for mean predictions
%   pred_CI_pred    - 95% prediction interval for new observations
%   RMSE_folds      - Validation RMSE across folds (from CV)
%   trainRMSE_folds - Training RMSE across folds (from CV)
% -------------------------------------------------------------------------
n = size(Xtrain,1);
n_test = size(Xtest,1);

% ---- Step 1: select lambda via CV ----
if isempty(OptimalLambda)
    DataOut = find_lambda_cv(Xtrain, ytrain, K, PP_train, lambda_grid, seed);

    lambda_opt = DataOut.lambda_opt;

    RMSE_folds = DataOut.rmse_folds_all;

    trainRMSE_folds = DataOut.rmse_train_all;
else
    lambda_opt= OptimalLambda; RMSE_folds = []; trainRMSE_folds = [];
end
% %% ---- Step 1: Select lambda via cross-validation ----
% DataOut = find_lambda_cv(Xtrain, ytrain, K, PP_train, lambda_grid, seed);
% lambda_opt = DataOut.lambda_opt;
% RMSE_folds = DataOut.rmse_folds_all;
% trainRMSE_folds = DataOut.rmse_train_all;
%% ---- Step 2: Normalize predictors ----
[Xtrain_norm, mu, sigma] = normalize(Xtrain,1);
Xtrain_norm = [ones(n,1) Xtrain_norm];  % Add intercept
Xtest_norm = (Xtest - mu) ./ sigma;
Xtest_norm = [ones(n_test,1) Xtest_norm];  % Add intercept
%% ---- Step 3: Ridge regression estimation ----
p = size(Xtrain_norm,2);
I = eye(p);
I(1,1)= 0; % intercept not penalized
XtX = Xtrain_norm' * Xtrain_norm;
XtX_reg = XtX + lambda_opt * I;
XtX_inv = inv(XtX_reg);
beta = XtX_inv * Xtrain_norm' * ytrain;
%%
% ---------------- Step 3b: Coefficients across lambda grid (stability check) ----------------
num_lambda = length(lambda_grid);

beta_grid = zeros(p, num_lambda);

for i = 1:num_lambda

    lambda = lambda_grid(i);

    XtX_reg_tmp = XtX + lambda * I;

    XtX_inv_tmp = inv(XtX_reg_tmp);

    beta_grid(:,i) = XtX_inv_tmp * Xtrain_norm' * ytrain;
end

% Compute stability metric (ignore intercept)
stability = mean(abs(beta_grid(2:end,:)), 2) ./ max(abs(beta_grid(2:end,:)), [], 2);
%% ---- Step 4: Predictions ----
y_pred_train = Xtrain_norm * beta;
y_pred_test  = Xtest_norm * beta;
rmse_test = sqrt(mean((ytest - y_pred_test).^2));
rmse_train= sqrt(mean((ytrain - y_pred_train).^2));
%% ---- Step 5: Residual variance and degrees of freedom ----
resid = ytrain - y_pred_train;
H = Xtrain_norm * XtX_inv * Xtrain_norm'; % ridge hat matrix
df = trace(H); % Effective degrees of freedom
sigma2 = (resid' * resid) / (n - df); % Residual variance
% ---- Step 6: covariance of ridge estimator ----
Cov_beta = sigma2 * XtX_inv * XtX * XtX_inv; % Generalized covariance formula
VarBeta = diag(Cov_beta); % Variance of each coefficient
% ---- Step 7: prediction intervals ----
alpha = 0.05;
tcrit = tinv(1 - alpha/2, max(1,n-df));
% variance of mean prediction
var_mean = sum((Xtest_norm * Cov_beta) .* Xtest_norm, 2);
se_mean = sqrt(var_mean);
% Variance of new observation (prediction interval)
se_pred = sqrt(var_mean + sigma2);
% 95% confidence interval for mean predictions
pred_CI_mean = [ ...
    y_pred_test - tcrit * se_mean, ...
    y_pred_test + tcrit * se_mean ];
% 95% prediction interval for new observations
pred_CI_pred = [ ...
    y_pred_test - tcrit * se_pred, ...
    y_pred_test + tcrit * se_pred ];

end

