function [DataOut] = find_lambda_cv(X, y, K, PP_train, lambda_grid, seed)
% FIND_LAMBDA_CV
% -------------------------------------------------------------
% Performs group K-fold cross-validation for Ridge regression
% to select the optimal regularization parameter (lambda).
%
% INPUTS:
%   X            - Feature matrix (n x p)
%   y            - Target vector (n x 1)
%   K            - Number of folds
%   PP_train     - Grouping variable to avoid data leakage
%   lambda_grid  - Vector of candidate lambda values
%   seed         - Random seed for reproducibility
%
% OUTPUT:
%   DataOut      - Struct containing:
%       lambda_opt       -> best lambda (min CV RMSE)
%       lambda_opt1se    -> lambda using 1-SE rule
%       rmse_cv          -> mean validation RMSE per lambda
%       rmse_se          -> standard error of RMSE
%       rmse_cvtrain     -> mean training RMSE
%       rmse_folds_all   -> RMSE per fold (validation)
%       rmse_train_all   -> RMSE per fold (training)
% -------------------------------------------------------------
rng(seed)
% % % data grouping group k fold to avoid leakage
PP_sub = PP_train;
[~,~,groupID] = unique(PP_sub, 'rows');
uniqueGroups = unique(groupID);
if length(uniqueGroups) < K
    error('Number of unique groups (%d) < K (%d)', length(uniqueGroups), K);
end
cv  = cvpartition(length(uniqueGroups), 'KFold', K);
% % % % % % % % % % % % % % % % % % % %
% Preallocate lambda grid
nLambda = length(lambda_grid);
rmse_folds_all = zeros(nLambda,K);
rmse_train_all = zeros(nLambda,K);
for l = 1:nLambda
    lambda = lambda_grid(l);
    % % % Group K fold loop
    for i = 1:K
        % Get training/validation group indices
        idx_tr  = training(cv,i);
        idx_val = test(cv,i);
        % Map group indices back to sample indices
        trainMask = ismember(groupID, find(idx_tr));
        valMask   = ismember(groupID, find(idx_val));
        % Split data
        Xtr1 = X(trainMask,:);
        ytr  = y(trainMask);
        Xval1 = X(valMask,:);
        yval  = y(valMask);
        % ---- Normalize using TRAIN fold only ----
        % Check if the first column is all ones (intercept)
        if all(Xtr1(:,1) == 1)
            % Do not normalize the first column
            Xtr_rest = Xtr1(:,2:end);
            % Normalize remaining features
            [Xtr_norm, xTrain_c, xTrain_s] = normalize(Xtr_rest, 1);
            % Rebuild training matrix with intercept
            Xtr = [ones(size(Xtr_norm,1),1), Xtr_norm];
            % Apply same normalization to validation set
            Xval = Xval1;
            Xval(:,2:end) = (Xval1(:,2:end) - xTrain_c) ./ xTrain_s;
        else
            % Normalize all features
            [Xtr, xTrain_c, xTrain_s] = normalize(Xtr1, 1);
            % Apply same transformation to validation
            Xval = (Xval1 - xTrain_c) ./ xTrain_s;
            % Add intercept term manually
            Xtr  = [ones(size(Xtr,1),1), Xtr];
            Xval = [ones(size(Xval,1),1), Xval];
        end
        % --- Ridge regression estimation ---
        % Intercept is NOT penalized
        p = size(Xtr,2);
        I = eye(p);
        % % % intercept always for the first column
        I(1,1) = 0;
        beta = (Xtr'*Xtr + lambda*I) \ (Xtr'*ytr);
        % --- Predictions ---
        y_pred = Xval * beta; % validation predictions
        y_pred_train= Xtr * beta; % training predictions
        rmse_folds_all(l,i) = sqrt(mean((yval - y_pred).^2));
        rmse_train_all(l,i) = sqrt(mean((ytr - y_pred_train).^2));
    end
end
%% --- Model selection ---
% Compute mean and standard error across folds
rmse_cv = mean(rmse_folds_all,2);
rmse_se = std(rmse_folds_all,0,2)/sqrt(K);
rmse_cvtrain = mean(rmse_train_all,2);
% Find lambda with minimum CV RMSE
[rmse_min_val, idx_min] = min(rmse_cv);
%% --- 1-SE Rule ---
% Select simplest model within 1 standard error of best
threshold_1se = rmse_min_val + rmse_se(idx_min);
lambda_1se_idx = find(rmse_cv <= threshold_1se, 1, 'first');
%% --- Store outputs ---
DataOut.lambda_opt = lambda_grid(idx_min);
DataOut.lambda_opt1se = lambda_grid(lambda_1se_idx);
DataOut.rmse_cv = rmse_cv;
DataOut.rmse_se = rmse_se;
DataOut.rmse_cvtrain = rmse_cvtrain;
DataOut.rmse_folds_all=rmse_folds_all;
DataOut.rmse_train_all=rmse_train_all;
end
