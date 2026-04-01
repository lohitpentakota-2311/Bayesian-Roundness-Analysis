%%
function bestLambda = findBestLambda(x, y, InittrainIdx, testIdx, ProcessParameters, K, seed)
% FINDBESTLAMBDA
% -------------------------------------------------------------------------
% Selects the optimal lambda for Bayesian LASSO using grouped K-fold
% cross-validation on the training set.
%
% INPUTS:
%   x                 - Feature matrix (n x p)
%   y                 - Target vector (n x 1)
%   InittrainIdx      - Indices for initial training/validation split
%   testIdx           - Indices for test set
%   ProcessParameters - Grouping variables (to prevent leakage)
%   K                 - Number of CV folds
%   seed              - Random seed for reproducibility
%
% OUTPUT:
%   bestLambda        - Selected lambda minimizing CV RMSE
% -------------------------------------------------------------------------

%% ============================================================
%  Train/Test split
% ============================================================
rng(seed)
xTrain_Valid = x(InittrainIdx, :);
yTrain_Valid = y(InittrainIdx);
xTest = x(testIdx, :);
%% ============================================================
%  Leakage check (train vs test)
% ============================================================
% Ensure no overlap between training and test sets (critical!)
if any(ismember(xTest(:,137:end), xTrain_Valid(:,137:end), 'rows'))
    error('Data leakage detected: test rows found in training set!');
end
%% ============================================================
%  Grouped Cross-Validation setup
% ============================================================
% Create group IDs to enforce grouped CV (avoid leakage across folds)
PP_train = ProcessParameters(InittrainIdx, :);
[~,~,groupID] = unique(PP_train, 'rows');
uniqueGroups  = unique(groupID);
% Partition groups (not individual samples)
cv2 = cvpartition(length(uniqueGroups), 'KFold', K);
p = size(xTrain_Valid, 2);
% ============================================================
%  Lambda grid via frequentist LASSO
% ============================================================
[~, FitInfo] = lasso(xTrain_Valid, yTrain_Valid, 'Standardize', true);
Lambda = FitInfo.Lambda .* ...
    sum(~any(isnan(xTrain_Valid),2)) ./ sqrt(FitInfo.MSE);
numval = numel(Lambda);
% ============================================================
%  Storage
% ============================================================
foldErrorsValid = zeros(numval, K);
% ============================================================
%  Cross-validation loop
% ============================================================
for j = 1:numval
    Cumfmse_cv = 0;
    for fold = 1:K
        trainIdx = training(cv2, fold);
        validIdx = test(cv2, fold);
        trainMask = ismember(groupID, uniqueGroups(trainIdx));
        valMask   = ismember(groupID, uniqueGroups(validIdx));
        % Leakage check inside CV
        leak = sum(ismember(PP_train(valMask,:), PP_train(trainMask,:), 'rows'));
        if leak > 0
            error('Data leakage detected in fold %d!', fold)
        end
        % Split
        xTrain = xTrain_Valid(trainMask,:);
        yTrain = yTrain_Valid(trainMask);

        xValid = xTrain_Valid(valMask,:);
        yValid = yTrain_Valid(valMask);
        % Standardize on TRAIN only
        [xTrain, mu, sigma] = normalize(xTrain, 1);
        xValid = (xValid - mu) ./ sigma;
        % Bayesian LASSO
        PriorMdl = bayeslm(p, 'ModelType', 'Lasso');
        % % Intercept not penalized
        PriorMdl.Lambda(2:end) = Lambda(j);
        PosteriorMdl = estimate( ...
            PriorMdl, xTrain, yTrain, ...
            'Display', false, ...
            'BurnIn', 5000, ...
            'NumDraws', 35000);
        % Validation error
        yhat = forecast(PosteriorMdl, xValid);
        fmse_Valid = rmse(yValid, yhat);
        Cumfmse_cv = Cumfmse_cv + fmse_Valid;
        foldErrorsValid(j, fold) = fmse_Valid;
    end
end
% ============================================================
%  Select best lambda
% ============================================================
meanValid = mean(foldErrorsValid, 2);
[~, minIdx] = min(meanValid);
bestLambda = Lambda(minIdx);
end