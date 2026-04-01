function [bestV1, bestV2] = findBestV1V2( ...
    xTrain_Valid, yTrain_Valid, InittrainIdx, ProcessParameters, V1, V2, K, seed)
% FINDBESTV1V2
% -------------------------------------------------------------------------
% Performs grouped K-fold cross-validation to select optimal hyperparameters
% (V1, V2) for a Bayesian MixConjugate linear model.
%
% INPUTS:
%   xTrain_Valid      - Training + validation feature matrix (n x p)
%   yTrain_Valid      - Training + validation target vector (n x 1)
%   InittrainIdx      - Indices used to extract grouping variables
%   ProcessParameters - Grouping variables (to prevent leakage)
%   V1                - Grid of first variance hyperparameter values
%   V2                - Grid of second variance hyperparameter values
%   K                 - Number of cross-validation folds
%   seed              - Random seed for reproducibility
%
% OUTPUTS:
%   bestV1            - Optimal V1 (minimizing CV RMSE)
%   bestV2            - Optimal V2 (minimizing CV RMSE)
% -------------------------------------------------------------------------
rng(seed)
% ============================================================
%  Grouped CV setup
% ============================================================
PP_train = ProcessParameters(InittrainIdx, :);
[~,~,groupID] = unique(PP_train, 'rows');
uniqueGroups  = unique(groupID);
cv2 = cvpartition(length(uniqueGroups), 'KFold', K);
% ============================================================
%  Initialization
% ============================================================
numv1 = numel(V1);
numv2 = numel(V2);
% Store RMSE for each (V1, V2, fold)
foldErrorsValid = zeros(numv1, numv2, K);
foldErrorsTrain = zeros(numv1, numv2, K);
% Number of features
p = size(xTrain_Valid, 2);
% ============================================================
%  Cross-validation
% ============================================================
for k = 1:numv2
    for j = 1:numv1
        Cumfmse_traincv = 0;
        Cumfmse_cv      = 0;
        % Loop over folds
        for fold = 1:K
            trainIdx = training(cv2, fold);
            validIdx = test(cv2, fold);
            % --- FIXED grouping logic (important!) ---
            trainMask = ismember(groupID, uniqueGroups(trainIdx));
            valMask   = ismember(groupID, uniqueGroups(validIdx));
            % Leakage check
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
            % ------------------------------------------------
            %  Bayesian MixConjugate model
            % ------------------------------------------------
            V = [V1(j)*ones(p+1,1), V2(k)*ones(p+1,1)];
            PriorMdl = bayeslm(p, 'ModelType', 'Mixconjugate', 'V', V);
            PosteriorMdl = estimate( ...
                PriorMdl, xTrain, yTrain, ...
                'Display', false, ...
                'BurnIn', 5000, ...
                'NumDraws', 50000);
            % ------------------------------------------------
            %  Training error
            % ------------------------------------------------
            yhattrain = forecast(PosteriorMdl, xTrain);
            fmse_train = sqrt(mean((yTrain - yhattrain).^2));
            Cumfmse_traincv = Cumfmse_traincv + fmse_train;
            foldErrorsTrain(j, k, fold) = fmse_train;
            % ------------------------------------------------
            %  Validation error
            % ------------------------------------------------
            yhat = forecast(PosteriorMdl, xValid);
            fmse_valid = sqrt(mean((yValid - yhat).^2));
            Cumfmse_cv = Cumfmse_cv + fmse_valid;
            foldErrorsValid(j, k, fold) = fmse_valid;

        end
    end
end
% ============================================================
%  Model selection
% ============================================================
meanValid = mean(foldErrorsValid, 3);
% --- Best V2 (averaging over V1) ---
avgValid_overV1 = mean(meanValid, 1);
[~, idxV2] = min(avgValid_overV1);
bestV2 = V2(idxV2);
% --- Best V1 (averaging over V2) ---
avgValid_overV2 = mean(meanValid, 2);
[~, idxV1] = min(avgValid_overV2);
bestV1 = V1(idxV1);
end