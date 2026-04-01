% % % Hybrid Error Calculation wiht the given threshold value.
function custom_error = computeHybridrror(yTrue, yPred, epsilon)
% computeCustomError computes a normalized absolute error:
%
% custom_error = mean(abs(yTrue - yPred) ./ (abs(yTrue) + epsilon))
%
% Inputs:
%   yTrue   : vector of true values
%   yPred   : vector of predicted values
%   epsilon : small constant to avoid division by zero (optional, default 3)
%
% Output:
%   custom_error : dimensionless normalized error
if nargin < 3
    epsilon = 3; % default value if not provided
end
yTrue = yTrue(:);
yPred = yPred(:);
custom_error = mean(abs(yTrue - yPred) ./ (abs(yTrue) + epsilon));
end