% % Bayesian Model Convergence Calculations
function [SummaryTable, rHat, ess, mcse] = computeConvergence(EstMdl)
% Compute R-hat, ESS, and MCSE for Bayesian model convergence
allDraws = EstMdl.BetaDraws(2:end,1:end);
numParams = size(allDraws,1);
totalIter = size(allDraws,2);
numChains = 10;
iterPerChain = floor(totalIter / numChains);
rHat = zeros(numParams,1);
ess  = zeros(numParams,1);
mcse = zeros(numParams,1);
for i = 1:numParams
    paramDraws = allDraws(i, 1:(numChains*iterPerChain));
    chainsMatrix = reshape(paramDraws, iterPerChain, numChains);
    chainMeans = mean(chainsMatrix,1);
    chainVars  = var(chainsMatrix,0,1);
    W = mean(chainVars);
    B = iterPerChain * var(chainMeans);
    if W==0
        rHat(i)=1;
    else
        varHat = ((iterPerChain-1)/iterPerChain)*W + (1/iterPerChain)*B;
        rHat(i) = sqrt(varHat/W);
    end
    [acf,~] = autocorr(paramDraws, 'NumLags', min(100,totalIter-1));
    sum_rho = sum(acf(acf>0)) - acf(1);
    ess(i) = totalIter/(1+2*sum_rho);
    mcse(i) = std(paramDraws)/sqrt(ess(i));
end
rHat = real(rHat);
ess  = real(ess);
mcse = real(mcse);
SummaryTable = table({'Split-Rhat'; 'ESS'; 'MCSE'}, ...
    [mean(rHat); mean(ess); mean(mcse)], ...
    [min(rHat); min(ess); min(mcse)], ...
    [max(rHat); max(ess); max(mcse)], ...
    {'< 1.1'; '> 1000'; 'Small'}, ...
    'VariableNames', {'Metric', 'Mean', 'Min', 'Max', 'Target'});
if max(rHat) > 1.1
    warning('Maximum R-hat is %.4f. Model may not have converged.', max(rHat));
else
    fprintf('Convergence Confirmed: Max R-hat is %.4f\n', max(rHat));
end
end