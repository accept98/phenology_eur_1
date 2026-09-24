% ========================================================================
% calibrate_parameters_example.m
% Example calibration code for the LNPP-based maize phenology model.
%
% This example uses the maturity stage at GC. In the study, the same
% calibration procedure was applied separately to flowering and maturity
% at both GC and HB.

% Required input files in the current working directory:
%   1) GC_phenology.xlsx
%      Columns: y, SD, YSD, MT, YMT
%      y   = year
%      SD  = sowing date (DOY)
%      YSD = cumulative LNPP at sowing
%      MT  = observed maturity date (DOY)
%      YMT = cumulative LNPP at maturity
%
%   2) GC_daily_CLNPP.xlsx
%      Columns: year, rixu, lnpp
%
% ========================================================================

clc; clear; close all;

baseDir =  pwd;
phenoFile = fullfile(baseDir, 'GC_phenology.xlsx');
clnppFile = fullfile(baseDir, 'GC_daily_CLNPP.xlsx');

%% 1. Read phenology observations
pheno = readtable(phenoFile);

Year = pheno.y;
Ds = pheno.SD;
CLNPPs = pheno.YSD;
Dobs = pheno.MT;
CLNPPobs = pheno.YMT;

%% 2. Estimate a and b from the basic LNPP-phenology relationship
valid_ab = isfinite(Dobs) & isfinite(CLNPPobs);
if sum(valid_ab) < 2
    error('Insufficient valid observations for estimating a and b.');
end

coef_ab = polyfit(Dobs(valid_ab), CLNPPobs(valid_ab), 1);
a = coef_ab(1);
b = coef_ab(2);

%% 3. Estimate c and d from the sowing-referenced relationship
X_rel = (Dobs - Ds) ./ Ds;
Y_rel = (CLNPPobs - CLNPPs) ./ CLNPPs;

valid_cd = isfinite(X_rel) & isfinite(Y_rel) & (Ds > 0) & (CLNPPs > 0);
if sum(valid_cd) < 2
    error('Insufficient valid observations for estimating c and d.');
end

coef_cd = polyfit(X_rel(valid_cd), Y_rel(valid_cd), 1);
c = coef_cd(1);
d = coef_cd(2);

%% 4. Fit the multi-year climatic baseline of cumulative LNPP
clnppData = readtable(clnppFile);

DOY = clnppData.rixu;
CLNPP = clnppData.lnpp;  % cumulative LNPP

valid_base = isfinite(DOY) & isfinite(CLNPP) & (DOY >= 1);
DOY_base = DOY(valid_base);
CLNPP_base_input = CLNPP(valid_base);


[G_doy, meanDOY] = findgroups(DOY_base);
meanCLNPP = splitapply(@(x) mean(x, 'omitnan'), CLNPP_base_input, G_doy);

valid_mean = isfinite(meanDOY) & isfinite(meanCLNPP);
meanDOY = meanDOY(valid_mean);
meanCLNPP = meanCLNPP(valid_mean);

logistic = @(p, x) p(1) ./ (1 + exp(-p(2) .* (x - p(3))));

L0 = max(meanCLNPP) * 1.05;
if ~isfinite(L0) || L0 <= 0
    error('The baseline CLNPP series does not contain valid positive values.');
end

[~, idx_half] = min(abs(meanCLNPP - 0.5 * L0));
d0_0 = meanDOY(idx_half);
k0 = 0.03;

p0 = [L0, k0, d0_0];
lb = [0, 0, 1];
ub = [Inf, 1, max(meanDOY)];

options = optimset('Display', 'off', 'MaxIter', 2000, 'TolFun', 1e-8);
[p_base, ~, ~, exitflag] = lsqcurvefit( ...
    logistic, p0, meanDOY, meanCLNPP, lb, ub, options);

if exitflag <= 0
    warning('The logistic baseline fit did not converge.');
end

L = p_base(1);
k = p_base(2);
d0 = p_base(3);

%% 5. Calculate the initial phenological date using calibrated a-d
denominator = a .* Ds - c .* CLNPPs;
D_initial = NaN(size(Ds));
valid_initial = isfinite(denominator) & (abs(denominator) > 1e-10) ...
    & isfinite(Ds) & isfinite(CLNPPs);

D_initial(valid_initial) = Ds(valid_initial) .* ...
    (CLNPPs(valid_initial) .* (1 + d - c) - b) ./ ...
    denominator(valid_initial);

%% 6. Estimate e and f from calibration residuals and lambda at sowing
CLNPP_base_s = logistic(p_base, Ds);
lambda_s = CLNPPs ./ CLNPP_base_s;


delta_obs = Dobs - D_initial;

valid_ef = isfinite(delta_obs) & isfinite(lambda_s) & (CLNPP_base_s > 0);
if sum(valid_ef) < 2
    error('Insufficient valid observations for estimating e and f.');
end

coef_ef = polyfit(lambda_s(valid_ef), delta_obs(valid_ef), 1);
e = coef_ef(1);
f = coef_ef(2);

%% 7. Final corrected phenological date
delta_pred = e .* lambda_s + f;
D_final = D_initial + delta_pred;

%% 8. Display and save results
fprintf('\nCalibrated parameters for the example stage/site:\n');
fprintf('a  = %.6f\n', a);
fprintf('b  = %.6f\n', b);
fprintf('c  = %.6f\n', c);
fprintf('d  = %.6f\n', d);
fprintf('e  = %.6f\n', e);
fprintf('f  = %.6f\n', f);
fprintf('L  = %.6f\n', L);
fprintf('k  = %.6f\n', k);
fprintf('d0 = %.6f\n', d0);

parameterTable = table( ...
    {'a';'b';'c';'d';'e';'f';'L';'k';'d0'}, ...
    [a;b;c;d;e;f;L;k;d0], ...
    'VariableNames', {'Parameter','Value'});

fitTable = table(Year, Ds, CLNPPs, Dobs, CLNPPobs, ...
    D_initial, lambda_s, delta_obs, delta_pred, D_final);

writetable(parameterTable, fullfile(baseDir, 'GC_maturity_parameters_example.xlsx'));
writetable(fitTable, fullfile(baseDir, 'GC_maturity_calibration_example.xlsx'));

fprintf('\nCalibration completed.\n');
