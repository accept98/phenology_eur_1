% ========================================================================
% simulate_sowing_window_example.m
% Example regional sowing-window simulation using calibrated maturity-stage
% parameters from the nearest calibration site (GC or HB).
%
% Required input file in the current working directory:
%   regional_CLNPP_example.xlsx
%   Columns: lon, lat, year, rixu, lnpp
%   NOTE: the historical column name "lnpp" must contain DAILY CUMULATIVE
%   LNPP (CLNPP), not daily LNPP.
% ========================================================================

clc; clear; close all; 

baseDir = pwd;
inputFile = fullfile(baseDir, 'regional_CLNPP_example.xlsx');
outputDailyFile = fullfile(baseDir, 'regional_CLNPP_simulated_example.xlsx');
outputWindowFile = fullfile(baseDir, 'regional_sowing_window_example.xlsx');


T_limit = 300;  % common latest allowable maturity date (DOY)

%% 1. Calibration-site coordinates
refName = {'GC', 'HB'};
refLat = [39.1333, 35.6667];
refLon = [115.6667, 114.5167];

%% 2. Calibrated maturity-stage parameters


params = struct();

params(1).a = 212.33;
params(1).b = -8316.9;
params(1).c = 6.0152;
params(1).d = -1.9212;
params(1).e = 1.3946;
params(1).f = -1.5265;

params(2).a = 114.95;
params(2).b = 5808.80;
params(2).c = 5.5214;
params(2).d = -2.3613;
params(2).e = 107.34;
params(2).f = -107.83;

%% 3. Read regional cumulative-LNPP data
T = readtable(inputFile);
T.Properties.VariableNames = strtrim(T.Properties.VariableNames);

requiredVars = {'lon','lat','year','rixu','lnpp'};
if ~all(ismember(requiredVars, T.Properties.VariableNames))
    error('Input file must contain lon, lat, year, rixu, and lnpp.');
end

lon = T.lon;
lat = T.lat;
year = T.year;
Ds = T.rixu;
CLNPPs = T.lnpp;  % cumulative LNPP at each candidate sowing date

%% 4. Identify actual regional stations
[stationID, stationLon, stationLat] = findgroups(lon, lat);
nStations = length(stationLon);

%% 5. Assign each station to the nearest calibration site
stationDistance = NaN(nStations, 2);
for j = 1:2
    stationDistance(:,j) = haversineKm( ...
        stationLat, stationLon, refLat(j), refLon(j));
end

[minDistanceKm, stationParamIndex] = min(stationDistance, [], 2);
assignedCalibrationSite = reshape(refName(stationParamIndex), [], 1);
rowParamIndex = stationParamIndex(stationID);

% Expand parameter values to all daily candidate rows
a = NaN(height(T),1);
b = NaN(height(T),1);
c = NaN(height(T),1);
d = NaN(height(T),1);
e = NaN(height(T),1);
f = NaN(height(T),1);

for j = 1:2
    mask = (rowParamIndex == j);
    a(mask) = params(j).a;
    b(mask) = params(j).b;
    c(mask) = params(j).c;
    d(mask) = params(j).d;
    e(mask) = params(j).e;
    f(mask) = params(j).f;
end

%% 6. Fit a station-specific multi-year CLNPP baseline
% Eq. (13): CLNPP_base(d) = L / (1 + exp(-k*(d-d0)))
CLNPPbase = NaN(height(T),1);

for s = 1:nStations
    idxStation = (stationID == s);
    doyStation = Ds(idxStation);
    clnppStation = CLNPPs(idxStation);
    yearStation = year(idxStation);

    [pBase, ok] = fitCLNPPBaseline(doyStation, clnppStation);

if ~ok
    error('CLNPP baseline fitting failed for station %d.', s);
end

    logistic = @(p, x) p(1) ./ (1 + exp(-p(2) .* (x - p(3))));
    CLNPPbase(idxStation) = logistic(pBase, doyStation);
end

%% 7. Calculate initial and climate-corrected maturity dates
denominator = a .* Ds - c .* CLNPPs;
D_initial = NaN(height(T),1);
validDenominator = isfinite(denominator) & (abs(denominator) > 1e-10);

D_initial(validDenominator) = Ds(validDenominator) .* ...
    (CLNPPs(validDenominator) .* ...
    (1 + d(validDenominator) - c(validDenominator)) ...
    - b(validDenominator)) ./ denominator(validDenominator);

lambda_s = CLNPPs ./ CLNPPbase;
lambda_s(~isfinite(lambda_s) | (CLNPPbase <= 0)) = NaN;

delta_pred = e .* lambda_s + f;
D_final = D_initial + delta_pred;

%% 8. Determine annual sowing-window boundaries for each actual station

sowingCondition = false(height(T),1);
validForCondition = isfinite(CLNPPs) & isfinite(Ds) & (Ds > 0) & ...
    isfinite(a) & isfinite(c) & (abs(c) > 1e-12);
sowingCondition(validForCondition) = ...
    (CLNPPs(validForCondition) ./ Ds(validForCondition)) > ...
    (a(validForCondition) ./ c(validForCondition));


validCandidate = sowingCondition & validDenominator & ...
    isfinite(D_final) & ...
    (D_final >= 1) & ...
    (D_final > Ds) & ...
    (D_final <= T_limit) & ...
    (Ds <= T_limit);

[groupID, groupStationID, groupYear] = findgroups(stationID, year);
nGroups = length(groupYear);

EarliestSowing = NaN(nGroups,1);
LatestSowing = NaN(nGroups,1);
SowingWindowDuration = NaN(nGroups,1);
EarliestMaturityFinal = NaN(nGroups,1);
LatestMaturityFinal = NaN(nGroups,1);

for g = 1:nGroups
    idxGroup = (groupID == g);
    groupRows = find(idxGroup);
    rowsValid = groupRows(validCandidate(groupRows));

    % Earliest sowing date: first valid candidate (forward search).
    if ~isempty(rowsValid)
        [~, posEarly] = min(Ds(rowsValid));
        rEarly = rowsValid(posEarly);
        EarliestSowing(g) = Ds(rEarly);
        EarliestMaturityFinal(g) = D_final(rEarly);

        % Latest sowing date: last valid candidate (backward search).
        [~, posLate] = max(Ds(rowsValid));
        rLate = rowsValid(posLate);
        LatestSowing(g) = Ds(rLate);
        LatestMaturityFinal(g) = D_final(rLate);
    end

    if isfinite(EarliestSowing(g)) && isfinite(LatestSowing(g))
        SowingWindowDuration(g) = ...
            LatestSowing(g) - EarliestSowing(g) + 1;
    end
end

%% 9. Build station-year output table
GroupLon = stationLon(groupStationID);
GroupLat = stationLat(groupStationID);
GroupAssignedSite = assignedCalibrationSite(groupStationID);
windowTable = table( ...
    groupStationID, GroupLon, GroupLat, groupYear, ...
    GroupAssignedSite, ...
    EarliestSowing, EarliestMaturityFinal, ...
    LatestSowing, LatestMaturityFinal, SowingWindowDuration, ...
    'VariableNames', { ...
    'StationID','Lon','Lat','Year','AssignedCalibrationSite', ...
    'EarliestSowing','EarliestMaturityFinal', ...
    'LatestSowing','LatestMaturityFinal', ...
    'SowingWindowDuration'});

%% 10. Save daily simulation variables and annual sowing-window results
rowAssignedSite = assignedCalibrationSite(stationID);

T.StationID = stationID;
T.AssignedCalibrationSite = rowAssignedSite;
T.CLNPPbase = CLNPPbase;
T.lambda_s = lambda_s;
T.D_initial = D_initial;
T.delta_pred = delta_pred;
T.D_final = D_final;

writetable(T, outputDailyFile);
writetable(windowTable, outputWindowFile);

fprintf('\nRegional sowing-window simulation completed.\n');
fprintf('Daily simulation output: %s\n', outputDailyFile);
fprintf('Annual sowing-window output: %s\n', outputWindowFile);

%% Local functions
function distanceKm = haversineKm(lat1, lon1, lat2, lon2)
    R = 6371.0088;  
    phi1 = deg2rad(lat1);
    phi2 = deg2rad(lat2);
    dphi = deg2rad(lat2 - lat1);
    dlambda = deg2rad(lon2 - lon1);

    h = sin(dphi ./ 2).^2 + ...
        cos(phi1) .* cos(phi2) .* sin(dlambda ./ 2).^2;
    h = min(max(h, 0), 1);
    distanceKm = 2 .* R .* asin(sqrt(h));
end

function [pBase, ok] = fitCLNPPBaseline(DOY, CLNPP)
    ok = false;
    pBase = [NaN, NaN, NaN];

    valid = isfinite(DOY) & isfinite(CLNPP) & (DOY >= 1);
    DOY = DOY(valid);
    CLNPP = CLNPP(valid);

    if numel(DOY) < 10
        return;
    end

    [G, meanDOY] = findgroups(DOY);
    meanCLNPP = splitapply(@(x) mean(x, 'omitnan'), CLNPP, G);

    validMean = isfinite(meanDOY) & isfinite(meanCLNPP);
    meanDOY = meanDOY(validMean);
    meanCLNPP = meanCLNPP(validMean);

    if numel(meanDOY) < 10 || max(meanCLNPP) <= 0
        return;
    end

    logistic = @(p, x) p(1) ./ (1 + exp(-p(2) .* (x - p(3))));

    L0 = max(meanCLNPP) * 1.05;
    [~, idxHalf] = min(abs(meanCLNPP - 0.5 * L0));
    d0_0 = meanDOY(idxHalf);
    p0 = [L0, 0.03, d0_0];

    lb = [0, 0, 1];
    ub = [Inf, 1, max(meanDOY)];
    options = optimset('Display', 'off', 'MaxIter', 2000, 'TolFun', 1e-8);

    try
        [pBase, ~, ~, exitflag] = lsqcurvefit( ...
            logistic, p0, meanDOY, meanCLNPP, lb, ub, options);
        ok = (exitflag > 0) && all(isfinite(pBase));
    catch
        ok = false;
    end
end
