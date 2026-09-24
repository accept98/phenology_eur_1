Supplementary Code

Reduced example datasets are provided to demonstrate the structure of the inputs.

1. Software requirements

MATLAB R2018b with Optimization Toolbox was used for LNPP calculation, model calibration, and phenological
simulation.

Python 3.10 was used for statistical analyses. Required Python packages include:
numpy, pandas, scipy, statsmodels, scikit-learn, openpyxl, libpysal, esda, and spreg.
Package versions are provided in requirements.txt.

2. Main scripts

calculate_lnpp_clnpp_example.m
    Performs meteorological-data quality checks and preprocessing, estimates solar
    radiation from sunshine duration, and calculates daily LNPP and cumulative LNPP.

calibrate_parameters_example.m
    Estimates the phenology-model parameters from the sowing-date experiment data.

simulate_sowing_window_example.m
    Applies the calibrated parameters to regional meteorological data and calculates
    simulated phenological dates and modelled sowing metrics. Regional stations are
    assigned the parameter set of the geographically nearest calibration site using
    Haversine great-circle distance.

regional_trend_analysis.py
    Calculates annual regional means and long-term temporal trends using linear
    regression with Newey-West heteroskedasticity- and autocorrelation-consistent
    standard errors.

station_trend_analysis.py
    Calculates station-level temporal trends, multiple-testing-adjusted significance,
    and spatial gradients in the modelled sowing metrics.

spatial_error_models.py
    Fits the spatial error models used to examine associations between climatic trends
    and trends in the modelled sowing metrics. Alternative temperature metrics are
    evaluated in separate models.

3. Recommended workflow

The scripts correspond to the following sequence of the main analytical procedures. Each script can be run independently using its accompanying example input file:

1) calculate_lnpp_clnpp_example.m
2) calibrate_parameters_example.m
3) simulate_sowing_window_example.m
4) regional_trend_analysis.py
5) station_trend_analysis.py
6) spatial_error_models.py


4. Data processing and model implementation

Meteorological preprocessing includes date and duplicate checks, range and consistency
checks, and treatment of missing observations before LNPP calculation. Calendar dates
are handled using MATLAB datetime functions, including leap years.


5. Data availability

The complete meteorological and agrometeorological datasets used in the study were
obtained from the China Meteorological Administration (CMA) and are subject to its
data-access policies. They are therefore not redistributed in this package.

Reduced example datasets with the same input structure are provided for reproducibility
of the computational workflow. Access to the original CMA datasets should be requested
through the corresponding CMA data services.
