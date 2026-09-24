# -*- coding: utf-8 -*-
"""
Station-level temporal trend analysis.

Input:
    yearlydata_example.xlsx

Required columns:
    Station
    Lon
    Lat
    Year

    EarliestSowing
    LatestSowing
    SowingRange

    Tmean_yr
    Tmax_yr
    Tmin_yr
    Sun_yr
    Prec_yr
    Rad_yr
    RH_yr

Methods:
    - OLS temporal trend
    - Newey-West HAC standard errors
    - BH-FDR correction for station-level sowing-metric trends
    - Pearson correlations with latitude and longitude
    - Trend-surface regression using latitude and longitude

No figures are generated.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests


# ============================================================
# 1. Paths and settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "yearlydata_example.xlsx"

OUTPUT_FILE = (
    BASE_DIR
    / "station_trend_results.xlsx"
)

STATION_COL = "Station"
YEAR_COL = "Year"

LON_COL = "Lon"
LAT_COL = "Lat"

MINIMUM_YEARS = 10
FDR_ALPHA = 0.05


SOWING_VARIABLES = [
    "EarliestSowing",
    "LatestSowing",
    "SowingRange",
]

METEOROLOGICAL_VARIABLES = [
    "Tmean_yr",
    "Tmax_yr",
    "Tmin_yr",
    "Sun_yr",
    "Prec_yr",
    "Rad_yr",
    "RH_yr",
]

TREND_VARIABLES = (
    SOWING_VARIABLES
    + METEOROLOGICAL_VARIABLES
)


# ============================================================
# 2. Functions
# ============================================================

def select_hac_lag(n):

    if n < 5:
        return 1

    lag = int(
        np.floor(
            4 * (n / 100.0) ** (2 / 9)
        )
    )

    return max(
        1,
        min(lag, n - 2)
    )


def fit_station_trend(
    years,
    values,
    minimum_n=10
):

    years = pd.to_numeric(
        pd.Series(years),
        errors="coerce"
    ).to_numpy(dtype=float)

    values = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).to_numpy(dtype=float)

    valid = (
        np.isfinite(years)
        & np.isfinite(values)
    )

    years = years[valid]
    values = values[valid]

    if len(years) < minimum_n:
        return None

    order = np.argsort(
        years
    )

    years = years[order]
    values = values[order]

    if len(np.unique(years)) != len(years):
        raise ValueError(
            "Duplicate years were found within a station."
        )

    if np.isclose(
        np.var(values),
        0
    ):

        return {
            "N_years":
                len(years),

            "Start_year":
                int(np.min(years)),

            "End_year":
                int(np.max(years)),

            "Trend_per_year":
                0.0,

            "Trend_per_decade":
                0.0,

            "CI95_lower_per_decade":
                0.0,

            "CI95_upper_per_decade":
                0.0,

            "HAC_p":
                1.0,

            "HAC_max_lag":
                0,
        }

    x_centered = (
        years
        - np.mean(years)
    )

    X = sm.add_constant(
        x_centered,
        has_constant="add"
    )

    max_lag = select_hac_lag(
        len(years)
    )

    model = sm.OLS(
        values,
        X
    ).fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags":
                max_lag,

            "use_correction":
                True,
        }
    )

    slope_year = float(
        model.params[1]
    )

    ci = np.asarray(
        model.conf_int(
            alpha=0.05
        )
    )

    slope_ci = ci[1, :]

    return {
        "N_years":
            int(len(years)),

        "Start_year":
            int(np.min(years)),

        "End_year":
            int(np.max(years)),

        "Trend_per_year":
            slope_year,

        "Trend_per_decade":
            slope_year * 10.0,

        "CI95_lower_per_decade":
            slope_ci[0] * 10.0,

        "CI95_upper_per_decade":
            slope_ci[1] * 10.0,

        "HAC_p":
            float(model.pvalues[1]),

        "HAC_max_lag":
            max_lag,
    }


def safe_pearson(x, y):

    x = np.asarray(
        x,
        dtype=float
    )

    y = np.asarray(
        y,
        dtype=float
    )

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if (
        len(x) < 3
        or np.std(x) == 0
        or np.std(y) == 0
    ):
        return np.nan, np.nan

    r, p = pearsonr(
        x,
        y
    )

    return float(r), float(p)


def trend_surface(
    data,
    response,
    lon_col="Lon",
    lat_col="Lat"
):
    """
    Trend surface:
        Y = beta0
            + beta_lat * latitude
            + beta_lon * longitude
            + error
    """

    subset = (
        data[
            [
                response,
                lon_col,
                lat_col,
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .copy()
    )

    if len(subset) < 5:
        return None

    X = subset[
        [
            lat_col,
            lon_col,
        ]
    ]

    X = sm.add_constant(
        X,
        has_constant="add"
    )

    model = sm.OLS(
        subset[response],
        X
    ).fit()

    return {
        "Variable":
            response,

        "N_stations":
            int(model.nobs),

        "Beta_lat":
            float(
                model.params[
                    lat_col
                ]
            ),

        "P_lat":
            float(
                model.pvalues[
                    lat_col
                ]
            ),

        "Beta_lon":
            float(
                model.params[
                    lon_col
                ]
            ),

        "P_lon":
            float(
                model.pvalues[
                    lon_col
                ]
            ),

        "R_squared":
            float(
                model.rsquared
            ),

        "Model_p":
            float(
                model.f_pvalue
            ),
    }


# ============================================================
# 3. Read data
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Input file not found:\n"
        f"{INPUT_FILE}"
    )


data = pd.read_excel(
    INPUT_FILE
)


required_columns = [
    STATION_COL,
    YEAR_COL,
    LON_COL,
    LAT_COL,
] + TREND_VARIABLES


missing_columns = [
    col
    for col in required_columns
    if col not in data.columns
]


if missing_columns:

    raise KeyError(
        f"Missing columns: "
        f"{missing_columns}\n"
        f"Existing columns: "
        f"{list(data.columns)}"
    )


numeric_columns = [
    YEAR_COL,
    LON_COL,
    LAT_COL,
] + TREND_VARIABLES


for col in numeric_columns:

    data[col] = pd.to_numeric(
        data[col],
        errors="coerce"
    )


data = data.dropna(
    subset=[
        STATION_COL,
        YEAR_COL,
    ]
).copy()


# ============================================================
# 4. Duplicate check
# ============================================================

duplicate_mask = data.duplicated(
    subset=[
        STATION_COL,
        YEAR_COL,
    ],
    keep=False
)


if duplicate_mask.any():

    duplicates = data.loc[
        duplicate_mask,
        [
            STATION_COL,
            YEAR_COL,
        ]
    ]

    raise ValueError(
        "Duplicate Station-Year records "
        "were found.\n"
        f"{duplicates.head(20)}"
    )


data = data.sort_values(
    [
        STATION_COL,
        YEAR_COL,
    ]
)


# ============================================================
# 5. Station coordinates
# ============================================================

coordinate_check = (
    data
    .groupby(
        STATION_COL
    )
    .agg(
        Lon_unique=(
            LON_COL,
            "nunique"
        ),
        Lat_unique=(
            LAT_COL,
            "nunique"
        ),
    )
    .reset_index()
)


inconsistent_coordinates = (
    coordinate_check[
        (
            coordinate_check[
                "Lon_unique"
            ] > 1
        )
        |
        (
            coordinate_check[
                "Lat_unique"
            ] > 1
        )
    ]
)


if not inconsistent_coordinates.empty:

    warnings.warn(
        "Some stations contain "
        "slightly different coordinate "
        "records. Median coordinates "
        "will be used."
    )


station_coordinates = (
    data
    .groupby(
        STATION_COL,
        as_index=False
    )
    .agg(
        {
            LON_COL:
                "median",

            LAT_COL:
                "median",
        }
    )
)


# ============================================================
# 6. Station-level trends
# ============================================================

records = []


for variable in TREND_VARIABLES:

    for station, station_data in data.groupby(
        STATION_COL
    ):

        result = fit_station_trend(
            station_data[
                YEAR_COL
            ],
            station_data[
                variable
            ],
            minimum_n=MINIMUM_YEARS,
        )

        if result is None:
            continue

        row = {
            "Station":
                station,

            "Variable":
                variable,
        }

        row.update(
            result
        )

        records.append(
            row
        )


trend_results = pd.DataFrame(
    records
)


trend_results = trend_results.merge(
    station_coordinates,
    on=STATION_COL,
    how="left",
    validate="many_to_one",
)


# ============================================================
# 7. BH-FDR for sowing trends
# ============================================================

trend_results["FDR_q"] = np.nan

trend_results[
    "Significant_FDR_q_lt_0_05"
] = False


for variable in SOWING_VARIABLES:

    mask = (
        (
            trend_results[
                "Variable"
            ] == variable
        )
        &
        trend_results[
            "HAC_p"
        ].notna()
    )

    p_values = (
        trend_results
        .loc[
            mask,
            "HAC_p"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if len(p_values) == 0:
        continue

    rejected, q_values, _, _ = (
        multipletests(
            p_values,
            alpha=FDR_ALPHA,
            method="fdr_bh",
        )
    )

    trend_results.loc[
        mask,
        "FDR_q"
    ] = q_values

    trend_results.loc[
        mask,
        "Significant_FDR_q_lt_0_05"
    ] = rejected


# ============================================================
# 8. Build one-row-per-station trend table
# ============================================================

trend_wide = (
    trend_results
    .pivot(
        index="Station",
        columns="Variable",
        values="Trend_per_decade",
    )
    .reset_index()
)


rename_columns = {
    "EarliestSowing":
        "Trend_EarliestSowing_10yr",

    "LatestSowing":
        "Trend_LatestSowing_10yr",

    "SowingRange":
        "Trend_SowingRange_10yr",

    "Tmean_yr":
        "Tmean_yr",

    "Tmax_yr":
        "Tmax_yr",

    "Tmin_yr":
        "Tmin_yr",

    "Sun_yr":
        "Sun_yr",

    "Prec_yr":
        "Prec_yr",

    "Rad_yr":
        "Rad_yr",

    "RH_yr":
        "RH_yr",
}


trend_wide = trend_wide.rename(
    columns=rename_columns
)


sem_input = station_coordinates.merge(
    trend_wide,
    on="Station",
    how="left",
    validate="one_to_one",
)


# ============================================================
# 9. Multiyear station means of sowing metrics
# ============================================================

station_means = (
    data
    .groupby(
        STATION_COL,
        as_index=False
    )[
        SOWING_VARIABLES
    ]
    .mean()
)


station_means = station_coordinates.merge(
    station_means,
    on=STATION_COL,
    how="left",
    validate="one_to_one",
)


# ============================================================
# 10. Spatial Pearson correlations
# ============================================================

spatial_correlation_records = []


spatial_variables = []


for variable in SOWING_VARIABLES:

    spatial_variables.append(
        (
            f"Mean_{variable}",
            station_means[
                variable
            ]
        )
    )


for variable in SOWING_VARIABLES:

    trend_column = (
        "Trend_"
        f"{variable}_10yr"
    )

    spatial_variables.append(
        (
            trend_column,
            sem_input[
                trend_column
            ],
        )
    )


for name, values in spatial_variables:

    if name.startswith(
        "Mean_"
    ):

        base = station_means

    else:

        base = sem_input

    r_lat, p_lat = safe_pearson(
        values,
        base[LAT_COL],
    )

    r_lon, p_lon = safe_pearson(
        values,
        base[LON_COL],
    )

    spatial_correlation_records.append(
        {
            "Variable":
                name,

            "r_lat":
                r_lat,

            "p_lat":
                p_lat,

            "r_lon":
                r_lon,

            "p_lon":
                p_lon,
        }
    )


spatial_correlations = pd.DataFrame(
    spatial_correlation_records
)


# ============================================================
# 11. Trend-surface regression
# ============================================================

trend_surface_records = []


mean_surface_data = (
    station_means
    .rename(
        columns={
            variable:
                f"Mean_{variable}"
            for variable
            in SOWING_VARIABLES
        }
    )
)


for variable in SOWING_VARIABLES:

    response = (
        f"Mean_{variable}"
    )

    result = trend_surface(
        mean_surface_data,
        response
    )

    if result is not None:

        trend_surface_records.append(
            result
        )


for variable in SOWING_VARIABLES:

    response = (
        f"Trend_{variable}_10yr"
    )

    result = trend_surface(
        sem_input,
        response
    )

    if result is not None:

        trend_surface_records.append(
            result
        )


trend_surface_results = pd.DataFrame(
    trend_surface_records
)


# ============================================================
# 12. Sowing-trend significance summary
# ============================================================

summary_records = []


for variable in SOWING_VARIABLES:

    subset = trend_results[
        trend_results[
            "Variable"
        ] == variable
    ].copy()

    n_total = len(
        subset
    )

    significant = subset[
        "Significant_FDR_q_lt_0_05"
    ]

    n_sig = int(
        significant.sum()
    )

    n_sig_positive = int(
        (
            significant
            &
            (
                subset[
                    "Trend_per_decade"
                ] > 0
            )
        ).sum()
    )

    n_sig_negative = int(
        (
            significant
            &
            (
                subset[
                    "Trend_per_decade"
                ] < 0
            )
        ).sum()
    )

    summary_records.append(
        {
            "Variable":
                variable,

            "Number_of_stations":
                n_total,

            "Significant_stations":
                n_sig,

            "Significant_percentage":
                (
                    n_sig
                    / n_total
                    * 100
                    if n_total > 0
                    else np.nan
                ),

            "Significant_positive":
                n_sig_positive,

            "Significant_negative":
                n_sig_negative,
        }
    )


sowing_trend_summary = pd.DataFrame(
    summary_records
)


# ============================================================
# 13. Save
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    trend_results.to_excel(
        writer,
        sheet_name="Station_trends",
        index=False
    )

    sowing_trend_summary.to_excel(
        writer,
        sheet_name="Sowing_summary",
        index=False
    )

    sem_input.to_excel(
        writer,
        sheet_name="SEM_input",
        index=False
    )

    station_means.to_excel(
        writer,
        sheet_name="Station_means",
        index=False
    )

    spatial_correlations.to_excel(
        writer,
        sheet_name="Spatial_correlations",
        index=False
    )

    trend_surface_results.to_excel(
        writer,
        sheet_name="Trend_surface",
        index=False
    )

    inconsistent_coordinates.to_excel(
        writer,
        sheet_name="Coordinate_check",
        index=False
    )


print(
    "\nStation-level trend analysis completed."
)

print(
    f"Results saved to:\n"
    f"{OUTPUT_FILE}"
)