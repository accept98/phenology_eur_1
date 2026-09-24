# -*- coding: utf-8 -*-
"""
Regional temporal trend analysis.

Input:
    yearlydata_example.xlsx

Required columns:
    Station
    Year
    EarliestSowing
    LatestSowing
    SowingRange

Method:
    1. Calculate the annual regional mean across stations.
    2. Fit a linear trend to the annual regional mean.
    3. Use Newey-West HAC standard errors for inference.
    4. Report slope, 95% confidence interval, and HAC-adjusted p-value.

No figures are generated.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================================================
# 1. Paths and settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "yearlydata_example.xlsx"
OUTPUT_FILE = BASE_DIR / "regional_trend_results.xlsx"

STATION_COL = "Station"
YEAR_COL = "Year"

TARGET_COLUMNS = [
    "EarliestSowing",
    "LatestSowing",
    "SowingRange",
]


# ============================================================
# 2. Functions
# ============================================================

def select_hac_lag(n):
    """
    Automatic Newey-West maximum lag.

    For approximately 45 annual observations, this usually
    gives a lag of about 3 years.
    """

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


def fit_hac_trend(years, values):
    """
    Linear regression with Newey-West HAC standard errors.

    Model:
        Y = intercept + slope * Year + error

    Year is centered for numerical stability.
    """

    years = np.asarray(years, dtype=float)
    values = np.asarray(values, dtype=float)

    valid = (
        np.isfinite(years)
        & np.isfinite(values)
    )

    years = years[valid]
    values = values[valid]

    if len(years) < 4:
        raise ValueError(
            "At least four valid annual observations are required."
        )

    year_centered = years - np.mean(years)

    X = sm.add_constant(
        year_centered,
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
            "maxlags": max_lag,
            "use_correction": True,
        }
    )

    slope_year = float(
        model.params[1]
    )

    ci = np.asarray(
        model.conf_int(alpha=0.05)
    )

    slope_ci_year = ci[1, :]

    p_value = float(
        model.pvalues[1]
    )

    return {
        "Start_year":
            int(np.min(years)),

        "End_year":
            int(np.max(years)),

        "Number_of_years":
            int(len(years)),

        "Trend_per_year":
            slope_year,

        "Trend_per_decade":
            slope_year * 10.0,

        "CI95_lower_per_decade":
            slope_ci_year[0] * 10.0,

        "CI95_upper_per_decade":
            slope_ci_year[1] * 10.0,

        "HAC_adjusted_p":
            p_value,

        "HAC_max_lag":
            max_lag,
    }


# ============================================================
# 3. Read and check data
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )

data = pd.read_excel(
    INPUT_FILE
)

required_columns = [
    STATION_COL,
    YEAR_COL,
] + TARGET_COLUMNS

missing_columns = [
    col
    for col in required_columns
    if col not in data.columns
]

if missing_columns:
    raise KeyError(
        f"Missing columns: {missing_columns}\n"
        f"Existing columns: {list(data.columns)}"
    )


for col in [YEAR_COL] + TARGET_COLUMNS:

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
# 4. Check duplicate station-year records
# ============================================================

duplicate_mask = data.duplicated(
    subset=[
        STATION_COL,
        YEAR_COL,
    ],
    keep=False
)

if duplicate_mask.any():

    duplicate_rows = data.loc[
        duplicate_mask,
        [STATION_COL, YEAR_COL]
    ]

    raise ValueError(
        "Duplicate Station-Year records were found.\n"
        f"{duplicate_rows.head(20)}"
    )


# ============================================================
# 5. Regional trends
# ============================================================

trend_records = []
annual_tables = {}

for variable in TARGET_COLUMNS:

    annual = (
        data
        .groupby(YEAR_COL, as_index=False)[variable]
        .agg(
            ["mean", "std", "count"]
        )
        .reset_index()
    )

    annual = annual.rename(
        columns={
            "mean": "Regional_mean",
            "std": "Regional_SD",
            "count": "Number_of_stations",
        }
    )

    annual = annual.dropna(
        subset=["Regional_mean"]
    )

    annual_tables[variable] = annual

    result = fit_hac_trend(
        annual[YEAR_COL],
        annual["Regional_mean"],
    )

    result["Variable"] = variable

    trend_records.append(
        result
    )


regional_trends = pd.DataFrame(
    trend_records
)

column_order = [
    "Variable",
    "Start_year",
    "End_year",
    "Number_of_years",
    "Trend_per_year",
    "Trend_per_decade",
    "CI95_lower_per_decade",
    "CI95_upper_per_decade",
    "HAC_adjusted_p",
    "HAC_max_lag",
]

regional_trends = regional_trends[
    column_order
]


# ============================================================
# 6. Save results
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    regional_trends.to_excel(
        writer,
        sheet_name="Regional_trends",
        index=False
    )

    for variable in TARGET_COLUMNS:

        annual_tables[variable].to_excel(
            writer,
            sheet_name=f"Annual_{variable}",
            index=False
        )


print("\nRegional trend analysis completed.")
print(f"Results saved to:\n{OUTPUT_FILE}")

print("\nRegional trends:")
print(regional_trends.to_string(index=False))