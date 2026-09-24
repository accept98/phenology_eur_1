# -*- coding: utf-8 -*-
"""
Spatial error models for associations between climatic trends
and modelled maize sowing-metric trends.

Input:
    station_trend_example.xlsx

Required columns:
    Station
    Lon
    Lat

    Trend_EarliestSowing_10yr
    Trend_LatestSowing_10yr
    Trend_SowingRange_10yr

    Tmean_yr
    Tmax_yr
    Tmin_yr
    Sun_yr
    Prec_yr
    Rad_yr
    RH_yr

Models:
    Primary:
        Tmean + Sun + Prec + RH

    Temperature-metric sensitivity:
        Tmax + Sun + Prec + RH
        Tmin + Sun + Prec + RH

Methods:
    - Pearson correlation and VIF diagnosis for the original
      seven climatic-trend variables
    - Standardization of response and predictor variables
    - Row-standardized 4-nearest-neighbour spatial weights
    - Maximum-likelihood spatial error models
    - Moran's I for spatially filtered residuals
    - BH-FDR correction within each response x model
      across the four predictor coefficients

No figures are generated.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor


try:
    from esda.moran import Moran
    from libpysal.weights import W
    from spreg import ML_Error

except ImportError as exc:
    raise ImportError(
        "Required spatial packages are missing.\n"
        "Please install: libpysal esda spreg"
    ) from exc


# ============================================================
# 1. Paths and settings
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = (
    BASE_DIR
    / "station_trend_example.xlsx"
)

OUTPUT_FILE = (
    BASE_DIR
    / "spatial_error_model_results.xlsx"
)


STATION_COL = "Station"
LON_COL = "Lon"
LAT_COL = "Lat"


RESPONSE_COLUMNS: Dict[str, str] = {

    "EarliestSowing":
        "Trend_EarliestSowing_10yr",

    "LatestSowing":
        "Trend_LatestSowing_10yr",

    "SowingRange":
        "Trend_SowingRange_10yr",
}


TEMPERATURE_PREDICTORS = [
    "Tmean_yr",
    "Tmax_yr",
    "Tmin_yr",
]


PRIMARY_SOLAR = "Sun_yr"


REDUCED_MODELS = {

    "Tmean_model": [
        "Tmean_yr",
        PRIMARY_SOLAR,
        "Prec_yr",
        "RH_yr",
    ],

    "Tmax_model": [
        "Tmax_yr",
        PRIMARY_SOLAR,
        "Prec_yr",
        "RH_yr",
    ],

    "Tmin_model": [
        "Tmin_yr",
        PRIMARY_SOLAR,
        "Prec_yr",
        "RH_yr",
    ],
}


PRIMARY_MODEL = "Tmean_model"


# Seven climatic-trend variables retained only
# for multicollinearity diagnosis.

ALL_SEVEN_PREDICTORS = [
    "Tmean_yr",
    "Tmax_yr",
    "Tmin_yr",
    "Rad_yr",
    "Sun_yr",
    "Prec_yr",
    "RH_yr",
]


# Variables used to define the common SEM sample.

COMMON_MODEL_PREDICTORS = [
    "Tmean_yr",
    "Tmax_yr",
    "Tmin_yr",
    "Sun_yr",
    "Prec_yr",
    "RH_yr",
]


PRIMARY_K = 4

MORAN_PERMUTATIONS = 999

RANDOM_SEED = 20260804

SPATIAL_METHOD = "LU"


# ============================================================
# 2. Utility functions
# ============================================================

def standardize_frame(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Column-wise z-score standardization.
    """

    scaler = StandardScaler()

    values = scaler.fit_transform(
        data
    )

    return pd.DataFrame(
        values,
        columns=data.columns,
        index=data.index,
    )


def calculate_vif(
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate variance inflation factors.
    """

    if data.shape[1] == 1:

        return pd.DataFrame(
            {
                "Predictor":
                    data.columns,

                "VIF":
                    [1.0],
            }
        )

    standardized = standardize_frame(
        data
    )

    design = sm.add_constant(
        standardized,
        has_constant="add",
    )

    records = []

    for i, predictor in enumerate(
        design.columns
    ):

        if predictor == "const":
            continue

        try:

            value = (
                variance_inflation_factor(
                    design.to_numpy(),
                    i,
                )
            )

        except Exception:

            value = np.inf

        records.append(
            {
                "Predictor":
                    predictor,

                "VIF":
                    float(value),
            }
        )

    return pd.DataFrame(
        records
    )


def build_knn_weights(
    lon: Sequence[float],
    lat: Sequence[float],
    k: int,
) -> W:
    """
    Construct row-standardized k-nearest-neighbour
    spatial weights using haversine distance.
    """

    lon = np.asarray(
        lon,
        dtype=float
    )

    lat = np.asarray(
        lat,
        dtype=float
    )

    n = len(lon)

    if n < k + 1:

        raise ValueError(
            f"At least {k + 1} stations are required "
            f"to construct {k}-nearest-neighbour "
            f"spatial weights; only {n} stations "
            f"are available."
        )

    coordinates = np.radians(
        np.column_stack(
            [
                lat,
                lon,
            ]
        )
    )

    nearest = NearestNeighbors(
        n_neighbors=k + 1,
        algorithm="ball_tree",
        metric="haversine",
    )

    nearest.fit(
        coordinates
    )

    neighbour_indices = (
        nearest.kneighbors(
            return_distance=False
        )
    )

    neighbours = {}
    weights = {}

    for row_index, row in enumerate(
        neighbour_indices
    ):

        selected = [
            int(index)
            for index in row
            if int(index) != row_index
        ][:k]

        if len(selected) != k:

            raise RuntimeError(
                f"Station index {row_index} "
                f"has fewer than {k} neighbours."
            )

        neighbours[
            row_index
        ] = selected

        weights[
            row_index
        ] = [1.0] * k

    spatial_weights = W(
        neighbours,
        weights,
        silence_warnings=True,
    )

    spatial_weights.transform = "r"

    return spatial_weights


def moran_test(
    values: np.ndarray,
    spatial_weights: W,
) -> Tuple[float, float]:
    """
    Moran's I permutation test.
    """

    values = np.asarray(
        values,
        dtype=float
    ).reshape(-1)

    if (
        len(values) != spatial_weights.n
        or np.std(values) == 0
    ):

        return np.nan, np.nan

    np.random.seed(
        RANDOM_SEED
    )

    result = Moran(
        values,
        spatial_weights,
        permutations=MORAN_PERMUTATIONS,
    )

    return (
        float(result.I),
        float(result.p_sim),
    )


# ============================================================
# 3. Spatial error model functions
# ============================================================

def fit_sem(
    data: pd.DataFrame,
    response_col: str,
    predictors: Sequence[str],
    spatial_weights: W,
    response_name: str,
    model_name: str,
) -> ML_Error:
    """
    Fit a maximum-likelihood spatial error model
    using standardized response and predictors.
    """

    X = standardize_frame(
        data[
            list(predictors)
        ]
    )

    y = (
        StandardScaler()
        .fit_transform(
            data[
                [response_col]
            ]
        )
        .reshape(-1, 1)
    )

    model = ML_Error(
        y,
        X.to_numpy(),
        spatial_weights,
        method=SPATIAL_METHOD,
        vm=True,
        name_y=response_name,
        name_x=list(predictors),
        name_w=model_name,
        name_ds=(
            "Station-level climatic "
            "and sowing trends"
        ),
    )

    return model


def model_term_names(
    model: ML_Error
) -> List[str]:
    """
    Obtain term names aligned with model coefficients.
    """

    names = list(
        getattr(
            model,
            "name_x",
            [],
        )
    )

    n_beta = len(
        np.asarray(
            model.betas
        ).reshape(-1)
    )

    if len(names) == n_beta:

        return names

    if n_beta >= 2:

        return (
            ["CONSTANT"]
            +
            [
                f"X_{i}"
                for i
                in range(
                    1,
                    n_beta - 1
                )
            ]
            +
            ["lambda"]
        )

    return [
        f"Term_{i}"
        for i
        in range(n_beta)
    ]


def extract_coefficients(
    model: ML_Error,
    response_name: str,
    model_name: str,
    predictors: Sequence[str],
) -> pd.DataFrame:
    """
    Extract standardized coefficients,
    standard errors, p-values and 95% CIs.
    """

    names = model_term_names(
        model
    )

    betas = np.asarray(
        model.betas
    ).reshape(-1)

    ses = np.asarray(
        model.std_err
    ).reshape(-1)

    z_stats = list(
        model.z_stat
    )

    n = min(
        len(names),
        len(betas),
        len(ses),
        len(z_stats),
    )

    records = []

    for i in range(n):

        term = names[i]

        if term not in predictors:
            continue

        beta = float(
            betas[i]
        )

        se = float(
            ses[i]
        )

        z_value = float(
            z_stats[i][0]
        )

        p_value = float(
            z_stats[i][1]
        )

        records.append(
            {
                "Response":
                    response_name,

                "Model":
                    model_name,

                "Term":
                    term,

                "Standardized_beta":
                    beta,

                "Standard_error":
                    se,

                "Z_value":
                    z_value,

                "P_value":
                    p_value,

                "CI95_lower":
                    beta - 1.96 * se,

                "CI95_upper":
                    beta + 1.96 * se,
            }
        )

    return pd.DataFrame(
        records
    )


def extract_model_summary(
    model: ML_Error,
    response_name: str,
    model_name: str,
    predictors: Sequence[str],
    spatial_weights: W,
    maximum_vif: float,
) -> Dict[str, object]:
    """
    Extract model-level statistics and
    filtered-residual Moran's I.
    """

    filtered_i, filtered_p = (
        moran_test(
            model.e_filtered,
            spatial_weights,
        )
    )

    return {

        "Response":
            response_name,

        "Model":
            model_name,

        "Temperature_metric":
            predictors[0],

        "Predictors":
            ", ".join(
                predictors
            ),

        "N_stations":
            int(
                model.n
            ),

        "Maximum_VIF":
            float(
                maximum_vif
            ),

        "Lambda":
            float(
                model.lam
            ),

        "Pseudo_R2":
            float(
                model.pr2
            ),

        "Log_likelihood":
            float(
                model.logll
            ),

        "AIC":
            float(
                model.aic
            ),

        "BIC":
            float(
                model.schwarz
            ),

        "Filtered_residual_Moran_I":
            filtered_i,

        "Filtered_residual_Moran_P":
            filtered_p,
    }


def add_bh_fdr(
    coefficients: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply BH-FDR separately within each
    response x reduced model.

    Each family contains four predictor tests.
    """

    output = coefficients.copy()

    output[
        "Q_value_BH"
    ] = np.nan

    output[
        "Significant_BH_q_lt_0_05"
    ] = False

    groups = output.groupby(
        [
            "Response",
            "Model",
        ]
    ).groups

    for _, indices in groups.items():

        indices = list(
            indices
        )

        p_values = (
            output
            .loc[
                indices,
                "P_value"
            ]
            .to_numpy(
                dtype=float
            )
        )

        valid = np.isfinite(
            p_values
        )

        if valid.sum() == 0:
            continue

        rejected, q_values, _, _ = (
            multipletests(
                p_values[valid],
                alpha=0.05,
                method="fdr_bh",
            )
        )

        valid_indices = (
            np.asarray(
                indices
            )[valid]
        )

        output.loc[
            valid_indices,
            "Q_value_BH"
        ] = q_values

        output.loc[
            valid_indices,
            "Significant_BH_q_lt_0_05"
        ] = rejected

    return output


# ============================================================
# 4. Read and check input data
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
    LON_COL,
    LAT_COL,
    *RESPONSE_COLUMNS.values(),
    *ALL_SEVEN_PREDICTORS,
]


required_columns = list(
    dict.fromkeys(
        required_columns
    )
)


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


for col in required_columns:

    if col != STATION_COL:

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )


# ============================================================
# 5. Check station records
# ============================================================

if data[
    STATION_COL
].duplicated().any():

    duplicated_stations = (
        data.loc[
            data[
                STATION_COL
            ].duplicated(
                keep=False
            ),
            STATION_COL
        ]
        .tolist()
    )

    raise ValueError(
        "The input must contain one row per station.\n"
        f"Duplicated stations: "
        f"{duplicated_stations[:20]}"
    )


# ============================================================
# 6. Analysis
# ============================================================

all_coefficients = []

all_model_summaries = []

all_reduced_vif = []

initial_vif_records = []

correlation_tables = {}


for response_name, response_col in (
    RESPONSE_COLUMNS.items()
):

    # --------------------------------------------------------
    # A. Seven-variable correlation and VIF diagnosis
    # --------------------------------------------------------

    diagnostic_columns = [
        STATION_COL,
        LON_COL,
        LAT_COL,
        response_col,
        *ALL_SEVEN_PREDICTORS,
    ]

    diagnostic_data = (
        data[
            diagnostic_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .copy()
        .sort_values(
            STATION_COL
        )
        .reset_index(
            drop=True
        )
    )

    if len(diagnostic_data) >= 3:

        correlation_tables[
            response_name
        ] = (
            diagnostic_data[
                ALL_SEVEN_PREDICTORS
            ]
            .corr(
                method="pearson"
            )
        )

        initial_vif = calculate_vif(
            diagnostic_data[
                ALL_SEVEN_PREDICTORS
            ]
        )

        initial_vif.insert(
            0,
            "Response",
            response_name,
        )

        initial_vif[
            "N_stations"
        ] = len(
            diagnostic_data
        )

        initial_vif_records.append(
            initial_vif
        )


    # --------------------------------------------------------
    # B. Common complete-case sample for Tmean/Tmax/Tmin SEMs
    # --------------------------------------------------------

    analysis_columns = [
        STATION_COL,
        LON_COL,
        LAT_COL,
        response_col,
        *COMMON_MODEL_PREDICTORS,
    ]

    response_data = (
        data[
            analysis_columns
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
        .dropna()
        .copy()
        .sort_values(
            STATION_COL
        )
        .reset_index(
            drop=True
        )
    )


    minimum_required = (
        PRIMARY_K + 1
    )


    if len(response_data) < minimum_required:

        raise ValueError(
            f"{response_name}: only "
            f"{len(response_data)} complete stations. "
            f"At least {minimum_required} stations "
            f"are required to construct "
            f"{PRIMARY_K}-nearest-neighbour "
            f"spatial weights."
        )


    print(
        f"\n{response_name}: "
        f"{len(response_data)} complete stations"
    )


    # --------------------------------------------------------
    # C. Row-standardized four-nearest-neighbour weights
    # --------------------------------------------------------

    spatial_weights = (
        build_knn_weights(
            response_data[
                LON_COL
            ],
            response_data[
                LAT_COL
            ],
            PRIMARY_K,
        )
    )


    # --------------------------------------------------------
    # D. Reduced SEMs
    # --------------------------------------------------------

    for model_name, predictors in (
        REDUCED_MODELS.items()
    ):

        vif_table = calculate_vif(
            response_data[
                predictors
            ]
        )

        vif_table.insert(
            0,
            "Response",
            response_name,
        )

        vif_table.insert(
            1,
            "Model",
            model_name,
        )

        vif_table[
            "N_stations"
        ] = len(
            response_data
        )

        all_reduced_vif.append(
            vif_table
        )


        maximum_vif = float(
            vif_table[
                "VIF"
            ].max()
        )


        model = fit_sem(
            response_data,
            response_col,
            predictors,
            spatial_weights,
            response_name,
            model_name,
        )


        coefficient_table = (
            extract_coefficients(
                model,
                response_name,
                model_name,
                predictors,
            )
        )

        all_coefficients.append(
            coefficient_table
        )


        summary = (
            extract_model_summary(
                model,
                response_name,
                model_name,
                predictors,
                spatial_weights,
                maximum_vif,
            )
        )

        all_model_summaries.append(
            summary
        )


# ============================================================
# 7. Combine results
# ============================================================

coefficients = pd.concat(
    all_coefficients,
    ignore_index=True
)


coefficients = add_bh_fdr(
    coefficients
)


model_summaries = pd.DataFrame(
    all_model_summaries
)


reduced_model_vif = pd.concat(
    all_reduced_vif,
    ignore_index=True
)


if initial_vif_records:

    initial_vif = pd.concat(
        initial_vif_records,
        ignore_index=True
    )

else:

    initial_vif = pd.DataFrame()


# ============================================================
# 8. Primary Tmean model
# ============================================================

primary_results = (
    coefficients[
        coefficients[
            "Model"
        ] == PRIMARY_MODEL
    ]
    .copy()
)


# ============================================================
# 9. Temperature sensitivity summary
# ============================================================

temperature_rows = []


for model_name, temperature_term in [

    (
        "Tmean_model",
        "Tmean_yr"
    ),

    (
        "Tmax_model",
        "Tmax_yr"
    ),

    (
        "Tmin_model",
        "Tmin_yr"
    ),
]:

    subset = coefficients[
        (
            coefficients[
                "Model"
            ] == model_name
        )
        &
        (
            coefficients[
                "Term"
            ] == temperature_term
        )
    ].copy()

    temperature_rows.append(
        subset
    )


temperature_sensitivity = (
    pd.concat(
        temperature_rows,
        ignore_index=True
    )
)


# ============================================================
# 10. Combined SEM result table
# ============================================================

sem_results = coefficients.merge(

    model_summaries[
        [
            "Response",
            "Model",
            "Temperature_metric",
            "N_stations",
            "Maximum_VIF",
            "Lambda",
            "Pseudo_R2",
            "Filtered_residual_Moran_I",
            "Filtered_residual_Moran_P",
            "AIC",
            "BIC",
        ]
    ],

    on=[
        "Response",
        "Model",
    ],

    how="left",
)


# ============================================================
# 11. Save results
# ============================================================

with pd.ExcelWriter(
    OUTPUT_FILE,
    engine="openpyxl"
) as writer:

    sem_results.to_excel(
        writer,
        sheet_name="SEM_results",
        index=False
    )

    primary_results.to_excel(
        writer,
        sheet_name="Primary_Tmean",
        index=False
    )

    temperature_sensitivity.to_excel(
        writer,
        sheet_name="Temperature_sensitivity",
        index=False
    )

    coefficients.to_excel(
        writer,
        sheet_name="All_coefficients",
        index=False
    )

    model_summaries.to_excel(
        writer,
        sheet_name="Model_summary",
        index=False
    )

    reduced_model_vif.to_excel(
        writer,
        sheet_name="Reduced_model_VIF",
        index=False
    )

    initial_vif.to_excel(
        writer,
        sheet_name="Initial_7var_VIF",
        index=False
    )

    for response_name, correlation_table in (
        correlation_tables.items()
    ):

        sheet_name = (
            "Corr_"
            + response_name
        )[:31]

        correlation_table.to_excel(
            writer,
            sheet_name=sheet_name
        )


print(
    "\nSpatial error model analysis completed."
)

print(
    f"Results saved to:\n"
    f"{OUTPUT_FILE}"
)