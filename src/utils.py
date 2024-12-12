import pathlib
import pandas as pd
import xarray as xr
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import GPy
from GPy.models import GPRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from typing import List, Union, Optional, Tuple

from math import sqrt
import numpy as np
import plotly.graph_objects as go

from pathlib import Path

DATASET_FOLDER = Path("./../assets/ice_sheet_simulation/")

sns.set_style("whitegrid")
sns.set_palette("colorblind")
mpl.rcParams["font.family"] = "Arial"


def predict_and_plot_SLC(
    data: pd.DataFrame,
    timestamp: float,
    max_val_un: float,
    resolution: int,
    plot: bool = False,
    sample_size: int = 10,
    prior_sigma: float = 0.5,
    mean_prior: float = 3,
    fit_gp: bool = True,
    X_m: Optional[np.ndarray] = None,
) -> List[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]]:
    """Predict sea level change (SLC) using Gaussian Process regression and optionally plot results.

    Args:
        data: DataFrame containing the ice sheet data
        timestamp: Time point for prediction
        max_val_un: Maximum value for y-axis in plot
        resolution: Spatial resolution in km
        plot: Whether to generate visualization
        sample_size: Number of samples for prediction
        prior_sigma: Standard deviation for lognormal prior
        mean_prior: Mean for lognormal prior
        fit_gp: Whether to optimize GP hyperparameters
        X_m: Optional pre-defined input points for prediction

    Returns:
        List containing:
        - Input points (X_m)
        - GP predictions (mean and variance)
    """
    # Filter data for given resolution and timestamp
    data_sub = data[data["resolution"] == resolution]
    data_sub = data_sub[data_sub["years"] == timestamp]

    # Generate or use provided sampling points
    if X_m is None:
        X_m = np.random.lognormal(
            mean=mean_prior, sigma=prior_sigma, size=sample_size
        ).reshape((-1, 1))

    # Prepare data for GP
    X, Y, X_m = prep_data(data_sub, X_m=X_m)
    # Initialize and fit GP model
    kernel = GPy.kern.RBF(input_dim=1, ARD=True, lengthscale=[1], variance=1)
    model_gpy = GPRegression(X, Y, kernel=kernel, normalizer=True, noise_var=1)
    if fit_gp:
        model_gpy.optimize()

    if plot:
        _plot_slc_predictions(model_gpy, resolution, max_val_un)

    return [X_m, model_gpy.predict(X_m.reshape((-1, 1)))]

def _plot_slc_predictions(
    model: GPRegression,
    resolution: int,
    max_val_un: float,
) -> None:
    """Generate plot for SLC predictions.

    Args:
        model: Fitted GP model
        resolution: Spatial resolution in km
        max_val_un: Maximum value for y-axis
    """
    FONT_SIZES = {"ticks": 22, "label": 32, "legend": 20}

    model.plot(title=f"{resolution} km")
    plt.xticks(fontsize=FONT_SIZES["ticks"])
    plt.yticks(fontsize=FONT_SIZES["ticks"])
    plt.xlabel("Melt average " + r"$ma^{-1}$", fontsize=FONT_SIZES["label"])
    plt.ylabel("SLC (mm)", fontsize=FONT_SIZES["label"])
    plt.title(f"{resolution}km", fontsize=FONT_SIZES["label"])
    plt.ylim(0, max_val_un)

    # Set custom x-axis ticks
    plt.xticks(
        [-1.6, -0.533, 0.5333, 1.6],
        ["0", "50", "100", "150"],
    )

    # Handle legend based on resolution
    if resolution == 8:
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=FONT_SIZES["legend"]
        )
        plt.gcf().set_size_inches(plt.gcf().get_size_inches() * [1.5, 1])
    else:
        plt.legend().set_visible(False)


def prep_data(
    data: pd.DataFrame,
    melt_average: float = 0,
    sub: int = -1,
    X_m: Optional[np.ndarray] = None,
    include_res: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Prepare data for model training by scaling and shuffling.

    Args:
        data: Input DataFrame containing SLC and melt_average columns
        melt_average: Default melt average value (unused)
        sub: Number of samples to use (-1 for all)
        X_m: Optional array of points to transform using same scaling
        include_res: Whether to include resolution as a feature

    Returns:
        Tuple containing:
        - X: Scaled and shuffled feature matrix
        - Y: Target values
        - X_m: Transformed X_m if provided, else None
    """
    # Select features and target
    features = ["melt_average", "resolution"] if include_res else ["melt_average"]
    X = np.array(data[features])
    Y = np.array(data["SLC"]).reshape((-1, 1))

    # Shuffle data
    X, Y = shuffle(X, Y)

    # Subset data if requested
    if sub != -1:
        X = X[:sub, :]
        Y = Y[:sub, :]

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Transform X_m if provided
    if X_m is not None:
        X_m = scaler.transform(X_m)

    return X, Y, X_m


def load_data(
    timestamp: Union[int, List[int]],
    datasets_folder: str = DATASET_FOLDER,
    extension: str = "",
    version: str = "",
) -> pd.DataFrame:
    """Load ice sheet data from NetCDF files for different resolutions.

    Args:
        timestamp: Single timestamp or list [start, end] for range
        datasets_folder: Directory containing the dataset files
        extension: Optional filename extension (unused)
        version: Optional version string (unused)

    Returns:
        DataFrame containing combined data from all resolutions with columns:
        melt_exp, melt_average, melt_partial, SLC, resolution, years
    """
    RESOLUTIONS = range(2, 9)  # 2km to 8km
    input_data = []

    # Process each resolution
    for fidelity in RESOLUTIONS:
        # Load dataset
        file_path = pathlib.Path(datasets_folder) / f"{fidelity}km.nc"
        dataset = xr.open_dataset(file_path)

        # Handle timestamp range or single timestamp
        if isinstance(timestamp, list):
            start, end = timestamp
            for year in range(start, end):
                input_data.append(_create_dataframe(dataset, fidelity, year))
        else:
            input_data.append(_create_dataframe(dataset, fidelity, timestamp))

    return pd.concat(input_data).reset_index().dropna()


def _create_dataframe(
    dataset: xr.Dataset, fidelity: int, timestamp: int
) -> pd.DataFrame:
    """Helper function to create a DataFrame for a specific dataset and timestamp.

    Args:
        dataset: xarray Dataset containing ice sheet data
        fidelity: Resolution value in km
        timestamp: Year index

    Returns:
        DataFrame with extracted data for the given timestamp
    """
    return pd.DataFrame(
        {
            "melt_exp": dataset["Inputs"][2],
            "melt_average": dataset["Inputs"][3],
            "melt_partial": dataset["Inputs"][0],
            "SLC": np.array(dataset["SLC"])[timestamp, :].squeeze(),
            "resolution": [fidelity] * dataset["Inputs"].shape[1],
            "years": timestamp,
        }
    )
