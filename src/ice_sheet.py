"""Module for running ice sheet simulations and processing their results."""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from typing import Tuple

DATASET_FOLDER = Path("./../assets/ice_sheet_simulation/")

class IceSheetRunner:
    """Runner class for ice sheet simulations with varying resolutions."""

    def __init__(self, timestamp: int = 150) -> None:
        """Initialize the ice sheet runner.

        Args:
            timestamp: Time point to analyze in the simulation data (default: 150)
        """
        self.timestamp = timestamp
        self.X_clf: StandardScaler = None
        self.load_data(timestamp)

    def load_data(self, timestamp: int) -> None:
        """Load ice sheet simulation data from NetCDF files.

        Args:
            timestamp: Time point to load from the simulation data
        """
        input_data = []
        for fidelity in [2, 3, 4, 5, 6, 8]:
            file = f"{fidelity}km.nc"
            dataset = xr.open_dataset(DATASET_FOLDER / file)
            input_data.append(
                pd.DataFrame(
                    {
                        "melt_exp": dataset["Inputs"][2],
                        "melt_average": dataset["Inputs"][3],
                        "melt_partial": dataset["Inputs"][0],
                        "SLC": np.array(dataset["SLC"])[timestamp, :].squeeze(),
                        "resolution": [fidelity] * dataset["Inputs"].shape[1],
                    }
                )
            )
        self.data = pd.concat(input_data).reset_index().dropna()
        self.X, self.Y = self.prep_data(self.data)

    def prep_data(self, data: pd.DataFrame, melt_average: float = 0, sub: int = -1) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training by scaling and shuffling.

        Args:
            data: Input DataFrame containing simulation results
            melt_average: Average melt value (default: 0)
            sub: Number of samples to subsample (-1 for all) (default: -1)

        Returns:
            Tuple of (X, Y) arrays containing processed features and targets
        """
        tmp = data
        Y = np.array(tmp["SLC"]).reshape((-1, 1))
        X = np.array(tmp[["melt_average", "resolution"]])
        X, Y = shuffle(X, Y)
        if sub != -1:
            idx_sub = np.concatenate(
                [
                    np.where(X[:, 1] == 5)[0][: sub // 2],
                    np.where(X[:, 1] == 10)[0][: sub // 2],
                ]
            )
            X = X[idx_sub, :]
            Y = Y[idx_sub, :]
        self.X_clf = StandardScaler()
        self.X_clf.fit(X)
        X = self.X_clf.transform(X)
        return X, Y

    def run(self, data: np.ndarray) -> np.ndarray:
        """Run the ice sheet simulation for given input data.

        Args:
            data: Input array containing simulation parameters

        Returns:
            Array of simulation results
        """
        y_return = []
        for i in range(data.shape[0]):
            sub_X = np.array([n for n in self.X if n[1] == data[i,1]])
            distance = np.round(np.sum(np.abs(sub_X - data[i, :]), axis=1), decimals=5)
            idx = np.random.choice(np.where(distance == distance.min())[0])
            y_return.append(self.Y[idx])
        return np.array(y_return)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Convenience method to run simulations.

        Args:
            X: Input array containing simulation parameters

        Returns:
            Array of simulation results
        """
        return self.run(X)
