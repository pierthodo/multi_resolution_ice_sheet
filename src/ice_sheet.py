from cgi import test
from math import sqrt
import pde
from pde import PDE, CartesianGrid, MemoryStorage, ScalarField, plot_kymograph
import numpy as np
from pde import PDE, FieldCollection, PlotTracker, ScalarField, UnitGrid
from pde import DiffusionPDE, ScalarField, UnitGrid
from emukit.core.acquisition.acquisition_per_cost import acquisition_per_expected_cost
from tqdm import tqdm
import pandas as pd
import pathlib
from typing import Union
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
from mfed.plot import plot
from IPython.display import Video
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle


class ice_sheet_runner:
    def __init__(self, timestamp=199) -> None:
        self.timestamp = timestamp
        self.load_data(timestamp)

    def load_data(self, timestamp):
        input_data = []
        for fidelity in [2, 3, 4, 5, 6, 8]:
            datasets_folder = "/Users/pierthodo/Documents/Research/Experiments/multi_fidelity_experimental_design/assets/WAVI-WAIS-setups/ensembles/multi_fidelity_paper_2023/"
            file = f"ensemble_{fidelity}km_vary_gamma.nc"
            datasets = pathlib.Path(datasets_folder)
            dataset = xr.open_dataset(datasets / file)
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

    def prep_data(self, data, melt_average=0, sub=-1):
        tmp = data
        # Y = StandardScaler().fit_transform(np.array(tmp["SLC"]).reshape((-1, 1)))
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
        # jitter = np.random.normal(0, 0.0001, size=X.shape)
        # jitter[:, 0] = 0
        # X += jitter
        # X[:, 1] = X[:, 1] * 0.1
        return X, Y

    def run(self, data):
        y_return = []
        for i in range(data.shape[0]):
            distance = np.round(np.sum(np.abs(self.X - data[i, :]), axis=1), decimals=5)
            idx = np.random.choice(np.where(distance == distance.min())[0])
            y_return.append(self.Y[idx])
        return np.array(y_return)

    def __call__(self, X):
        return self.run(X)
