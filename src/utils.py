import pathlib
import pandas as pd
import xarray as xr
import numpy as np
import numpy as np
import pathlib
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import GPy
from GPy.models import GPRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns

plt.style.use("seaborn-whitegrid")
sns.set_palette("colorblind")
mpl.rcParams["font.family"] = "Arial"


def predict_and_plot_SLC(
    data, timestamp, max_val_un, resolution, plot=False, sample_size=10
):
    data_sub = data[data["resolution"] == resolution]
    data_sub = data_sub[data_sub["years"] == timestamp]
    X_m = np.random.lognormal(mean=3, sigma=0.5, size=sample_size).reshape((-1, 1))
    X, Y, X_m = prep_data(data_sub, X_m=X_m)

    kernel = GPy.kern.RBF(input_dim=1, ARD=True, lengthscale=[1], variance=1)
    model_gpy = GPRegression(X, Y, kernel=kernel, normalizer=True, noise_var=1)
    model_gpy.optimize()

    # Ploting
    if plot:
        model_gpy.plot(title=f"{resolution} km")
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Melt average (m/a)", fontsize=18)
        plt.ylabel("SLC (mm)", fontsize=18)
        plt.title(str(resolution) + "km", fontsize=18)
        plt.ylim(0, max_val_un)
        plt.xticks(
            [-1.6, -0.533, 0.5333, 1.6],
            [
                "0",
                "50",
                "100",
                "150",
            ],
        )
        plt.legend(loc="upper left", fontsize=12)
    return [X_m, model_gpy.predict(X_m.reshape((-1, 1)))]


def load_data(timestamp):
    # timestamp=[0,199]
    input_data = []
    for fidelity in [2, 3, 4, 5, 6, 7, 8]:
        datasets_folder = "./../assets/ice_sheet_simulation/"

        file = f"ensemble_{fidelity}km_vary_gamma_quad_corrected.nc"
        datasets = pathlib.Path(datasets_folder)
        dataset = xr.open_dataset(datasets / file)
        if type(timestamp) == list:
            for i in range(timestamp[0], timestamp[1], 1):
                input_data.append(
                    pd.DataFrame(
                        {
                            "melt_exp": dataset["Inputs"][2],
                            "melt_average": dataset["Inputs"][3],
                            "melt_partial": dataset["Inputs"][0],
                            "SLC": np.array(dataset["SLC"])[i, :].squeeze(),
                            "resolution": [fidelity] * dataset["Inputs"].shape[1],
                            "years": i,
                        }
                    )
                )
        else:
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
    return pd.concat(input_data).reset_index().dropna()


def prep_data(data, melt_average=0, sub=-1, X_m=None, include_res=False):
    tmp = data
    Y = StandardScaler().fit_transform(np.array(tmp["SLC"]).reshape((-1, 1)))
    Y = np.array(tmp["SLC"]).reshape((-1, 1))
    if include_res:
        X = np.array(tmp[["melt_average", "resolution"]])
    else:
        X = np.array(tmp[["melt_average"]])
    X, Y = shuffle(X, Y)
    if sub != -1:
        X = X[:sub, :]
        Y = Y[:sub, :]
    clf = StandardScaler()
    clf.fit(X)
    X = clf.transform(X)
    if X_m is not None:
        X_m = clf.transform(X_m)
    return X, Y, X_m
