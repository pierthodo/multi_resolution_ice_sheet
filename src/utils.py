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

from math import sqrt
import numpy as np
import plotly.graph_objects as go

sns.set_style("whitegrid")
sns.set_palette("colorblind")
mpl.rcParams["font.family"] = "Arial"


def predict_and_plot_SLC(
    data,
    timestamp,
    max_val_un,
    resolution,
    plot=False,
    sample_size=10,
    prior_sigma=0.5,
    mean_prior=3,
    fit_gp=True,
    X_m=None,
):
    data_sub = data[data["resolution"] == resolution]
    data_sub = data_sub[data_sub["years"] == timestamp]
    if X_m is None:
        X_m = np.random.lognormal(
            mean=mean_prior, sigma=prior_sigma, size=sample_size
        ).reshape((-1, 1))
    X, Y, X_m = prep_data(data_sub, X_m=X_m)

    kernel = GPy.kern.RBF(input_dim=1, ARD=True, lengthscale=[1], variance=1)
    model_gpy = GPRegression(X, Y, kernel=kernel, normalizer=True, noise_var=1)
    if fit_gp:
        model_gpy.optimize()

    # Ploting
    if plot:
        ticksize = 22
        label = 32
        legend = 20
        model_gpy.plot(title=f"{resolution} km")
        plt.xticks(fontsize=ticksize)
        plt.yticks(fontsize=ticksize)
        plt.xlabel("Melt average "+ r'$ma^{-1}$', fontsize=label)
        plt.ylabel("SLC (mm)", fontsize=label)
        plt.title(str(resolution) + "km", fontsize=label)
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
        if resolution == 8:
            print("High")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=legend)
            plt.gcf().set_size_inches(plt.gcf().get_size_inches() * [1.5, 1])
        else:
            plt.legend().set_visible(False)
    return [X_m, model_gpy.predict(X_m.reshape((-1, 1)))]


def load_data(
    timestamp,
    datasets_folder="./../assets/ice_sheet_simulation/",
    extension="",
    version="",
):
    # timestamp=[0,199]
    input_data = []
    for fidelity in [2, 3, 4, 5, 6, 7, 8]:
        if version == "vs" and fidelity in [2]:
            file = f"ensemble_{fidelity}km_vary_gamma_quad_corrected{extension}_vs.nc"
        elif version == "r" and fidelity in [2]:
            file = f"ensemble_{fidelity}km_vary_gamma_quad_corrected{extension}_r.nc"
        else:
            file = f"ensemble_{fidelity}km_vary_gamma_quad_corrected{extension}.nc"
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
    # Y = StandardScaler().fit_transform(np.array(tmp["SLC"]).reshape((-1, 1)))
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


def plot(X, model_variance):
    mesh_size = 50
    x_1_space = np.linspace(X[:, 0].min(), X[:, 0].max(), mesh_size)
    x_2_space = np.linspace(X[:, 1].min(), X[:, 1].max(), mesh_size)
    xx, yy = np.meshgrid(x_1_space, x_2_space)
    X_mesh = np.hstack([xx.reshape((-1, 1)), yy.reshape((-1, 1))])
    utility = model_variance.evaluate(X_mesh).reshape((mesh_size, mesh_size))
    fig = go.Figure(
        data=[
            go.Surface(
                x=xx, y=yy, z=utility, opacity=0.7, showscale=False, colorscale="RdBu"
            ),
            go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=[utility.min()] * X.shape[0],
                mode="markers",
                marker_symbol="x",
                marker_size=5,
            ),
        ]
    )

    dic_scene = dict(
        xaxis_title="Melt Average", yaxis_title="Resolution", zaxis_title="Utility"
    )

    fig.update_layout(
        scene=dic_scene,
    )  # title="Partial melt: "+str(melt_partial)+" lengthscale: "+str(model_gpy.param_array[2]),

    fig.show()