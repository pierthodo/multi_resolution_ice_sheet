from modal import Image
import sys
import pathlib
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pathlib
from GPy.models import GPRegression
import GPy
from datetime import datetime
import pickle
import argparse
import matplotlib.colors as mcolors
from sklearn.neighbors import KernelDensity
import sys
import os
sys.path.insert(0, "./../")
#
import random
from pathos.multiprocessing import ProcessingPool as Pool
import time
#
#from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.stats import wasserstein_distance
from sklearn.utils import shuffle
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from GPy.models import GPRegression
import GPy
from emukit.model_wrappers import GPyModelWrapper
from emukit.experimental_design.acquisitions import IntegratedVarianceReduction
from emukit.core import ContinuousParameter, ParameterSpace
from emukit.core.acquisition.acquisition_per_cost import acquisition_per_expected_cost
from emukit.core.interfaces.models import IModel
from emukit.experimental_design import ExperimentalDesignLoop
import plotly.graph_objects as go
import itertools
from sklearn.svm import SVR
#
import copy
from IPython.display import Video
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('colorblind')
mpl.rcParams['font.family'] = 'Arial'
import modal

datascience_image = Image.debian_slim(python_version="3.10").pip_install(
    "numpy==1.24", "GPy==1.10.0", "xarray", "matplotlib==3.7.5", "scikit-learn", "emukit","seaborn", "pathos", "scipy"
)
start = 2

budget_list = []
NUM_BUDGET = 24
NUM_REP = 100

for i in range(30):
    new_start = int(start * (1.5))
    if new_start - start > 500:
        start += 500
    else:
        start = new_start
    budget_list.append(start)
max_budget = budget_list[-1]
cost = {2:70,3:12,4:8,5:4,6:3,7:2,8:1}

app = modal.App("ice-sheet")

def filter_nan(x,y=None):
    if y is not None:
        valid_X = ~np.isnan(x).any(axis=1)
        valid_Y = ~np.isnan(y).flatten()
        valid_indices = valid_X & valid_Y

        # Remove the rows with NaN from both X and Y
        X_cleaned = x[valid_indices]
        Y_cleaned = x[valid_indices]
        return X_cleaned, Y_cleaned
    else:
        valid_X = ~np.isnan(x).any(axis=1)

        # Remove the rows with NaN from both X and Y
        X_cleaned = x[valid_X]
        return X_cleaned

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


def control_distribution(x, lower=2, upper=8, threshold=1.5):
    # Ensure x is within the valid range [0, 1]
    x = max(0.0, min(1.0, x))

    # Define the mean of the Gaussian based on x
    mean = lower + x * (upper - lower)

    # Values range from lower to upper
    values = np.arange(lower, upper + 1)

    # Compute Gaussian weights based on the defined mean
    weights = np.exp(-0.5 * ((values - mean) ** 2) / threshold**2)

    # Accumulate outliers into the nearest extreme values directly
    if lower + threshold > lower:
        accumulated_lower = weights[values < lower + threshold].sum()
        weights[0] += accumulated_lower  # Assume the first element corresponds to 'lower'

    if upper - threshold < upper:
        accumulated_upper = weights[values > upper - threshold].sum()
        weights[-1] += accumulated_upper  # Assume the last element corresponds to 'upper'

    # Normalizing weights
    weights /= weights.sum()

    # Sample from the distribution
    return np.random.choice(values, p=weights)

def mfed_selection(data_sub, X_m, fit_lengthscale=False,percentile=50):
    X_m = np.hstack([np.zeros((7,1)), np.arange(2, 9).reshape((-1, 1))])
    X, Y, X_m = prep_data(data_sub, include_res=True,X_m=X_m)


    # Select 100 random samples from X and Y and assign directly to X_sub and Y_sub
    indices = np.random.choice(len(X), size=min(250, len(X)), replace=False)
    X_sub = X[indices]
    Y_sub = Y[indices]

    # Boiler plate for plotting
    mesh_size = 50
    space = ParameterSpace([
                            ContinuousParameter('Melt Average', X[:,0].min(), X[:,0].max()),
                            ContinuousParameter('Resolution', X_m[:,1].min(), X_m[:,1].max())])



    kernel = GPy.kern.RBF(input_dim=2,ARD=True,lengthscale=[1,1],variance=1)

    # Create a mask for valid (non-NaN) entries in both X and Y
    X_sub,Y_sub = filter_nan(X_sub,Y_sub)


    model_gpy = GPRegression(X_sub,Y_sub,kernel = kernel,normalizer=True,noise_var=0.05)
    model_gpy.parameters[0].variance.fix()
    model_gpy.parameters[1].variance.fix()
    if fit_lengthscale:
        try:
            hmc = GPy.inference.mcmc.HMC(model_gpy,stepsize=0.01)
            #for i in range(10):
            #    s = hmc.sample(num_samples=10)
            #
            s = hmc.sample(num_samples=100)
            s = hmc.sample(num_samples=100)
            print("Iteration", [np.percentile(s[:,1], p, axis=0) for p in [10, 25,50,75, 90]])
            model_gpy.parameters[0].lengthscale[1] = np.percentile(s[:,1], 100-percentile, axis=0)
            #rint("Mfed", model_gpy.parameters[0].lengthscale[1])
        except:
            #print("Failed")
            pass
                #print("Double failed")
        #model_gpy.parameters[0].lengthscale[0] = 1

    x_monte_carlo = np.vstack([
                    np.linspace(X[:,0].min(),X[:,0].max(),mesh_size),
                    np.ones((mesh_size))*space.parameters[1].min,]).T
    model_emukit = GPyModelWrapper(model_gpy)
    model_variance = IntegratedVarianceReduction(model=model_emukit,space=space,x_monte_carlo=x_monte_carlo)
    class CostModel(IModel):
        def __init__(self,cost):
            self.cost = cost

        def predict(self, X: np.ndarray):
            cost_dict = self.cost
            keys = np.array(list(cost_dict.keys()))
            values = np.array(list(cost_dict.values()))
            sorted_indices = np.argsort(keys)
            sorted_keys = keys[sorted_indices]
            sorted_values = values[sorted_indices]
            costs = np.interp(X[:, 1], sorted_keys, sorted_values).reshape((-1, 1))
            return (costs, X)

    cost_model = CostModel({x[1]: cost[idx+2] for idx, x in enumerate(X_m)}) # Fit a linear model on the costs
    weighted_variance = acquisition_per_expected_cost(model_variance,cost_model)
    weighted_utility = {}
    for idx,(_,res) in enumerate(X_m):
        x_monte_carlo = np.vstack([
                    np.linspace(X[:,0].min(),X[:,0].max(),mesh_size),
                    np.ones((mesh_size))*res]).T
        #raise "Fix this"
        weighted_utility[idx+2] = np.array(weighted_variance.evaluate(x_monte_carlo)).mean()
    return weighted_utility

def point_selection_algo(data_sub, resolution, budget, X_m, prev_val):
    max_res = None
    match resolution:
        case "average":
            coef = 1-(budget / max_budget)
            tmp_budget = budget - sum([cost[i] for i in prev_val])
            while tmp_budget > 0:
                min_lower = min(cost.keys(), key=lambda x: abs(cost[x] - tmp_budget))
                sample = control_distribution(coef, lower=min_lower)
                prev_val.append(sample)
                tmp_budget -= cost[sample]
                if len(prev_val) > 500:
                    break
            closest_points = sample_points_res(data_sub, prev_val, X_m)
            if len(closest_points) < 2:
                return None, None, None
            closest_points = pd.DataFrame(closest_points)
        case "mfed" | "mfed_opt" | "mfed_75" | "mfed_50" | "mfed_90" | "mfed_early_75" | "mfed_early" | "mfed_early_90" | "mfed_early_50" | "mfed_early_10" | "mfed_10":
            closest_points = pd.DataFrame(sample_points_res(data_sub, prev_val +[6,7,8], X_m))
            if resolution in ["mfed", "mfed_early"]:
                weighted_utility = mfed_selection(closest_points, X_m)
            elif resolution == "mfed_opt":
                weighted_utility = mfed_selection(closest_points, X_m,fit_lengthscale=True)
            elif resolution in ["mfed_75", "mfed_early_75"]:
                weighted_utility = mfed_selection(closest_points, X_m,fit_lengthscale=True, percentile=75)
            elif resolution in  ["mfed_50", "mfed_early_50"]:
                weighted_utility = mfed_selection(closest_points, X_m,fit_lengthscale=True, percentile=50)
            elif resolution in ["mfed_90", "mfed_early_90"]:
                weighted_utility = mfed_selection(closest_points, X_m,fit_lengthscale=True, percentile=90)
            elif resolution in ["mfed_10", "mfed_early_10"]:
                weighted_utility = mfed_selection(closest_points, X_m,fit_lengthscale=True, percentile=10)
            tmp_budget = budget - sum([cost[i] for i in prev_val])
            max_res = max((k for k, v in cost.items()), key=lambda x: weighted_utility[x], default=None)
            if "early" in resolution:
                threshold = cost[max_res]
            else:
                threshold = 0

            # remove cost[max_res] to perform another strategy. Plot both
            while tmp_budget > threshold:
                sample = max((k for k, v in cost.items() if v <= tmp_budget), key=lambda x: weighted_utility[x], default=None)
                tmp_budget -= cost[sample]
                prev_val.append(sample)
            closest_points = sample_points_res(data_sub, prev_val, X_m)
            if len(closest_points) < 2:
                return None, None, None
            closest_points = pd.DataFrame(closest_points)
        case 2 | 3 | 4 | 8:
            num_points = budget//cost[resolution]
            if num_points < 2:
                return None, None, None
            if num_points > 500:
                num_points = 500
            data_sub = data_sub[data_sub["resolution"] == resolution]
            closest_points_indices = []
            for x_m_value in X_m[:num_points].flatten():
                closest_point_index = (data_sub['melt_average'] - x_m_value).abs().argsort()[:1]
                closest_points_indices.append(closest_point_index.values[0])
            closest_points = data_sub.iloc[closest_points_indices]
    return closest_points, prev_val, max_res

def sample_points_res(data_sub, res_list, X_m):
    closest_points = []
    #print(data_list)
    for idx, res in enumerate(res_list):
        x_m_value = random.choice(X_m)
        data_tmp = data_sub[data_sub["resolution"] == res]
        closest_point_index = (data_tmp['melt_average'] - x_m_value).abs().argsort()[:1].values[0]
        closest_points.append(data_tmp.iloc[closest_point_index])
    return closest_points



@app.function(image=datascience_image, concurrency_limit=100, timeout=6000)
def calculate_optimal_density(data_dic):
    data = data_dic["data"]
    comp_budget = data_dic["budget_list"]
    prev_val = data_dic["prev_val"]
    resolution = data_dic["resolution"]
    mean_melt = data_dic["mean_melt"]
    data_sub = data[data["years"] == 150]
    X_m = np.random.lognormal(mean=mean_melt, size=500).reshape(
        (-1, 1)
    )
    closest_points, prev_val, max_res = point_selection_algo(data_sub, resolution, comp_budget, X_m,prev_val)
    if closest_points is None:
        return None
    X, Y, X_m = prep_data(closest_points, X_m=X_m)
    kernel = GPy.kern.RBF(input_dim=1, ARD=True, lengthscale=[1], variance=1)
    model_gpy = GPRegression(X, Y, kernel=kernel, normalizer=True, noise_var=1)
    model_gpy.optimize()
    mean,var  = model_gpy.predict(X_m.reshape((-1, 1)))
    data_list = []
    size = 500 // len(mean)
    for m,v in zip(mean,var):
        data_list += list(np.random.normal(m,np.sqrt(v),size=size))

    X_lin = np.linspace(0,500,100)[:, np.newaxis]
    filtered_X = filter_nan(np.array(data_list).reshape((-1,1)))
    if filtered_X.shape[0] == 0:
        return None
    kde = KernelDensity(kernel='gaussian',bandwidth=1).fit(filtered_X) #you can supply a bandwidth

    log_density_values=kde.score_samples(X_lin)
    y_opt_pdf=np.exp(log_density_values)
    # Calculate optimal CDF
    max_val = 400
    x = []
    y = []
    for thres in range(0,max_val,5):
        # Count how many elements in data list are abov thres
        # and divide by the total number of elements
        # to get the percentage of elements above thres
        x.append(thres)
        y.append(len([i for i in data_list if i > thres])/len(data_list))
    y_opt_cdf = np.array(y)
    return y_opt_cdf, y_opt_pdf, data_list, prev_val, max_res

def calc_result(data):
    melt_rate = data["melt_rate"]
    resolution = data["resolution"]
    opt_pdf = data["opt_pdf"]
    observed_2k = data["observed_2k"]
    dir_path = data["dir_path"]

    data = data["ice_sheet_data"]


    if melt_rate == "low":
        mean_melt = 2.2
        melt_label = 10
    else:
        mean_melt = 2.872
        melt_label = 20

    #from src.utils import load_data, prep_data
    #data = load_data([0,151], datasets_folder="./../assets/v2_ice_sheet_simulation/", extension="_dt", version="vs")

    data_sub = data[data["years"] == 150]
    X_m = np.hstack([np.linspace(0,150,7).reshape((-1, 1)), np.arange(2, 9).reshape((-1, 1))])
    X, Y, X_m = prep_data(data_sub, include_res=True,X_m=X_m)


    mean_errors = {}
    pdf_errors = {}
    prev_val = []
    dt_list = {}
    #for mean_melt, melt_label in [(2.2,10),(2.872,20)]: #,


    res = resolution
    mean_errors[melt_label] = {}
    pdf_errors[melt_label] = {}
    dt_list[melt_label] = {}
        #for res in [2,4,8]: #"mfed", ,
            #if res == "mfed":
            #    prev_val = [2,4,8]
    prev_val = []
    new_val = []
    mean_errors[melt_label][res] = {}
    pdf_errors[melt_label][res] = {}
    dt_list[melt_label][res] = {}

    max_res_list = {}
    max_res_list[melt_label] = {}
    max_res_list[melt_label][res] = {}
    prev_val_list = [[] for i in range(NUM_REP)]
    for budget in budget_list[:NUM_BUDGET]:
        errors = []
        pdf_error_list = []
        max_res_tmp = []
        data_tmp_list = []
        for i in range(NUM_REP):
            data_tmp_list.append({
                "data": data,
                "budget_list": budget,
                "prev_val": prev_val_list[i],
                "resolution": res,
                "mean_melt": mean_melt
            })

        failure_count = 0
        while failure_count < 2:
            try:
                results = [i for i in calculate_optimal_density.map(data_tmp_list)]
                break  # If successful, exit the while loop
            except:
                failure_count += 1
                if failure_count == 2:
                    print(f"Skipping budget {budget} due to repeated failures")
                    break  # Exit the while loop after two failures

        if failure_count == 2:
            continue  # Skip to the next iteration of the for loop
        prev_val_list = [[] for i in range(NUM_REP)]

        dt_list[melt_label][res][budget] = []
        for idx,result in enumerate(results):
            if result is not None:
                y_opt_cdf, new_pdf, new_observed, new_val, max_res = result
                prev_val_list[idx] = new_val
            else:
                continue
            pdf_error_list.append(np.abs(opt_pdf-new_pdf).sum())
            max_res_tmp.append(max_res)
            dt_list[melt_label][res][budget].append(new_val)
            errors.append(wasserstein_distance(observed_2k, new_observed))
        if new_val is not None:
            prev_val = new_val
        mean_errors[melt_label][res][budget] = errors
        pdf_errors[melt_label][res][budget] = pdf_error_list
        max_res_list[melt_label][res][budget] = np.array(max_res_tmp)
        try:
            print("Selected resolution", dt_list[melt_label][res][budget][0])
        except:
            pass
        print(f"Resolution: {res}, Budget: {budget}, Mean Error: {np.array(pdf_errors[melt_label][res][budget]).mean()}, max res: {max_res_list[melt_label][res][budget]}")
        with open(f"{dir_path}/{melt_rate}_{resolution}.pkl", 'wb') as file:
            pickle.dump({"mean_errors":mean_errors, "pdf_errors":pdf_errors, "dt_list":dt_list, "opt_res":max_res_list},file)
    return True



def create_timestamped_directory(base_path):
    # Get current timestamp
    now = datetime.now()

    # Format the timestamp as "month_date_hour_minute"
    dir_name = now.strftime("%m_%d_%H_%M")

    # Create the full path for the new directory
    new_dir_path = os.path.join(base_path, dir_name)

    # Create the directory if it doesn't exist
    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)
        print(f"Directory '{dir_name}' created successfully in {base_path}")
    else:
        print(f"Directory '{dir_name}' already exists in {base_path}")

    return new_dir_path


@app.local_entrypoint()
def main():
    fixed_base_path = "/Users/pierrethodoroff/development/personal/multi_resolution_ice_sheet/assets/plots_data/fig_7"
    created_dir = create_timestamped_directory(fixed_base_path)
    data = load_data([0,151], datasets_folder="./../assets/v2_ice_sheet_simulation/", extension="_dt", version="vs")
    for melt_rate in ["low"]: #, "high"
        if melt_rate == "low":
            mean_melt = 2.2
        else:
            mean_melt = 2.872

        data_tmp = {
            "data": data,
            "budget_list": budget_list[-1],
            "prev_val": [],
            "resolution": 2,
            "mean_melt": mean_melt
        }
        opt_cdf,opt_pdf, observed_2k,_, _ = calculate_optimal_density.remote(data_tmp)
        #for resolution in ["mfed_75", "mfed_early_75", "mfed", "mfed_early", 2,4,8]: #"mfed", "mfed_75", : #, "average", 2,4,8
        #for resolution in ["mfed_early_90", "mfed_early_50", "mfed_early_10","mfed","mfed_early", 2, 3, 4, 5, 6, 7, 8]: #,
        for resolution in ["mfed_early"]: #,
            calc_result({"melt_rate":melt_rate,"resolution": resolution, "ice_sheet_data": data, "opt_pdf": opt_pdf, "observed_2k": observed_2k, "dir_path":created_dir})



