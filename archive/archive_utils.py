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
import pickle
import matplotlib.colors as mcolors
from sklearn.neighbors import KernelDensity
import sys
sys.path.insert(0, "./../")
from src.ice_sheet import ice_sheet_runner
import random
from pathos.multiprocessing import ProcessingPool as Pool
import time
from src.utils import load_data, prep_data, predict_and_plot_SLC
from tqdm import tqdm
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
from src.utils import plot
import copy
from IPython.display import Video
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('colorblind')
mpl.rcParams['font.family'] = 'Arial'


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

def mfed_selection(data_sub, X_m):
    X_m = np.hstack([np.zeros((7,1)), np.arange(2, 9).reshape((-1, 1))])
    X, Y, X_m = prep_data(data_sub, include_res=True,X_m=X_m)
    X_sub = X
    Y_sub = Y
    # Boiler plate for plotting
    mesh_size = 50
    space = ParameterSpace([ContinuousParameter('Resolution', X[:,0].min(), X[:,0].max()),
                            ContinuousParameter('Melt Average', X[:,1].min(), X[:,1].max())])



    kernel = GPy.kern.RBF(input_dim=2,ARD=True,lengthscale=[1,1],variance=1)
    model_gpy = GPRegression(X_sub,Y_sub,kernel = kernel,normalizer=True,noise_var=0.05)
    model_gpy.parameters[0].variance.fix()
    model_gpy.parameters[1].variance.fix()

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

        weighted_utility[idx+2] = np.array(weighted_variance.evaluate(x_monte_carlo)).mean()
    return weighted_utility


def sample_points_res(data_sub, res_list, X_m):
    closest_points = []
    #print(data_list)
    for idx, res in enumerate(res_list):
        x_m_value = random.choice(X_m)
        data_tmp = data_sub[data_sub["resolution"] == res]
        closest_point_index = (data_tmp['melt_average'] - x_m_value).abs().argsort()[:1].values[0]
        closest_points.append(data_tmp.iloc[closest_point_index])
    return closest_points


def point_selection_algo(data_sub, resolution, budget, X_m, prev_val):
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
                return None, None
            closest_points = pd.DataFrame(closest_points)
        case "mfed":

            closest_points = pd.DataFrame(sample_points_res(data_sub, prev_val +[2,4,8], X_m))
            weighted_utility = mfed_selection(closest_points, X_m)
            tmp_budget = budget - sum([cost[i] for i in prev_val])
            while tmp_budget > 0:
                sample = max((k for k, v in cost.items() if v <= tmp_budget), key=lambda x: weighted_utility[x], default=None)
                tmp_budget -= cost[sample]
                prev_val.append(sample)
            closest_points = sample_points_res(data_sub, prev_val, X_m)
            if len(closest_points) < 2:
                return None, None
            closest_points = pd.DataFrame(closest_points)
        case 2 | 4  | 8:
            num_points = budget//cost[resolution]
            if num_points < 2:
                return None, None
            if num_points > 500:
                num_points = 500
            data_sub = data_sub[data_sub["resolution"] == resolution]
            closest_points_indices = []
            for x_m_value in X_m[:num_points].flatten():
                closest_point_index = (data_sub['melt_average'] - x_m_value).abs().argsort()[:1]
                closest_points_indices.append(closest_point_index.values[0])
            closest_points = data_sub.iloc[closest_points_indices]
    return closest_points, prev_val

def calculate_optimal_density(data, comp_budget, prev_val, resolution=2, mean_melt=2.2):
    data_sub = data[data["years"] == 150]
    X_m = np.random.lognormal(mean=mean_melt, size=500).reshape(
        (-1, 1)
    )
    closest_points, prev_val = point_selection_algo(data_sub, resolution, comp_budget, X_m,prev_val)
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
    kde = KernelDensity(kernel='gaussian',bandwidth=1).fit(np.array(data_list).reshape((-1,1))) #you can supply a bandwidth

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
    return y_opt_cdf, y_opt_pdf, data_list, prev_val