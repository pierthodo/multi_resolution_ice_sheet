import numpy as np
import matplotlib as mpl
import numpy as np
from GPy.models import GPRegression
import GPy
from datetime import datetime
import pickle
from sklearn.neighbors import KernelDensity
import sys
import os
import random
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import numpy as np
from GPy.models import GPRegression
from src.utils import load_data, prep_data
import GPy
import seaborn as sns
sns.set_style('whitegrid')
sns.set_palette('colorblind')
mpl.rcParams['font.family'] = 'Arial'



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





def point_selection_algo(data_sub, resolution, budget, X_m, prev_val):
    max_res = None
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

fixed_base_path = "/Users/pierrethodoroff/development/personal/multi_resolution_ice_sheet/assets/plots_data/fig_7"


data = load_data([0,151])
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



