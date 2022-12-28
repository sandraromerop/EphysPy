
#%%
%load_ext autoreload
%autoreload 2

import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
# import ssm
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn.pipeline
import sklearn.decomposition
import scipy.stats as scistats
import scipy.stats as scistatscd
import scipy.io as scio
import sklearn.pipeline
import sklearn.decomposition
import scipy.stats
from scipy import linalg
from sklearn.decomposition import PCA
from utils import *
from plots import *
from sql_utils import *
import ssm.plots as splt
import cluster_metrics.metrics as metrics
from cluster_metrics.params import QualityMetricsParams
from sklearn.model_selection import train_test_split
import seaborn as sns
color_names = ["windows blue", "red", "amber", "faded green"]
colors = sns.xkcd_palette(color_names)
sns.set_style("white")
sns.set_context("talk")
#%%
X_df_folder ='/Volumes/GoogleDrive-116748251018268574178/My Drive/BeliefStateData/Sandra/Extracted_Data/BeliefState/Spikes/SingleUnitPk/BS005_20200922_ORB_X_df_res.pickle'
with open(X_df_folder, 'rb') as f:
    dat = pickle.load(f)
xdf = dat['xdf']
#%% Population activity : tile space 
ng = list(xdf.filter(like='_Zscore'))
ng.append('bins_trial_numb')
sp_trial_list = xdf[ng].groupby('bins_trial_numb').apply(lambda x:np.asarray(x))
sp_trial_list = list([ss[:,:-1] for ss in sp_trial_list])

# inputs_sep_trial_list = xdf[["bins_cue_1",'bins_cue_2','bins_cue_3','bins_cue_4', "bins_reward",'bins_trial_numb']].groupby('bins_trial_numb').apply(lambda x:np.asarray(x))
inputs_sep_trial_list = xdf[['bins_cue_type','bins_trial_numb']].groupby('bins_trial_numb').apply(lambda x:np.asarray(x))
inputs_sep_trial_list= list([ss[:,:-1] for ss in inputs_sep_trial_list])
# X_train, X_test, y_train, y_test = train_test_split(sp_trial_list, \
#                                     inputs_sep_trial_list, \
#                                     test_size=0.3, random_state=0)
# one_hot_cue = np.int0(np.asarray([np.sum(ii[:,:-1],axis=0) for ii in inputs_sep_trial_list])>0)
# idcue = np.argwhere(one_hot_cue>0) 
idcue = np.asarray([ii[0][0].squeeze() for ii in inputs_sep_trial_list])
#%%
itype  = 8
idcue_ = list(np.argwhere(idcue==itype).squeeze())
sp_cue = [sp_trial_list[ic] for ic in idcue_]
ll = np.min([ii.shape[0] for ii in sp_cue])
sp_cue = np.asarray([ii[:ll,:] for ii in sp_cue])
musp_cue = np.nanmean(sp_cue,axis=0).squeeze()
resp_filt = np.asarray([savgol_filter(musp_cue[:,rr], 11, 3) for rr in range(musp_cue.shape[1])])
idmx = np.argmax(resp_filt,axis=1)

idsort = np.argsort(idmx)
resp_filt_s = resp_filt[idsort,:]
musp_cue_s = musp_cue[:,idsort].T
plt.imshow(resp_filt_s, interpolation='nearest', aspect='auto')
#%%


#%% 
# measure tuning ? 

#%% Dimensionality reduction 
