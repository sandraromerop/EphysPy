# %load_ext autoreload
# %autoreload 2

#%%
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import ssm
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

params = vars(QualityMetricsParams())
nas_drive ='/Volumes/GoogleDrive-116748251018268574178/My Drive/BeliefStateData/'  #'/Volumes/homes' #'Z:'# '/Volumes/homes' #'Z:' #'/Volumes/homes' #'Z:'
protocol_name = 'BeliefState'
table_name = 'neuron'
sql_path = os.path.join(nas_drive,'Sandra','db_','session_log_config.sqlite')
fig_dir = os.path.join(nas_drive,'Sandra','Figures',protocol_name,'Spikes')

fields = '"name","date","brainReg","cluId","cluGro","single_unit_mat","contam_rate","isi_viol","presence_ratio","firing_rate"'
type_data = 'Spikes'
type_dir = 'extracted_su'
con = create_connection(sql_path)
cur = con.cursor()
rows = get_sql_rows_by_fields(cur, table_name, fields)
sp_dir_pkl = os.path.join(nas_drive,'Sandra','Extracted_Data',protocol_name,'Spikes','SingleUnitPk')
if not os.path.exists(sp_dir_pkl):
    os.mkdir(sp_dir_pkl)
# Get date and protocol /phase for each session
fields = '"name","file_date","protocol","phase"'
table_name = 'session'
con = create_connection(sql_path)
cur = con.cursor()
sessions = get_sql_rows_by_fields(cur, table_name, fields)
name_sessions = np.asarray([ii[0] for ii in sessions])
date_sessions = np.asarray([ii[1] for ii in sessions])
prot_sessions = np.asarray([ii[2] for ii in sessions])
phase_sessions = np.asarray([ii[3] for ii in sessions])
date_ephys = np.asarray([ii[1] for ii in rows])
date_sessions = [ii.replace('-','') for ii in date_sessions]
protocol_phase_batch = [get_protocol_phase_BS(irow,name_sessions,date_sessions,prot_sessions,phase_sessions) for irow in rows]
d_ = pd.DataFrame(rows) # how to assign columen names!
d_.columns = ['name','date','brainReg','cluId','cluGro','single_unit_mat','contam_rate','isi_viol','presence_ratio','firing_rate']
protocol_= [pr[0] for pr in protocol_phase_batch]
d_['protocol']= [pr[0] for pr in protocol_phase_batch]
d_['phase']= [pr[1] for pr in protocol_phase_batch]
d_['batch']= [pr[2] for pr in protocol_phase_batch]
categories = [  ['BeliefState_Task1',1,'Batch1_Task1'],
                ['BeliefState_Task1',2,'Batch2_Task1'],
                ['BeliefState_Task2',2,'Batch2_Task2'],
                ['BeliefState_SwtichDay',2,'Batch2_Switch']]
brain_reg_interest = ['IL','PL','ORB','MOp','MOs','ACB','CP','STR','AON','MOB',"FRP",'ACA']
plot_metr = ['contam_rate','isi_viol','presence_ratio','firing_rate']
thresholds = [.5,.7]

ic,ib  = 2,3
icat = categories[ic]
brain_reg  = brain_reg_interest[ib]

id_cat = np.intersect1d(np.argwhere(np.asarray(d_['phase']== icat[0])==True),np.argwhere(np.asarray(d_['batch']== icat[1])==True))
id_units = np.argwhere(np.asarray([brain_reg in bb for bb in d_.brainReg]) == True)
id_regioncategory = np.intersect1d(id_cat,id_units)
isi_v = np.asarray(d_['isi_viol'][id_regioncategory],dtype=float)
pres_r =np.asarray(d_['presence_ratio'][id_regioncategory],dtype=float)
id_pass = np.intersect1d(np.argwhere(isi_v<thresholds[0]),np.argwhere(pres_r>thresholds[1]))
rows_pass = id_regioncategory[id_pass]
sel_sessions = d_['date'].to_numpy()[rows_pass]
sel_mice =  d_['name'].to_numpy()[rows_pass] 
sel_neurons = d_['single_unit_mat'].to_numpy()[rows_pass]  
sel_neurons_ids = d_['cluId'].to_numpy()[rows_pass]  
n_sessions  = np.unique(sel_sessions)
#%%

for ib in  [2]:#range(2,len(brain_reg_interest)):
    for ic in  [0]:#range(3,len(categories)):
        icat = categories[ic]
        brain_reg  = brain_reg_interest[ib]

        id_cat = np.intersect1d(np.argwhere(np.asarray(d_['phase']== icat[0])==True),np.argwhere(np.asarray(d_['batch']== icat[1])==True))
        id_units = np.argwhere(np.asarray([brain_reg in bb for bb in d_.brainReg]) == True)
        id_regioncategory = np.intersect1d(id_cat,id_units)
        isi_v = np.asarray(d_['isi_viol'][id_regioncategory],dtype=float)
        pres_r =np.asarray(d_['presence_ratio'][id_regioncategory],dtype=float)
        id_pass = np.intersect1d(np.argwhere(isi_v<thresholds[0]),np.argwhere(pres_r>thresholds[1]))
        rows_pass = id_regioncategory[id_pass]
        sel_sessions = d_['date'].to_numpy()[rows_pass]
        sel_mice =  d_['name'].to_numpy()[rows_pass] 
        sel_neurons = d_['single_unit_mat'].to_numpy()[rows_pass]  
        sel_neurons_ids = d_['cluId'].to_numpy()[rows_pass]  
        n_sessions  = np.unique(sel_sessions)
        print(n_sessions,brain_reg)

        psth_resolution = 0.01
        smoothing_time_const  = .02
        t = np.arange(0,2.5*smoothing_time_const,psth_resolution)
        smoothing_kernel     = smoothing_func(t,smoothing_time_const)
        smoothing_kernel     = smoothing_kernel / sum(smoothing_kernel)
        sm_length = len(smoothing_kernel) 
#% %
        for isess in [0]:#range(1,len(n_sessions)):
            id_this = np.argwhere(sel_sessions==n_sessions[isess])
            n_neurons_this_sess = len(id_this)
            if n_neurons_this_sess>20:
                for (id_neuron_this_sess,id_count) in zip(id_this,np.arange(n_neurons_this_sess)):
                    id_neuron_this_sess = id_neuron_this_sess[0]        
                    irow = rows_pass[id_neuron_this_sess]
                    mouse = sel_mice[id_neuron_this_sess]
                    this_session = sel_sessions[id_neuron_this_sess]
                    this_neuron = sel_neurons[id_neuron_this_sess]
                    this_neuron_id = sel_neurons_ids[id_neuron_this_sess]
                    print('Unit # :' + str(this_neuron_id) + ' || mouse: '+  mouse + ' || session: ' + this_session + '||'  + str(id_count) + ' of '+ str(n_neurons_this_sess))
                    sp_dir = os.path.join(nas_drive,'Sandra','Extracted_Data',protocol_name,'Spikes',mouse,'SingleUnitData')
                    sp_mat = loadmat(os.path.join(sp_dir,this_neuron))
                    sp = sp_mat['responses']['spikes']
                    odor_on = sp_mat['events']['odorOn'] 
                    rew_on =  sp_mat['events']['rewardOn']
                    n_trials = len(sp_mat['events']['trialType'])
                    trial_types = sp_mat['events']['trialType']
                    
                    delays = rew_on-odor_on
                    if np.sum(delays)==0:
                        delays = sp_mat['events']['rewardOnPerTrial']-sp_mat['events']['odorOnPerTrial']
                        rew_on = odor_on+delays
                    iti = odor_on[1:]-rew_on[:-1]
                    inter_trial_interval = np.round(np.percentile(iti[iti<40] ,99))
                    post_odor =  rew_on - odor_on
                    post_odor_interval = np.round(np.max(post_odor))
                    tr_win = [-inter_trial_interval/2, post_odor_interval+inter_trial_interval ] # re -odor
                    st_time_s = odor_on[0] 
                    st_time_e = odor_on[-1] 
                    psth_length = np.int0(   np.ceil((tr_win[1] - tr_win[0]) / psth_resolution) )
                    total_session_length =  np.int0(   np.ceil((tr_win[1]+odor_on[-1] - (tr_win[0]+odor_on[0])) / psth_resolution) )
                    
                    if id_count==0:
                        sp_bins = np.zeros((total_session_length+1,n_neurons_this_sess))
                        psth_bins_full = np.zeros((total_session_length+1,n_neurons_this_sess))
                        psth_zsc_full = np.zeros((total_session_length+1,n_neurons_this_sess))
                        sp_bins_zsc = np.zeros((total_session_length+1,n_neurons_this_sess))
                        input_full = np.zeros((total_session_length+1,2))

                    # For total session:            
                    win_indices = (sp > st_time_s + tr_win[0])*(sp < st_time_e + tr_win[1])
                    sp_times_full = sp[win_indices]
                    psth_indices_full = np.int0(   np.round((sp_times_full - st_time_s) / psth_resolution) - tr_win[0] / psth_resolution)        
                    ps_full = np.zeros((total_session_length+1))
                    sp_full = np.zeros((total_session_length+1))
                    for i_sp in np.arange(len(psth_indices_full)):
                        sp_full[psth_indices_full[i_sp]] += 1
                        tbs = psth_indices_full[i_sp] # time bin start
                        tbe = min(tbs+sm_length-1, total_session_length)
                        ps_full[tbs:tbe] += smoothing_kernel[:tbe-tbs]
                    sp_bins[:,id_count] = sp_full
                    psth_bins_full[:,id_count] = ps_full

                cue_indices_full = np.int0(np.round((odor_on - st_time_s) / psth_resolution) - tr_win[0] / psth_resolution)      
                rew_indices_full = np.int0(np.round((rew_on - st_time_s) / psth_resolution) - tr_win[0] / psth_resolution)      
                input_full[cue_indices_full,0] =1
                input_full[rew_indices_full,1] =1
#%%
# initialize: 
input_full = np.zeros((total_session_length+1,2))
bins_cue = np.zeros((total_session_length+1))
bins_reward = np.zeros((total_session_length+1))
bins_cue_1 = np.zeros((total_session_length+1))
bins_cue_2 = np.zeros((total_session_length+1))
bins_cue_3 = np.zeros((total_session_length+1))
bins_cue_4 = np.zeros((total_session_length+1))
bins_cue_type = np.zeros((total_session_length+1))
bins_reward_type = np.zeros((total_session_length+1))
bins_trial_numb =  np.zeros((total_session_length+1))
cumcount =  np.zeros((total_session_length+1))
# Fill in with events: 
input_full[get_indeces(odor_on,st_time_s,tr_win,psth_resolution),0] =1
input_full[get_indeces(rew_on,st_time_s,tr_win,psth_resolution),1] =1
bins = np.linspace(0,(len(input_full)-1)*psth_resolution,len(input_full))
bins_cue_1[get_indeces(odor_on[trial_types==1],st_time_s,tr_win,psth_resolution)] =1
bins_cue_2[get_indeces(odor_on[trial_types==2],st_time_s,tr_win,psth_resolution)] =1
bins_cue_3[get_indeces(odor_on[trial_types==3],st_time_s,tr_win,psth_resolution)] =1
bins_cue_4[get_indeces(odor_on[trial_types==4],st_time_s,tr_win,psth_resolution)] =1
bins_cue[get_indeces(odor_on,st_time_s,tr_win,psth_resolution)] =1
bins_reward[get_indeces(rew_on,st_time_s,tr_win,psth_resolution)] =1
end_trials = np.concatenate((odor_on-1,[bins[-1]]))
start_trials = odor_on-1

for i_trial in np.arange(n_trials):
    ids = get_indeces(start_trials[i_trial],st_time_s,tr_win,psth_resolution)
    ide = get_indeces(end_trials[i_trial+1],st_time_s,tr_win,psth_resolution)
    bins_reward_type[ids:ide] =delays[i_trial] 
    bins_trial_numb[ids:ide] = i_trial
    bins_cue_type[ids:ide] =trial_types[i_trial] 
    cumcount[ids:ide] = np.arange(ids,ide)-ids
#%%
np.sum(bins_reward-bins_cue)
#%%
this_neuron_ids = sel_neurons_ids[id_this]
data_pd = dict()
keys_names =['bins', 'bins_cue_type', 'bins_reward', 'bins_cue', 'bins_reward_type','bins_trial_numb', 'bins_cue_1', 'bins_cue_2', 'bins_cue_3', 'bins_cue_4','cumcount'] 
list_keys =[bins, bins_cue_type, bins_reward, bins_cue, bins_reward_type,bins_trial_numb, bins_cue_1, bins_cue_2, bins_cue_3, bins_cue_4,cumcount] 
for (ik,im)  in zip(keys_names,list_keys):
    data_pd[ik] = im
for (id_this_neuron,icount) in zip(this_neuron_ids,range(len(id_this))):
    kn = 'neuron_' + str(id_this_neuron[0])
    data_pd[kn] = psth_bins_full[:,icount]
    kn = 'spc_n_' + str(id_this_neuron[0])
    data_pd[kn] = sp_bins[:,icount]
#%%
xdf = pd.DataFrame(data_pd)

# z score of the firing rate across recording for each neuron
neurons = list(xdf.filter(like='neuron_'))
xdf_neurons = xdf[neurons]
xdf[xdf_neurons.add_suffix("_zscore").columns] = ((xdf_neurons - xdf_neurons.mean())/xdf_neurons.std())
neurons_zscore = list(xdf.filter(like='_zscore'))
neurons = list(xdf.filter(like='spc_n_'))
xdf_neurons = xdf[neurons]
xdf[xdf_neurons.add_suffix("_Zscore").columns] = ((xdf_neurons - xdf_neurons.mean())/xdf_neurons.std())
neurons_zscore_spc = list(xdf.filter(like='_Zscore'))
#%%
fig ,ax = plt.subplots(figsize=(8,8))
xdf_cue = xdf.loc[xdf["bins_cue_type"]==1]
fig, neurons_mean = evaluate_drift(xdf_cue, neurons, interval=[0,20], fig=fig, label="[0,20]") # mostly in PC1
fig, neurons_mean = evaluate_drift(xdf_cue, neurons, interval=[33,41], fig=fig, label="[33,41]") # mostly in PC1
fig, neurons_mean = evaluate_drift(xdf_cue, neurons, interval=[43,50], fig=fig, label="[43,50]") # mostly in PC1
fig.show()
#%%
# X_drift = xdf.copy()
X_drift = pd.DataFrame(data_pd)
neurons_zscore_res = [s + "_res" for s in neurons_zscore]
y_res = pd.DataFrame(columns=neurons_zscore_res)
cumcount_mean = xdf.groupby(["bins_cue_type", "cumcount"]).transform("mean")[neurons_zscore] # mean across trials at each time step
drift = (xdf[neurons_zscore] - cumcount_mean) # activity with no within-trial variability
drift["bins_trial_numb"] = xdf["bins_trial_numb"]
trials_mean = drift.groupby(["bins_trial_numb"]).mean()[neurons_zscore] # mean activity per trial for each neuron
_, _, v = linalg.svd(trials_mean, full_matrices=False)
no_drift_space = v[2:,:] # I tried both because wasn't sure which dim are the eigenvectors --> rows
#no_drift_space = v[:,2:]
y_res[neurons_zscore_res] = drift[neurons_zscore] @ (no_drift_space.T @ no_drift_space)
y_res[neurons_zscore_res] = y_res[neurons_zscore_res] + cumcount_mean.add_suffix("_res")
X_new = pd.concat([X_drift, y_res], axis=1)
for neuron in neurons_zscore_res:
    X_new[neuron] = pd.to_numeric(X_new[neuron])
plot_drift_set(X_new, neurons_zscore_res)
#%%
neurons_zscore_res = [s + "_res" for s in neurons_zscore_spc]
y_res = pd.DataFrame(columns=neurons_zscore_res)
cumcount_mean = xdf.groupby(["bins_cue_type", "cumcount"]).transform("mean")[neurons_zscore_spc] # mean across trials at each time step
drift = (xdf[neurons_zscore_spc] - cumcount_mean) # activity with no within-trial variability
drift["bins_trial_numb"] = xdf["bins_trial_numb"]
trials_mean = drift.groupby(["bins_trial_numb"]).mean()[neurons_zscore_spc] # mean activity per trial for each neuron
_, _, v = linalg.svd(trials_mean, full_matrices=False)
no_drift_space = v[2:,:] # I tried both because wasn't sure which dim are the eigenvectors --> rows
#no_drift_space = v[:,2:]
y_res[neurons_zscore_res] = drift[neurons_zscore_spc] @ (no_drift_space.T @ no_drift_space)
y_res[neurons_zscore_res] = y_res[neurons_zscore_res] + cumcount_mean.add_suffix("_res")
X_new = pd.concat([X_drift, y_res], axis=1)
for neuron in neurons_zscore_res:
    X_new[neuron] = pd.to_numeric(X_new[neuron])
plot_drift_set(X_new, neurons_zscore_res)
#%%
# Store data (serialize)
data = dict()
data['mouse'] = mouse
data['brain_reg'] = brain_reg
data['session'] = n_sessions[isess]
data['category'] = icat
data['xdf'] = xdf
fname = mouse + '_' + n_sessions[isess] + '_'+ brain_reg + '_X_df.pickle'
fullname = os.path.join(sp_dir_pkl,fname)
print('saving in: --------------------' + fullname + '--------------------')

data = dict()
data['mouse'] = mouse
data['brain_reg'] = brain_reg
data['session'] = n_sessions[isess]
data['category'] = icat
data['xdf'] = X_new
fname = mouse + '_' + n_sessions[isess] + '_'+ brain_reg + '_X_df_res.pickle'
fullname = os.path.join(sp_dir_pkl,fname)
print('saving in: --------------------' + fullname + '--------------------')
with open(fullname, 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
print('saved. --------------------' + fname + '--------------------')