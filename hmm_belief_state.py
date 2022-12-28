
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

#%%
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
print([n_sessions, brain_reg,len(sel_neurons)])
#%%
X_df_folder ='/Volumes/GoogleDrive-116748251018268574178/My Drive/BeliefStateData/Sandra/Extracted_Data/BeliefState/Spikes/SingleUnitPk/BS005_20200922_ORB_X_df_res.pickle'
with open(X_df_folder, 'rb') as f:
    dat = pickle.load(f)
xdf = dat['xdf']

cue_type_onset = xdf['bins_cue_type']*xdf['bins_cue']
if np.sum(xdf['bins_cue_2']==0):
    xdf['bins_cue_1'][:]=0.
    xdf['bins_cue_1'][cue_type_onset==1]=1.

    xdf['bins_cue_2'][:]=0.
    xdf['bins_cue_2'][cue_type_onset==3]=1.

    xdf['bins_cue_3'][:]=0.
    xdf['bins_cue_3'][cue_type_onset==5]=1.

    xdf['bins_cue_4'][:]=0.
    xdf['bins_cue_4'][cue_type_onset==8]=1.

ids=2500
ide=15000
plt.plot(xdf['bins_cue_1'][ids:ide])
plt.plot(xdf['bins_cue_2'][ids:ide])
plt.plot(xdf['bins_cue_3'][ids:ide])
plt.plot(xdf['bins_cue_4'][ids:ide])
plt.plot(xdf['bins_reward'][ids:ide],color='black')

#Get trial list 
ng = list(xdf.filter(like='_Zscore'))
ng.append('bins_trial_numb')
sp_trial_list = xdf[ng].groupby('bins_trial_numb').apply(lambda x:np.asarray(x))
sp_trial_list = list([ss[:,:-1] for ss in sp_trial_list])

# ng = list(xdf_res.filter(like='_res'))
# ng.append('bins_trial_numb')
# sp_trial_list_res = xdf_res[ng].groupby('bins_trial_numb').apply(lambda x:np.asarray(x))
# sp_trial_list_res = list([ss[:,:-1] for ss in sp_trial_list_res])

#Together:
inputs_tog_trial_list = xdf[["bins_cue", "bins_reward",'bins_trial_numb']].groupby('bins_trial_numb').apply(lambda x:np.asarray(x))
inputs_tog_trial_list= list([ss[:,:-1] for ss in inputs_tog_trial_list])

#Separate:
inputs_sep_trial_list = xdf[["bins_cue_1",'bins_cue_2','bins_cue_3','bins_cue_4', "bins_reward",'bins_trial_numb']].groupby('bins_trial_numb').apply(lambda x:np.asarray(x))
inputs_sep_trial_list= list([ss[:,:-1] for ss in inputs_sep_trial_list])

X_train, X_test, y_train, y_test = train_test_split(sp_trial_list, \
                                    inputs_sep_trial_list, \
                                    test_size=0.3, random_state=0)
print([len(X_train),X_train[0].shape,y_train[0].shape])
#%%
# First fit an HMM -- to see if it detects discrete states at diff time points within trials : 
# Now create a new HMM and fit it to the data with EM
num_states =[10,11,12,13,14,15] #[2,3,4,5,6,7,8,9,10,11,12,13,14,15] 
N_iters = 20
train_data = X_train
input_data = y_train
obs_dim = train_data[0].shape[1]
input_dim = input_data[0].shape[1]
# list_hmm = []
# list_hmm_lps = []
for (inum,ic) in zip(num_states,range(len(num_states))):
    print('-------- For  nb of states :' + str(inum) + '|| ---------- Model:' + str(ic) + ' of: ' + str(len(num_states)) )
    hmm = ssm.HMM(inum, obs_dim, input_dim, 
            observations="gaussian", 
            transitions="inputdriven")
    hmm_lps = hmm.fit(train_data, inputs=input_data, method="em", num_iters=N_iters)
    list_hmm.append(hmm)
    list_hmm_lps.append(hmm_lps)
#%%
# # Plot the log probabilities of  fit models
for (inum,icount) in zip(num_states,range(len(num_states))):
    hmm_lps = list_hmm_lps[icount]
    plt.plot(hmm_lps, label="EM " + str(inum),color=cm.jet(icount/len(num_states)))
    plt.xlabel("EM Iteration")
    plt.xlim(0, N_iters)
    plt.ylabel("Log Probability")
plt.show()
#%%

# Compute log prob of test data given the input & fit HMM model for each nb of states
__,ax = plt.subplots(1,2,figsize=(6,3))
n_samples = len(X_test)
lps_test = np.zeros((len(num_states), n_samples))
for (inum,ic) in zip(num_states,range(len(num_states))):
    hmm = list_hmm[ic]
    print('-------- For state nb:' + str(inum) + '||----- '+str(ic) + 'of: ' + str(len(num_states)) )
    for itr in range(n_samples):
        # Compute the true log probability of the data, summing out the discrete states
        obs = X_test[itr]
        inpt = y_test[itr]
        time_bins = len(inpt)
        test_lp = hmm.log_probability(obs, inputs=inpt)
        lps_test[ic,itr] = test_lp
lps_test[lps_test<0]=np.nan

ax[1].errorbar(num_states,np.nanmean(lps_test,axis=1),
        yerr=np.nanstd(np.nanmean(lps_test,axis=1))/(np.sqrt(n_samples)),
        color='blue')
lps_a =np.asarray(list_hmm_lps)
ax[0].errorbar(num_states,np.nanmean(lps_a[:len(num_states),:],axis=1),
        yerr=np.nanstd(np.nanmean(lps_a[:len(num_states),:],axis=1))/(np.sqrt(n_samples)),
        color='grey')   
title_(ax[0],'Train data (log prob)')
title_(ax[1],'Test data (log prob)')
plot_config(ax[0],'num states','log(P)',14,False)
plot_config(ax[1],'num states','log(P)',14,False)

#%% Compute correlation between first two pcs of : Model Obs vs Test Obs
from sklearn.decomposition import PCA
__,ax = plt.subplots(figsize=(4,3))

corr_pca_obs = np.zeros((len(num_states), n_samples,3))
for (inum,ic) in zip(num_states,range(len(num_states))):
    hmm = list_hmm[ic]
    print('-------- For state nb:' + str(inum) + '||----- '+str(ic) + ' of: ' + str(len(num_states)) )
    for itr in range(n_samples):
        obs = X_test[itr]
        inpt = y_test[itr]

        pca = PCA(n_components=21)
        pca.fit(obs) 
        x_pca_data = pca.transform(obs)

        pca_test = PCA(n_components=21) 
        test_st, obs_test = hmm.sample(len(inpt), input=inpt)
        pca_test.fit(obs_test) 
        x_pca_test = pca_test.transform(obs_test)
        
        corr_pca_obs[ic,itr,0]=np.corrcoef(x_pca_data[:,0],x_pca_test[:,0])[0,1]
        corr_pca_obs[ic,itr,1]=np.corrcoef(x_pca_data[:,1],x_pca_test[:,1])[0,1]
        corr_pca_obs[ic,itr,2]=np.corrcoef(x_pca_data[:,2],x_pca_test[:,2])[0,1]

ax.errorbar(num_states,np.nanmean(corr_pca_obs[:,:,0],axis=1),
    yerr=np.nanstd(np.nanmean(corr_pca_obs[:,:,0],axis=1))/np.sqrt(n_samples),
    label= 'PC 1')
ax.errorbar(num_states,np.nanmean(corr_pca_obs[:,:,1],axis=1),
    yerr=np.nanstd(np.nanmean(corr_pca_obs[:,:,1],axis=1))/np.sqrt(n_samples),
    label= 'PC 2')
ax.errorbar(num_states,np.nanmean(corr_pca_obs[:,:,2],axis=1),
    yerr=np.nanstd(np.nanmean(corr_pca_obs[:,:,2],axis=1))/np.sqrt(n_samples),
    label= 'PC 3')
plot_config(ax,'nb states','correlation PCs (train vs test)',14,True)


#%%  cues separated

cuetypes = [np.argwhere(np.sum(ii,axis=0)==1)[0][0] for ii in input_data]
ucues = np.unique(cuetypes)
len_ = np.max([len(ii) for ii in train_data] )
__,ax = plt.subplots(len(num_states),len(ucues),figsize=(15,40))
for (inum,icount) in zip(num_states,range(len(num_states))):
    hmm = list_hmm[icount]
    for (ic,icc) in zip(ucues,range(len(ucues))):
        ids = np.argwhere(cuetypes==ic).flatten()
        inf_states=  np.nan*np.ones((len(ids),len_))
        for (ii,itr) in zip(range(len(ids)),ids):
            obs = train_data[itr]
            inpt = input_data[itr]
            st = hmm.most_likely_states(obs, input=inpt)
            inf_states[ii,:len(st)] =st
        ax[icount,icc].imshow(inf_states,aspect='auto',cmap='Dark2')
        # ax[icount,icc].plot(np.nanmean(inf_states,axis=0),color=cm.jet(icount/len(num_states)))
        ax[icount,icc].set_xlim((0,1000))
        plot_config(ax[icount,icc],'','trials',12,False)
        title_(ax[icount,icc],'Nb states ' + str(inum) + ' Cue : ' + str(ic))

cuetypes = [np.argwhere(np.sum(ii,axis=0)==1)[0][0] for ii in input_data]
ucues = np.unique(cuetypes)
len_ = np.max([len(ii) for ii in train_data] )
__,ax = plt.subplots(len(num_states),len(ucues),figsize=(15,40))
for (inum,icount) in zip(num_states,range(len(num_states))):
    hmm = list_hmm[icount]
    for (ic,icc) in zip(ucues,range(len(ucues))):
        ids = np.argwhere(cuetypes==ic).flatten()
        inf_states=  np.nan*np.ones((len(ids),len_))
        for (ii,itr) in zip(range(len(ids)),ids):
            obs = train_data[itr]
            inpt = input_data[itr]
            st = hmm.most_likely_states(obs, input=inpt)
            inf_states[ii,:len(st)] =st
        tt = np.arange(0,(1000)*psth_resolution,step=psth_resolution)
        ax[icount,icc].plot(tt,np.nanmean(inf_states[:,:1000],axis=0),color=cm.jet(icount/len(num_states)))
        ax[icount,icc].set_xlim((0,10))
        plot_config(ax[icount,icc],'','trials',12,False)
        title_(ax[icount,icc],'Nb states ' + str(inum) + ' Cue : ' + str(ic))
#%%
psth_resolution= .01
from scipy import stats
# All cues together
len_ = np.max([len(ii) for ii in train_data] )
__,ax = plt.subplots(len(num_states),2,figsize=(10,40))
for (inum,icount) in zip(num_states,range(len(num_states))):
    hmm = list_hmm[icount]
    inf_states=  np.nan*np.ones((len(ids),len_))
    for (ii,itr) in zip(range(len(ids)),ids):
        obs = train_data[itr]
        inpt = input_data[itr]
        st = hmm.most_likely_states(obs, input=inpt)
        inf_states[ii,:len(st)] =st
    ax[icount,0].imshow(inf_states,aspect='auto',cmap='Dark2')
    tt = np.arange(0,(1000)*psth_resolution,step=psth_resolution)
    ax[icount,1].plot(tt,np.nanmean(inf_states[:,0:1000],axis=0),color=cm.jet(icount/len(num_states)))
    ax[icount,0].set_xlim((0,1000))
    ax[icount,1].set_xlim((0,10))
    plot_config(ax[icount,0],'','trials',14,False)
    plot_config(ax[icount,1],'nb samples','mean state',10,False)
    title_(ax[icount,0],'Nb states ' + str(inum) )
# Plot:
#  -baseline log transition probabilities (the log of the state-transition matrix) 
# - input weights $w$. 
plt.figure(figsize=(8, 4))

vlim = max(abs(true_hmm.transitions.log_Ps).max(),
           abs(true_hmm.transitions.Ws).max(),
           abs(hmm.transitions.log_Ps).max(),
           abs(hmm.transitions.Ws).max())

plt.subplot(141)
plt.imshow(true_hmm.transitions.log_Ps, vmin=-vlim, vmax=vlim, cmap="RdBu", aspect=1)
plt.xticks(np.arange(num_states))
plt.yticks(np.arange(num_states))
plt.title("True\nBaseline Weights")
plt.grid(b=None)

plt.subplot(142)
plt.imshow(true_hmm.transitions.Ws, vmin=-vlim, vmax=vlim, cmap="RdBu", aspect=num_states/input_dim)
plt.xticks(np.arange(input_dim))
plt.yticks(np.arange(num_states))
plt.title("True\nInput Weights")
plt.grid(b=None)
