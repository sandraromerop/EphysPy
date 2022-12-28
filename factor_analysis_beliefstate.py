#%%

from elephant.gpfa import GPFA
import numpy as np
import numpy as np
from scipy.integrate import odeint
import quantities as pq
import neo
import pandas as pd
import matplotlib.pyplot as plt

from utils import *
from plots import *
from sql_utils import *
import cluster_metrics.metrics as metrics
from cluster_metrics.params import QualityMetricsParams


from elephant.spike_train_generation import inhomogeneous_poisson_process
from neo.core import Segment, SpikeTrain, AnalogSignal
#%%

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

# %%

ib = 2
ic= 0
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
isess = 0
id_this = np.argwhere(sel_sessions==n_sessions[isess])
n_neurons_this_sess = len(id_this)
max_trials=150
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
        if id_count==0:
            list_sp = [list() for ii in range(len(odor_on))]
            list_sp2 = [list() for ii in range(len(odor_on))]
        sp = sp_mat['responses']['spikes']
        odor_on = sp_mat['events']['odorOn'][:max_trials]
        rew_on =  sp_mat['events']['rewardOn'][:max_trials]
        n_trials = len(sp_mat['events']['trialType'][:max_trials])
        trial_types = sp_mat['events']['trialType'][:max_trials]
        if id_count==0:
            list_sp = [list() for ii in range(len(odor_on))]
        delays = rew_on-odor_on
        
        if np.sum(delays)==0:
            delays = sp_mat['events']['rewardOnPerTrial'][:max_trials]-sp_mat['events']['odorOnPerTrial'][:max_trials]
            rew_on = odor_on+delays
        
        iti = odor_on[1:]-rew_on[:-1]
        post_odor =  rew_on - odor_on
        post_odor_interval = np.round(np.max(post_odor))
        inter_trial_interval = np.round(np.percentile(iti[iti<40] ,99))
        tr_win = [-inter_trial_interval/3, post_odor_interval+inter_trial_interval ]
        sp_list = []
        for itrial in range(len(odor_on)):
            id_sp = np.argwhere((sp>(odor_on[itrial]+tr_win[0]))*(sp<(odor_on[itrial]+tr_win[1])))
            sp_trial = sp[id_sp].flatten()
            sp_trial = sp_trial-(odor_on[itrial]+tr_win[0])
            t_stop = tr_win[1]-tr_win[0]
            t_start = 0
            if len(sp_trial)==0 and itrial>0:
                train0 = list_sp2[itrial-1]#SpikeTrain(times=[0], units='sec', t_stop=t_stop,t_start=t_start)
            elif  len(sp_trial)==0 and itrial==0:
                itt = itrial+1
                while len(sp_trial)==0:
                    id_sp = np.argwhere((sp>(odor_on[itt]+tr_win[0]))*(sp<(odor_on[itt]+tr_win[1])))
                    sp_trial = sp[id_sp].flatten()
                    sp_trial = sp_trial-(odor_on[itt]+tr_win[0])
                    itt=itt+1
            else:
                train0 = SpikeTrain(times=sp_trial, units='sec', t_stop=t_stop,t_start=t_start)
            list_sp2[itrial].append(train0)
            
#%%
# --- list of num trials 
#       --- each element is a list of nb of neurons
#           ---  each element is a 2 element list of [idneuron, spiketrain object]
list_sp2[0][0].t_stop
a=[[ll.t_stop for ll in trial_l] for trial_l in list_sp2]
#%%
a =[[np.float64(ll) for ll in trial_l] for trial_l in a]
#%%
tstops = np.asarray(a)
iddisc = np.argwhere(tstops>(tr_win[1]-tr_win[0]))
np.unique(iddisc[:,0])
# ValueError: structure of the spiketrains is not correct: 
# 0-axis should be trials, 
# 1-axis neo.SpikeTrain
# 2-axis spike times
#%%

# specify fitting parameters
bin_size = 20 * pq.ms
latent_dimensionality = 2
gpfa_2dim = GPFA(bin_size=bin_size, x_dim=latent_dimensionality)
gpfa_2dim.fit(list_sp2)
# print(gpfa_2dim.params_estimated.keys())
#%%
# trajectories = gpfa_2dim.transform(spiketrains_oscillator[num_trials//2:])
