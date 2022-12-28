#%%
import os
import json
import pickle
import numpy as np
import pandas as pd 
from plots import*
from sql_utils import *
from utils import *

groot_drive = '/Volumes/GoogleDrive-116748251018268574178/My Drive/Data/'
protocol_name = 'BeliefState'
nas_drive = os.path.join(groot_drive,protocol_name,'BeliefStateData')
save_path = os.path.join(groot_drive,protocol_name,'Spikes','Extracted')

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
d_ = pd.DataFrame(rows) 
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

for ib in range(3,len(brain_reg_interest)):
    for ici in range(len(categories)):
        icat = categories[ici]
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

        for isess in range(len(n_sessions)):
            id_this = np.argwhere(sel_sessions==n_sessions[isess])
            n_neurons_this_sess = len(id_this)
            sp_info = dict()
            sp_info['mouse'] = sel_mice[id_this]
            sp_info['brain_reg'] = brain_reg
            sp_info['category_task'] = icat
            sp_info['metrics_cluster'] = ['isi_viol','presence_ratio']
            sp_info['metrics_thresholds'] = thresholds
            sp_info['neuron_ids'] = sel_neurons_ids[id_this]
            sp_info['sessions'] = sel_sessions[id_this]

            for (id_neuron_this_sess,id_count) in zip(id_this,np.arange(n_neurons_this_sess)):
            # for (id_neuron_this_sess,id_count) in zip(id_this,np.arange(n_neurons_this_sess)):
                id_neuron_this_sess = id_neuron_this_sess[0]        
                irow = rows_pass[id_neuron_this_sess]
                mouse = sel_mice[id_neuron_this_sess]
                session_n  = sel_sessions[id_neuron_this_sess]
                save_path_n = os.path.join(save_path,session_n)
                if not os.path.isdir(save_path_n):
                    os.mkdir(save_path_n)
                this_session = sel_sessions[id_neuron_this_sess]
                this_neuron = sel_neurons[id_neuron_this_sess]
                this_neuron_id = sel_neurons_ids[id_neuron_this_sess]
                
                print('Unit # :' + str(this_neuron_id) + ' || mouse: '+  mouse + ' || session: ' + this_session + '||'  + str(id_count) + ' of '+ str(n_neurons_this_sess))
                sp_dir = os.path.join(nas_drive,'Sandra','Extracted_Data',protocol_name,'Spikes',mouse,'SingleUnitData')
                sp_mat = loadmat(os.path.join(sp_dir,this_neuron))
                sp = sp_mat['responses']['spikes']
                odor_on = sp_mat['events']['odorOn'] 
                rew_on =  sp_mat['events']['rewardOn']
                delays = rew_on-odor_on
                if np.sum(delays)==0:
                    delays = sp_mat['events']['rewardOnPerTrial']-sp_mat['events']['odorOnPerTrial']
                iti = odor_on[1:]-rew_on[:-1]
                inter_trial_interval = np.round(np.percentile(iti[iti<40] ,99))
                post_odor =  rew_on - odor_on
                post_odor_interval = np.round(np.max(post_odor))
                tr_win = [-2, post_odor_interval+inter_trial_interval ] # re -odor
                st_trials,en_trials = odor_on+tr_win[0],odor_on+tr_win[1]
                sp_times_ms = [list(1000*(sp[(sp<en_trials[ii])*(sp>st_trials[ii])]-st_trials[ii])) for ii in range(len(st_trials))]
                tr_types = sp_mat['events']['trialType']
                tr_common = [1,2,3,4]
                if sum(tr_types==8)>0:
                    utr = np.unique(tr_types)
                    tr_types2 = np.zeros_like(tr_types)
                    tr_types2[:] =tr_types
                    for (uu,vv) in zip(utr,tr_common):
                        tr_types2[tr_types==uu] = vv
                    tr_types[:] = tr_types2

                for vv in tr_common:
                    sp_save = [sp_times_ms[ii[0]] for ii in np.argwhere(tr_types==vv)]
                    id_keep = np.argwhere(np.asarray([len(ii) for ii in sp_save])>0)
                    sp_save = [sp_save[ii[0]] for ii in id_keep]
                    save_path_ = os.path.join(save_path_n,'trial_type_' + str(vv))
                    if not os.path.isdir(save_path_):
                        os.mkdir(save_path_)
                    save_path_ = os.path.join(save_path_n,'trial_type_' + str(vv),'spikes')
                    if not os.path.isdir(save_path_):
                        os.mkdir(save_path_)
                    with open((save_path_ + "/{0}.json").format(id_count), 'w') as f:
                        json.dump(sp_save, f, sort_keys=False, indent=4, separators=(',', ': '))

                save_path_sp = os.path.join(save_path_n,'spikes')
                if not os.path.isdir(save_path_sp):
                    os.mkdir(save_path_sp)
                
                
                with open((save_path_n + "/{0}.json").format(id_count), 'w') as f:
                        json.dump(sp_times_ms, f, sort_keys=False, indent=4, separators=(',', ': '))
            
            fname = os.path.join(save_path_n,brain_reg+ '_' +session_n + '_' +  mouse + '_' + 'spikes_info.pkl')
            f = open(fname,"wb")
            pickle.dump(sp_info,f)
            f.close()
            print('writing .pkl file for session...')
#%%
