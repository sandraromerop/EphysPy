#%%
import sys 
import os
import glob
import pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from plots import*
from sql_utils import *
from utils import *
from cluster_metrics.params import QualityMetricsParams


params = vars(QualityMetricsParams())
groot_drive = '/Volumes/GoogleDrive-116748251018268574178/My Drive/Data/'
protocol_name = 'BeliefState'
nas_drive = os.path.join(groot_drive,protocol_name,'BeliefStateData')
save_path = os.path.join(groot_drive,protocol_name,'Spikes','Extracted')
fig_path = os.path.join(groot_drive,protocol_name,'Figures','mutual_info')
analysis_path = os.path.join(groot_drive,protocol_name,'Analysis','mutual_info')

table_name = 'neuron'
sql_path = os.path.join(nas_drive,'Sandra','db_','session_log_config.sqlite')

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
cue_on_t =2000
pre_cue_t = 500
post_rew_t = 500
ms_per_bin = 50
win_range = [0, 10000]
nshuff = 100
tr_common = [1,2,3,4]
#  ORB Batch2_Task1 20220524 BS007
# 
for ib in [len(brain_reg_interest)-1]:#range(2,len(brain_reg_interest)):
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
            session_n = sel_sessions[id_this[0]][0]
            n_neurons_this_sess = len(id_this)
            mouse = sel_mice[id_this[0]][0]
            this_neuron = sel_neurons[id_this[0]][0]
            results = dict()
            results['info']= dict()
            results['info']['brain_reg'] = brain_reg
            results['info']['category_task'] = icat
            results['info']['mouse'] = mouse
            results['info']['session_n'] = session_n

            results['parameters']= dict()
            results['parameters']['win_range']=win_range
            results['parameters']['cue_on_t']=cue_on_t
            results['parameters']['pre_cue_t']=pre_cue_t
            results['parameters']['post_rew_t']=post_rew_t
            results['parameters']['ms_per_bin']=ms_per_bin
            results['parameters']['nshuff']=nshuff

            print('Processing for MI for: ' + brain_reg+ ' ' + icat[-1] +' ' +session_n + ' ' +  mouse )
            
            sp_dir = os.path.join(nas_drive,'Sandra','Extracted_Data',protocol_name,'Spikes',mouse,'SingleUnitData')
            sp_mat = loadmat(os.path.join(sp_dir,this_neuron))
            odor_on = sp_mat['events']['odorOn'] 
            if ici==1 or ici==2 or ici ==3:
                rew_on =  sp_mat['events']['rewardOn']+1
            else:
                rew_on =  sp_mat['events']['rewardOn']
            delays = rew_on-odor_on
            if np.sum(delays)==0:
                delays = sp_mat['events']['rewardOnPerTrial']-sp_mat['events']['odorOnPerTrial']
            tr_types = sp_mat['events']['trialType']
            
            if sum(tr_types==8)>0:
                utr = np.unique(tr_types)
                tr_types2 = np.zeros_like(tr_types)
                tr_types2[:] =tr_types
                for (uu,vv) in zip(utr,tr_common):
                    tr_types2[tr_types==uu] = vv
                tr_types[:] = tr_types2
            delays_common = []
            for (tr_type,iit) in zip(tr_common,np.arange(len(tr_common))):
                delay = np.floor(np.max(np.unique(delays[tr_types==tr_type]))*10)/10
                delays_common.append(delay)
            cell_range = range(0,n_neurons_this_sess )
            spbins_all =[]
            print('loading spike bins ...')
            for (tr_type,iit) in zip(tr_common,np.arange(len(tr_common))):
                res_path =  os.path.join(save_path,session_n+ '_' + brain_reg + '_'+ mouse,'trial_type_'+str(tr_type),'results')
                path_to_data = os.path.join(save_path,session_n+ '_' + brain_reg + '_'+ mouse,'trial_type_' + str(tr_type))
                file_list = glob.glob(os.path.join(res_path,"*.json"))
                
                cell_ids = np.unique([ii.split('/')[-1].split('.json')[0].split('_')[-1] for ii in file_list])
                data_processor = DataProcessor(path_to_data, cell_range, win_range)
                spbins = data_processor.spikes_binned
                spbins = [spbins[ii] for ii in range(len(spbins))]
                spbins_all.append(spbins)
            
            #%% Calculate x variables 
            max_delay = 1000*np.round(10*np.max(delays))/10
            isi_bins_max = np.floor(np.linspace(1,14,np.int0(max_delay/ms_per_bin)))
            isi_bins_max[-1] = 13
            new_len_t= np.int0((win_range[-1]-win_range[0])/ms_per_bin)
            new_time_bins = np.linspace(0,14,new_len_t)

            ntrials_per_t= [np.int0(np.nanmin([spbins_all[itr][icell].shape[0] for icell in range(len(spbins_all[itr])) ])) for itr in range(len(spbins_all))]
            ntrials_min = np.nanmin(ntrials_per_t)
            n_neurons = len(spbins_all[0])
            sp_new_mat =  np.zeros((n_neurons,len(tr_common),ntrials_min,new_len_t))
            ybins_mat =   np.zeros((len(tr_common),ntrials_min,new_len_t))
            ymicro_mat ,ytypes_mat,ymicro_fortype, ytype_formicro= [],[],[],[]
            
            sp_micro_mat, sp_types_mat,sp_micro_mat_sh =np.zeros((n_neurons,0)), np.zeros((n_neurons,0)),np.zeros((n_neurons,0,nshuff))
            for (tr_type,iit) in zip(tr_common,np.arange(len(tr_common))):
                print('For trial type' + str(iit)+' of ' +str(len(tr_common)))
                spbins = spbins_all[iit]
                delays_trial_types=delays[tr_types ==tr_type]
                for itr in range(np.min([len(spbins[0]),ntrials_min])):
                    
                    st_bin = cue_on_t-pre_cue_t
                    en_bin = cue_on_t + ( np.round(delays_trial_types[itr]*10)/10)*1000 + post_rew_t
                    bin_len = np.int0((en_bin-st_bin)/ms_per_bin)
                    time_bin = np.linspace(-pre_cue_t, delays_trial_types[itr]*1000 + post_rew_t,bin_len)
                    time_micro = np.zeros_like(time_bin)
                    time_micro[time_bin<0] = 0
                    time_micro[time_bin>( np.round(delays_trial_types[itr]*10)/10)*1000] = 14
                    isi_id = np.argwhere((time_bin>0)*(time_bin< (np.round(delays_trial_types[itr]*10)/10)*1000)).flatten()
                    time_micro[isi_id]=isi_bins_max[:len(isi_id)]
                    ybins_mat[iit,itr,:] = np.round(new_time_bins) # saving uniformly spaced time bins
                    ymicro_mat = np.concatenate((ymicro_mat,time_micro),axis=0)
                    type_bins = tr_type*np.ones((len(isi_id)))
                    type_bins_formicro = tr_type*np.ones((len(time_micro)))
                    type_bins_formicro[time_bin<0] = 0
                    type_bins_formicro[time_bin>( np.round(delays_trial_types[itr]*10)/10)*1000] = 0
                    ytypes_mat = np.concatenate((ytypes_mat,type_bins),axis=0)
                    ymicro_fortype = np.concatenate((ymicro_fortype,time_micro[isi_id]),axis=0)
                    ytype_formicro = np.concatenate((ytype_formicro,type_bins_formicro),axis=0)
                    sp_micro = np.zeros((n_neurons,len(time_micro)))
                    sp_micro_sh = np.zeros((n_neurons,len(time_micro),nshuff))
                    sp_types = np.zeros((n_neurons,len(type_bins)))
                    for icell in range(n_neurons):
                        spcell = spbins[icell][itr]
                        
                        time_old = np.linspace(-cue_on_t,spcell.shape[0]-cue_on_t,spcell.shape[0])
                        time_new = np.linspace(-cue_on_t,spcell.shape[0]-cue_on_t,np.int0(spcell.shape[0]/ms_per_bin))
                        event_times = time_old[spcell>0]
                        event_times = event_times[(event_times>=time_bin[0])*(event_times<=time_bin[-1])]
                        sp_new = np.bincount(np.digitize(event_times, time_bin, right = 1))
                        sp_new1 = np.zeros((len(time_bin)))
                        sp_new1[:len(sp_new)] = sp_new
                        sp_micro[icell,:] =sp_new1 
                        for iter in range(nshuff):
                            nbroll= np.random.randint(1,len(sp_new1)-1)
                            spcell_shuf = np.roll(sp_new1,nbroll)
                            sp_micro_sh[icell,:,iter] =spcell_shuf 
                        sp_types[icell,:] =sp_new1[isi_id]
                    sp_micro_mat = np.concatenate((sp_micro_mat,sp_micro),axis=1)
                    sp_micro_mat_sh= np.concatenate((sp_micro_mat_sh,sp_micro_sh),axis=1)
                    sp_types_mat = np.concatenate((sp_types_mat,sp_types),axis=1)
            sp_types_mat = sp_types_mat.T
            # sp_micro_mat_sh = sp_micro_mat_sh.T
            sp_micro_mat = sp_micro_mat.T

            #% % Test for mutual info between activity and :
            # Representation of micro-states
            # Representation of types of cue (in the ISI)
            # Joint representation of micro -states and type of trials :
            # - ISI micro states are given the type of the cue 
            # - Pre-cue and post-reward micro states are given the type of 0 

            #% % Representation of micro -states
            # probability density of the mouse occupying location x
            # calculated by counting the number of frames the mouse spent in each bin 
            # across trials and normalized to have sum=1
            # $I = \int_x \lambda(x) \log_2 \frac{\lambda(x)}{\lambda}p(x)dx

            # The randomized variables (__R__E or __R__Y) were created by using uniform random 
            # sampling with replacement from the joint distribution of discrete evidence (__E__) 
            # and position (__Y__) values. More specifically, for the __R__E × __Y__ space, 
            # in which __Y__ is the non-randomized dimension, we first found the distribution 
            # of __E__ values present in the data for each __Y__ value. This created 30 separate
            #  __E__ distributions with respect to __Y__. The __R__E value for each frame was 
            # generated by randomly sampling from the sole __E__ distribution that corresponded 
            # to the non-randomized __Y__ value for that frame.

            # Randomize the type_for_micro variable 
            y2d = np.concatenate((ymicro_mat[:,np.newaxis],ytype_formicro[:,np.newaxis]),axis=1)
            um = np.unique(ymicro_mat)
            for (iu,iiu) in zip(um,range(len(um))):
                ids = np.argwhere(ymicro_mat==iu).flatten()
                dist_type = ytype_formicro[ids]
                id_samp = np.random.randint(0,len(dist_type),len(ids))
                y_temp = ytype_formicro[ids][id_samp]
                y2d[ids,1] = y_temp

            nb_neurons = sp_micro_mat.shape[1]
            um = np.unique(ymicro_mat)
            ut = np.unique(ytype_formicro)
            px_micro = np.zeros((len(um)))
            px_micro2d = np.zeros((len(um)*len(ut)))
            px_micro2d_sh = np.zeros((len(um)*len(ut)))
            lambda_micro = np.zeros((len(um),nb_neurons))
            lambda_micro2d = np.zeros((len(um)*len(ut),nb_neurons))
            lambda_micro_sh = np.zeros((len(um),nb_neurons,nshuff))
            lambda_micro2d_sh = np.zeros((len(um)*len(ut),nb_neurons,nshuff))

            lambda_sp = np.nanmean(sp_micro_mat,axis=0)
            int_micro =  np.zeros((len(um),nb_neurons))
            int_micro2d =  np.zeros((len(um)*len(ut),nb_neurons))
            int_micro_sh =  np.zeros((len(um),nb_neurons,nshuff))
            int_micro2d_sh =  np.zeros((len(um)*len(ut),nb_neurons,nshuff))
            ic = 0

            for (iu,iiu) in zip(um,range(len(um))):
                ids = np.argwhere(ymicro_mat==iu).flatten()
                px_micro[iiu] = len(ids)/len(ymicro_mat)
                sp_ = sp_micro_mat[ids,:]
                lambda_micro[iiu,:]= np.nanmean(sp_,axis=0)
                int_micro[iiu,:]=lambda_micro[iiu,:]*np.log2(lambda_micro[iiu,:]/lambda_sp)*px_micro[iiu]
                for (it,iit) in zip(ut,range(len(ut))):
                    ids2d = np.argwhere((ymicro_mat==iu)*(ytype_formicro==it)).flatten()
                    px_micro2d[ic] = len(ids2d)/len(ymicro_mat)
                    sp_ = sp_micro_mat[ids2d,:]
                    lambda_micro2d[ic,:]= np.nanmean(sp_,axis=0)
                    int_micro2d[ic,:] = lambda_micro2d[ic,:]*np.log2(lambda_micro2d[ic,:]/lambda_sp)*px_micro2d[ic]
                    ic+=1

            for iter in range(nshuff):
                y2d = np.concatenate((ymicro_mat[:,np.newaxis],ytype_formicro[:,np.newaxis]),axis=1)
                um = np.unique(ymicro_mat)
                for (iu,iiu) in zip(um,range(len(um))):
                    ids = np.argwhere(ymicro_mat==iu).flatten()
                    dist_type = ytype_formicro[ids]
                    id_samp = np.random.randint(0,len(dist_type),len(ids))
                    y_temp = ytype_formicro[ids][id_samp]
                    y2d[ids,1] = y_temp
                ic = 0
                for (iu,iiu) in zip(um,range(len(um))):
                    ids = np.argwhere(ymicro_mat==iu).flatten()
                    sp_sh = sp_micro_mat_sh[:,ids,iter].T
                    lambda_micro_sh[iiu,:,iter]= np.nanmean(sp_sh,axis=0)
                    int_micro_sh[iiu,:,iter]=lambda_micro_sh[iiu,:,iter]*np.log2(lambda_micro_sh[iiu,:,iter]/lambda_sp)*px_micro[iiu]
                    for (it,iit) in zip(ut,range(len(ut))):
                        ids2d_ = np.argwhere((y2d[:,0]==iu)*(y2d[:,1]==it)).flatten()
                        px_micro2d_sh[ic] = len(ids2d_)/len(ymicro_mat)
                        sp_ = sp_micro_mat[ids2d_,:]
                        lambda_micro2d_sh[ic,:,iter]= np.nanmean(sp_,axis=0)
                        int_micro2d_sh[ic,:,iter] = lambda_micro2d_sh[ic,:,iter]*np.log2(lambda_micro2d_sh[ic,:,iter]/lambda_sp)*px_micro2d_sh[ic]
                        ic+=1

            mi_micro2d = np.nansum(int_micro2d,axis=0)
            mi_micro2d_sh = np.nansum(int_micro2d_sh,axis=0)
            mi_micro = np.nansum(int_micro,axis=0)
            mi_micro_sh = np.nansum(int_micro_sh,axis=0)
            thrs_sig = np.nanstd(mi_micro_sh,axis=1)*2+np.nanmean(mi_micro_sh,axis=1)
            thrs_sig2d= np.nanstd(mi_micro2d_sh,axis=1)*2+np.nanmean(mi_micro2d_sh,axis=1)

            idsort = np.argsort(mi_micro)
            mi_micro_sh = mi_micro_sh[idsort]
            thrs_sig = thrs_sig[idsort]
            mi_micro = mi_micro[idsort]
            idsort = np.argsort(mi_micro2d)

            mi_micro2d = mi_micro2d[idsort]
            thrs_sig2d= thrs_sig2d[idsort]

            
            results['mi_micro'] = mi_micro
            results['mi_micro2d'] = mi_micro2d
            results['mi_micro_sh'] = mi_micro_sh
            results['mi_micro2d_sh'] = mi_micro2d_sh
            # create a binary pickle file 
            fname = os.path.join(analysis_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' + 'mutual_info.pkl')
            f = open(fname,"wb")
            pickle.dump(results,f)
            f.close()
            print('writing .pkl file ...')

            #%%
            fig,ax = plt.subplots(1,2,figsize=(10,7))
            ax[0].plot(np.argwhere(mi_micro>=thrs_sig),np.log((mi_micro[mi_micro>=thrs_sig])),'o',color='red',label='MI with micro-state')
            ax[0].plot(np.argwhere(mi_micro<thrs_sig),np.log((mi_micro[mi_micro<thrs_sig])),'+',color='grey')
            ax[0].set_ylim((-7,1))
            plot_config(ax[0],'cell id','log(MI)',14,True)
            ax[0].set_title((session_n + ' ' +  mouse + ' '+  brain_reg))
            ax[1].plot(np.argwhere(mi_micro2d>=thrs_sig2d),np.log(mi_micro2d[mi_micro2d>=thrs_sig2d]),'o',color='blue',label='MI with micro-state * trial type')
            ax[1].plot(np.argwhere(mi_micro2d<thrs_sig2d),np.log(mi_micro2d[mi_micro2d<thrs_sig2d]),'+',color='grey')
            ax[1].set_ylim((-7,1))
            plot_config(ax[1],'cell id','log(MI)',14,True)
            ax[1].set_title((session_n + ' ' +  mouse + ' '+  brain_reg))
            fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' + 'mutual_info.png'))
            fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' + 'mutual_info.pdf'))

            print('DONE' + brain_reg+ '_' +session_n + '_' +  mouse )