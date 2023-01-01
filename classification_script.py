#%%
import os
import glob
import pickle
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from plots import*
from sql_utils import *
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score,ShuffleSplit

groot_drive = '/Volumes/GoogleDrive-116748251018268574178/My Drive/Data/'
protocol_name = 'BeliefState'
nas_drive = os.path.join(groot_drive,protocol_name,'BeliefStateData')
save_path = os.path.join(groot_drive,protocol_name,'Spikes','Extracted')
fig_path = os.path.join(groot_drive,protocol_name,'Figures','classification')
analysis_path = os.path.join(groot_drive,protocol_name,'Analysis','classification')
table_name = 'neuron'
sql_path = os.path.join(nas_drive,'Sandra','db_','session_log_config.sqlite')
fields = '"name","date","brainReg","cluId","cluGro","single_unit_mat","contam_rate","isi_viol","presence_ratio","firing_rate"'
type_data = 'Spikes'
type_dir = 'extracted_su'
con = create_connection(sql_path)
cur = con.cursor()
rows = get_sql_rows_by_fields(cur, table_name, fields)

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
thresholds = [.5,.7]
tr_common = [1,2,3,4]
cue_on_t =2000
pre_cue_t = 500
post_rew_t = 500
ms_per_bin = 25
n_iterations =50
win_range = [0, 10000]
for ib in [11]:#range(2):#range(len(brain_reg_interest)):
    for ici in [2]:#range(3):#range(len(categories)):
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
        print([n_sessions, brain_reg,len(sel_neurons),icat[-1]])
        for isess in range(len(n_sessions)):
            
            id_this = np.argwhere(sel_sessions==n_sessions[isess])
            session_n = sel_sessions[id_this[0]][0]
            n_neurons_this_sess = len(id_this)
            mouse = sel_mice[id_this[0]][0]
            results = dict()
            results['info']= dict()
            results['info']['brain_reg'] = brain_reg
            results['info']['category_task'] = icat
            results['info']['mouse'] = mouse
            results['info']['session_n'] = session_n

            results['parameters']= dict()
            results['parameters']['cue_on_t']=cue_on_t
            results['parameters']['pre_cue_t']=pre_cue_t
            results['parameters']['post_rew_t']=post_rew_t
            results['parameters']['ms_per_bin']=ms_per_bin
            results['parameters']['n_iterations']=n_iterations
            results['parameters']['win_range']=win_range

            this_neuron = sel_neurons[id_this[0]][0]
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
            for (tr_type,iit) in zip(tr_common,np.arange(len(tr_common))):
                
                res_path =  os.path.join(save_path,session_n+ '_' + brain_reg + '_'+ mouse,'trial_type_'+str(tr_type),'results')
                path_to_data = os.path.join(save_path,session_n+ '_' + brain_reg + '_'+ mouse,'trial_type_' + str(tr_type))
                
                file_list = glob.glob(os.path.join(res_path,"*.json"))
                cell_ids = np.unique([ii.split('/')[-1].split('.json')[0].split('_')[-1] for ii in file_list])
                data_processor = DataProcessor(path_to_data, cell_range, win_range)
                spbins = data_processor.spikes_binned
                spbins = [spbins[ii] for ii in range(len(spbins))]
                spbins_all.append(spbins)
            

            # Re bin into 0.010 (10 ms) instead of 1 ms
            # Classification for time bins: only from 1sec pre + cue-reward + 2 sec post reward
            # - This will be different for each trial type: [1000: 2000+delay+2000]
            # Classification for trial types: only from cue-reward
            # - This will be different for each trial type: [2000: 2000+delay]

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
            sp_micro_mat, sp_types_mat =np.zeros((n_neurons,0)), np.zeros((n_neurons,0))
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
                    type_bins = iit*np.ones((len(isi_id)))
                    type_bins_formicro = iit*np.ones((len(time_micro)))
                    ytypes_mat = np.concatenate((ytypes_mat,type_bins),axis=0)
                    ymicro_fortype = np.concatenate((ymicro_fortype,time_micro[isi_id]),axis=0)
                    ytype_formicro = np.concatenate((ytype_formicro,type_bins_formicro),axis=0)
                    sp_micro = np.zeros((n_neurons,len(time_micro)))
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
                        sp_types[icell,:] =sp_new1[isi_id]
                    sp_micro_mat = np.concatenate((sp_micro_mat,sp_micro),axis=1)
                    sp_types_mat = np.concatenate((sp_types_mat,sp_types),axis=1)
            sp_types_mat = sp_types_mat.T
            sp_micro_mat = sp_micro_mat.T
            #% %
            clf = LinearDiscriminantAnalysis()
            cv = ShuffleSplit(n_splits=20, test_size=0.3, random_state=0)
            scs_types = cross_val_score(clf, sp_types_mat, ytypes_mat, cv=cv)
            scs_bins = cross_val_score(clf, sp_micro_mat, ymicro_mat, cv=cv)
            clf = LinearDiscriminantAnalysis()
            cv = ShuffleSplit(n_splits=20, test_size=0.3, random_state=0)
            irand_type= np.random.randint(0,len(ytypes_mat),size=len(ytypes_mat))
            irand_micro= np.random.randint(0,len(ytypes_mat),size=len(ymicro_mat))
            scs_types_rand = cross_val_score(clf, sp_types_mat, ytypes_mat[irand_type], cv=cv)
            scs_bins_rand = cross_val_score(clf, sp_micro_mat, ymicro_mat[irand_micro], cv=cv)
            fig,ax = plt.subplots(figsize=(6,6))
            ax.bar(1,np.nanmean(scs_types),color='red')
            ax.errorbar(1,np.nanmean(scs_types),np.nanstd(scs_types),color='black')
            ax.bar(2,np.nanmean(scs_types_rand),color='grey')
            ax.errorbar(2,np.nanmean(scs_types_rand),np.nanstd(scs_types_rand),color='black')
            ax.bar(3.5,np.nanmean(scs_bins),color='blue')
            ax.errorbar(3.5,np.nanmean(scs_bins),np.nanstd(scs_bins),color='black')
            ax.bar(4.5,np.nanmean(scs_bins_rand),color='grey')
            ax.errorbar(4.5,np.nanmean(scs_bins_rand),np.nanstd(scs_bins_rand),color='black')
            plot_config(ax,'',' classification accuracy',14,False)
            ax.set_title('LDA')
            ax.set_xticks([1,2,3.5,4.5])
            ax.set_xticklabels(['score trial type','shuffle','score time bin','shuffle'])
            fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' + 'LDA_overall_classification_accuracy' '.png'))
            fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' + 'LDA_overall_classification_accuracy' '.pdf'))

            
            results['score_type_lda_cv'] = scs_types
            results['score_type_lda_cv_shuff'] = scs_types_rand
            results['score_micro_lda_cv'] = scs_bins
            results['score_micro_lda_cv_shuff'] = scs_bins_rand

            #% %
            
            sc_cv = np.zeros((n_iterations,len(np.unique(ytypes_mat))))
            sc_cv_sh = np.zeros((n_iterations,len(np.unique(ytypes_mat))))
            sc_cv_b = np.zeros((n_iterations,len(np.unique(ymicro_mat))))
            sc_cv_b_sh = np.zeros((n_iterations,len(np.unique(ymicro_mat))))
            for it in range(n_iterations):
                X_train, X_test, y_train, y_test = train_test_split(sp_types_mat, ytypes_mat, test_size=0.5)
                clf = LinearDiscriminantAnalysis()
                clf.fit(X_train, y_train)
                utype= np.unique(y_test)
                for (iu,iiu) in zip(utype,range(len(utype))):
                    ids = np.argwhere(y_test==iu).flatten()
                    sc = clf.score(X_test[ids,:], y_test[ids])
                    sc_cv[it,iiu]=sc
                irand_type= np.random.randint(0,len(ytypes_mat),size=len(ytypes_mat))
                X_train, X_test, y_train, y_test = train_test_split(sp_types_mat, ytypes_mat[irand_type], test_size=0.5)
                clf = LinearDiscriminantAnalysis()
                clf.fit(X_train, y_train)
                utype= np.unique(y_test)
                for (iu,iiu) in zip(utype,range(len(utype))):
                    ids = np.argwhere(y_test==iu).flatten()
                    sc = clf.score(X_test[ids,:], y_test[ids])
                    sc_cv_sh[it,iiu]=sc

                X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(sp_micro_mat, ymicro_mat, test_size=0.5)
                clf = LinearDiscriminantAnalysis()
                clf.fit(X_train_b, y_train_b)
                utype= np.unique(y_test_b)
                for (iu,iiu) in zip(utype,range(len(utype))):
                    ids = np.argwhere(y_test_b==iu).flatten()
                    sc = clf.score(X_test_b[ids,:], y_test_b[ids])
                    sc_cv_b[it,iiu] = sc
                
                irand_micro= np.random.randint(0,len(ymicro_mat),size=len(ymicro_mat))
                X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(sp_micro_mat, ymicro_mat[irand_micro], test_size=0.5)
                clf = LinearDiscriminantAnalysis()
                clf.fit(X_train_b, y_train_b)
                utype= np.unique(y_test_b)
                for (iu,iiu) in zip(utype,range(len(utype))):
                    ids = np.argwhere(y_test_b==iu).flatten()
                    sc = clf.score(X_test_b[ids,:], y_test_b[ids])
                    sc_cv_b_sh[it,iiu] = sc

            results['score_type_each_lda_cv'] = sc_cv
            results['score_type_each_lda_cv_shuff'] = sc_cv_sh
            results['score_micro_each_lda_cv'] = sc_cv_b
            results['score_micro_each_lda_cv_shuff'] = sc_cv_b_sh
            #% %
            fig,ax = plt.subplots(3,1,figsize=(6,12))
            X_train, X_test, y_train, y_test = train_test_split(sp_types_mat, ytypes_mat, test_size=0.3)
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            utype= np.unique(y_test)
            lab_matrix = np.nan*np.ones((len(utype),len(utype)))
            lab_matrix_sh = np.nan*np.ones((len(utype),len(utype)))
            clf_sh = LinearDiscriminantAnalysis()
            id_rand = np.random.randint(0,len(y_train),size=len(y_train))
            clf_sh.fit(X_train, y_train[id_rand])
            for (iu,iiu) in zip(utype,range(len(utype))):
                ids = np.argwhere(y_test==iu).flatten()
                yl = clf.predict(X_test[ids,:]) 
                yl_sh = clf_sh.predict(X_test[ids,:]) 
                upred = np.unique(yl)
                for (iul,iiul) in zip(upred,range(len(upred))):
                    if len( np.argwhere(utype==iul).flatten())>0:
                        idrow = np.argwhere(utype==iul).flatten()[0]
                        idlabs = np.argwhere(yl==iul)
                        lab_matrix[idrow,iiu] = len(idlabs)/len(ids)
                        idlabs = np.argwhere(yl_sh==iul)
                        lab_matrix_sh[idrow,iiu] = len(idlabs)/len(ids)
                    
                        
                    
            results['label_mat_trial_type'] = lab_matrix
            results['label_mat_trial_type_sh'] = lab_matrix_sh
            pos=ax[0].imshow(lab_matrix,aspect='auto',cmap='hot')
            plot_config(ax[0],'True label','Predicted label',14,False)
            fig.colorbar(pos, ax=ax[0])
            ax[0].set_xticks(np.arange(len(utype)))
            ax[0].set_yticks(np.arange(len(utype)))
            ax[0].set_title('LDA classification for trial type')

            pos=ax[1].imshow(lab_matrix_sh,aspect='auto',cmap='hot')
            plot_config(ax[1],'True label','Predicted label',14,False)
            fig.colorbar(pos, ax=ax[1])
            ax[1].set_xticks(np.arange(len(utype)))
            ax[1].set_yticks(np.arange(len(utype)))
            ax[1].set_title('LDA classification for trial type (shuffle)')

            ax[2].errorbar(np.arange(len(utype)),np.nanmean(sc_cv,axis=0),
                np.nanstd(sc_cv,axis=0),color='black')
            ax[2].errorbar(np.arange(len(utype)),np.nanmean(sc_cv_sh,axis=0),
                np.nanstd(sc_cv,axis=0),color='grey')
            plot_config(ax[2],'True label (trial type)','Accuracy',14,False)
            ax[2].set_xticks(np.arange(len(utype)))
            fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' +'LDA_classification_accuracy_trial_type' '.png'))
            fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' +'LDA_classification_accuracy_trial_type' '.pdf'))
            #%  Accuracy per trial type per time bin
            fig,ax = plt.subplots(figsize=(6,4))
            cols = ['green','red','blue','purple']
            X_train, X_test, id_train, id_test = train_test_split(np.zeros((sp_types_mat.shape[0],2)), np.arange(sp_types_mat.shape[0]), test_size=0.3)
            X_train = sp_types_mat[id_train,:]
            X_test = sp_types_mat[id_test,:]
            y_train = ytypes_mat[id_train]
            y_test = ytypes_mat[id_test]
            y_testmicro = ymicro_fortype[id_test]
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            utype = np.unique(y_test)
            umicro = np.unique(y_testmicro)
            acc_micro_x_type= np.zeros((len(utype), len(umicro)))
            for (iu,iiu) in zip(utype,range(len(utype))):
                ids = np.argwhere(y_test==iu).flatten()
                yl = clf.predict(X_test[ids,:]) 
                bins_ = y_testmicro[ids]
                acc_im = np.zeros((len(umicro)))
                for im in range(len(umicro)):
                    idm = np.argwhere(bins_==im)
                    ylm = yl[idm]
                    if len(idm)>0:
                        acc_im[im] = sum(ylm==iu)/len(idm)
                    else:
                        acc_im[im] = np.nan
                    acc_micro_x_type[iiu,im] = acc_im[im]
                ax.plot(umicro,acc_im,color=cols[iiu])
                ax.set_xticks(umicro)
                ax.set_ylim([0,1])
                plot_config(ax,'Micro-state','Accuracy for trial type',14,False)

            results['acc_micro_x_type'] = acc_micro_x_type
            fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' +'LDA_classification_accuracy_trial_type_per_microstate' '.png'))
            fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' +'LDA_classification_accuracy_trial_type_per_microstate' '.pdf'))
            #% %
            fig,ax = plt.subplots(3,1,figsize=(6,12))
            X_train, X_test, y_train, y_test = train_test_split(sp_micro_mat, ymicro_mat,test_size=0.3)
            clf = LinearDiscriminantAnalysis()
            clf.fit(X_train, y_train)
            clf_sh = LinearDiscriminantAnalysis()
            id_rand = np.random.randint(0,len(y_train),size=len(y_train))
            clf_sh.fit(X_train, y_train[id_rand])
            utype= np.unique(y_test)
            lab_matrix = np.nan*np.ones((len(utype),len(utype)))
            lab_matrix_sh = np.nan*np.ones((len(utype),len(utype)))
            for (iu,iiu) in zip(utype,range(len(utype))):
                ids = np.argwhere(y_test==iu).flatten()
                yl = clf.predict(X_test[ids,:]) 
                yl_sh = clf_sh.predict(X_test[ids,:]) 
                upred = np.unique(yl)
                for (iul,iiul) in zip(upred,range(len(upred))):
                    if len( np.argwhere(utype==iul).flatten())>0:
                        idrow = np.argwhere(utype==iul).flatten()[0]
                        idlabs = np.argwhere(yl==iul)
                        lab_matrix[idrow,iiu] = len(idlabs)/len(ids)
                        idlabs = np.argwhere(yl_sh==iul)
                        lab_matrix_sh[idrow,iiu] = len(idlabs)/len(ids)

            results['label_mat_micro_state'] = lab_matrix
            results['label_mat_micro_state_sh'] = lab_matrix_sh
            ax[0].imshow(lab_matrix,aspect='auto',cmap='hot')
            plot_config(ax[0],'True label','Predicted label',14,False)
            fig.colorbar(pos, ax=ax[0])
            ax[0].set_title('LDA classification for trial bins')
            ax[0].set_xticks(np.arange(len(utype)))

            ax[1].imshow(lab_matrix_sh,aspect='auto',cmap='hot')
            plot_config(ax[1],'True label','Predicted label',14,False)
            fig.colorbar(pos, ax=ax[1])
            ax[1].set_title('LDA classification for trial bins (shuffle)')
            ax[1].set_xticks(np.arange(len(utype)))
            ax[2].errorbar(np.arange(len(utype)),np.nanmean(sc_cv_b,axis=0),
                                np.nanstd(sc_cv_b,axis=0)/np.sqrt(sc_cv_b.shape[0]),color='black')
            ax[2].errorbar(np.arange(len(utype)),np.nanmean(sc_cv_b_sh,axis=0),
                                np.nanstd(sc_cv_b,axis=0)/np.sqrt(sc_cv_b.shape[0]),color='grey')
            plot_config(ax[2],'True label (trial type)','Accuracy',14,False)
            ax[2].set_xticks(np.arange(len(utype)))
            fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' +'LDA_classification_accuracy_trial_bins' '.png'))
            fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' +'LDA_classification_accuracy_trial_bins' '.pdf'))


            #% % Same but separated by trial bin
            utypes = np.unique(ytype_formicro)
            fig,ax = plt.subplots(3,len(utypes),figsize=(20,12))
            for iutype in np.arange(len(utypes)):
                ids_type = np.argwhere(ytype_formicro==utypes[iutype]).flatten()
                X_train, X_test, ids_train, ids_test = train_test_split(sp_micro_mat[ids_type,:], 
                                                                    np.arange(len(ids_type)),
                                                                    test_size=0.5)
                y_train = ymicro_mat[ids_type][ids_train]
                clf = LinearDiscriminantAnalysis()
                clf.fit(X_train, y_train)
                ytypes_formicro_test = ytype_formicro[ids_type][ids_test]
                y_test = ymicro_mat[ids_type][ids_test]
                x_test = sp_micro_mat[ids_type,:][ids_test]
                utype= np.unique(y_test)
                lab_matrix = np.zeros((len(utype),len(utype)))
                lab_matrix_sh = np.zeros((len(utype),len(utype)))
                sc_matrix =  np.zeros((len(utype),))
                sc_matrix_sh =  np.zeros((len(utype),))
                clf_sh = LinearDiscriminantAnalysis()
                id_rand = np.random.randint(0,len(y_train),size=len(y_train))
                clf_sh.fit(X_train, y_train[id_rand])
                for (iu,iiu) in zip(utype,range(len(utype))):
                    ids = np.argwhere(y_test==iu).flatten()
                    if len(ids)>0:
                        yl = clf.predict(x_test[ids,:]) 
                        yl_sh = clf_sh.predict(x_test[ids,:]) 
                        sc_matrix[iiu] = clf.score(X_test_b[ids,:], y_test[ids])
                        sc_matrix_sh[iiu] = clf_sh.score(X_test_b[ids,:], y_test[ids])
                        upred = np.unique(yl)
                        for (iul,iiul) in zip(upred,range(len(upred))):
                            idrow = np.argwhere(utype==iul).flatten()[0]
                            idlabs = np.argwhere(yl==iul)
                            lab_matrix[idrow,iiu] = len(idlabs)/len(ids)
                            idlabs = np.argwhere(yl_sh==iul)
                            lab_matrix_sh[idrow,iiu] = len(idlabs)/len(ids)
                    else:
                        lab_matrix[idrow,iiu] = np.nan
                        lab_matrix_sh[idrow,iiu] = np.nan
                        sc_matrix[iiu] = np.nan
                        sc_matrix_sh[iiu] = np.nan
                results['label_mat_micro_state' + '_type_' + str(iutype)] = lab_matrix
                results['label_mat_micro_state_sh'+ '_type_' + str(iutype)] = lab_matrix_sh
                ax[0,iutype].imshow(lab_matrix,aspect='auto',cmap='hot')
                plot_config(ax[0,iutype],'True label','Predicted label',14,False)
                fig.colorbar(pos, ax=ax[0,iutype])
                ax[0,iutype].set_title('LDA classification for trial bins')
                ax[0,iutype].set_xticks(np.arange(len(utype)))

                ax[1,iutype].imshow(lab_matrix_sh,aspect='auto',cmap='hot')
                plot_config(ax[1,iutype],'True label','Predicted label',14,False)
                fig.colorbar(pos, ax=ax[1,iutype])
                ax[1,iutype].set_title('LDA classification for trial bins (shuffle)')
                ax[1,iutype].set_xticks(np.arange(len(utype)))

                ax[2,iutype].plot(np.arange(len(utype)),sc_matrix,color='black')
                ax[2,iutype].plot(np.arange(len(utype)),sc_matrix_sh,color='grey')
                plot_config(ax[2,iutype],'True label (trial type)','Accuracy',14,False)
                ax[2,iutype].set_xticks(np.arange(len(utype)))
                ax[2,iutype].set_ylim([0,.5])
                fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' +'LDA_classification_accuracy_trial_bins_per_trialtype' '.png'))
                fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' +'LDA_classification_accuracy_trial_bins_per_trialtype' '.pdf'))


            #% % create a binary pickle file 
            fname = os.path.join(analysis_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' + 'classification.pkl')
            f = open(fname,"wb")
            pickle.dump(results,f)
            f.close()
            print('writing .pkl file ...')
            print('done')
# %% From paper: 
# mubins = [np.asarray([np.nanmean(ss,axis=0) for ss in spbins ]) for spbins in spbins_all]
# cos_sims = []
# for itr in range(len(tr_common)):
#     print('for ' + str(itr) + ' of ' + str(len(tr_common)))
#     mub = mubins[itr]
#     mubins_zsc = zscore(mub,axis=1)
#     nbins = mubins_zsc.shape[1]
#     cos_sim = np.zeros((nbins,nbins))
#     for ib in range(nbins):
#         for ib2 in range(nbins):
#             v1 = mubins_zsc[:,ib]
#             v2 = mubins_zsc[:,ib2]
#             v1[np.isnan(v1)]=.000001
#             v2[np.isnan(v2)]=.000001
#             cos_sim[ib,ib2] = np.dot(v1,v2)/(norm(v1)*norm(v2))
#     cos_sims.append(cos_sim)
# #% %
# fig,ax = plt.subplots(1,len(tr_common),figsize=(25,5))
# for itr in range(len(tr_common)):
#     norml=matplotlib.colors.Normalize(vmin=np.percentile(cos_sims[itr],.1),vmax=np.percentile(cos_sims[itr],95))
#     ax[itr].imshow(cos_sims[itr],aspect='auto',norm=norml,cmap='gnuplot2')
#     ax[itr].set_title(labs[itr] + r' (cosine similarity between time bins)')
#     plot_config(ax[itr],'time bin (ms)','time bin (ms)',18,False)
# fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' +'cosine_similarity_timebins' '.png'))
# fig.savefig(os.path.join(fig_path,brain_reg+ '_' +session_n + '_' +  mouse + '_' +'cosine_similarity_timebins' '.pdf'))



#%% 




#%%
# noise correlations based on sequence in time ?
# other metrics of'connectivity'
# %%
# peak of activity vsti increases in ick rates : trial to trial
# for a given cell, if activity is moved fwd or bwd wrt to mean timing field-
# # - is this related to changes in lick rates?
#%%
# do again with longer sp_bins range: to see whether it would  still change tuning
#%% 
# decoding of stimulus identity across time bins
# decoding of time (i.e., microstimujlus) across time bins


#%% From paper: 
# We found the maximum likelihood fit  for each trial independently. 
# Prior to the fitting  all the parameters except μt and σt were fixed 
# to the values found through the fit of the entire spike train spanning all the 
# trials. 
# The trial-averaged μt and σt were still significantly correlated 
# (Pearson’s correlation 0.42, < −P 10 3). 
# This suggests that the dynamics of the spreading time fields is noticeable on 
# a trial level as well and the increase of the Width of the Time Fields 
# with the Peak Time Was Not an Artifact of Trial Averaging

#%%
# check whether the mu across different trial types is similar 
# For those cells that have a mu in the late ISI-- what happens in the short cues? 
# What about the variables delay cue? 
