#%%
# Sandra Romero Pinto 2022, based  on Sam Gershman, June 2015

#  
sys.path.insert(0,'/Volumes/GoogleDrive/My Drive/Dropbox (HMS)/EphysPy/')
sys.path.append('/Volumes/GoogleDrive-116748251018268574178/My Drive/Dropbox (HMS)/EphysPy')

from distutils.util import copydir_run_2to3
import model_fitting as fit
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import pandas as pd
import scipy.io as scio
import scipy.stats as stats

import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sklearn.linear_model as lm
from sklearn.model_selection import KFold

from utils import *
from plots import *


#  
g_dir = '/Volumes/GoogleDrive-116748251018268574178/My Drive/Dropbox (HMS)/LHb/Ju_LHb_Lesions/'
suffs = ['control','lesion']
save_path = '/Users/sromeropinto/Dropbox (HMS) orig/Distributional_RL/codes_paper/analysis'
fig_path = '/Users/sromeropinto/Dropbox (HMS) orig/Distributional_RL/codes_paper/figures'
suffs = ['control','lesion']

u_dates_ =[]
id_unique_ = []
for i_suff in np.arange(len(suffs)):
    suffix = suffs[i_suff]
    data_path = os.path.join(g_dir,  'lightidunits',suffix)
    ff = glob.glob(data_path + '/*formatted.mat')
    dates_ = []
    for iff in np.arange(len(ff)):
        f_name = (ff[iff].split('/')[-1])
        id_start = f_name.find('_',2)+1
        fn_st = f_name[id_start:]
        id_end  = fn_st.find('_')
        dates_.append(fn_st[:id_end])

    u_dates,id_unique =np.unique(dates_, return_index=True)
    u_dates_.append(u_dates)
    id_unique_.append(id_unique)

u_max = np.max([len(ii) for ii in id_unique_])

# % % 
# def rlfit_predict(data,results,model):
#     '''
#       Compute log predictive probability of new data.
     
#       USAGE: logp = mfit_predict(data,results)
     
#       INPUTS:
#         data - [S x 1] data structure
#         results - results structure
     
#       OUTPUTS:
#         logp - [S x 1] log predictive probabilities for each subject
     
#       Sam Gershman, June 2015
#     '''
    
#     # logp = results.likfun(results.x(s,:),data(s));
#     logp = results['likfun'](results['x'],data,model)

#     return logp
# #% %

#  prob10: [5 6]
#  prob50: [3 4]
#  prob90: [1 2]
#  cuepuff: [7 8]
#  freeR : 9
#  freeP: 10
def map_cues(map_,trial_types):
    cue_types = np.zeros((len(trial_types)))
    for ic in range(len(map_)):
        ids = np.argwhere(trial_types==ic+1)
        cue_types[ids] = map_[ic]

    return cue_types

def map_rewards(rew_on, puff_on,trial_types):
    rew_types = np.zeros((len(trial_types,)))
    rew_types[~np.isnan(rew_on)] = 1
    rew_types[~np.isnan(puff_on)] = -1

    return rew_types
#%%
tr_win = [500, 2000]
e_values = [.9,.5,.1,-.8]
map_= [0,0,1,1,2,2,3,3]

psth_resolution = 10
psth_length = np.int0( 1 + np.ceil((tr_win[1] - tr_win[0]) / psth_resolution) )
models = ['symmetric','reward_sensitivity', 'asymmetric']
results_groups = dict(zip(suffs,[dict() for i in range(len(suffs))]))
for suffix , i_suff in zip(suffs, range(len(suffs))):
    results_groups[suffix] = dict(zip(models,[dict() for i in range(len(models))]))
    for model in models:
        results_groups[suffix][model]=[]        
        data_path = os.path.join(g_dir,  'lightidunits',suffix)
        ff = glob.glob(data_path + '/*formatted.mat')
        id_unique = id_unique_[i_suff]
        for iff in np.arange(len(id_unique)):
            print('Simulations for group ' + suffix + '|| Model ' + model + ' || Subject nb: ' + str(iff) + ' of ' + str(len(id_unique)))
            idu = id_unique[iff]
            r = loadmat(ff[idu])
            trial_types = r['S']['TrialTypes']
            licks =  r['S']['responses']['lick']
            odor_on = r['S']['events']['odorOn']
            rew_on = r['S']['events']['rewardOn']
            puff_on = r['S']['events']['airpuffOn']
            n_trials = len(odor_on)
            lick_raster = np.zeros((n_trials, psth_length+1))
            rew_types = map_rewards(rew_on, puff_on,trial_types)
            cue_types = map_cues(map_,trial_types)
            icount = 0
            itrial_ids = []
            for i_trial in np.arange(n_trials):
                if trial_types[i_trial] != 10 and trial_types[i_trial] != 9 and  trial_types[i_trial] != 7 and  trial_types[i_trial] != 8:
                    st_time = odor_on[i_trial]
                    win_indices = (licks > st_time + tr_win[0])*(licks < st_time + tr_win[1])
                    lick_times = licks[win_indices]
                    lick_psth_indices = np.int0( 1 + np.round((lick_times - st_time) / psth_resolution) - tr_win[0] / psth_resolution)
                    lick_r = np.zeros((psth_length+1))
                    for i_lick in np.arange(len(lick_psth_indices)):
                        lick_r[lick_psth_indices] += 1
                    lick_raster[icount,:] = lick_r
                    icount +=1
                    itrial_ids.append(i_trial)
            ids_ = np.asarray(itrial_ids)
            lick_mu = np.nanmean(lick_raster,axis=1)
            # lick_mu = (lick_mu-np.nanmin(lick_mu))/(np.nanmax(lick_mu)-np.nanmin(lick_mu))

            cue_types = cue_types[ids_]
            rew_types = rew_types[ids_]
            if np.sum(lick_mu)>0:
                lick_mu = lick_mu[np.arange(len(ids_))]
                ids_ = np.arange(len(ids_))
                kf = KFold(n_splits=2)
                train, test = kf.split(ids_)

                id_train = train[0]
                id_test = train[1]
                data  = dict()
                data['N'] = len(id_train)
                data['c'] = cue_types[id_train]
                data['r'] = rew_types[id_train]
                data['v'] = lick_mu[id_train]
                data['trial'] = np.arange(len(id_train))
                data['v0'] = e_values

                data_test  = dict()
                data_test['N'] = len(id_test)
                data_test['c'] = cue_types[id_test]
                data_test['r'] = rew_types[id_test]
                data_test['v'] = lick_mu[id_test]
                data_test['trial'] = np.arange(len(id_test))
                data_test['v0'] = e_values

                param = fit.create_param_opt(model)
                _llfun = fit.likfun
                results = fit.rlfit_optimize(_llfun, param, data, nstarts=10)
                xx = results['x']
                lf, yhat= fit.likfun_full(xx,data_test,model)
                results['y_test'] = data_test['v']
                results['c_test'] = data_test['c']
                results['r_test'] = data_test['r']
                results['y_hat'] = yhat
                results['corr_pred'] = np.corrcoef(yhat,data_test['v'])[0,1]

                log_p_predict = fit.rlfit_predict(data_test,results,model)
                results['log_predictive_prob'] = log_p_predict
                results_groups[suffix][model].append(results)
                
                f_name = 'rlsim_train_data_' +os.path.split(ff[idu])[-1][2:-4] +'.mat'
                scio.savemat(os.path.join(save_path,'rl_mfit',suffix,f_name),data)
                f_name = 'rlsim_test_data_' +os.path.split(ff[idu])[-1][2:-4]+'.mat'
                scio.savemat(os.path.join(save_path,'rl_mfit',suffix,f_name),data_test)
                f_names = 'rlsim_results_' +os.path.split(ff[idu])[-1][2:-4]+'.mat'
                scio.savemat(os.path.join(save_path,'rl_mfit',suffix,f_name),results)
#%%
cue_names = ['90%','50%','10%']
cc = ['blue','green','red']
# for model, imod in zip(models,range(len(models))):
corr_coeff=dict()
for suffix in suffs:
    n_sessions = len(results_groups[suffix][models[0]])
    corr_coeff[suffix] = np.zeros((n_sessions,3,len(models)))
    print(n_sessions)
    fix,ax  = plt.subplots(len(models),3,figsize=(20,20))
    for iss in range(n_sessions):
        for model, imod in zip(models,range(len(models))):    
            results = results_groups[suffix][model][iss]
            yy = results['y_test']
            yh = results['y_hat']
            c = results['c_test']
            for (ic,iic) in zip(np.unique(c),range(len(np.unique(c)))):
                ids_ = np.argwhere(c==ic).flatten()
                yy_ = yy[ids_]
                yh_ = yh[ids_]
                ax[imod,iic].plot(yy_,yh_,'+',color=cc[iic])
                id_keep = np.unique(np.concatenate((np.argwhere(~np.isnan(yy_)),np.argwhere(~np.isnan(yh_)))).flatten())
                
                corr_ =np.corrcoef(yy_[id_keep],yh_[id_keep])[0,1]

                corr_coeff[suffix][iss,iic,imod] = corr_
                plot_config(ax[imod,iic],'lick/sec (data)','lick/sec (pred)',13,False )
                ax[imod,iic].set_title(cue_names[iic] + ' ' + model)
    fix.savefig(os.path.join(fig_path,'data_vs_predicted_all_models_' + suffix + '.pdf'))

#%%
cg = ['black','red']
alpha_p =.05

fix,ax  = plt.subplots(3,len(suffs),figsize=(20,20))
for suffix , i_suff in zip(suffs, range(len(suffs))):
    for iic in range(len(cue_names)):
        metr = corr_coeff[suffix][:,iic,:]
        metr = [metr[:,ii] for ii in range(metr.shape[1])]
        metr = [ii[~np.isnan(ii)] for ii in metr]
        norm_test = [stats.normaltest(ii) for ii in metr]
        pp = [ii.pvalue for ii in norm_test]
        if np.sum(np.asarray(pp)<alpha_p) ==2:
            print('null rejected: not normally distributed')
            p_=[[ii,jj,stats.kruskal(metr[ii],metr[jj]).pvalue] for ii in range(len(metr)) for jj in range(ii+1,len(metr))]
            mus_ = np.asarray([np.nanmedian(x) for x in metr])
        else:
            print('null accepted:  normally distributed')
            p_=[[ii,jj,stats.ttest_ind(metr[ii],metr[jj]).pvalue] for ii in range(len(metr)) for jj in range(ii+1,len(metr))]
            mus_ = np.asarray([np.nanmean(x) for x in metr])
        [ax[iic,i_suff].plot([ii-.05,ii+.1],[mus_[ii],mus_[ii]], color='black') for ii in range(len(metr))]
        [ax[iic,i_suff].plot(ii+np.random.rand(len(metr[ii]))*.05,metr[ii],'o',color=cg[i_suff]) for ii in range(len(metr))]
        [ax[iic,i_suff].text( p_[ii][0]+(p_[ii][1]-p_[ii][0])*.35, np.max(metr[p_[ii][1]]),'p= '+ str( np.round(p_[ii][2]*1000)/1000))for ii in range(len(p_))]
        ax[iic,i_suff].set_xticks(np.arange(3))
        ax[iic,i_suff].set_xticklabels(models)
        ax[iic,i_suff].set_title(suffix+ ' '+ cue_names[iic])


        # [ax[iic,i_suff].plot([ii-.05,ii+.1],[mus_[ii],mus_[ii]], color='black') for ii in range(len(mus_))]
        # [ax[iic,i_suff].plot(ii+np.random.rand(len(metr[:,ii]))*.05,metr[:,ii],'o',color=cg[i_suff]) for ii in range(metr.shape[1])]


    
#%%
from statsmodels.stats.multitest import multipletests
alpha_p =.05
metrics_ = ['aic','bic','log_predictive_prob','corr_pred']
cg = ['black','red']
# curr
for metric_ in metrics_:
    fix,ax  = plt.subplots(1,len(suffs),figsize=(10,4))
    for suffix , i_suff in zip(suffs, range(len(suffs))):
        metr =[]
        for model, imod in zip(models,range(len(models))):
            n_sessions = len( results_groups[suffix][model])
            metr_ = []
            for iff in np.arange(n_sessions):
                metr_.append(results_groups[suffix][model][iff][metric_])
            metr.append(metr_)

        norm_test = [stats.normaltest(ii) for ii in metr]
        pp = [ii.pvalue for ii in norm_test]

        st =[[ii,jj,stats.kruskal(metr[ii],metr[jj]).pvalue] for ii in range(len(metr)) for jj in range(ii+1,len(metr))]
        p_ = [st[ii][-1] for ii in range(len(st))]
        multipletests(p_)
        if np.sum(np.asarray(pp)<alpha_p) ==2:
            print('null rejected: not normally distributed')
            p_=[[ii,jj,stats.kruskal(metr[ii],metr[jj]).pvalue] for ii in range(len(metr)) for jj in range(ii+1,len(metr))]
            mus_ = np.asarray([np.nanmedian(x) for x in metr])
        else:
            print('null accepted:  normally distributed')
            p_=[[ii,jj,stats.ttest_ind(metr[ii],metr[jj]).pvalue] for ii in range(len(metr)) for jj in range(ii+1,len(metr))]
            mus_ = np.asarray([np.nanmean(x) for x in metr])

        p_corr = multipletests(p_)[1]
        [ax[i_suff].plot([ii-.05,ii+.1],[mus_[ii],mus_[ii]], color='black') for ii in range(len(metr))]
        [ax[i_suff].plot(ii+np.random.rand(len(metr[ii]))*.05,metr[ii],'o',color=cg[i_suff]) for ii in range(len(metr))]
        [ax[i_suff].text( p_[ii][0]+(p_[ii][1]-p_[ii][0])*.35, np.max(metr[p_[ii][1]]),'p= '+ str( np.round(p_[ii][2]*1000)/1000))for ii in range(len(p_))]
        plot_config(ax[i_suff],'',metric_,14,False)
        ax[i_suff].set_xticks(np.arange(3))
        ax[i_suff].set_xticklabels(models)
        ax[i_suff].set_title(suffix)
        fix.savefig(os.path.join(fig_path,'fit_lr_models_model_metrics_' + metric_ + '.pdf'))
# %%
ylims_ = [[0.,1.],[0,2.],[0.,1.]]
cg = ['black','red']
param_names =[[r'$\alpha$',r'$\beta$'],[r'$\alpha$','reward sens.',r'$\beta$'],[r'$\alpha^+$',  r'$\alpha^-$',r'$\beta$']]
alpha_p = .05
for model, imod in zip(models,range(len(models))):
    params = [];    
    param_name = param_names[imod]
    for suffix , i_suff in zip(suffs, range(len(suffs))):
        params_ = []
        n_sessions = len( results_groups[suffix][model])
        for iff in np.arange(n_sessions):
            if model =='asymmetric':
                x_ = results_groups[suffix][model][iff]['x']
                taus_ = x_[0]/(x_[0]+x_[1])
                params_.append(np.asarray([taus_,1-taus_,x_[2]]))
            else:
                params_.append(results_groups[suffix][model][iff]['x'])
        params_ = np.asarray(params_)
        params.append(params_)
    n_params = params[0].shape[1]
    norm_test = [stats.normaltest(ii[:,jj]) for ii in params for jj in np.arange(n_params)]

    pp = [ii.pvalue for ii in norm_test]
    if np.sum(np.asarray(pp)<alpha_p) ==2:
        
        print('null rejected: not normally distributed')
        st = [stats.kruskal(params[0][:,jj],params[1][:,jj]).pvalue  for jj in np.arange(n_params)]
        mus_ = np.asarray([np.nanmedian(x[:,ii]) for x in params for ii in np.arange(n_params)])
    else:
        print('null accepted:  normally distributed')
        st = [stats.ttest_ind(params[0][:,jj],params[1][:,jj]).pvalue  for jj in np.arange(n_params)]
        mus_ = np.asarray([np.nanmean(x[:,ii]) for x in params for ii in np.arange(n_params)])

    fix,ax  = plt.subplots(1,n_params,figsize=(5*n_params,5))    
    if n_params==1:  ax = np.asarray(ax)[...,np.newaxis]
    [ax[ii].bar(np.arange(2),mus_[np.arange(ii,len(mus_),step=n_params)],color=cg) for ii in np.arange(n_params)]
    [ax[ii].plot(ix+np.random.rand(len(x[:,ii]))*.05,x[:,ii].T,'o',color='grey') for (x,ix) in zip(params,np.arange(len(params))) for ii in np.arange(n_params)]
    [ax[ii].text(.2,ylims_[imod][0]+np.diff(ylims_[imod])*.8,'p='+str(np.round(st[ii]*1000)/1000)) for ii in np.arange(n_params)]
    [plot_config(ax[ii],'',param_name[ii],14,False) for ii in np.arange(n_params)]
    # [ax[ii].set_ylim((ylims_[imod])) for ii in np.arange(n_params)]
    [ax[ii].set_xlim(([-1,2])) for ii in np.arange(n_params)]
    [ax[ii].set_xticks([0,1]) for ii in np.arange(n_params)]
    [ax[ii].set_xticklabels(suffs) for ii in np.arange(n_params)]
    [ax[ii].set_title(model) for ii in np.arange(n_params)]
    fix.savefig(os.path.join(fig_path,'fit_lr_models_' + model + '.pdf'))
#%%
