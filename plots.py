from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

#%%
# general ones
def box_off(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def plot_config(ax,xlabel,ylabel,f_size,legend):
    ax.set_xlabel(xlabel,fontsize=f_size)
    ax.set_ylabel(ylabel,fontsize=f_size)
    box_off(ax)
    if legend is True:
        ax.legend(fontsize=f_size,framealpha=0)

def title_(ax,title):
    ax.set_title(title)

def xticks_(ax,xticks,xticklabels=None):
    ax.set_xticks((xticks))
    if not xticklabels.all()==None:
        ax.set_xticklabels((xticklabels))

def yticks_(ax,yticks,yticklabels=None):
    ax.set_yticks((yticks))
    if not yticklabels.all()==None:
        ax.set_yticklabels((yticklabels))


'''Photometry specific'''   

def plot_full_session_ttls(df_,ttls_time,fs,mouse,date,figname):
    yymin =np.nanmin(df_)
    yymax =np.nanmax(df_)
    time_ = np.linspace(0,len(df_)/fs,len(df_))
    time_x = np.stack((ttls_time,ttls_time))
    yy = np.stack((yymin*np.ones((len(ttls_time))),yymax*np.ones((len(ttls_time)))))
    fig,ax = plt.subplots(figsize=(20,5))
    plt.plot(time_x/60,yy,color='grey',linewidth=.4)
    plt.plot(time_/60,df_,color='green',linewidth=.7)
    ax.set_title(mouse + ' ' + date )
    plot_config(ax,'time(min)','df/f0 ',14,False)
    plt.savefig(figname)


'''Behavior specific'''   

def plot_responses_per_trial_type(resp_trials,time_trial,reward_types,trial_types,cue_names,figname=None,filtsize=None):
    if filtsize is not None:
        filtsize = filtsize
    else:
        filtsize =501
    utr_types = np.unique(trial_types)
    urew_types = np.unique(reward_types)
    urew_types = urew_types[::-1]
    yy=[]
    colors = [['red','gray'],['green','gray'],['blue','gray'],['purple','gray'],['purple','gray']]
    fig,ax = plt.subplots(2,len(utr_types),figsize=(len(utr_types)*5,len(urew_types)*3.5))
    for (iiu,iu) in zip(range(len(utr_types)),utr_types):
        mat_ = 0
        for (iir,ir) in zip(range(len(urew_types)),urew_types):
            id_ = np.intersect1d(np.argwhere(trial_types==iu),np.argwhere(reward_types==ir))
            resp_ = resp_trials[id_]
            if filtsize==0:
                # mu_ = gaussian_filter1d(np.nanmean(lickr[ids_all,:,:],axis=1),1)
                resp_filt = [gaussian_filter1d(rr, 1) for rr in resp_]
            else:   
                resp_filt = [savgol_filter(rr, filtsize, 3) for rr in resp_]
            if len(resp_filt)>0:
                ax[0,iiu].plot(time_trial,np.nanmean(resp_filt,axis=0),color=colors[iiu][iir])
                yy.append(np.max(np.nanmean(resp_filt,axis=0)))
                yy.append(np.min(np.nanmean(resp_filt,axis=0)))
            if mat_==0 and len(resp_filt)>0:
                mat_ = resp_filt
            elif len(resp_filt)>0:
                mat_ = np.concatenate((mat_,resp_filt))
        mat_ =np.array(mat_)
        ax[1,iiu].imshow(mat_, extent=[-2, 2, -1, 1])
        
    for (iiu,iu) in zip(range(len(utr_types)),utr_types):
        ax[0,iiu].set_ylim([np.nanmin(yy),np.nanmax(yy)*1.1])
        ax[0,iiu].plot([0,0],[np.nanmin(yy),np.nanmax(yy)*1.1],color='gray')
        ax[0,iiu].plot([1.00,1.00],[np.nanmin(yy),np.nanmax(yy)*1.1],'--',color='gray')
        ax[0,iiu].plot([2.00,2.00],[np.nanmin(yy),np.nanmax(yy)*1.1],'--',color='gray')
        ax[0,iiu].plot([2.50,2.50],[np.nanmin(yy),np.nanmax(yy)*1.1],color='gray')
        ax[0,iiu].plot([3.50,3.50],[np.nanmin(yy),np.nanmax(yy)*1.1],'--',color='gray')
        plot_config(ax[0,iiu],'time(sec)','df ',14,False)
        ax[0,iiu].set_title(cue_names[iiu])
    
    if figname is not None:
        plt.savefig(figname)

def plot_single_trial_type(resp_trials,time_trial,reward_types,trial_types,rew_id,type_id,cue_name,figname=None,filtsize=None):
    #Plot of all trials averaged
    if filtsize is not None:
        filtsize = filtsize
    else:
        filtsize =501
    id_ = np.intersect1d(np.argwhere(trial_types==type_id),np.argwhere(reward_types==rew_id))
    resp_trials = resp_trials[id_]
    mu_resp = np.nanmean(resp_trials,axis=0)
    if filtsize==0:
        mu_resp = gaussian_filter1d(mu_resp, 1)
    else:   
        mu_resp = savgol_filter(mu_resp, filtsize, 3) 
    fig,ax = plt.subplots(1,2,figsize=(15,4))
    ax[0].plot(time_trial,mu_resp,color='green')
    ax[1].imshow(resp_trials, extent=[-2, 2, -1, 1])
    plot_config(ax[0],'time(sec)','df ',14,False)
    ax[0].set_title(cue_name)
    ax[0].plot([0,0],[np.nanmin(np.nanmean(resp_trials,axis=0)),np.nanmax(np.nanmean(resp_trials,axis=0))*1.1],'--',color='gray')
    ax[0].plot([2.5,2.5],[np.nanmin(np.nanmean(resp_trials,axis=0)),np.nanmax(np.nanmean(resp_trials,axis=0))*1.1],'--',color='gray')
    
    if figname is not None:
        plt.savefig(figname)

def plot_evoked_responses_tr_type(resp_licks,trial_types,cue_names,resp_name,colors,mouse,date,figname=None):
    utr_types = np.unique(trial_types)
    fig,ax = plt.subplots(figsize=(6,4))
    mus_, std_,n_= [], [],[]
    for (iiu,iu) in zip(range(len(utr_types)),utr_types):
        id_ = np.argwhere(trial_types==iu)
        resp_ = resp_licks[id_]
        mus_.append(np.nanmean(resp_))
        std_.append(np.nanstd(resp_))
        n_.append(len(resp_))
    for iiu in  range(len(utr_types)) :
        ax.errorbar(iiu,mus_[iiu],yerr=std_[iiu]/np.sqrt(n_[iiu])/2,color=colors[iiu])
    ax.plot(np.arange(len(utr_types)),mus_[0:len(utr_types)],color='gray')
    plot_config(ax,'',resp_name,14,False)
    ax.set_xticks(np.arange(len(utr_types)))
    ax.set_xticklabels(cue_names[0:len(utr_types)])
    ax.set_title(mouse + ' ' + date)

    if figname is not None:
        plt.savefig(figname)

def plot_evoked_responses_tr_type_rew_type(resp_rew,trial_types,reward_types,cue_names,resp_name,mouse,date,figname=None):
    utr_types = np.unique(trial_types)
    tr_types_plot = utr_types[0:3]
    utr_types = tr_types_plot[::-1]
    cue_names = cue_names[0:3]
    cue_names = cue_names[::-1]
    urew_  = np.unique(reward_types)
    fig,ax = plt.subplots(figsize=(6,4))
    mus_, std_,names_= [], [],[]
    for (iir,ir) in zip(range(len(urew_)),urew_):        
        for (iiu,iu) in zip(range(len(tr_types_plot)),tr_types_plot):
            id_ = np.intersect1d(np.argwhere(trial_types==iu),np.argwhere(reward_types==ir))
            resp_ =np.nanmean(resp_rew[id_],axis=1)
            mus_.append(np.nanmean(resp_))
            std_.append(np.nanstd(resp_))
            names_.append(cue_names[iiu] + ' rew:' + str(ir))
    plt.errorbar(np.arange(len(mus_)),mus_,yerr=std_/np.sqrt(len(std_))/2,color='black')
    plot_config(ax,'',resp_name,14,False)
    ax.set_xticks(np.arange(len(mus_)))
    ax.set_xticklabels(names_)
    ax.set_title(mouse + ' ' + date)
    if figname is not None:
        plt.savefig(figname)