import numpy as np
import matplotlib.pyplot as plt


def diff_durations(vlist,x0):
    len_ = np.min([len(ii) for ii in vlist])
    short_v = np.argmin([len(ii) for ii in vlist])
    long_v = np.argmax([len(ii) for ii in vlist])
    x0 = np.int0(x0)
    v1_ = vlist[short_v]
    v2_ = vlist[long_v][np.arange(x0,x0+len_)]
    diff_ = (v1_-v2_)

    return short_v,long_v,len_,diff_
    

def get_trials_for_block(ttls_time,fs,n_trials,stimulus_delivery,trial_durations):
    dttls = np.diff(ttls_time)
    if len(dttls)==len(trial_durations):
        dttls = np.concatenate((dttls,np.array(trial_durations[-1])[np.newaxis]))
    id_block = np.argmin(np.abs(len(dttls)-n_trials))
    trials_block = np.concatenate([ib*np.ones((n_trials[ib])) for ib in range(len(n_trials))])
    trial_durations_block = trial_durations[np.argwhere(trials_block==id_block)].squeeze()
    id_trials_bpod = np.argwhere(trials_block==id_block)
    stimulus_block=stimulus_delivery[np.argwhere(trials_block==id_block)].squeeze()

    fig,ax = plt.subplots(figsize=(20,3))
    ttls_bpod=np.cumsum(np.concatenate([[0],trial_durations_block]))
    ax.plot(ttls_time ,np.ones(len(ttls_time )),'+')
    ax.plot(ttls_bpod+ttls_time[0] ,2*np.ones(len(ttls_bpod )),'+')
    vlist = [dttls,trial_durations_block]
    ll=[len(ii) for ii in vlist]
    ub = np.abs(np.diff(ll))
    xv = np.int0(np.arange(0,ub))
    id_discard= [[]]
    ttls_trials_time = ttls_time
    for x0 in xv:
        [short_v,long_v,len_,obj_] = diff_durations(vlist,x0)
        id_ = np.argwhere(np.abs(obj_)>np.max(trial_durations_block)).squeeze()+x0
        id_ =np.setdiff1d(id_,np.concatenate(id_discard))
        print('Discarded sample: ' + str(id_))
        id_discard.append(id_)
        ax.plot(ttls_time[id_] ,np.ones(len(id_)),'o',color='black')
        ttls_trials_time= np.delete(ttls_trials_time,id_)
    if len(ttls_trials_time)>len(stimulus_block):
        ttls_trials_time = ttls_trials_time[0:len(stimulus_block)]
    ttls_trials_time = ttls_trials_time+stimulus_block
    ttls_trials  = np.int0(np.round(ttls_trials_time*fs))

    return ttls_trials_time,ttls_trials,id_discard,id_trials_bpod,id_block

def get_baseline_subtracted(df_,ttls_trials,window,window_base,fs):

    plot_samples = np.int0(np.ceil(np.multiply(window,fs)))
    st_trials = ttls_trials+plot_samples[0]
    end_trials = ttls_trials+plot_samples[1]
    samp_base = np.int0(np.ceil(np.multiply(window_base,fs)))
    st_base = ttls_trials+samp_base[0]
    end_base = ttls_trials+samp_base[1]
    time_trial =np.linspace(window[0],window[-1],len(np.arange(plot_samples[0],plot_samples[-1])))
    base_trials = np.nanmean([df_[np.arange(st,en)] for (st,en) in zip(st_base ,end_base)],axis=1)[...,np.newaxis]
    resp_trials = np.asarray([df_[np.arange(st,en)] for (st,en) in zip(st_trials ,end_trials )])
    base_trials = np.repeat(base_trials,resp_trials.shape[1],axis=1)
    resp_trials = (resp_trials - base_trials)

    return resp_trials,base_trials,time_trial

def reformat_lick_raster(trial_types,lick_raster):

    lick_v = np.zeros((len(trial_types),lick_raster.shape[2]))
    ic = np.zeros((len(np.unique(trial_types))))
    for (itr,ii) in zip(trial_types,range(len(trial_types))):
        lick_v[ii,:]  = lick_raster[itr-1][np.int0(ic[itr-1])][:]
        ic[itr-1] =ic[itr-1] +1

    return lick_v

def get_lick_responses(window_licks,stimulus_delivery,fs_licks,lick_raster,baseline_subtr=False):
    stimulus_samp = np.int0(np.ceil(np.multiply(stimulus_delivery,fs_licks))) 
    plot_samples = np.int0(np.ceil(np.multiply(window_licks,fs_licks)))
    st_trials = plot_samples[0]+stimulus_samp
    end_trials = plot_samples[1]+stimulus_samp
    resp_licks = np.asarray([ll[np.arange(st,en)] for (st,en,ll) in zip(st_trials ,end_trials,lick_raster )])
    if baseline_subtr:
        base_licks = np.asarray([ll[np.arange(st,en)] for (st,en,ll) in zip(np.int0(np.zeros((len(st_trials)))),stimulus_samp,lick_raster )])
        base_licks = np.repeat( np.nanmean(base_licks,axis=1)[...,np.newaxis],resp_licks.shape[1],axis=1)
        resp_licks = resp_licks- base_licks
    time_trial =np.linspace(window_licks[0],window_licks[-1],len(np.arange(plot_samples[0],plot_samples[-1])))
    return resp_licks, time_trial
