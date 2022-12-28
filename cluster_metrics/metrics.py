import numpy as np
from cluster_metrics.epoch import Epoch


def calculate_metrics(spike_times, params, epochs = None, verbose=False):
    #  spike_times : numpy.ndarray (num_spikes x 0) Spike times in seconds (same timebase as epochs)
    if epochs is None:
        epochs = [Epoch('complete_session', 0, np.inf)]
        
    for epoch in epochs:
        in_epoch = (spike_times > epoch.start_time) * (spike_times < epoch.end_time)
        
        if verbose is True:
            print("Calculating isi violations")
        isi_viol, num_viol = calculate_isi_violations(spike_times[in_epoch], params['isi_threshold'], params['min_isi'])

        if verbose is True:
            print("Calculating contamination rate")
        contam_rate = calculate_contam_rate(spike_times[in_epoch],  params['tbin_sec'], params['isi_threshold'])

        if verbose is True:
            print("Calculating presence ratio")
        presence_ratio = calculate_presence_ratio(spike_times[in_epoch] )

        if verbose is True:
            print("Calculating firing rate")
        firing_rate = calculate_firing_rate(spike_times[in_epoch])


    return isi_viol, num_viol, contam_rate, presence_ratio, firing_rate

def calculate_presence_ratio(spike_times, num_bins=100):

    """Calculate fraction of time the unit is present within an epoch.

    Inputs:
    -------
    spike_train : array of spike times

    Outputs:
    --------
    presence_ratio : fraction of time bins in which this unit is spiking

    """
    min_time = np.min(spike_times)
    max_time = np.max(spike_times)
    h, b = np.histogram(spike_times, np.linspace(min_time, max_time, num_bins))

    presence_ratio = np.sum(h > 0) / num_bins

    return presence_ratio

def calculate_isi_violations(spike_times, isi_threshold, min_isi):
    
    """Calculate ISI violations for a spike train.

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705
    Modified by Dan Denman  Jennifer Colonell 

    Inputs:
    -------
    spike_times : array of spike times
    min_time : minimum time for potential spikes
    max_time : maximum time for potential spikes
    isi_threshold : threshold for isi violation
    min_isi : threshold for duplicate spikes

    Outputs:
    --------
    viol_rate : rate of contaminating spikes as a fraction of overall rate
        A perfect unit has a fpRate = 0
        A unit with some contamination has a fpRate < 0.5
        A unit with lots of contamination has a fpRate > 1.0
    num_violations : total number of violations

    """
    min_time = np.min(spike_times)
    max_time = np.max(spike_times)

    duplicate_spikes = np.where(np.diff(spike_times) <= min_isi)[0]

    spike_times = np.delete(spike_times, duplicate_spikes + 1)
    isis = np.diff(spike_times)

    num_spikes = len(spike_times)
    num_violations = sum(isis < isi_threshold) 
    violation_time = 2*num_spikes*(isi_threshold - min_isi)
    total_rate = calculate_firing_rate(spike_times, min_time, max_time)
    c = num_violations/(violation_time*total_rate)
    if c < 0.25:        # valid solution to quadratic eq. for fpRate:
        viol_rate  = (1 - np.sqrt(1-4*c))/2
    else:               # no valid solution to eq, call fpRate = 1
        viol_rate  = 1.0
   
    return viol_rate, num_violations

def calculate_contam_rate(spike_times, tbin_sec, isi_threshold):
    '''
    Given a set of spike times in sec, calculate the KS2 contamination percent
    Modified by Jennifer Colonell 
    Differences from the KS2 standard calc:
         -  When calculating an auto-correlogram (acg), as here, only remove self-counted spikes (rather than zeroing the lowest bin)
            This will give higher values for the contamination rate when there are duplicate spikes
    
         -  Instead of just taking the range of the acg with the lowest contamination, take the range corresponding
            to the user specified refractory period. This will also usually give higher values for the contamination rate.
    
    Inputs:
    -------
    spike_train : numpy.ndarray
        Array of spike times in seconds
    tbin_sec :  float
        time bin in seconds for cross-correlogram

    Outputs:
    --------
    Contamination rate: (modified from KS2)  float
        This is the estimated contamination, based on the number of refractory period violations. 
        This is roughly the ratio of the event rate in the central 2ms bin of the histogram to the baseline of the auto-correlogram (the "shoulders"). 
        Some people report the fraction of refractory violations to total number of spikes, which is a meaningless measure for assessing contamination. 
        Cut-off of 20% is what they use in KS2 to qualify a unit as 'good'
    '''

    refPerBin = int(isi_threshold/tbin_sec)
    if refPerBin == 0:
        refPerBin = 1   # if refractory period < bin size, take the first bin
    
    K, Qi, Q00, Q01, rir = ccg(spike_times, spike_times, 500, tbin_sec, True); # compute the auto-correlogram with 500 bins at 1ms bins
    
    normFactor = (max(Q00, Q01))
    
    if normFactor > 0:
        contam_rate  = Qi[refPerBin]/normFactor # get the Q[i] that includes the refractory period
    else:
        contam_rate = 1
    
    return contam_rate

def calculate_firing_rate(spike_train, min_time = None, max_time = None):
    """Calculate firing rate for a spike train.

    If no temporal bounds are specified, the first and last spike time are used.

    Inputs:
    -------
    spike_train : numpy.ndarray
        Array of spike times in seconds
    min_time : float
        Time of first possible spike (optional)
    max_time : float
        Time of last possible spike (optional)

    Outputs:
    --------
    fr : float
        Firing rate in Hz

    """

    if min_time is not None and max_time is not None:
        duration = max_time - min_time
    else:
        duration = np.max(spike_train) - np.min(spike_train)

    fr = spike_train.size / duration

    return fr

def ccg(st1, st2, nbins, tbin, auto):
    
    """ calculate crosscorrelogram between two sets of spike times (st1, st2)
        in seconds, with bin width tbin, time lags = plus/minus nbins.
        Algorithm from Kilosort2, written by Marius Pachitariu
    
    Inputs:
    -------
    st1 : spike times for set #1 in sec
    st2 : spike times for set #2 in sec
    nbins : ccg will be calculated for 2*nbins + 1, 
    tbin : bin width in seconds
    
    Output:
    K = ccg histogram
    Qi
    Q00
    Q01
    
    """
    st1 = np.sort(np.squeeze(st1))
    st2 = np.sort(np.squeeze(st2))
    
    dt = nbins*tbin  # cross correlogram spans -dt-dt
    T = max(np.max(st1),np.max(st2)) - min(np.min(st1),np.min(st2))
    
    ilow, ihigh, j, n_st2, n_st1, K = 0, 0, 0,  len(st2), len(st1), np.zeros((2*nbins+1,))
    # Traverse both spike trains together, keeping track of the spikes in the first
    # spike train that are within dt of the second spike train
   
    while j < n_st2:                      # walk over all spikes in 2nd spike train
        while (ihigh < n_st1) and (st1[ihigh] < st2[j]+dt):            
            ihigh = ihigh + 1             # increase upper bound until its outisde the dt range
        while (ilow < n_st1) and (st1[ilow] <= st2[j]-dt):
            ilow = ilow + 1                # increase lower bound until it is inside the dt range
        if ilow > n_st1:
            break
        if st1[ilow] > st2[j] + dt:
            # if the lower bound is actually outside of the dt range, means there were no spikes in range of the ccg
            # just move on to next spike st2
            j = j + 1
            continue
        for k in range(ilow,ihigh):
            # for all spikes within the plus/minus dt range
            ibin = int(np.round((st2[j]-st1[k])/tbin))    # calculate which bin
            K[ibin + nbins] = K[ibin + nbins] + 1    # increment corresponding bin in correlogram
        j = j + 1   # go to next spike in st2
        
    if auto:
        # if this is an autocorrelogram, remove the self-found spikes from the zero bin
        K[nbins] = K[nbins] - n_st1     # remove "self found" spikes from 
    
    irange1 = np.concatenate((np.arange(1, int(nbins/2)), np.arange(int(3/2*nbins), 2*nbins-1)),0) # this index range corresponds to the CCG shoulders, excluding end bins
    irange2 = np.arange(nbins-50, nbins-10)  # 40 channels to negative side of peak
    irange3 = np.arange(nbins+10, nbins+50)  # 40 channels to positive side of peak
    
    # Normalize the firing rate in the shoulders by the mean firing rate
    # A Poisson process has a flat ACG (equal numbers of spikes at all ISIs) and these ratios would = 1
    mean_firing_rate = (n_st2)/T
    Q00 = (sum(K[irange1])/(n_st1 * tbin * len(irange1)))/mean_firing_rate
    Q01_neg = (sum(K[irange2])/(n_st1 * tbin * len(irange2)))/mean_firing_rate
    Q01_pos = (sum(K[irange3])/(n_st1 * tbin * len(irange3)))/mean_firing_rate
    Q01 = max(Q01_neg, Q01_pos)
    
    # Get highest spike rate of the sampled time regions
    R00 = max(np.mean(K[irange2]), np.mean(K[irange3])) # Larger of the two shoulders near t = 0
    R00 = max(R00, np.mean(K[irange1])) # compare this to the asymptotic shoulder
    
    # Calculate "refractoriness for periods from 1*tbin to 10*tbin
    Qi = np.zeros((11,))
    Ri = np.zeros((11,))
    for i in range(1,11):
        irange = np.arange(nbins-i,nbins+i)
        Qi[i] = (sum(K[irange])/(2*i*tbin+1))/mean_firing_rate    #rate in this time period/mean rate
        # Marius note: this is tricky: we approximate the Poisson likelihood with a gaussian of equal mean and variance
        # That allows us to integrate the probability that we would see <N spikes in the center of the cross-correlogram from a distribution with mean R00*i spikes        
    return K, Qi, Q00, Q01, Ri