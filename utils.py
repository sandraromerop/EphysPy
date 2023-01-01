import scipy.io as scio
import os 
import pickle
import mat73
import numpy as np
import csv
import matplotlib.pyplot as plt

import sklearn.pipeline
import sklearn.decomposition
import scipy.stats
import errno
import json
import datetime

#from glm.model import *
# from bases_tools import *
# from scipy import linalg
# from scipy import signal
# from numpy.random import default_rng


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    
    from: `StackOverflow <http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries>`_
    '''
    try:
        data = scio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    except:
        data = mat73.loadmat(filename)

    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], scio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def get_data_dir(nas_drive, protocol, mouse, type_dir, type_data):


    if type_dir == 'extracted_su':
        dir_ = os.path.join(nas_drive, 'Sandra','Extracted_Data', protocol, type_data, mouse, 'SingleUnitData')
    elif type_dir == 'extracted_session' and type_data=='Photometry':
        dir_ = os.path.join(nas_drive, 'Sandra','Extracted_Data', protocol, type_data, mouse, 'SessionData')
    elif type_dir == 'extracted_session' and type_data=='Bpod':
        dir_ = os.path.join(nas_drive, 'Sandra','Extracted_Data', protocol, type_data, mouse )
    elif type_dir == 'extracted_session' and type_data=='Bonsai':
        dir_ = os.path.join(nas_drive, 'Sandra','Extracted_Data', protocol, type_data, mouse )
    elif type_dir == 'raw' and type_data=='Bonsai':
        dir_ = os.path.join(nas_drive, 'Sandra','Data', protocol, type_data, mouse, 'Data')
    elif type_dir == 'raw' and type_data=='Bpod':
        dir_ = os.path.join(nas_drive, 'Sandra','Data', protocol, type_data, mouse,protocol, 'SessionData')
        



    return dir_

def get_cluster_group_name(clu_group):
    gr_names = ['mua','good']

    return  gr_names[clu_group]


def indeces_for_conditions(condit_list,ss):
    id_process = np.arange(len(ss.name))
    for ic in condit_list:
        id_process = np.intersect1d(id_process,np.argwhere(ic))
    
    return id_process

def read_cvs_row(full_name, irow):
    with open(full_name,'r') as file:
            reader = csv.reader(file)
            gp = []
            for row in reader:
                if len(row)>0:
                    gp.append(row[irow])
            
    return gp

def get_gpio_idx(gp, thres_gpio):
    gp = np.array([np.float64(g) for g in gp])
    gp_norm = gp/np.max(gp)
    gp_norm[0] = thres_gpio+0.1
    detection_points = np.float64(np.argwhere(gp_norm > thres_gpio).squeeze())
    if not np.isscalar(detection_points):
        change_points = np.concatenate([[0],np.diff(detection_points)])
        gpio_trials = detection_points[change_points > 1]
    else:
        gpio_trials = None

    return gpio_trials

def load_pickle(file_name):
    with open(file_name, 'rb') as handle:
        b = pickle.load(handle)
    
    return b


def get_protocol_phase_BS(irow,name_sessions,date_sessions,prot_sessions,phase_sessions):
    m_ = irow[0]
    d_ = irow[1]
    id_session = np.intersect1d(np.argwhere([ns==m_ for ns in name_sessions]),np.argwhere([ds==d_ for ds in date_sessions]))
    if len(id_session)>1: id_session = id_session[0]
    protocol_ = prot_sessions[id_session]
    phase_ = phase_sessions[id_session]
    if protocol_ =='BeliefState':
        protocol_ = 'BeliefState'
    if phase_ =='BeliefState':
        phase_= 'BeliefState_Task1'
    
    if np.double(d_[0:4])>2020:
        batch_ = 2
    else:
        batch_ = 1

    return protocol_, phase_, batch_


def get_indeces_from_time(time_stamps,time0,psth_resolution,window):
    indices_ = np.int0( 1 + np.round((time_stamps - time0) / psth_resolution) - window[0] / psth_resolution)

    return indices_

def get_trend(y,newx):
    plt.style.use('seaborn-poster')

    x = np.linspace(0, 1, len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    y = y[:, np.newaxis] # turn y into a column vector
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y) # Direct least square regression
    trend_= newx*alpha[0]+alpha[1]

    return trend_

def get_trend_param(y,x=None):
    if x is None:
        x = np.linspace(0, 1, len(y))
    A = np.vstack([x, np.ones(len(x))]).T
    y = y[:, np.newaxis] # turn y into a column vector
    alpha = np.dot((np.dot(np.linalg.inv(np.dot(A.T,A)),A.T)),y) # Direct least square regression

    return alpha 

def get_trend_from_params(m,b,newx):
    trend_= newx*m+b

    return trend_

def initialize_sp_arrays(psth_length,total_session_length,n_neurons_this_sess,n_trials):
    sp_bins_trials_zsc =  [np.zeros((psth_length+1,n_neurons_this_sess))] * n_trials
    sp_bins_trials_zsc_detr =  [np.zeros((psth_length+1,n_neurons_this_sess))] * n_trials
    sp_bins_trials  =  [np.zeros((psth_length+1,n_neurons_this_sess))] * n_trials
    sp_bins = np.zeros((total_session_length+1,n_neurons_this_sess))
    sp_bins_zsc = np.zeros((total_session_length+1,n_neurons_this_sess))
    sp_bins_zsc_detr = np.zeros((total_session_length+1,n_neurons_this_sess))
    input_bins_trials =  [np.zeros((psth_length+1,2))] * n_trials
    input_full = np.zeros((total_session_length+1,2))

    return sp_bins_trials,sp_bins_trials_zsc,sp_bins_trials_zsc_detr,sp_bins,sp_bins_zsc,sp_bins_zsc_detr,input_bins_trials,input_full

def smoothing_func(t,smoothingTimeConst):
    kern =  (1-np.exp(-t))*np.exp(-t/smoothingTimeConst)

    return kern

def define_inputs(data_df, together, tau, n_bases=8):
    """Create input matrix to the rSLDS from cue and reward
    """

    if together:
        inputs = data_df[["bins_cue", "bins_reward"]].to_numpy().T
    else:
        inputs = data_df[["bins_cue_1", "bins_cue_2",
                                "bins_cue_3", "bins_reward"]].to_numpy().T

    exp_inputs = []
    for input in inputs:
        exp_inputs.append(np.convolve(input, np.array(
            [(np.exp(-i*tau)) for i in range(n_bases)]), mode="full"))
            
    return np.array(exp_inputs)[:,:-int(n_bases-1)]

def display_space(data_full, columns=["x1", "x2"], ax_mins = [-4, -10], ax_maxs = [13, 2], space="Latent space", sampling_rate=0.1, cue=None):
    
    if cue is not None: 
        data_cue = data_full.loc[data_full["bins_cue_type"]==cue].reset_index()
    else:
        data_cue = data_full.copy()
    
    # Extract dynamics matrix from the true model
    sample_trials = random.choices(
        data_cue["bins_trial_numb"].unique(), k=3)

    print(sample_trials)

    max_trial_bins = 100

    # get random trials
    latent_sets = []
    for idx in range(len(sample_trials)):
        data_selected = data_cue.loc[(data_cue["bins_trial_numb"] == int(sample_trials[idx]))]
        latent_sets.append(data_selected)
        
    A_est = rslds.dynamics.A
    b_est = rslds.dynamics.b

    _, axs = plt.subplots(1, 3, constrained_layout=True, figsize=(16, 4))

    #Plot the dynamics vector field
    for idx, data in enumerate(latent_sets):
        cue_type = data.head(1)["bins_cue_type"].to_numpy()[0]
        
        if cue_type < 8:
            reward_cumcount = data.loc[(data["bins_reward"]==1)]["cumcount"].to_numpy()[0]
        cue_cumcount = data.loc[(data["bins_cue"]==1)]["cumcount"].to_numpy()[0]
        data = data[columns].to_numpy()

        ssm.plots.plot_dynamics_2d(A_est,
                b_est,
                mins=ax_mins,
                maxs=ax_maxs,
                #mins=data.min(axis=0),
                #maxs=data.max(axis=0),
                axis=axs[idx])

        axs[idx].plot(data[:cue_cumcount+1, 0], data[:cue_cumcount+1, 1], ".-", lw=1, label="lateITI", color="cornflowerblue")
        axs[idx].scatter(data[cue_cumcount, 0], data[cue_cumcount, 1], lw=1, label="cue", color="b")
        
        if cue_type < 8:
            axs[idx].plot(data[cue_cumcount:reward_cumcount+1, 0], data[cue_cumcount:reward_cumcount+1, 1], ".-", lw=1, label="ISI", color="lightcoral")
            axs[idx].scatter(data[reward_cumcount, 0], data[reward_cumcount, 1], lw=1, label="reward", color="firebrick")
            axs[idx].plot(data[reward_cumcount:max_trial_bins, 0], data[reward_cumcount:max_trial_bins, 1], ".-", lw=1, label="earlyITI", color="lightseagreen")
        else: 
            axs[idx].plot(data[cue_cumcount:max_trial_bins, 0], data[cue_cumcount:max_trial_bins, 1], ".-", lw=1, label="earlyITI", color="lightseagreen")
        
        axs[idx].set_ylim(ax_mins[1], ax_maxs[1])
        axs[idx].set_xlim(ax_mins[0], ax_maxs[0])
        axs[idx].set_xlabel("$x_1$")
        axs[idx].set_ylabel("$x_2$")
        axs[idx].set_title(f"{space} for trial {sample_trials[idx]} cue {cue_type}")

    plt.legend()
    plt.show()

def get_indeces(event_time,st_time_s,tr_win,psth_resolution):
    ids =  np.int0(np.round((event_time - st_time_s) / psth_resolution) - tr_win[0] / psth_resolution)      

    return ids

def evaluate_drift(X_df, neurons, interval=[0,20], fig=None, label="[0,20]"):
    """Compute PCA on trial means and plot two first PCs for given interval
    """

    # mean for each neuron and each trial
    neurons_mean = X_df.loc[(X_df["cumcount"]>interval[0]) & (X_df["cumcount"]<interval[1])].groupby("bins_trial_numb").mean()[neurons]

    # PCA 
    pipe = sklearn.pipeline.Pipeline([('scaler', sklearn.preprocessing.StandardScaler(
        )), 
        ('pca', sklearn.decomposition.PCA(n_components=2))])

    xhat_pca = pipe.fit_transform(neurons_mean.to_numpy())
    neurons_mean["pc1"] = xhat_pca[:,0]
    neurons_mean["pc2"] = xhat_pca[:,1]

    if fig is None:
        fig = plt.figure(1)
    else: 
        fig = fig
    plt.plot(neurons_mean["pc1"], neurons_mean["pc2"], ".-", label=label)
    plt.legend()
    
    if fig is None:
        plt.legend()
        plt.show()
    return fig, neurons_mean

def plot_drift_set(X_new, neurons):
    fig = plt.figure(1)
    X_new_cue = X_new.loc[X_new["bins_cue_type"]==1]
    fig, neurons_mean = evaluate_drift(X_new_cue, neurons, interval=[0,20], fig=fig, label="[0,20]") # mostly in PC1
    fig, neurons_mean = evaluate_drift(X_new_cue, neurons, interval=[33,41], fig=fig, label="[33,41]") # mostly in PC1
    fig, neurons_mean = evaluate_drift(X_new_cue, neurons, interval=[43,50], fig=fig, label="[43,50]") # mostly in PC1
    #fig, neurons_mean = evaluate_drift(X_new_cue, neurons_zscore, interval=[0,90], fig=fig, label="[0,90]") # mostly in PC1
    fig.show()

    neurons_mean[neurons].plot(legend=False)
    plt.show()


# # # # Helper functions for plotting results
def plot_trajectory(z, x, ax=None, ls="-"):
    zcps = np.concatenate(([0], np.where(np.diff(z))[0] + 1, [z.size]))
    fig = plt.figure(figsize=(4, 4))
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.gca()
    for start, stop in zip(zcps[:-1], zcps[1:]):
        ax.plot(x[start:stop + 1, 0],
                x[start:stop + 1, 1],
                lw=1, ls=ls,
                color=colors[z[start] % len(colors)],
                alpha=1.0)
    return ax


def plot_most_likely_dynamics(model,
    xlim=(-4, 4), ylim=(-3, 3), nxpts=20, nypts=20,
    alpha=0.8, ax=None, figsize=(3, 3)):
    
    K = model.K
    assert model.D == 2
    x = np.linspace(*xlim, nxpts)
    y = np.linspace(*ylim, nypts)
    X, Y = np.meshgrid(x, y)
    xy = np.column_stack((X.ravel(), Y.ravel()))

    # Get the probability of each state at each xy location
    z = np.argmax(xy.dot(model.transitions.Rs.T) + model.transitions.r, axis=1)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)

    for k, (A, b) in enumerate(zip(model.dynamics.As, model.dynamics.bs)):
        dxydt_m = xy.dot(A.T) + b - xy

        zk = z == k
        if zk.sum(0) > 0:
            ax.quiver(xy[zk, 0], xy[zk, 1],
                      dxydt_m[zk, 0], dxydt_m[zk, 1],
                      color=colors[k % len(colors)], alpha=alpha)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

    plt.tight_layout()

    return ax



class DataProcessor(object):

    """Extracts data from given python-friendly formatted dataset.

    Parameters
    ----------
    path : str
        Path to data directory. Must contain a spikes subdirectory with spikes in ms.
    window : ndarray
        Array that holds timing information including the beginning and
        end of the region of interest and the time bin. All in milliseconds.
        If not supplied, it is assumed that trial lengths are unequal and will be loaded from file.
    cell_range : range
        Beginning and end cell to be analyzed, in range format.

    Attributes
    ----------
    path : str
        Path to data directory. Must contain a spikes subdirectory with spikes in ms.
    window : ndarray
        Array that holds timing information including the beginning and
        end of the region of interest and the time bin. All in milliseconds.
    cell_range : range
        Beginning and end cell to be analyzed, in range format.
    num_conditions : int
        Integer signifying the number of experimental conditions in the
        dataset.
    spikes : numpy.ndarray
        Array of spike times in milliseconds, of dimension (trials × time).
    num_trials : numpy.ndarray
        Array containing integers signifying the number of trials a given cell has data for.
    spikes_summed : dict (int: numpy.ndarray)
        Dict containing summed spike data for all cells, indexed by cell.
    spikes_binned : dict (int: numpy.ndarray)
        Dict containing binned spike data for all cells, indexed by cell.
    spikes_summed_cat : dict (int: numpy.ndarray)
        Dict containing binned spike data for all cells, indexed by cell and category.
    conditions_dict : dict (tuple of int, int: numpy.ndarray of int)
        Dict containing condition information for all cells.
        Indexed by cell and condition.

    """

    def __init__(self, path, cell_range, window=None):
        self.path = path
        self._check_results_dir(path)
        self.cell_range = cell_range
        self.spikes = self._extract_spikes()
        self.num_trials = self._extract_num_trials()
        conditions = self._extract_conditions()
        if conditions is not None:
            # finds total number of different conditions in supplied file
            self.num_conditions = len(
                set([item for sublist in list(conditions.values()) for item in sublist]))
        self.conditions_dict = self._associate_conditions(conditions)
        # if window is not provided, a default window will be constructed
        # based off the min and max values found in the data
        if window:
            print("Time window provided. Assuming all trials are of equal length")
            num_cells = len(self.cell_range)
            self.window = {}
            for cell in self.cell_range:
                if self.num_trials[cell] == 0:
                    print("cell with no trials or spikes detected")
                    self.num_trials[cell] = 1
                min_time = np.full((max(self.num_trials.values())), window[0])
                max_time = np.full((max(self.num_trials.values())), window[1])
                self.window[cell] = np.stack((min_time, max_time), axis=1)
        elif not window:
            self.window = self._extract_trial_lengths()
        self.spike_info = self._extract_spike_info()
        self.spikes_binned = self._bin_spikes()
        self.spikes_summed = self._sum_spikes()
        self.spikes_summed_cat = self._sum_spikes_conditions(conditions)

    def _check_results_dir(self, path):
        """Creates directories for artifacts if they don't exist.

        """
        os.makedirs(path+"/results/figs/", mode=0o777, exist_ok=True)

    def _set_default_time(self):
        """Parses spikes and finds min and max spike times for bounds.

        """
        max_time = sys.float_info.min
        min_time = sys.float_info.max
        for cell in self.spikes:
            for trial in self.spikes[cell]:
                for t in trial:
                    if t > max_time:
                        max_time = t
                    if t < min_time:
                        min_time = t
        return [min_time, max_time]

    def _extract_trial_lengths(self):
        """Extracts trial lengths from file.

        """
        path = self.path + "/trial_lengths.json"
        window = {}
        try:
            with open(path, 'r') as f:
                trial_lengths = np.array(json.load(f))
                for i, cell in enumerate(trial_lengths):
                    window[i] = np.array(cell, dtype=int)
                    # trial_lengths[i][:, 0] = int(trial_lengths[i][:,0])
                return window
        except:
            raise(FileNotFoundError("trial_lengths.json not found"))
            return None

    def _extract_spikes(self):
        """Extracts spike times from data file.

        Returns
        -------
        dict (int: numpy.ndarray of float)
            Contains per cell spike times.

        """
        # HERE:
        spikes = {}
        if os.path.exists(self.path + "/spikes/"):
            for i in self.cell_range:
                spike_path = self.path + '/spikes/%d.json' % i
                with open(spike_path, 'rb') as f:
                    spikes[i] = np.array(json.load(f))
        else:
            print("Spikes folder not found.")
            raise FileNotFoundError(
                errno.ENOENT, os.strerror(errno.ENOENT), self.path+"/spikes/")

        return spikes

    def check_unit_trials(self, num_trials):
        if num_trials == 0:
            return False
        else:
            return True

    def _extract_num_trials(self):
        """Extracts number of trials per cell.

        Returns
        -------
        numpy.ndarray of int
            Array of dimension [Number of cells] that provides the number of
            trials of a given cell.

        """
        num_trials = {}
        if os.path.exists(self.path + "/number_of_trials.json"):
            with open(self.path + "/number_of_trials.json", 'rb') as f:
                
                # nt_load = json.load(f, encoding="bytes")
                nt_load = json.load(f)
                for cell in self.cell_range:
                    loaded_trials = nt_load[cell]
                    if self.check_unit_trials(loaded_trials):
                        num_trials[cell] = nt_load[cell]
                    else:
                        raise ValueError(
                            "cell has 0 trials, check number_of_trials.json"
                        )
        else:
            for cell in self.cell_range:
                calc_trials = len(self.spikes[cell])
                if self.check_unit_trials(calc_trials):
                    num_trials[cell] = calc_trials
                else:
                    raise ValueError(
                        "cell has 0 trials, check input spikes"
                    )

        return num_trials

        

    def _extract_conditions(self):
        """Extracts trial conditions per cell per trial.

        Returns
        -------
        numpy.ndarray of int
            Array of dimension [Number of cells] × [Trials] that provides the condition
            for the trial.

        """
        # convert keys to int
        if os.path.exists(self.path + "/conditions.json"):
            with open(self.path + "/conditions.json", 'rb') as f:
                loaded = json.load(f)
                if type(loaded) is dict:
                    return {int(k): v for k, v in loaded.items()}
                if type(loaded) is list:
                    return {int(k): v for k, v in enumerate(loaded)}

        else:
            print("conditions.json not found")

            return None

    def _extract_spike_info(self):
        """Extracts spike info from file.

        """
        if os.path.exists(self.path+"/spike_info.json"):
            with open(self.path + "/spike_info.json", 'rb') as f:
                data = json.load(f)

            return data

        else:
            print("spike_info not found")
            return None

    def _sum_spikes(self):
        """Sums spike data over trials.

        Returns
        -------
        dict (int: numpy.ndarray of int)
            Summed spike data.

        """
        spikes = self.spikes_binned
        summed_spikes = {}
        for cell in self.cell_range:
            summed_spikes[cell] = np.nansum(spikes[cell], 0)

        return summed_spikes

    def _sum_spikes_conditions(self, conditions):
        """Sums spike data over trials per condition

        Parameters
        ----------
        conditions : numpy.ndarray of int
            Array of dimension [Number of cells] × [Trials] that provides the condition
            for the trial.

        Returns
        -------
        numpy.ndarray of int
            Array of dimension [Cells] × [Condition] × [Time].

        """
        spikes = self.spikes_binned

        if conditions is None:
            return None
        else:
            summed_spikes_condition = {}
            for cell in self.cell_range:
                summed_spikes_condition[cell] = {}
                for condition in range(self.num_conditions):
                    summed_spikes_condition[cell][condition+1] = {}
                    summed_spikes_condition[cell][condition+1] = np.sum(
                        spikes[cell].T * self.conditions_dict[cell][condition + 1].T, 1)

            return summed_spikes_condition

    def _bin_spikes(self):
        """Bins spikes within the given time range into 1 ms bins.

        """
        spikes_binned = {}
        max_upper = 0
        min_lower = np.inf
        # for cell in self.window:
        #     if max(self.window[cell][:, 1]) > max_upper:
        #         max_upper = max(self.window[cell][:, 1])
        #     if min(self.window[cell][:, 0]) < min_lower:
        #         min_lower = min(self.window[cell][:,0])
        # total_bins = int(max_upper) - int(min_lower)
        for cell in self.spikes:
            
            max_upper = max(self.window[cell][:, 1])
            min_lower = min(self.window[cell][:,0])
            total_bins = int(max_upper) - int(min_lower)
            lower_bounds, upper_bounds = self.window[cell][:, 0], self.window[cell][:, 1]
            
            spikes_binned[cell] = np.zeros(
                (int(self.num_trials[cell]), total_bins))
            for trial_index, trial in enumerate(self.spikes[cell][:self.num_trials[cell]]):
                time_low, time_high = lower_bounds[trial_index], upper_bounds[trial_index]
                # total_bins = time_high - time_low
                if type(trial) is float or type(trial) is int or type(trial) is np.float64:
                    trial = [trial]
                for value in trial:
                    if value < time_high and value >= time_low:
                        spikes_binned[cell][trial_index][int(
                            value - time_low)] = 1
                if trial_index < self.num_trials[cell]:
                    spikes_binned[cell][trial_index][int(upper_bounds[trial_index]- lower_bounds[trial_index]):] = np.nan
                    # print("test1111")

        return spikes_binned

    def _associate_conditions(self, conditions):
        """Builds dictionary that associates trial and condition.

        Returns
        -------
        dict (int, int: np.ndarray of int)
            Dict indexed by condition AND cell number returning array of trials.
            Array contains binary data: whether or not the trial is of the indexed condition.

        """
        if conditions is None:
            return None
        else:
            conditions_dict = {}
            for cell in self.cell_range:
                conditions_dict[cell] = {
                    i+1: np.zeros((self.num_trials[cell], 1)) for i in range(self.num_conditions)}
                cond = conditions[cell][0:self.num_trials[cell]]
                for trial, condition in enumerate(cond):
                    if condition:
                        conditions_dict[cell][condition][trial] = 1

            return conditions_dict

    def save_attribute(self, attribute, filename, path=""):
        """Saves data_processor attribute to disk.

        """
        with open((os.getcwd() + path + "/{0}.json").format(filename), 'w') as f:
            json.dump(attribute, f)
