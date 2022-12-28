
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.pipeline
import sklearn.decomposition
import scipy.stats as scistats
import scipy.io as scio
from scipy import linalg


#import ssm.plots
from utils import *
from plots import *
from sql_utils import *
import cluster_metrics.metrics as metrics
from cluster_metrics.params import QualityMetricsParams


# Initialize
params = vars(QualityMetricsParams())
nas_drive = 'Z:'
protocol_name = 'BeliefState'
table_name = 'neuron'
sql_path = os.path.join(nas_drive,'Sandra','db',protocol_name,'session_log_config.sqlite')
fields = '"name","date","cluId","cluGro","contam_rate"'
type_data = 'Spikes'
type_dir = 'extracted_su'
con = create_connection(sql_path)
cur = con.cursor()
rows = get_sql_rows_by_fields(cur, table_name, fields)
update_fields = ['isi_viol','contam_rate','presence_ratio','single_unit_mat']
for irow in rows:
    cont_r = irow[4] 
    date =  irow[1]
    if len(date)<8:
        date = '0'+ date
    if np.int0(date[0:3])<202:
        date = date[4:] +date[:4]
        condition = 'name = "' + mouse  + '" AND date = '+  irow[1]  + ' AND cluId =' + str(clu_id) 
        update_table_field(cur, con, table_name, 'date', str(date), condition)  
    if cont_r is None:
        mouse = irow[0]
        clu_id =  irow[2]
        clu_gr = get_cluster_group_name(np.int0( irow[3]) -1)
        dir_ = get_data_dir(nas_drive, protocol_name, mouse, type_dir, type_data)
        fname_ = protocol_name + '_' + mouse + '_' +  str(date) + '_' + str(clu_id) + '_npx_formatted_' + clu_gr + '.mat'  
        sp = loadmat(os.path.join(dir_,fname_))
        spike_times = np.asarray(sp['responses']['spikes'])
        if len(spike_times.shape)>0:
            isi_viol, num_viol, contam_rate, presence_ratio, firing_rate = metrics.calculate_metrics(spike_times, params)
            condition = 'name = "' + mouse  + '" AND date = '+  irow[1]  + ' AND cluId =' + str(clu_id) 
            update_values = [isi_viol,contam_rate,presence_ratio, fname_]
            [update_table_field(cur, con, table_name, ifi, ivi, condition) for (ifi,ivi) in zip(update_fields,update_values) ]
    
cur.close()

