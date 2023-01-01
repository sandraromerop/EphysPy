import sqlite3
from sqlite3 import Error
import numpy as np
from utils import *


def create_connection(path):
    
    connection = None
    try:
        connection = sqlite3.connect(path)
        print("Connection to SQLite DB successful")
    except Error as e:
        
        print(f"The error '{e}' occurred")
        
    return connection
    


def get_sql_rows_by_fields(cur, table_name, fields, condition=None):

    if condition is not None:
        command = 'SELECT ' + fields + ' FROM ' + table_name+ ' WHERE ' + condition
    else:
        command = 'SELECT ' + fields + ' FROM ' + table_name #+ ' WHERE ' + condition
    rows =  np.asarray(cur.execute(command).fetchall()).squeeze()

    return rows

def update_table_field(cur, con, table_name, field_name, field_value, condition):

    command = 'UPDATE ' + table_name + ' SET "' + field_name  + '" = "' + str(field_value) + '" WHERE ' + condition 
    print(command)
    cur.execute(command)
    con.commit()



def get_essential_sql_info(sql_path):
    table_name = 'neuron'
    fields = '"name","date","brainReg","cluId","cluGro","single_unit_mat","contam_rate","isi_viol","presence_ratio","firing_rate"'
    con = create_connection(sql_path)
    cur = con.cursor()
    rows_neuron = get_sql_rows_by_fields(cur, table_name, fields)
    fields = '"name","file_date","protocol","phase"'
    table_name = 'session'
    con = create_connection(sql_path)
    cur = con.cursor()
    rows_session = get_sql_rows_by_fields(cur, table_name, fields)
    name_sessions = np.asarray([ii[0] for ii in rows_session])
    date_sessions = np.asarray([ii[1] for ii in rows_session])
    prot_sessions = np.asarray([ii[2] for ii in rows_session])
    phase_sessions = np.asarray([ii[3] for ii in rows_session])
    date_ephys = np.asarray([ii[1] for ii in rows_neuron])
    date_sessions = [ii.replace('-','') for ii in date_sessions]
    protocol_phase_batch = [get_protocol_phase_BS(irow,name_sessions,date_sessions,prot_sessions,phase_sessions) for irow in rows_neuron]


    return name_sessions,date_sessions,prot_sessions,phase_sessions,date_ephys,date_sessions,protocol_phase_batch,rows_neuron,rows_session

def get_region_cat_id(d_,icat,brain_reg):
    id_cat = np.intersect1d(np.argwhere(np.asarray(d_['phase']== icat[0])==True),np.argwhere(np.asarray(d_['batch']== icat[1])==True))
    id_units = np.argwhere(np.asarray([brain_reg in bb for bb in d_.brainReg]) == True)
    id_regioncategory = np.intersect1d(id_cat,id_units)

    return id_regioncategory