import sqlite3
from sqlite3 import Error
import numpy as np

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



