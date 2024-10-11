import requests
import json
import pandas as pd
import numpy as np


def send_print_statement(message):
    url = 'http://74.234.35.205:5000/log'
    data = {'message': message}
    requests.post(url, json=data)



def send_dataframe(df, filename):
    url = 'http://74.234.35.205:5000/save_dataframe'
    if isinstance(df, pd.DataFrame):
        data = df.to_json(orient='split')
    elif isinstance(df, np.ndarray):
        data = pd.DataFrame(df).to_json(orient='split')
    else:
        raise ValueError("Input must be a pandas DataFrame or a numpy array.")    
    payload = {
        'data': data,
        'filename': filename
    }
    json_data = json.dumps(payload)
    headers = {'Content-Type': 'application/json'}
    requests.post(url, data=json_data, headers=headers)
    