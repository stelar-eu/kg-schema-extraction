# import flask 
from flask import Flask, request, jsonify, make_response

import threading
import time
import sqlite3
import subprocess
import json 

# import all functions from functions.py
from functions import *

# Load the configuration file
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# insert here the correct paths to the folders
result_folder = config['result_folder']
pickle_folder = config['pickle_folder']
comparison_folder = config['comparison_folder']

app = Flask(__name__)

def dbscan_handler(cs_method, similarity, eps, min_samples, file_input=None):
    """
    This function handles the dbscan extraction method requests. 
    It creates the final hierarchy and saves it in a pickle file and a json file.
    """
    if file_input == None:
        filename = config['test_filename']
    else:
        filename = file_input
    metric = similarity
    final_hierarchy = cluster_api_DBSCAN(filename, float(eps), int(min_samples), str(metric))
    timestamp = str(int(time.time()))
    tmp_title = "nobelprize_dbscan_"+str(metric)+"_"+timestamp
    tmp_analysis_loc = result_folder+tmp_title+".json"
    final_hierarchy.get_description_json(tmp_analysis_loc)
    # pickle the file
    tmp_title_pickle = pickle_folder+tmp_title+".pkl"
    with open(tmp_title_pickle, 'wb') as f:
        pickle.dump(final_hierarchy, f)
    return "Files Generated"

def frequent_itemset_handler(min_support, min_similarity, file_input=None):
    """
    This function handles the frequent itemset extraction method requests. 
    It creates the final hierarchy and saves it in a pickle file and a json file.
    """
    if file_input == None:
        filename = filename = config['test_filename']
    else:
        filename = file_input
    final_hierarchy = cluster_api_frequent_itemsets(filename, float(min_support), min_similarity)
    timestamp = str(int(time.time()))
    tmp_title = "nobelprize_frequent_itemset_"+timestamp
    tmp_analysis_loc = result_folder+tmp_title+".json"
    final_hierarchy.get_description_json(tmp_analysis_loc)
    # pickle the file
    tmp_title_pickle = pickle_folder+tmp_title+".pkl"
    with open(tmp_title_pickle, 'wb') as f:
        pickle.dump(final_hierarchy, f)
    return "Files Generated" 


def comparison_handler(file_1, file_2):
    """
    Generates a comparison between two files in .json format
    Stores it in the comparison folder
    """
    # load the two files
    file_1 = pickle_folder+file_1
    file_2 = pickle_folder+file_2
    # load the two files
    with open(file_1, 'rb') as f:
        comp_file_1 = pickle.load(f)
    with open(file_2, 'rb') as f:
        comp_file_2 = pickle.load(f)
    # compare the two files
    comp_title = comparison_folder+"comparison_"+str(time.time())+".json"
    get_comparison_json(comp_file_1.one_hot_matrix, comp_file_1.cluster_cs_dict, comp_file_1.weight_matrix, comp_file_2.one_hot_matrix, comp_file_2.cluster_cs_dict, comp_file_2.weight_matrix, comp_title)
    return comp_title


# base URL route to receive schema extraction requests
# the route receives a [POST] request
@app.route('/', methods=['POST'])
def index():
    # parse the request body as JSON
    data = request.json
    # make a dict from the json file 
    data = dict(data)
    # if the extraction method is dbscan
    if data['extraction-method'] == 'dbscan':
        #similarity = "jaccard"
        similarity = data['similarity']
        cs_method = data['cs-method']
        eps = data['input-0']
        min_samples = data['input-1']
        # create a thread for dbscan
        t = threading.Thread(target=dbscan_handler, args=(cs_method, similarity, eps, min_samples))
        t.start()
    elif data['extraction-method'] == 'frequent-itemset':        
        min_support = data['input-0']
        min_similarity = data['input-1']
        # create a thread for frequent-itemset
        t = threading.Thread(target=frequent_itemset_handler, args=(min_support, min_similarity))
        t.start()
    # create a response with code 200 (OK)
    # and contents 'Data received'
    response = make_response('Data received', 200)
    return response

# base URL route to receive schema comparison requests
# the route receives a [POST] request
@app.route('/compare', methods=['POST'])
def compare():
    print("Started comparing!")
    data = request.json
    # make a dict from the json file 
    data = dict(data)
    file_1 = data['file_1']
    file_2 = data['file_2']

    file_1_new = file_1.replace(".json", ".pkl")
    file_2_new = file_2.replace(".json", ".pkl")

    comp_location = comparison_handler(file_1_new, file_2_new)

    response_data = {
        'comp_location': comp_location
    }

    return jsonify(response_data), 200

if __name__ == '__main__':

    print("App is running!")
    app.run()