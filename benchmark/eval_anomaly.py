import os
import json
import time
import argparse
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from .anomaly_detection.knn import run_KNN
from .anomaly_detection.evaluation.metrics import get_metrics
from .util.tools import create_directory

from TSB_Symbolic.onennclassifier.sax_classifier import SAXDictionaryClassifier
from TSB_Symbolic.onennclassifier.sfa_classifier import SFADictionaryClassifier
from TSB_Symbolic.onennclassifier.spartan_classifier import SPARTANClassifier

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=False, default="../Univariate_ts/")
    parser.add_argument("-p", "--problem", required=False, default="NAB")  # see data_loader.regression_datasets
    parser.add_argument("-c", "--classifier", required=False, default="sax")  # see regressor_tools.all_models
    parser.add_argument("-g", '--config',required=False,default='./configs/sax/sax_boss.json')
    parser.add_argument("-i", "--itr", required=False, default=13)
    parser.add_argument("-n", "--norm", required=False, default="zscore")  # none, standard, minmax
    parser.add_argument("-s","--save_model",required=False, default=None)
    parser.add_argument("-r","--skip_repeat",required=False,default=True)
    parser.add_argument("-m","--dataset_num",required=False,default=1, type=int)
    parser.add_argument("-w","--store_words",default=None)
    parser.add_argument("-e","--repeat_num",default=10, type=int)
    parser.add_argument("-b","--data_split",default='split', type=str, choices=['split', 'merge'])
    parser.add_argument("-t","--repr_type",default='single', type=str, choices=['single', 'bop'])


    arguments = parser.parse_args()
    return arguments

if __name__ == "__main__":

    arguments = parse_arguments()
    module = 'SymbolicRepresentationExperiments'

    data_path = arguments.data
    classifier_name = arguments.classifier
    normalization = arguments.norm
    problem = arguments.problem
    itr = arguments.itr
    config = arguments.config
    skip_repeat = arguments.skip_repeat
    data_id = arguments.dataset_num
    repeat_num = arguments.repeat_num
    data_split = arguments.data_split
    repr_type = arguments.repr_type


    print("[{}] Data Dir: {}".format(module,data_path))
    print("[{}] Problem: {}".format(module,problem))
    print("[{}] Classifier: {}".format(module,classifier_name))
    #Load Model Config
    if config is not None:
        model_kwargs = json.load(open(config))
        print("[{}] Model Args: {}".format(module,model_kwargs))

    output_directory = "output/anomaly/result/"
    output_directory = output_directory + classifier_name + '/' + data_path.split("/")[-1] + "-" + problem.split(".out")[0] + '/itr_' + str(itr) + '/'


    img_savedir =  "output/anomaly/vis/"
    img_savedir = img_savedir + classifier_name + '/' + data_path.split("/")[-1] + "-" + problem.split(".out")[0] + '/itr_' + str(itr) + '/'
    create_directory(img_savedir)

    create_directory(output_directory)

    window_size = 100
    model_kwargs['window_size'] = window_size

    if classifier_name == 'sax':
        if config is None:
            clf = SAXDictionaryClassifier(save_words = True)
        else:
            clf = SAXDictionaryClassifier(**model_kwargs)
    elif classifier_name == 'sfa':
        if config is None:
            clf = SFADictionaryClassifier(save_words=True)
        else:
            clf = SFADictionaryClassifier(**model_kwargs)
    elif classifier_name == 'spartan':
        clf = SPARTANClassifier(**model_kwargs)

    comp_start = time.time()

    # Specify Anomaly Detector to use and data directory
    AD_Name = 'KNN'

    data_direc = os.path.join(data_path, problem)

    # Loading Data
    df = pd.read_csv(data_direc).dropna()
    data = df.iloc[:, 0:-1].values.astype(float).reshape(1, -1)
    label = df.iloc[:, -1].astype(int).to_numpy()

    data = (data - np.mean(data)) / np.std(data)
    print(f"[{module}] Input shape: {data.shape}")

    X_train_transform = X_test_transform = data.copy()
    y_train_transformed = np.zeros(1)
    clf.fit(X_train_transform,y_train_transformed)
    _ = clf.predict(X_test_transform)

    sliding_symbol = clf.predict_words_bps

    print(f"[{module}] Symbolic Rep shape: {sliding_symbol.shape}")

    # make prediction
    output = run_KNN(data.reshape(-1,1), sliding_symbol[0], n_neighbors=50, method='largest', slidingWindow=model_kwargs["window_size"], metric='l1')

    # Evaluation
    comp_end = time.time()
    evaluation_result = get_metrics(output, label)

    # plot figures
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6))

    data = data.reshape(-1, 1)
    x = np.arange(len(data))
    mask = label > 0

    data_xmin = 0
    data_xmax = len(output)

    ax1.set_title("Raw Time Series")
    ax1.plot(x[data_xmin:data_xmax], data[data_xmin:data_xmax])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Value")
    ax1.plot(x[mask], data[mask], color='red')

    ax2.set_title(f"{classifier_name}")
    ax2.plot(x[data_xmin:data_xmax], output[data_xmin:data_xmax])
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Anomaly Score")
    ax2.set_ylim([0,1.0])

    fig.tight_layout(pad=2.0)

    img_savepath = os.path.join(img_savedir, "anomaly_prediction.jpg")
    plt.savefig(img_savepath, dpi=300)
    plt.show()
    plt.close()

    anomaly_xmin = np.min(x[mask])
    anomaly_xmax = np.max(x[mask])
    anomaly_range = anomaly_xmax - anomaly_xmin
    anomaly_xmin = int(max(0,anomaly_xmin - 0.5*anomaly_range))
    anomaly_xmax = int(min(max(x),anomaly_xmax + 0.5*anomaly_range))

    plt.figure(figsize=(5,6))

    plt.plot(x[anomaly_xmin:anomaly_xmax], data[anomaly_xmin:anomaly_xmax])
    plt.plot(x[mask], data[mask], color='red') # , linewidths=1
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.xticks(np.arange(anomaly_xmin, anomaly_xmax+1, 500))
    plt.savefig(os.path.join(img_savedir, "anomaly_zoom.jpg"), dpi=300)
    plt.show()
    plt.close()

    plt.figure(figsize=(5,6))
    plt.plot(x[anomaly_xmin:anomaly_xmax], output[anomaly_xmin:anomaly_xmax])
    plt.xlabel("Time")
    plt.ylabel("Anomaly Score")
    plt.ylim([0,1.0])
    plt.xticks(np.arange(anomaly_xmin, anomaly_xmax+1, 500))
    plt.savefig(os.path.join(img_savedir, "anomaly_output_zoom.jpg"), dpi=300)
    plt.show()
    plt.close()

    results = pd.DataFrame(
        [evaluation_result]
    )
    model_params = pd.DataFrame([model_kwargs])
    model_params['runtime'] = comp_end - comp_start
    results = pd.concat([results,model_params],ignore_index=False,axis=1).fillna(0)

    print("Final results\n")

    print(results)

    filename = output_directory + 'anomaly_results.csv'
    with open(filename, 'a') as f:
        results.to_csv(f, mode='a', header=f.tell()==0,index=False)