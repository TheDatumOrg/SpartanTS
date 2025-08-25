import os

import argparse
import json
import time
import numpy as np
import pandas as pd

from .util.dataset import create_numpy_dataset
from .util.normalization import create_normalizer
from .util.tools import create_directory,compute_classification_metrics

from sklearn.preprocessing import LabelEncoder


from TSB_Symbolic.onennclassifier.sax_classifier import SAXDictionaryClassifier
from TSB_Symbolic.onennclassifier.esax_classifier import ESAXDictionaryClassifier
from TSB_Symbolic.onennclassifier.oned_sax_classifier import OneDSAXDictionaryClassifier
from TSB_Symbolic.onennclassifier.tfsax_classifier import TFSAXDictionaryClassifier
from TSB_Symbolic.onennclassifier.sax_dr_classifier import SAXDRDictionaryClassifier
from TSB_Symbolic.onennclassifier.sax_vfd_classifier import SAXVFDDictionaryClassifier
from TSB_Symbolic.onennclassifier.sfa_classifier import SFADictionaryClassifier
from TSB_Symbolic.onennclassifier.spartan_classifier import SPARTANClassifier

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=False, default="data/ucr/Univariate_ts/")
    parser.add_argument("-p", "--problem", required=False, default="ArrowHead")  # see data_loader.regression_datasets
    parser.add_argument("-c", "--classifier", required=False, default="sax")  # see regressor_tools.all_models
    parser.add_argument("-g",'--config',required=False,default='')
    parser.add_argument("-i", "--itr", required=False, default=13)
    parser.add_argument("-n", "--norm", required=False, default="zscore")  # none, standard, minmax
    parser.add_argument("-s","--save_model",required=False, default=None)
    parser.add_argument("-r","--skip_repeat",required=False,default=True)
    parser.add_argument("-m","--dataset_num",required=False,default=1, type=int)
    parser.add_argument("-w","--store_words",default=None)
    parser.add_argument("-t","--downsample",default=1.0, type=float)

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    
    module = 'SymbolicRepresentationExperiments'

    arguments = parse_arguments()
    data_path = arguments.data
    classifier_name = arguments.classifier
    normalization = arguments.norm
    problem = arguments.problem
    itr = arguments.itr
    config = arguments.config
    skip_repeat = arguments.skip_repeat
    data_id = arguments.dataset_num
    downsample_rate = arguments.downsample

    # create output directory
    output_directory = "output/classification/"
    if normalization != None:
        output_directory = "output/classification/"
    output_directory = output_directory + classifier_name + '/' + problem + '/itr_' + str(itr) + '/'
    create_directory(output_directory)

    print("=======================================================================")
    print("[{}] Starting Classification Experiment".format(module))
    print("=======================================================================")
    print("[{}] Data path: {}".format(module, data_path))
    print("[{}] Output Dir: {}".format(module, output_directory))
    print("[{}] Iteration: {}".format(module, itr))
    print("[{}] Problem: {} | {}".format(module, data_id, problem))
    print("[{}] Classifier: {}".format(module, classifier_name))
    print("[{}] Config: {}".format(module,config))
    print("[{}] Normalisation: {}".format(module, normalization))

    #Call Datasets
    print("[{}] Loading data".format(module))
    X_train,y_train,X_test,y_test = create_numpy_dataset(name=problem,path=data_path)

    #Create Normalizer & Normalize Data
    print("[{}] X_train: {}".format(module, X_train.shape))
    print("[{}] X_test: {}".format(module, X_test.shape))
    # normalizer, X_train_transform = create_normalizer(normalization,X_train)
    # X_test_transform = normalizer.transform(X_test)
    X_train = X_train[:,0,:]
    X_test = X_test[:,0,:]

    train_means = np.mean(X_train,axis=1,keepdims=True)
    train_stds = np.std(X_train,axis=1,keepdims=True)
    test_means = np.mean(X_test,axis=1,keepdims=True)
    test_stds = np.std(X_test,axis=1,keepdims=True)

    test_stds[test_stds == 0] = 1

    X_train_transform = (X_train - train_means) / train_stds
    X_test_transform = (X_test - test_means) / test_stds

    #Normalize Labels
    label_encode = LabelEncoder()
    y_train_transformed = label_encode.fit_transform(y_train)
    y_test_transformed = label_encode.transform(y_test)


    #Load Model Config
    if os.path.exists(config):
        model_kwargs = json.load(open(config))
    else:
        model_kwargs = {}


    if classifier_name == 'sax':
        if config is None:
            clf = SAXDictionaryClassifier(save_words = True)
        else:
            clf = SAXDictionaryClassifier(**model_kwargs)
    elif classifier_name =='esax':
        clf = ESAXDictionaryClassifier(**model_kwargs)
    elif classifier_name =='1dsax':
        clf = OneDSAXDictionaryClassifier(**model_kwargs)
    elif classifier_name =='tfsax':
        clf = TFSAXDictionaryClassifier(**model_kwargs)
    elif classifier_name =='sax_dr':
        clf = SAXDRDictionaryClassifier(**model_kwargs)
    elif classifier_name =='sax_vfd':
        clf = SAXVFDDictionaryClassifier(**model_kwargs)
    elif classifier_name == 'sfa':
        if config is None:
            clf = SFADictionaryClassifier(save_words=True)
        else:
            clf = SFADictionaryClassifier(**model_kwargs)
    elif classifier_name == 'spartan':
        model_kwargs['downsample'] = downsample_rate
        clf = SPARTANClassifier(**model_kwargs)

    print("[{}] Model Args: {}".format(module,model_kwargs))

    if classifier_name == 'spartan' and downsample_rate < 1.0:

        repeat_num = 5
    else:
        repeat_num = 1

    avg_runtime = 0.0
    avg_results = pd.DataFrame(columns=['acc','precision','recall','f1'])
    for itr in range(repeat_num):

        # initialize the symbolic model
        if classifier_name == 'sax':
            if config is None:
                clf = SAXDictionaryClassifier(save_words = True)
            else:
                clf = SAXDictionaryClassifier(**model_kwargs)
        elif classifier_name =='esax':
            clf = ESAXDictionaryClassifier(**model_kwargs)
        elif classifier_name =='1dsax':
            clf = OneDSAXDictionaryClassifier(**model_kwargs)
        elif classifier_name =='tfsax':
            clf = TFSAXDictionaryClassifier(**model_kwargs)
        elif classifier_name =='sax_dr':
            clf = SAXDRDictionaryClassifier(**model_kwargs)
        elif classifier_name =='sax_vfd':
            clf = SAXVFDDictionaryClassifier(**model_kwargs)
        elif classifier_name =='asax':
            clf = ASAXDictionaryClassifier(**model_kwargs)
        elif classifier_name =='twa_sax':
            clf = TWASAXDictionaryClassifier(**model_kwargs)
        elif classifier_name == 'sfa':
            if config is None:
                clf = SFADictionaryClassifier(save_words=True)
            else:
                clf = SFADictionaryClassifier(**model_kwargs)
        elif classifier_name == 'spartan':
            # if downsample_rate < 1.0:
            #     model_kwargs['downsample'] = downsample_rate
            model_kwargs['downsample'] = downsample_rate
            clf = SPARTANClassifier(**model_kwargs)


        comp_start = time.time()

        fit_start = time.time()
        clf.fit(X_train_transform,y_train_transformed)
        fit_end = time.time()

        pred_start = time.time()
        model_pred = clf.predict(X_test_transform)
        pred_end = time.time()

        comp_end = time.time()

        avg_runtime += comp_end - comp_start
        
        results = compute_classification_metrics(y_test_transformed,model_pred)
        avg_results = pd.concat([avg_results, results], ignore_index=True)

        print(f'Fit time: {(fit_end - fit_start):.4f}s')
        print(f'Pred time: {(pred_end - pred_start):.4f}s')

        avg_results = avg_results.mean().to_frame().T
        model_params = pd.DataFrame([model_kwargs])
        model_params['runtime'] = avg_runtime / repeat_num

        final_results = pd.concat([avg_results,model_params],ignore_index=False,axis=1)

        print(final_results)

        filename = output_directory + 'classification_results.csv'
        with open(filename, 'a') as f:
            final_results.to_csv(f, mode='a', header=f.tell()==0,index=False)
