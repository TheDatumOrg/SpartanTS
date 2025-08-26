import argparse
import json
import os
import time
import numpy as np
import pandas as pd

from .util.dataset import create_numpy_dataset
from .util.normalization import create_normalizer
from .util.tools import create_directory,compute_classification_metrics
from .util.tools import eval_cluster, compute_clustering_metrics

from .util.distance_vectorized import hamming_vectorized,symbol_vectorized

from sklearn.preprocessing import LabelEncoder


from TSB_Symbolic.onennclassifier.sax_classifier import SAXDictionaryClassifier
from TSB_Symbolic.onennclassifier.esax_classifier import ESAXDictionaryClassifier
from TSB_Symbolic.onennclassifier.oned_sax_classifier import OneDSAXDictionaryClassifier
from TSB_Symbolic.onennclassifier.tfsax_classifier import TFSAXDictionaryClassifier
from TSB_Symbolic.onennclassifier.sax_dr_classifier import SAXDRDictionaryClassifier
from TSB_Symbolic.onennclassifier.sax_vfd_classifier import SAXVFDDictionaryClassifier
from TSB_Symbolic.onennclassifier.sfa_classifier import SFADictionaryClassifier
from TSB_Symbolic.onennclassifier.spartan_classifier import SPARTANClassifier

from sklearn.cluster import KMeans


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=False, default="data/ucr/Univariate_ts/")
    parser.add_argument("-p", "--problem", required=False, default="ArrowHead")  # see data_loader.regression_datasets
    parser.add_argument("-c", "--classifier", required=False, default="sax")  # see regressor_tools.all_models
    parser.add_argument("-g", '--config',required=False,default='')
    parser.add_argument("-i", "--itr", required=False, default=13)
    parser.add_argument("-n", "--norm", required=False, default="zscore")  # none, standard, minmax
    parser.add_argument("-s","--save_model",required=False, default=None)
    parser.add_argument("-r","--skip_repeat",required=False,default=True)
    parser.add_argument("-m","--dataset_num",required=False,default=1, type=int)
    parser.add_argument("-w","--store_words",default=None)
    parser.add_argument("-k","--clust_model",default='kmedoids')
    parser.add_argument("-l","--linkage",default='complete')
    parser.add_argument("-o","--kmedoids_type",default='pam')
    parser.add_argument("-e","--repeat_num",default=10, type=int)
    parser.add_argument("-b","--data_split",default='merge', type=str, choices=['split', 'merge'])
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
    clust_model = arguments.clust_model
    linkage = arguments.linkage
    repeat_num = arguments.repeat_num
    kmedoids_type = arguments.kmedoids_type
    data_split = arguments.data_split
    repr_type = arguments.repr_type

    assert clust_model in ['kmedoids', 'hierarchical', 'symb_kmeans']

    # create output directory
    output_directory = "output/clustering/"
    if normalization != None:
        output_directory = "output/clustering/"
    output_directory = output_directory + classifier_name + '/' + problem + '/itr_' + str(itr) + '/'
    create_directory(output_directory)



    print("=======================================================================")
    print("[{}] Starting Clustering Experiment".format(module))
    print("=======================================================================")
    print("[{}] Iteration: {}".format(module, itr))
    print("[{}] Problem: {} | {}".format(module, data_id, problem))
    print("[{}] Classifier: {}".format(module, classifier_name))

    #Call Datasets
    print("[{}] Loading data".format(module))
    X_train,y_train,X_test,y_test = create_numpy_dataset(name=problem,path=data_path)

    #Create Normalizer & Normalize Data
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

    n_clusters = len(np.unique(y_train_transformed))
    assert n_clusters == len(np.unique(y_test_transformed))

    #Load Model Config
    if os.path.exists(config):
        model_kwargs = json.load(open(config))
        print("[{}] Model Args: {}".format(module,model_kwargs))
    else:
        model_kwargs = {}

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


    X_train_origin = X_train_transform.copy()
    X_test_origin = X_test_transform.copy()
    y_train_origin = y_train_transformed.copy()
    y_test_origin = y_test_transformed.copy()


    if data_split == 'split':
        pass 
    elif data_split == 'merge':

        X_train_transform = np.concatenate((X_train_origin, X_test_origin), axis=0)
        X_test_transform = X_train_transform.copy()
        y_train_transformed = np.concatenate((y_train_origin, y_test_origin), axis=0)
        y_test_transformed = y_train_transformed.copy()


    print("[{}] X_train_transformed: {}".format(module, X_train_transform.shape))
    print("[{}] X_test_transformed: {}".format(module, X_test_transform.shape))

    comp_start = time.time()

    fit_start = time.time()
    clf.fit(X_train_transform,y_train_transformed)
    fit_end = time.time()

    pred_start = time.time()
    model_pred = clf.predict(X_test_transform)

    # extract symbolic representation on testset
    if repr_type == 'single':
        test_repr = clf.predict_words_bps
        test_repr = test_repr.reshape(len(test_repr), -1)
        
        test_dist_mat = symbol_vectorized(test_repr,test_repr)
        pred_end = time.time()

        print("nclusters: ", n_clusters)
        comp_end = time.time()

    elif repr_type == 'bop':
        test_repr = clf.pred_histogram
        test_repr = test_repr.reshape(len(test_repr), -1)
        
        test_dist_mat = symbol_vectorized(test_repr,test_repr)
        pred_end = time.time()


    print(f'Fit time: {(fit_end - fit_start):.4f}s')
    print(f'Pred time: {(pred_end - pred_start):.4f}s')

    print(f"X_test: {X_test_transform.shape}, X_test_symb_repr: {test_repr.shape}")
    print("Ground Truth: ", y_test_transformed[:20])

    results = pd.DataFrame()

    for rand_itr in range(repeat_num):
        if clust_model == 'kmedoids':

            import kmedoids

            # print(test_dist_mat.shape)
            if kmedoids_type == 'pam':
                symb_clustering = kmedoids.pam(diss=test_dist_mat, medoids=n_clusters, max_iter=100, init='random', random_state=rand_itr)
                comp_end = time.time()
                y_pred_symb = symb_clustering.labels


            elif kmedoids_type == 'fasterpam':
                symb_clustering = kmedoids.fasterpam(diss=test_dist_mat, medoids=n_clusters, max_iter=100, init='random', random_state=rand_itr)
                comp_end = time.time()
                y_pred_symb = symb_clustering.labels

        elif clust_model == 'hierarchical':

            if rand_itr > 0:
                continue

            # symbolic representation
            from sklearn.cluster import AgglomerativeClustering

            symb_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, metric='l1').fit(test_repr)
            comp_end = time.time()
            y_pred_symb = symb_clustering.labels_

        elif clust_model == 'symb_kmeans':
            from sklearn.cluster import KMeans
            symb_kmeans = KMeans(n_clusters=n_clusters, random_state=rand_itr, n_init="auto").fit(test_repr)
            comp_end = time.time()
            y_pred_symb = symb_kmeans.labels_

        
        results = pd.concat([results, compute_clustering_metrics(y_test_transformed, y_pred_symb)], ignore_index=False, axis=0)

    model_params = pd.DataFrame([model_kwargs])
    model_params['clust_model'] = clust_model
    model_params['clust_param'] = None
    if clust_model == 'kmedoids':
        model_params['clust_param'] = kmedoids_type
    elif clust_model == 'hierarchical':
        model_params['clust_param'] = f'linkage-{linkage}'
    model_params['repeat_num'] = repeat_num
    model_params['data_split'] = data_split
    model_params['runtime'] = comp_end - comp_start

    ri_mean, ri_std   = results.loc[:, 'ri'].mean(),  results.loc[:, 'ri'].std()
    ari_mean, ari_std = results.loc[:, 'ari'].mean(), results.loc[:, 'ari'].std()
    nmi_mean, nmi_std = results.loc[:, 'nmi'].mean(), results.loc[:, 'nmi'].std()

    results = pd.DataFrame({
        'ri': [ri_mean],
        'ari': [ari_mean],
        'nmi': [nmi_mean],
        'ri_std': [ri_std],
        'ari_std': [ari_std],
        'nmi_std': [nmi_std]
    })

    results = pd.concat([results,model_params],ignore_index=False,axis=1).fillna(0)

    print("Final results\n")
    print(results)

    filename = output_directory + 'clustering_results.csv'
    with open(filename, 'a') as f:
        results.to_csv(f, mode='a', header=f.tell()==0,index=False)




