# Major changes
## - BUG FIX: Update SFA with the correct version (lower bounding property)
## - Experimental setting change: splitting training and testing
import os
import time
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tslearn.piecewise import SymbolicAggregateApproximation

from .util.distance_vectorized import euclidean_vectorized, univariate_sfa_distance, spartan_pca_mindist
from .util.dataset import create_numpy_dataset
from .util.normalization import create_normalizer
from TSB_Symbolic.onennclassifier.spartan_classifier import SPARTANClassifier
# from TSB_Symbolic.onennclassifier.sax_classifier import SAXDictionaryClassifier
# from TSB_Symbolic.onennclassifier.sfa_classifier import SFADictionaryClassifier

# bug fix: load updated implementation of SFA from the authors (for lower bounding property)
from TSB_Symbolic.symbolic.sfa.sfa_whole import SFAWhole


from sklearn.preprocessing import LabelEncoder

import argparse

from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=False, default="../data/TSC/Univariate_ts/")
    parser.add_argument("-p", "--problem", required=False, default="Computers") 
    parser.add_argument("-x", "--problem_idx", required=False, default=0) 
    parser.add_argument("--alpha_min", required=False, default=4, type=int) 
    parser.add_argument("--alpha_max", required=False, default=10, type=int) 
    parser.add_argument("--wordlen_min", required=False, default=4, type=int) 
    parser.add_argument("--wordlen_max", required=False, default=10, type=int) 

    arguments = parser.parse_args()

    return arguments

if __name__ == "__main__":

    arguments = parse_arguments()
    dataset   = arguments.problem
    data_path = arguments.data
    data_id   = arguments.problem_idx
    alphabet_max = arguments.alpha_max
    alphabet_min = arguments.alpha_min
    wordlen_max  = arguments.wordlen_max
    wordlen_min  = arguments.wordlen_min

    normalization = 'zscore'

    module = 'SymbolicRepresentationExperiments'
    print("=======================================================================")
    print("[{}] Starting TLB Experiment".format(module))
    print("=======================================================================")
    print("[{}] Data path: {}".format(module, data_path))
    print("[{}] Problem: {} | {}".format(module, data_id, dataset))

    # load dataset

    X_train,y_train,X_test,y_test = create_numpy_dataset(name=dataset,path=data_path)

    X_train = X_train[:,0,:]
    X_test = X_test[:,0,:]

    train_means = np.mean(X_train,axis=1,keepdims=True)
    train_stds = np.std(X_train,axis=1,keepdims=True)
    test_means = np.mean(X_test,axis=1,keepdims=True)
    test_stds = np.std(X_test,axis=1,keepdims=True)

    train_stds[train_stds == 0] = 1
    test_stds[test_stds == 0] = 1

    X_train_transform = (X_train - train_means) / train_stds
    X_test_transform = (X_test - test_means) / test_stds

    n_instances,n_timepoints = X_train_transform.shape

    X_all = np.concatenate([X_train_transform,X_test_transform],axis=0)

    label_encode = LabelEncoder()
    y_train_transformed = label_encode.fit_transform(y_train)
    y_test_transformed = label_encode.transform(y_test)

    y_all = np.concatenate([y_train_transformed,y_test_transformed],axis=0)

    print("[{}] X_train: {}".format(module, X_train_transform.shape))
    print("[{}] X_test: {}".format(module, X_test_transform.shape))

    n_instances,n_timepoints = X_all.shape

    window_size = 0

    alphabet_max = max(min(alphabet_max, n_instances, n_timepoints), 2)
    alphabet_min = max(min(alphabet_min, n_instances, n_timepoints), 2)

    wordlen_max = max(min(wordlen_max, n_instances, n_timepoints), 2)
    wordlen_min = max(min(wordlen_min, n_instances, n_timepoints), 2)

    assert alphabet_max >= alphabet_min and wordlen_max >=  wordlen_min
    alphabet_sizes = np.arange(alphabet_min, alphabet_max+1)
    word_sizes = np.arange(wordlen_min, wordlen_max+1)
    
    print("[{}] Alphabet Size Range: {}".format(module, alphabet_sizes))
    print("[{}] Word length Range: {}".format(module, word_sizes))

    spartan_DAA_tlb_values = np.zeros((len(word_sizes),len(alphabet_sizes)))
    spartan_woDAA_tlb_values = np.zeros((len(word_sizes),len(alphabet_sizes)))
    sax_tlb_values = np.zeros((len(word_sizes),len(alphabet_sizes)))
    sfa_tlb_values = np.zeros((len(word_sizes),len(alphabet_sizes)))

    euclidean_dist_mat = euclidean_vectorized(X_test_transform,X_test_transform)

    tlb_results = pd.DataFrame()

    start_time = time.time()

    for n in tqdm(range(len(word_sizes))):

        word_len = word_sizes[n]
        
        for m,alphabet_size in tqdm(enumerate(alphabet_sizes)):
            
            ######################
            # I. Initialize models
            ######################

            # print(f"ALPHABET: {alphabet_size} | WORDLEN: {word_len} | BUDGET: {np.log2(alphabet_size) * word_len}")
            spartan_woDAA = SPARTANClassifier(
                alphabet_size=int(alphabet_size),
                word_length=word_len,
                metric='pca_mindist',
                assignment_policy='direct',
                pca_solver='full'
            )

            assignment_policy = 'DAA'
            lamda = 0.5

            spartan_DAA = SPARTANClassifier(
                alphabet_size=int(alphabet_size),
                word_length=word_len,
                metric='pca_mindist',
                assignment_policy='DAA',
                lamda=lamda,
                bit_budget=int(np.log2(alphabet_size) * word_len),
                pca_solver='full'
            )
            
            sfa = SFAWhole(word_length=int(word_len),
               alphabet_size=int(alphabet_size)
            )

            ####################
            # II. Fit models
            ####################

            # print("SPARTAN FITTING")
            spartan_woDAA.fit(X_train_transform,y_train_transformed)
            pred_woDAA = spartan_woDAA.predict(X_test_transform)

            spartan_DAA.fit(X_train_transform,y_train_transformed)
            pred_DAA = spartan_DAA.predict(X_test_transform)

            # print("SFA FITTING")
            sfa_train_words, _ = sfa.fit_transform(X_train_transform, None)
            sfa_words, _ = sfa.transform(X_test_transform, None)
            
            # print("SAX FITTING")
            tsl_sax = SymbolicAggregateApproximation(
                n_segments=word_len,
                alphabet_size_avg=alphabet_size
            )

            tsl_words = tsl_sax.fit_transform(np.expand_dims(X_test_transform, axis=-1))
            

            #######################################
            # III. Calculate distance & measure TLB
            #######################################
            
            # print("SPARTAN TLB")
            # SPARTAN without DAA TLB
            pred_words = np.squeeze(spartan_woDAA.pred_words,axis=1)
            breakpoints = spartan_woDAA.spartan.mindist_breakpoints
            spartan_woDAA_dist_mat = spartan_pca_mindist(pred_words,pred_words,breakpoints)

            spartan_woDAA_tlbs = []
            for i,j in zip(spartan_woDAA_dist_mat.ravel(),euclidean_dist_mat.ravel()):
                if i == 0 and j == 0:
                    spartan_woDAA_tlbs.append(1)
                else:
                    spartan_woDAA_tlbs.append(i / j)

            spartan_woDAA_mean_tlb = np.mean(spartan_woDAA_tlbs)

            spartan_woDAA_tlb_values[n,m] = spartan_woDAA_mean_tlb

            # SPARTAN with DAA TLB

            pred_words = np.squeeze(spartan_DAA.pred_words,axis=1)
            breakpoints = spartan_DAA.spartan.mindist_breakpoints
            spartan_DAA_dist_mat = spartan_pca_mindist(pred_words,pred_words,breakpoints)

            spartan_DAA_tlbs = []
            for i,j in zip(spartan_DAA_dist_mat.ravel(),euclidean_dist_mat.ravel()):
                if i == 0 and j == 0:
                    spartan_DAA_tlbs.append(1)
                else:
                    spartan_DAA_tlbs.append(i / j)

            spartan_DAA_mean_tlb = np.mean(spartan_DAA_tlbs)

            spartan_DAA_tlb_values[n,m] = spartan_DAA_mean_tlb
            
            # print("SAX TLB")
            # SAX TLB
            tsl_sax_dist_mat = np.zeros((tsl_words.shape[0],tsl_words.shape[0]))
            for i in range(tsl_words.shape[0]):
                for j in range(tsl_words.shape[0]):

                    tsl_sax_dist_mat[i,j] = tsl_sax.distance_sax(tsl_words[i],tsl_words[j])

            sax_tlbs = []
            for i,j in zip(tsl_sax_dist_mat.ravel(),euclidean_dist_mat.ravel()):
                if i == 0 and j == 0:
                    sax_tlbs.append(1)
                else:
                    sax_tlbs.append(i / j)

            sax_mean_tlb = np.mean(sax_tlbs)

            sax_tlb_values[n,m] = sax_mean_tlb
            
            # print("SFA TLB")
            # SFA TLB 
            sfa_tlbs = []
            sfa_dist_mat = np.zeros((sfa_words.shape[0],sfa_words.shape[0]))
            sfa_bkpt = sfa.breakpoints
            for i in range(sfa_words.shape[0]):
                for j in range(sfa_words.shape[0]):

                    sfa_dist_mat[i,j] = univariate_sfa_distance(sfa_words[i],sfa_words[j], sfa_bkpt)

            assert sfa_dist_mat.shape == euclidean_dist_mat.shape

            for i,j in zip(sfa_dist_mat.ravel(),euclidean_dist_mat.ravel()):
                if i == 0 and j == 0:
                    sfa_tlbs.append(1)
                else:
                    if j == 0 or i > j:
                        sfa_tlbs.append(np.nan) # error
                        print("sfa: ", i, j)
                        continue
                    else:
                        sfa_tlbs.append(i / j)

            sfa_mean_tlb = np.mean(sfa_tlbs)

            sax_record = pd.DataFrame([{'dataset':dataset,'a':alphabet_size,'w':word_len,'method':'sax','tlb':sax_mean_tlb, 'param': 'none'}])
            sfa_record = pd.DataFrame([{'dataset':dataset,'a':alphabet_size,'w':word_len,'method':'sfa','tlb':sfa_mean_tlb, 'param': 'none'}])
            spartan_DAA_record = pd.DataFrame([{'dataset':dataset,'a':alphabet_size,'w':word_len,'method':'spartan','tlb':spartan_DAA_mean_tlb,'param': f'{assignment_policy}-lamda{lamda}'}])
            spartan_woDAA_record = pd.DataFrame([{'dataset':dataset,'a':alphabet_size,'w':word_len,'method':'spartan','tlb':spartan_woDAA_mean_tlb,'param': 'woDAA'}])

            tlb_results = pd.concat([tlb_results,sax_record,sfa_record,spartan_woDAA_record,spartan_DAA_record],ignore_index=True)


    print(tlb_results.tail())

    outpath = "output/tlb" # change as needed
    os.makedirs(outpath, exist_ok=True)

    tlb_results.to_csv(os.path.join(outpath, f'{dataset}_tlb_results.csv'), index=False)

    print(f"runtime for {dataset}: {(time.time()-start_time)/60:.2f}min")

        
