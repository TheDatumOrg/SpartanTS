import numpy as np
from numba import prange,njit

import textdistance

#Hamming distance for processed strings
@njit(parallel=True)
def matching_distance(word_list1,word_list2,normalize=True):
    dist = 0
    for word1,word2 in zip(word_list1,word_list2):
        matching = np.sum((word1 != word2))

        if normalize:
            matching = matching / len(word1)

        dist = dist + matching
    return dist

#Edit distance
@njit(parallel=True)
def editdistance(wordlist1,wordlist2):
    dist = 0

    partials = np.zeros(len(wordlist1))
    for i in prange(len(wordlist1)):
        word1 = wordlist1[i]
        word2 = wordlist2[i]

        partials[i] = textdistance.levenshtein.distance(word1,word2)

    dist = np.sum(partials)
    return dist

# @njit(parallel=True)
def cell(r,c,breakpoints):
    if np.abs(r-c) <= 1:
        return 0
    else:
        return breakpoints[int(max(r,c) - 1)] - breakpoints[int(min(r,c))]
 
#Lower bounding distance from: A Symbolic Representation of Time Series, with Implications for Streaming Algorithms, Lin et al.
# @njit(parallel=True)
def mindist(word_list1,word_list2,breakpoints,n,w):
    sum = 0
    for word1, word2 in zip(word_list1,word_list2):
        # print(word1, word2)
        partial_dist=0
        for (q,c) in zip(word1,word2):
            partial_dist = partial_dist + (cell(q,c,breakpoints))**2
        sum = sum + (np.sqrt(n/w)*np.sqrt(partial_dist))

    return sum
#Lower bounding distance from: Symbolic Fourier Approximation
def dft_dist(word_list1,word_list2,breakpoints,n,w):
    pass

@njit(parallel=True)
def hist_euclidean_dist(hist1,hist2):
    return np.sum(np.sqrt((hist1-hist2)**2))

@njit(parallel=True)
def euclidean_rep(wordlist1,wordlist2):
    dist = 0
    for word1,word2 in zip(wordlist1,wordlist2):
        dist = dist + np.sum(np.sqrt(word1-word2)**2)
    return dist

@njit(parallel=True)
def symbol_dist(wordlist1,wordlist2):
    dist = 0
    for word1,word2 in zip(wordlist1,wordlist2):
        dist = dist + np.sum(np.abs(word1-word2))
    return dist
@njit(parallel=True)
def boss_distance(wordbags1,wordbags2):
    
    diff = (wordbags1 - wordbags2)**2

    diff[(wordbags1 == 0)] = 0

    dist = np.sum(diff)

    return dist

# @njit(parallel=True,fastmath=True)
def pairwise_distance(X_wordlists,Y_wordlists=None,symmetric = True,metric='matching', sax_param=None):
    if Y_wordlists is None:
        Y_wordlists=X_wordlists
    X_samples = X_wordlists.shape[0]
    Y_samples = Y_wordlists.shape[0]

    pairwise_matrix = np.zeros((X_samples,Y_samples))

    for i in range(X_samples):
        for j in range(Y_samples):
            x_curr = X_wordlists[i]
            y = Y_wordlists[j]

            if metric == 'matching':
                pairwise_matrix[i,j] = matching_distance(x_curr,y)
            elif metric == 'euclidean':
                pairwise_matrix[i,j] = euclidean_rep(x_curr,y)
            elif metric == 'symbol':
                pairwise_matrix[i,j] = symbol_dist(x_curr,y)
            elif metric == 'editdistance':
                # pairwise_matrix[i,j] = editdistance(x_curr,y)
                pass
            elif metric == 'mindist':
                breakpoints, n, w = sax_param
                pairwise_matrix[i,j] = mindist(x_curr,y,breakpoints,n,w)
            else:    
                pass

    return pairwise_matrix

@njit(parallel=True)
def pairwise_histogram_distance(X_hists,Y_hists,symmetric=True,metric='hist_euclidean'):
    if Y_hists is None:
        Y_hists=X_hists
    X_samples = X_hists.shape[0]
    Y_samples = Y_hists.shape[0]

    pairwise_matrix = np.zeros((X_samples,Y_samples))

    for i in range(X_samples):
        curr_x = X_hists[i]
        for j in range(Y_samples):
            y = Y_hists[j]
            if metric == 'hist_euclidean':
                pairwise_matrix[i,j] = hist_euclidean_dist(curr_x,y)
            elif metric == 'boss':
                pairwise_matrix[i,j] = boss_distance(curr_x,y)
    return pairwise_matrix
            