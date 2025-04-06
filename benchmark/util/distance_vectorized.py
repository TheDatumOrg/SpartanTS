import numpy as np
from numba import njit, prange

def symbol_vectorized(X,Y):
    dist = np.sum(np.abs(X[:,None,:] - Y[None,:,:]),axis=2)
    return dist
def hamming_vectorized(X,Y):
    n_inst, word_length = X.shape
    dist = np.sum(X[:,None,:] != Y[None,:,:],axis=2) / word_length
    return dist
# def euclidean_vectorized(X,Y):
#     dist = np.sum(np.sqrt(((X[:,None,:] - Y[None,:,:])**2)),axis=2)
#     return dist
def euclidean_vectorized(X,Y):
    dist = np.sqrt(np.sum(((X[:,None,:] - Y[None,:,:])**2),axis=2))
    return dist

def cosine_similarity_vectorized(X,Y):
    dist = 0

    dot = np.sum(X[:,None,:] * Y[None,:,:],axis=2)

    x_norm = np.linalg.norm(X,axis=1)
    y_norm = np.linalg.norm(Y,axis=1)

    xy_norm = x_norm[:,None] * y_norm[None,:]

    dist = dot / xy_norm

    dist = 1 - dist

    return dist

def kl_divergence(X,Y):
    X = X.astype(np.int16)
    Y = Y.astype(np.int16) 
    X_norm = (X + 1e-6) / np.sum(X + 1e-6,axis=1,keepdims=True)
    Y_norm = (Y + 1e-6) / np.sum(Y + 1e-6,axis=1,keepdims=True) 

    X_norm = X_norm.astype(np.float32)
    Y_norm = Y_norm.astype(np.float32)

    kl_dist = np.sum(X_norm[:,None,:] * np.log(X_norm[:,None,:] / Y_norm[None,:,:]),axis=2)
    
    return kl_dist



def boss_vectorized(X,Y):
    # diff = (wordbags1 - wordbags2)**2

    # diff[(wordbags1 == 0)] = 0

    # dist = np.sum(diff)

    X = X.astype(np.int16)
    Y = Y.astype(np.int16) 

    diff = X[:,None,:] - Y[None,:,:]
    z = np.zeros_like(diff)
    X_nm = X[:,None,:]+z
    diff[X_nm == 0] = 0

    dist = np.sum(diff**2,axis=2)
    return dist
def symbol_weighted(X,Y,weights):
    dist = np.sum(np.abs(X[:,None,:] - Y[None,:,:]) * weights[None,None,:],axis=2)
    return dist
def hamming_weighted(X,Y,weights):
    dist = np.sum(X[:,None,:] != Y[None,:,:] * weights[None,None,:],axis=2)
    return dist
def mindist_vectorized(X,Y,breakpoints):
    n_instances, word_length = X.shape

    X_breakpoint = np.concatenate([np.expand_dims(np.array(breakpoints[i][X[:,i].astype(np.int32)]),axis=1) for i in range(word_length)],axis=1)
    Y_breakpoint = np.concatenate([np.expand_dims(np.array(breakpoints[i][Y[:,i].astype(np.int32)]),axis=1) for i in range(word_length)],axis=1)
    dist = np.sum(np.sqrt((X_breakpoint[:,None,:] - Y_breakpoint[None,:,:])**2),axis=2)

    return dist

def mindist_weighted(X,Y,breakpoints,weights):
    n_instances, word_length = X.shape

    X_breakpoint = np.concatenate([np.expand_dims(np.array(breakpoints[i][X[:,i].astype(np.int32)]),axis=1) for i in range(word_length)],axis=1)
    Y_breakpoint = np.concatenate([np.expand_dims(np.array(breakpoints[i][Y[:,i].astype(np.int32)]),axis=1) for i in range(word_length)],axis=1)
    dist = np.sum(np.sqrt(((X_breakpoint[:,None,:] - Y_breakpoint[None,:,:]) * weights)**2),axis=2)

    return dist

def l1_mindist(X,Y,breakpoints):
    n_instances, word_length = X.shape
    X_breakpoint = np.concatenate([np.expand_dims(np.array(breakpoints[i][X[:,i].astype(np.int32)]),axis=1) for i in range(word_length)],axis=1)
    Y_breakpoint = np.concatenate([np.expand_dims(np.array(breakpoints[i][Y[:,i].astype(np.int32)]),axis=1) for i in range(word_length)],axis=1)

    dist = np.sum(np.abs((X_breakpoint[:,None,:] - Y_breakpoint[None,:,:])),axis=2)

    return dist

def sax_mindist(X,Y,breakpoints, ts_len=None):
    
    # bug fix: \sqrt{n/w} scaling
    n_instances,word_length = X.shape
    n_yinstances,word_length = Y.shape
    z = np.zeros((n_instances,n_yinstances,word_length))

    # print(breakpoints)

    X_ind = X + 1
    X_ind = X_ind.astype(np.int32)
    Y_ind = Y + 1
    Y_ind = Y_ind.astype(np.int32)

    cond = (np.abs((X_ind[:,None,:] - Y_ind[None,:,:])) <= 1)

    ind_max = np.maximum(X_ind[:,None,:],Y_ind[None,:,:]) - 1
    ind_min = np.minimum(X_ind[:,None,:],Y_ind[None,:,:])
    breakpoint_max = np.concatenate([np.expand_dims(breakpoints[i][ind_max[:,:,i]],axis=2) for i in range(word_length)],axis=2)
    breakpoint_min = np.concatenate([np.expand_dims(breakpoints[i][ind_min[:,:,i]],axis=2) for i in range(word_length)],axis=2)
    
    bp_diff = breakpoint_max - breakpoint_min

    # print(z.shape)
    # print(bp_diff.shape)

    cell = np.where(cond,z,bp_diff)

    if ts_len is None:
        dist_mat = np.sqrt(np.sum(np.abs(cell)**2,axis=2))
    else:
        dist_mat = np.sqrt(np.sum(np.abs(cell)**2,axis=2)*ts_len/word_length)

    # print(dist_mat)
    return dist_mat

# Lower bounding distance for Spartan PCA
def spartan_pca_mindist(X,Y,breakpoints):
    n_instances,word_length = X.shape
    n_yinstances,word_length = Y.shape
    z = np.zeros((n_instances,n_yinstances,word_length))

    X_ind = X + 1
    Y_ind = Y + 1

    cond = (np.abs((X_ind[:,None,:] - Y_ind[None,:,:])) <= 1)

    ind_max = np.maximum(X_ind[:,None,:],Y_ind[None,:,:]) - 1
    ind_max = ind_max.astype(np.int32)
    ind_min = np.minimum(X_ind[:,None,:],Y_ind[None,:,:])
    ind_min = ind_min.astype(np.int32)
        

    breakpoint_max = np.concatenate([np.expand_dims(breakpoints[i][ind_max[:,:,i]],axis=2) for i in range(word_length)],axis=2)
    breakpoint_min = np.concatenate([np.expand_dims(breakpoints[i][ind_min[:,:,i]],axis=2) for i in range(word_length)],axis=2)
    
    bp_diff = breakpoint_max - breakpoint_min

    cell = np.where(cond,z,bp_diff)

    dist_mat = np.sqrt(np.sum((cell**2),axis=2))
    return dist_mat
def mindist_minmax(X,Y,breakpoints):
    n_instances,word_length = X.shape
    n_yinstances,word_length = Y.shape
    z = np.zeros((n_instances,n_yinstances,word_length))

    X_ind = X + 1
    Y_ind = Y + 1

    cond = ((X_ind[:,None,:] - Y_ind[None,:,:]) <= 1)

    ind_max = np.maximum(X_ind[:,None,:],Y_ind[None,:,:]) - 1
    ind_max = ind_max.astype(np.int32)
    ind_min = np.minimum(X_ind[:,None,:],Y_ind[None,:,:]) - 1
    ind_min = ind_min.astype(np.int32)
        

    breakpoint_max = np.concatenate([np.expand_dims(breakpoints[i][ind_max[:,:,i]],axis=2) for i in range(word_length)],axis=2)
    breakpoint_min = np.concatenate([np.expand_dims(breakpoints[i][ind_min[:,:,i]],axis=2) for i in range(word_length)],axis=2)
    
    bp_diff = breakpoint_max - breakpoint_min

    # cell = np.where(cond,z,bp_diff)
    cell = bp_diff

    dist_mat = np.sqrt(np.sum(cell**2,axis=2))
    return dist_mat

@njit(cache=True, fastmath=True)
def univariate_sfa_distance(
    x: np.ndarray, y: np.ndarray, breakpoints: np.ndarray
) -> float:
    
    # adapted from aeon for SFA's lower bounding distance: https://github.com/aeon-toolkit/aeon/blob/6ee0597fd855b00b64d787db5d469420c064251b/aeon/distances/mindist/_sfa.py#L65
    dist = 0.0
    for i in range(x.shape[0]):
        if np.abs(x[i] - y[i]) <= 1:
            continue
        else:
            dist += (
                breakpoints[i, max(x[i], y[i]) - 1] - breakpoints[i, min(x[i], y[i])]
            ) ** 2

    return np.sqrt(2 * dist)
    

    