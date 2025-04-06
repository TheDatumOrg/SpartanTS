import os
import multiprocessing

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import matplotlib
import matplotlib.pyplot as plt

def draw_pairwise_plot(results, methods, output_dir, output_filename):

    matplotlib.rcParams.update({'font.size': 30})
    font_size = 35
    marker_size = 65
    # prepare the sub dataframe for each
    method1_accs = results[results['method'] == methods[0]]
    method2_accs = results[results['method'] == methods[1]]
    
    # find the intersection of datasets (completed)
    method1_completed_datasets = method1_accs[method1_accs['accuracy'] != -1]['dataset']
    method2_completed_datasets = method2_accs[method2_accs['accuracy'] != -1]['dataset']
    completed_datasets = list(set(method1_completed_datasets).intersection(set(method2_completed_datasets)))
    print(f"Completed datasets (total: {len(completed_datasets)}) \n", completed_datasets)

    results_nz = results[results['dataset'].isin(completed_datasets)]
    print(f"Result dataframe shape: ", results_nz.shape)

    method1_accs = results_nz[results_nz['method'] == methods[0]].sort_values(by=['dataset'])
    method2_accs = results_nz[results_nz['method'] == methods[1]].sort_values(by=['dataset'])

    method1_accs = method1_accs['accuracy']
    method2_accs = method2_accs['accuracy']

    method1_wins = method1_accs.values > method2_accs.values
    method1_ties = method1_accs.values == method2_accs.values
    method1_loss = method1_accs.values < method2_accs.values

    print(f'{methods[0]} vs {methods[1]} Accuracy')
    print(f'Win %:{method1_wins.sum() / len(completed_datasets)}')
    print(f'Tie %:{method1_ties.sum() / len(completed_datasets)}')
    print(f'Loss %:{method1_loss.sum() / len(completed_datasets)}')

    colors = method1_wins
    print(colors)
    print(np.array(sorted(completed_datasets))[method1_wins])


    rows = ['Win','Tie','Loss']
    columns = [f'{methods[1]} vs {methods[0]}']
    cell_text = [[str(method1_loss.sum())],
                [str(method1_ties.sum())],
                [str(method1_wins.sum())]]

    # add upper triangle
    fig = plt.figure(figsize=(8, 8))
    t1 = plt.Polygon([[0,0], [0, 1], [1, 1]], color= "Green", alpha = 0.25)
    plt.gca().add_patch(t1)

    # add scatter plot
    plt.scatter(x=method1_accs.values,y=method2_accs.values,c='darkblue',s=marker_size)
    plt.xlabel(f'{methods[0]}', fontsize=font_size)
    plt.ylabel(f'{methods[1]}', fontsize=font_size)


    # Plot the line to represent ratio = 1
    plt.axline((0, 0), (1, 1), color='tomato', linewidth=3)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.xticks([0,0.2,0.4,0.6,0.8,1.0])
    plt.yticks([0,0.2,0.4,0.6,0.8,1.0])

    tab = plt.table(cellText=cell_text,
            rowLabels=rows,
            colLabels=columns,
            colWidths=[0.15] * 3,
            loc='lower right'
            )
    tab.set_fontsize(30)
    tab.scale(3.3, 3.3)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, output_filename), dpi=500, bbox_inches='tight')




def initialise_multithread(num_cores=-1):
    """
    Initialise pool workers for multi processing
    :param num_cores:
    :return:
    """
    if (num_cores == -1) or (num_cores >= multiprocessing.cpu_count()):
        num_cores = multiprocessing.cpu_count() - 1
    p = multiprocessing.Pool(num_cores)
    return p


def create_directory(directory_path):
    """
    Create a directory if path doesn't exists
    :param directory_path:
    :return:
    """
    if os.path.exists(directory_path):
        return None
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path
    
def compute_classification_metrics(y_true,y_pred,y_true_val=None,y_pred_val=None):
    res = pd.DataFrame(data=np.zeros((1, 4), dtype=np.float32), index=[0],
                       columns=['acc', 'precision','recall','f1'])
    res['acc'] = accuracy_score(y_true,y_pred)
    res['precision'] = precision_score(y_true,y_pred,average='macro')
    res['recall'] = recall_score(y_true,y_pred,average='macro')
    res['f1'] = f1_score(y_true,y_pred,average='macro')
    
    return res

# 128 UCR univariate time series classification problems [1]
univariate = {
    "ACSF1",
    "Adiac",
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "ArrowHead",
    "Beef",
    "BeetleFly",
    "BirdChicken",
    "BME",
    "Car",
    "CBF",
    "Chinatown",
    "ChlorineConcentration",
    "CinCECGTorso",
    "Coffee",
    "Computers",
    "CricketX",
    "CricketY",
    "CricketZ",
    "Crop",
    "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup",
    "DistalPhalanxOutlineCorrect",
    "DistalPhalanxTW",
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "Earthquakes",
    "ECG200",
    "ECG5000",
    "ECGFiveDays",
    "ElectricDevices",
    "EOGHorizontalSignal",
    "EOGVerticalSignal",
    "EthanolLevel",
    "FaceAll",
    "FaceFour",
    "FacesUCR",
    "FiftyWords",
    "Fish",
    "FordA",
    "FordB",
    "FreezerRegularTrain",
    "FreezerSmallTrain",
    "Fungi",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "GunPoint",
    "GunPointAgeSpan",
    "GunPointMaleVersusFemale",
    "GunPointOldVersusYoung",
    "Ham",
    "HandOutlines",
    "Haptics",
    "Herring",
    "HouseTwenty",
    "InlineSkate",
    "InsectEPGRegularTrain",
    "InsectEPGSmallTrain",
    "InsectWingbeatSound",
    "ItalyPowerDemand",
    "LargeKitchenAppliances",
    "Lightning2",
    "Lightning7",
    "Mallat",
    "Meat",
    "MedicalImages",
    "MelbournePedestrian",
    "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxOutlineAgeGroup",
    "MiddlePhalanxTW",
    "MixedShapesRegularTrain",
    "MixedShapesSmallTrain",
    "MoteStrain",
    "NonInvasiveFetalECGThorax1",
    "NonInvasiveFetalECGThorax2",
    "OliveOil",
    "OSULeaf",
    "PhalangesOutlinesCorrect",
    "Phoneme",
    "PickupGestureWiimoteZ",
    "PigAirwayPressure",
    "PigArtPressure",
    "PigCVP",
    "PLAID",
    "Plane",
    "PowerCons",
    "ProximalPhalanxOutlineCorrect",
    "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxTW",
    "RefrigerationDevices",
    "Rock",
    "ScreenType",
    "SemgHandGenderCh2",
    "SemgHandMovementCh2",
    "SemgHandSubjectCh2",
    "ShakeGestureWiimoteZ",
    "ShapeletSim",
    "ShapesAll",
    "SmallKitchenAppliances",
    "SmoothSubspace",
    "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2",
    "StarLightCurves",
    "Strawberry",
    "SwedishLeaf",
    "Symbols",
    "SyntheticControl",
    "ToeSegmentation1",
    "ToeSegmentation2",
    "Trace",
    "TwoLeadECG",
    "TwoPatterns",
    "UMD",
    "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX",
    "UWaveGestureLibraryY",
    "UWaveGestureLibraryZ",
    "Wafer",
    "Wine",
    "WordSynonyms",
    "Worms",
    "WormsTwoClass",
    "Yoga",
}

def permutations_w_constraints(n_perm_elements, sum_total, min_value, max_value):
    # base case
    if n_perm_elements == 1:
        if (sum_total <= max_value) & (sum_total >= min_value):
            yield (sum_total,)
    else:
        for value in range(min_value, max_value + 1):
            for permutation in permutations_w_constraints(
                n_perm_elements - 1, sum_total - value, min_value, max_value
            ):
                if value >= permutation[0]:
                    yield (value,) + permutation

def entropy(signal, prob="standard"):
    """Computes the entropy of the signal using the Shannon Entropy.

    Description in Article:
    Regularities Unseen, Randomness Observed: Levels of Entropy Convergence
    Authors: Crutchfield J. Feldman David

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which entropy is computed
    prob : string
        Probability function (kde or gaussian functions are available)

    Returns
    -------
    float
        The normalized entropy value

    """

    if prob == "standard":
        value, counts = np.unique(signal, return_counts=True)
        p = counts / counts.sum()
    elif prob == "kde":
        p = kde(signal)
    elif prob == "gauss":
        p = gaussian(signal)

    if np.sum(p) == 0:
        return 0.0

    # Handling zero probability values
    p = p[np.where(p != 0)]

    # If probability all in one value, there is no entropy
    if np.log2(len(signal)) == 1:
        return 0.0
    elif np.sum(p * np.log2(p)) / np.log2(len(signal)) == 0:
        return 0.0
    else:
        return -np.sum(p * np.log2(p)) / np.log2(len(signal))


def slope(X):

    max_pt = np.argmax(X)
    min_pt = np.argmin(X)

    max_val = X[max_pt]
    min_val = X[min_pt]

    return (max_val - min_val) / (max_pt - min_pt)



def dynamic_bit_allocation(total_bit, EV, min_bit, max_bit):

    K = len(EV)
    N = total_bit
    DP = np.zeros((K+1,N+1))
    alloc = np.zeros_like(DP).astype(np.int32) # store the num of bits for each component

    # init
    for i in range(0, K+1):
        for j in range(0, N+1):
            
            DP[i][j] = -1e9

    DP[0][0] = 0
    
    # non-recursive
    for i in range(1, K+1):
        for j in range(0, N+1):
            
            max_reward = -1e9

            for x in range(min_bit, min(max_bit, j)+1):
                
                current_reward = DP[i-1][j-x]+x*EV[i-1]
                
                if current_reward > max_reward:

                    alloc[i][j] = x
                    max_reward = current_reward
                    DP[i][j] = current_reward

    
    def print_sol(alloc, K, N):
        
        bit_arr = []  
        unused_bit = N
        for i in range(K, 1, -1):
            bit_arr.append(alloc[i][unused_bit])
            unused_bit -= alloc[i][unused_bit]

        bit_arr.append(unused_bit)
        return bit_arr
    
    bit_arr = print_sol(alloc, K, N)

    return DP[K][N], bit_arr[::-1]


def dynamic_bit_allocation_old(total_bit, EV, min_bit, max_bit, delta=0.5):

    def ScaleFactor(x, A, delta=0.5):
        return 1 - delta * max(0, (x - A) / A)

    K = len(EV)
    N = total_bit
    A = N/K
    DP = np.zeros((K+1,N+1))
    alloc = np.zeros_like(DP).astype(np.int32) # store the num of bits for each component

    # init
    for i in range(0, K+1):
        for j in range(0, N+1):
            
            DP[i][j] = -1e9

    DP[0][0] = 0
    
    # non-recursive
    for i in range(1, K+1):
        for j in range(0, N+1):
            
            max_reward = -1e9

            for x in range(min_bit, min(max_bit, j)+1):
                
                current_reward = DP[i-1][j-x]+x*EV[i-1]*ScaleFactor(x,A,delta)
                
                if current_reward > max_reward:

                    alloc[i][j] = x
                    max_reward = current_reward
                    DP[i][j] = current_reward

    
    def print_sol(alloc, K, N):
        
        bit_arr = []  
        unused_bit = N
        for i in range(K, 1, -1):
            bit_arr.append(alloc[i][unused_bit])
            unused_bit -= alloc[i][unused_bit]

        bit_arr.append(unused_bit)
        return bit_arr
    
    bit_arr = print_sol(alloc, K, N)

    assert np.sum(bit_arr) == N

    return DP[K][N], bit_arr[::-1]


def dynamic_bit_allocation_update(total_bit, EV, min_bit, max_bit, delta=0.5):

    def regularization_term(x, ev_value, delta=0.5):
        return - delta*x*(x-1)*ev_value

    K = len(EV)
    N = total_bit
    A = N/K
    DP = np.zeros((K+1,N+1))
    alloc = np.zeros_like(DP).astype(np.int32) # store the num of bits for each component

    max_bit = int(np.max(EV)*N)

    print("max_bit", max_bit)
    # init
    for i in range(0, K+1):
        for j in range(0, N+1):
            
            DP[i][j] = -1e9

    DP[0][0] = 0
    
    # non-recursive
    for i in range(1, K+1):
        for j in range(0, N+1):
            
            max_reward = -1e9

            for x in range(min_bit, min(max_bit, j)+1):
                
                # current_reward = DP[i-1][j-x]+x*EV[i-1]*ScaleFactor(x,A,delta)
                current_reward = DP[i-1][j-x] + x*EV[i-1] + regularization_term(x, EV[i-1], delta)
                
                if current_reward > max_reward:

                    alloc[i][j] = x
                    max_reward = current_reward
                    DP[i][j] = current_reward

    print(max_bit)
    
    def print_sol(alloc, K, N):
        
        bit_arr = []  
        unused_bit = N
        for i in range(K, 1, -1):
            bit_arr.append(alloc[i][unused_bit])
            unused_bit -= alloc[i][unused_bit]

        bit_arr.append(unused_bit)
        return bit_arr
    
    bit_arr = print_sol(alloc, K, N)

    assert np.sum(bit_arr) == N

    return DP[K][N], bit_arr[::-1]


def dynamic_bit_allocation_debug(total_bit, EV, min_bit, max_bit, delta=0.5, regularization='prev'):

    def regularization_term_prev(x, ev_value, delta=0.5):
        return - delta*x*(x-1)*ev_value

    def regularization_term_1(x, ev_value, avg_bit, pos=0.5, delta=0.5):
        return -delta*abs(x-avg_bit)*(0.5-abs(0.5-pos))
    def regularization_term_2(x, ev_value, avg_bit, pos=0.5, delta=0.5):
        return -delta*abs(x-avg_bit)*(0.5-abs(0.5-pos))**2
    def regularization_term_3(x, ev_value, avg_bit, pos=0.5, delta=0.5):
        return -delta*np.exp(abs(x/N-1/K))*ev_value**2
    def regularization_term_4(x, x_prev, ev_value, ev_value_prev, avg_bit, pos=0.5, delta=0.5):
        
        if x_prev == N:
            return -delta* (np.exp(abs(x/N-1/K)))

        else:
            reg1 = abs(x-avg_bit)* ev_value**2
            reg2 = (x_prev - x) * ev_value**2
            # print(-delta*reg1, -delta*reg2)
            return -delta*( reg1+ reg2)

    def regularization_term_5(x, ev_value, delta=0.5, pos=1.0):
        
        # return - delta*(x-1)*ev_value**2
        return -delta * max(0, x-A) **2 * max(ev_value-1/N, 0)
        
    def regularization_term_6(x, ev_value, delta=0.5, pos=1.0):
        
        # return -delta * max(0, x-A) **2 * abs(ev_value-1/K)
        # return -delta * max(0, (x-A)**2) * ev_value
        return -delta * x*(x-1) * abs(ev_value-1/N) # 0.37
    
    def regularization_term_7(x, ev_value, delta=0.5, pos=1.0):
        
        return -delta * max(0, (x-A)**2) * ev_value # 0.99
    
    def regularization_term_8(x, ev_value, delta=0.5, pos=1.0):
        
        return -delta * max(0, (x-A)**2) * ev_value # 0.99

    



    K = len(EV)
    N = total_bit
    A = int(N/K)
    DP = np.zeros((K+1,N+1))
    min_bit = 1
    max_bit = int(np.max(EV) * N)
    alloc = np.zeros_like(DP).astype(np.int32) + N # store the num of bits for each component

    # init
    for i in range(0, K+1):
        for j in range(0, N+1):
            
            DP[i][j] = -1e9

    DP[0][0] = 0

    
    
    # print(f"Max bit: {max_bit} | Min bit: {min_bit}")
    # non-recursive
    for i in range(1, K+1):
        for j in range(0, N+1):
            
            max_reward = -1e9

            # for x in range(min_bit, min(max_bit, j)+1):
            for x in range(min_bit, max_bit+1):

                if j - x >= 0 and x <= alloc[i-1][j-x]:  
                    # current_reward = DP[i-1][j-x]+x*EV[i-1]*ScaleFactor(x,A,delta)
                    if regularization == 'prev':
                        current_reward = DP[i-1][j-x] + x*EV[i-1] + regularization_term_prev(x, EV[i-1], delta)
                    elif regularization == '1':
                        current_reward = DP[i-1][j-x] + x*EV[i-1] + regularization_term_1(x, EV[i-1], A, pos=i/K, delta=delta)
                    elif regularization == '2':
                        current_reward = DP[i-1][j-x] + x*EV[i-1] + regularization_term_2(x, EV[i-1], A, pos=i/K, delta=delta)
                    elif regularization == '3':
                        current_reward = DP[i-1][j-x] + x*EV[i-1] + regularization_term_3(x, EV[i-1], A, pos=i/K, delta=delta)
                    elif regularization == '4':
                        current_reward = DP[i-1][j-x] + x*EV[i-1] + regularization_term_4(x, alloc[i-1][j-x], EV[i-1], EV[i-2] ,A, pos=i/K, delta=delta)
                    elif regularization == '5':
                        current_reward = DP[i-1][j-x] + x*EV[i-1] + regularization_term_5(x, EV[i-1], delta, i/K)

                    elif regularization == '6':
                        current_reward = DP[i-1][j-x] + x*EV[i-1] + regularization_term_6(x, EV[i-1], delta, i/K)
                    
                    elif regularization == '7':
                        current_reward = DP[i-1][j-x] + x*EV[i-1] + regularization_term_7(x, EV[i-1], delta, i/K)


                        # print(x*EV[i-1])

                    if current_reward > max_reward:

                        alloc[i][j] = x
                        max_reward = current_reward
                        DP[i][j] = current_reward

    
    def print_sol(alloc, K, N):
        
        bit_arr = []  
        unused_bit = N
        for i in range(K, 1, -1):
            bit_arr.append(alloc[i][unused_bit])
            unused_bit -= alloc[i][unused_bit]

        bit_arr.append(unused_bit)
        return bit_arr
    
    bit_arr = print_sol(alloc, K, N)

    assert np.sum(bit_arr) == N

    return DP[K][N], bit_arr[::-1]


def eval_cluster(labels_test, y_pred):

    from sklearn import metrics

    ri = metrics.rand_score(labels_test, y_pred)
    ari = metrics.adjusted_rand_score(labels_test, y_pred)
    nmi = metrics.normalized_mutual_info_score(labels_test, y_pred)

    return ri, ari, nmi

def compute_clustering_metrics(y_true,y_pred):
    res = pd.DataFrame(data=np.zeros((1, 3), dtype=np.float32), index=[0],
                       columns=['ri', 'ari','nmi'])

    ri,ari,nmi = eval_cluster(y_true, y_pred)
    res['ri'] = ri
    res['ari'] = ari
    res['nmi'] = nmi
    
    return res