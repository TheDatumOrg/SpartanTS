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