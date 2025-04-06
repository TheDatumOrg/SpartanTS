import os
import argparse
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", required=False, default="data/ucr/Univariate_ts/")
    parser.add_argument("-c", "--classifier", required=False, default="spartan")
    parser.add_argument("-g",'--config',required=False,default=None) # config file for methods
    parser.add_argument("-i", "--itr", required=False, default=3)
    parser.add_argument("-n", "--norm", required=False, default="zscore")  # zscore as default
    parser.add_argument("-s","--save_model",required=False, default=None)
    parser.add_argument("-t","--top_num",required=False, default=0, type=int) # test top t datasets
    parser.add_argument("-p","--downsample",default=1.0, type=float)
    parser.add_argument("-e","--eval_task",required=False, default='classification', type=str) # classfication, clustering, tlb, anomaly
    parser.add_argument("-k","--clust_model",required=False, default='kmedoids', type=str)  # -- clustering experiments
    parser.add_argument("-l","--linkage",default='complete')
    parser.add_argument("-o","--kmedoids_type",default='pam')
    parser.add_argument("-b","--data_split",default='split', type=str)
    parser.add_argument("-r","--repr_type",default='single', type=str)
    parser.add_argument("--alpha_min", required=False, default=4, type=int)  # -- tlb experiments
    parser.add_argument("--alpha_max", required=False, default=10, type=int) 
    parser.add_argument("--wordlen_min", required=False, default=4, type=int) 
    parser.add_argument("--wordlen_max", required=False, default=10, type=int) 

    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":

    arguments = parse_arguments()
    data_path = arguments.data
    classifier_name = arguments.classifier
    normalization = arguments.norm
    itr = arguments.itr
    config = arguments.config
    top_num = arguments.top_num
    eval_task = arguments.eval_task
    
    clust_model = arguments.clust_model
    linkage = arguments.linkage
    kmedoids_type = arguments.kmedoids_type
    data_split = arguments.data_split
    repr_type = arguments.repr_type
    downsample_rate = arguments.downsample
    
    alphabet_max = arguments.alpha_max
    alphabet_min = arguments.alpha_min
    wordlen_max  = arguments.wordlen_max
    wordlen_min  = arguments.wordlen_min


    dset_info = pd.read_csv('benchmark/util/summaryUnivariate.csv')
    dset_info = dset_info.sort_values(by=['numTrainCases','numTestCases'])

    if eval_task in ['classification', 'clustering', 'tlb']:
        for i in range(dset_info.shape[0]):

            if top_num - 1 < i and top_num != 0:
                continue

            dataset = dset_info['problem'].iloc[i]

            print("Dataset No.: ", i, dataset)

            if eval_task == 'classification':
                call_string = 'python -m benchmark.eval_classfication --data {} --classifier {} --norm {} --problem {} --itr {} --config {} --downsample {}'.format(data_path,classifier_name,normalization,dataset,itr,config, downsample_rate)
            elif eval_task == 'clustering':
                call_string = 'python -m benchmark.eval_clustering --data {} --classifier {} --norm {} --problem {} --itr {} --config {} --clust_model {} --linkage {} --kmedoids_type {} -b {} -t {}'.format(data_path,classifier_name,normalization,dataset,itr,config, clust_model, linkage, kmedoids_type, data_split, repr_type)
            elif eval_task == 'tlb':
                call_string = 'python -m benchmark.eval_tlb --data {} --problem {} -x {} --alpha_max {} --alpha_min {} --wordlen_max {} --wordlen_min {}'.format(data_path, dataset, i, alphabet_max, alphabet_min, wordlen_max, wordlen_min)

            os.system(call_string)

    elif eval_task == 'anomaly':

        dataset_list = ['KDD21'] # change the dataset name as you need


        for dataset in dataset_list:

            datadir = f"../data/TSAD/TSB-UAD-Public/{dataset}"
            for i, filename in enumerate(sorted(os.listdir(datadir))):
                if not filename.endswith('.out'):
                    continue
                
                if top_num - 1 < i and top_num != 0:
                    continue

                call_string = 'python -m benchmark.eval_anomaly --data {} --classifier {} --norm {} --problem {} --itr {} --config {} -t {}'.format(datadir,classifier_name,normalization,filename,itr,config, repr_type)

                os.system(call_string)   
