from sklearn.decomposition import PCA

import os
import sys
import time
import numpy as np

from itertools import product
from sklearn.cluster import KMeans

class SPARTAN:
    def __init__(self,
                 alphabet_size=[8,4,4,2],
                 window_size=0,
                 word_length=4,
                 binning_method='equi-depth',
                 remove_repeat_words = False,
                 assignment_policy = 'DAA', # direct or DAA
                 bit_budget = 16,
                 lamda=0.5,
                 build_histogram=True,
                 downsample = 1.0,
                 pca_solver = 'auto'):
        
        if isinstance(alphabet_size,int):
            self.alphabet_size = [alphabet_size]*word_length
        else:    
            self.alphabet_size = alphabet_size
        self.window_size = window_size
        self.word_length = word_length
        self.binning_method = binning_method
        self.remove_repeat_words = remove_repeat_words
        self.assignment_policy = assignment_policy
        self.bit_budget = bit_budget
        self.lamda = lamda
        self.build_histogram = build_histogram
        self.downsample = downsample
        self.pca_solver = pca_solver

        self.word_to_num = None
        

    def fit(self, X, y=None):

        self.pca = PCA(n_components=self.word_length, svd_solver=self.pca_solver)
        self._X = X
        self._y = y


        # do random (down)sampling if required
        if self.downsample < 1.0:
            sampling_num = min(max(int(np.ceil(len(X)*self.downsample)), 10), 1000)
            random_indices = np.random.choice(X.shape[0], sampling_num, replace=False)
            self._X_downsampled = self._X[random_indices]

        # check data statistics
        n_instances, series_length = X.shape
        if self.window_size == 0:
            window_size = series_length
            self.window_size = window_size
        elif self.window_size < 1:
            window_size = int(series_length *  self.window_size)
            self.window_size = window_size
        
        self.window_size = min(self.window_size, series_length)
        self.window_size = max(self.word_length, self.window_size)
        window_size = self.window_size
        
        # split data (for BOP)
        num_windows_per_inst = series_length - window_size + 1
        split = X[:,np.arange(window_size)[None,:] + np.arange(num_windows_per_inst)[:,None]]

        start_time = time.time()

        # --- Numeric Approximation ---
        if num_windows_per_inst == 1:
            if self.downsample < 1.0:
                
                print("original shape: ", X.shape)
                print("downsampled shape: ", self._X_downsampled.shape)
                
                self.pca.fit(self._X_downsampled)
                X_transform = self.pca.transform(X)
            else:
                X_transform = self.pca.fit_transform(X)
        elif self.window_size != 0 and self.window_size != series_length:
            
            flat_split = np.reshape(split,(-1,split.shape[2]))
            if self.downsample < 1.0:
                print("bop original shape: ", flat_split.shape)
                print("bop downsampled shape: ", flat_split[random_indices].shape)
                self.pca.fit(flat_split[random_indices])
                split_transorm = self.pca.transform(flat_split)
            else:
                split_transorm = self.pca.fit_transform(flat_split)

        self.evcr = self.pca.explained_variance_ratio_

        end_time = time.time()

        # print(f"[Training] PCA time: {(end_time-start_time)/n_instances:.2e}")

        # --- Discretization ---

        # alphabet allocation
        if self.assignment_policy == 'direct':
            self.alphabet_size = self.alphabet_size
            self.avg_alphabet_size = int(np.mean(self.alphabet_size))
            assigned_evc = self.evcr[0:self.word_length]

            assigned_evc = assigned_evc / np.sum(assigned_evc)

            # print("after norm: ", assigned_evc)
        elif self.assignment_policy in ['DAA', 'dp']:

            start_time = time.time()
            if isinstance(self.alphabet_size, list):
                self.alphabet_size = int(np.mean(self.alphabet_size))

            self.avg_alphabet_size = self.alphabet_size
            if self.bit_budget != int(np.log2(self.alphabet_size)*self.word_length):
                total_bit = self.bit_budget = int(np.log2(self.alphabet_size)*self.word_length)
            else:
                total_bit = self.bit_budget

            # truncate and re-norm
            assigned_evc = self.evcr[0:self.word_length]
            # print("before norm: ", assigned_evc)
            assigned_evc = assigned_evc / np.sum(assigned_evc)
            # print("after norm: ", assigned_evc)

            avg_allocation = self.bit_budget // self.word_length


            DP_reward, bit_arr = self.dynamic_alphabet_allocation(total_bit=total_bit, 
                                                                  EV=assigned_evc, 
                                                                  lamda=self.lamda)
            self.alphabet_size = [int(2**bit_arr[i]) for i in range(len(bit_arr))]
            # print("dp result: ", bit_arr)

            end_time = time.time()
            # print(f"[Training] DAA time: {(end_time-start_time)/n_instances:.2e}")
        
        # binning
        if num_windows_per_inst == 1:

            kept_components = X_transform[:,0:self.word_length]
            self.pca_repr = kept_components
            start_time = time.time()
            self.breakpoints = self.binning(kept_components)

            # print("bkpts: ", self.breakpoints)
            end_time = time.time()
            # print(f"[Training] Binning time: {(end_time-start_time)/n_instances:.2e}")
             
        else:
            breakpoints = self.binning(split_transorm)

            self.pca_repr = split_transorm[:,0:self.word_length]
            self.breakpoints = breakpoints

        return 
    
    def transform(self, X):
        n_instances, series_length = X.shape

        self.window_size = min(self.window_size, X.shape[-1])
        window_size = self.window_size
        if self.window_size == 0:
            window_size = series_length

        num_windows_per_inst = series_length - window_size + 1

        if self.window_size == 0 or self.window_size == X.shape[1]:
            
            start_time = time.time()
            X_transform = self.pca.transform(X)
            end_time = time.time()
            # print(f"[Inference] PCA time: {(end_time-start_time)/n_instances:.2e}")
                        
            start_time = time.time()
            kept_components = X_transform[:,0:self.word_length]
            words = self.generate_words(kept_components,self.breakpoints)
            end_time = time.time()
            # print(f"[Inference] Digitize time: {(end_time-start_time)/n_instances:.2e}")

            if self.build_histogram:
                self.pred_histogram = self.bag_to_hist_DAA(self.create_bags(words[:,None,:]))
            else:
                self.pred_histogram = np.zeros((1,1))
        else:
            split = X[:,np.arange(window_size)[None,:] + np.arange(num_windows_per_inst)[:,None]]

            flat_split = np.reshape(split,(-1,split.shape[2]))
            split_transorm = self.pca.transform(flat_split)

            breakpoints = self.binning(split_transorm)

            self.breakpoints = breakpoints
            flat_words = self.generate_words(split_transorm,breakpoints)

            words = np.reshape(flat_words,(n_instances,num_windows_per_inst,self.word_length)) 
            
            if self.build_histogram:
                self.pred_histogram = self.bag_to_hist_DAA(self.create_bags(words))
            else:
                self.pred_histogram = np.zeros((1,1))
        return words
    
    def fit_transform(self, X, y=None):

        self.fit(X, y=None)
        return self.transform(X)
        

    def fit_transform2(self,X,y=None):
        
        self.pca = PCA(n_components=self.word_length, svd_solver=self.pca_solver)
        self._X = X
        self._y = y


        # do random (down)sampling if required
        if self.downsample < 1.0:
            sampling_num = min(max(int(np.ceil(len(X)*self.downsample)), 10), 1000)
            random_indices = np.random.choice(X.shape[0], sampling_num, replace=False)
            self._X_downsampled = self._X[random_indices]

        # check data statistics
        n_instances, series_length = X.shape
        if self.window_size == 0:
            window_size = series_length
            self.window_size = window_size
        elif self.window_size < 1:
            window_size = int(series_length *  self.window_size)
            self.window_size = window_size
        
        self.window_size = min(self.window_size, series_length)
        self.window_size = max(self.word_length, self.window_size)
        window_size = self.window_size
        
        # split data (for BOP)
        num_windows_per_inst = series_length - window_size + 1
        split = X[:,np.arange(window_size)[None,:] + np.arange(num_windows_per_inst)[:,None]]

        start_time = time.time()

        # --- Numeric Approximation ---
        if num_windows_per_inst == 1:
            if self.downsample < 1.0:
                
                print("original shape: ", X.shape)
                print("downsampled shape: ", self._X_downsampled.shape)
                
                self.pca.fit(self._X_downsampled)
                X_transform = self.pca.transform(X)
            else:
                X_transform = self.pca.fit_transform(X)
        elif self.window_size != 0 and self.window_size != series_length:
            
            flat_split = np.reshape(split,(-1,split.shape[2]))
            if self.downsample < 1.0:
                print("bop original shape: ", flat_split.shape)
                print("bop downsampled shape: ", flat_split[random_indices].shape)
                self.pca.fit(flat_split[random_indices])
                split_transorm = self.pca.transform(flat_split)
            else:
                split_transorm = self.pca.fit_transform(flat_split)

        self.evcr = self.pca.explained_variance_ratio_

        end_time = time.time()

        # print(f"[Training] PCA time: {(end_time-start_time)/n_instances:.2e}")

        # --- Discretization ---

        # alphabet allocation
        if self.assignment_policy == 'direct':
            self.alphabet_size = self.alphabet_size
            self.avg_alphabet_size = int(np.mean(self.alphabet_size))
            assigned_evc = self.evcr[0:self.word_length]

            assigned_evc = assigned_evc / np.sum(assigned_evc)

            # print("after norm: ", assigned_evc)
        elif self.assignment_policy in ['DAA', 'dp']:

            start_time = time.time()
            if isinstance(self.alphabet_size, list):
                self.alphabet_size = int(np.mean(self.alphabet_size))

            self.avg_alphabet_size = self.alphabet_size
            if self.bit_budget != int(np.log2(self.alphabet_size)*self.word_length):
                total_bit = self.bit_budget = int(np.log2(self.alphabet_size)*self.word_length)
            else:
                total_bit = self.bit_budget

            # truncate and re-norm
            assigned_evc = self.evcr[0:self.word_length]
            # print("before norm: ", assigned_evc)
            assigned_evc = assigned_evc / np.sum(assigned_evc)
            # print("after norm: ", assigned_evc)

            avg_allocation = self.bit_budget // self.word_length


            DP_reward, bit_arr = self.dynamic_alphabet_allocation(total_bit=total_bit, 
                                                                  EV=assigned_evc, 
                                                                  lamda=self.lamda)
            self.alphabet_size = [int(2**bit_arr[i]) for i in range(len(bit_arr))]
            # print("dp result: ", bit_arr)

            end_time = time.time()
            # print(f"[Training] DAA time: {(end_time-start_time)/n_instances:.2e}")
        
        # binning
        if num_windows_per_inst == 1:

            kept_components = X_transform[:,0:self.word_length]
            self.pca_repr = kept_components
            start_time = time.time()
            self.breakpoints = self.binning(kept_components)

            # print("bkpts: ", self.breakpoints)
            end_time = time.time()
            # print(f"[Training] Binning time: {(end_time-start_time)/n_instances:.2e}")

            start_time = time.time()
            words = self.generate_words(kept_components,self.breakpoints)
            end_time = time.time()
            # print(f"[Training] Digitize time: {(end_time-start_time)/n_instances:.2e}")
            
            if self.build_histogram:
                self.train_histogram = self.bag_to_hist_DAA(self.create_bags(words[:,None,:]))
            else:
                self.train_histogram = np.zeros((1,1))
             
        else:
            breakpoints = self.binning(split_transorm)

            self.pca_repr = split_transorm[:,0:self.word_length]
            self.breakpoints = breakpoints
            flat_words = self.generate_words(split_transorm,breakpoints)

            words = np.reshape(flat_words,(n_instances,num_windows_per_inst,self.word_length))
            # print("sliding win: ", words.shape)
            if self.build_histogram:
                self.train_histogram = self.bag_to_hist_DAA(self.create_bags(words)) 
            else:
                self.train_histogram = np.zeros((1,1))
        
        # print(words)
        return words

    def generate_words(self,pca,breakpoints):
        words = np.zeros((pca.shape[0],self.word_length))
        for a in range(pca.shape[0]):
            for i in range(self.word_length):
                words[a,i] = np.digitize(pca[a,i],breakpoints[i],right=True)
        
        return words

    def binning(self,pca):
        if self.binning_method == 'equi-depth' or self.binning_method == 'equi-width' or self.binning_method == 'kmeans':
            breakpoints = self._mcb(pca)
        return breakpoints

    def _mcb(self,pca):
        breakpoints = []
        mindist_breakpoints =[]

        pca = np.round(pca,4)
        for letter in range(self.word_length):
            column = np.sort(pca[:,letter])
            column_min = np.min(column)
            bin_index = 0
            
            letter_alphabet_size = self.alphabet_size[letter]
            breakpoint_i = np.zeros(letter_alphabet_size)
            mindist_breakpoint_i = np.zeros(letter_alphabet_size+1)

            mindist_breakpoint_i[0] = - sys.float_info.max

            #use equi-depth binning
            if self.binning_method == "equi-depth":
                target_bin_depth = len(pca) / letter_alphabet_size

                for bp in range(letter_alphabet_size - 1):
                    bin_index += target_bin_depth 
                    breakpoint_i[bp] = column[int(bin_index)]
                    mindist_breakpoint_i[bp+1] = column[int(bin_index)]
            
            #equi-width binning aka equi-frequency binning
            elif self.binning_method == "equi-width":
                target_bin_width = (column[-1] - column[0]) / letter_alphabet_size

                for bp in range(letter_alphabet_size - 1):
                    breakpoint_i[bp] = (bp + 1) * target_bin_width + column[0]
                    mindist_breakpoint_i[bp+1] = (bp + 1) * target_bin_width + column[0]

            elif self.binning_method == "kmeans":
                
                binning_clust_model = KMeans(n_clusters=letter_alphabet_size, random_state=0, n_init="auto").fit(column.reshape(-1,1))
                centroids = np.sort(binning_clust_model.cluster_centers_, axis=0)
                bkps = (centroids[:-1] + centroids[1:]) / 2
                
                # print(f"Letter: {letter}, kmeans breakpoints: {bkps}")
                breakpoint_i[:-1] = bkps.reshape(-1,)

            breakpoint_i[letter_alphabet_size - 1] = sys.float_info.max
            mindist_breakpoint_i[letter_alphabet_size] = sys.float_info.max
            breakpoints.append(breakpoint_i)
            mindist_breakpoints.append(mindist_breakpoint_i)

            # print(f"Letter: {letter}, breakpoints: {breakpoint_i}")
            
        # breakpoints[:, self.alphabet_size - 1] = sys.float_info.max
        self.mindist_breakpoints = mindist_breakpoints

        return breakpoints
    
    def create_bags(self,wordslists):
        n_instances,n_words_per_inst,word_length = wordslists.shape

        remove_repeat_words = self.remove_repeat_words
        wordslists = wordslists.astype(np.int32)
        bags = []
        last_word = None
        for i in range(n_instances):
            bag = {}
            wordlist = wordslists[i]
            for j in range(n_words_per_inst):
                word = wordlist[j]
                # print(word)
                text = ''.join(map(str,word))
                if (not remove_repeat_words) or (text != last_word):
                    bag[text] = bag.get(text, 0) + 1 

                last_word = text
            bags.append(bag)

        return bags

    def bag_to_hist(self,bags):
        n_instances = len(bags)

        word_length = self.word_length

        possible_words = self.alphabet_size[0] ** word_length
        print("possible_words: ", possible_words)
        word_to_num = [np.base_repr(i,base=self.alphabet_size[0]) for i in range(possible_words)]

        word_to_num = ['0'*(word_length - len(word)) + word for word in word_to_num]
        all_win_words = np.zeros((n_instances,possible_words))

        # print(word_to_num)

        for j in range(n_instances):
            bag = bags[j]
            # print(bag)

            for key in bag.keys():
                v = bag[key]

                n = word_to_num.index(key)

                all_win_words[j,n] = v
        return all_win_words


    def combination_words_DAA(self, alphabet_sizes):
        """
        Generate all possible words given the alphabet sizes for each letter and the word length.
        
        :param alphabet_sizes: List[int], List where each element specifies the number of choices for that letter position.
        :param word_length: int, Length of the word to be generated.
        :return: List[str], List of all possible words.
        """
        # Create a list of ranges based on the alphabet sizes
        ranges = [range(size) for size in alphabet_sizes]
        
        # Generate all possible combinations
        all_combinations = product(*ranges)
        
        # Convert each combination to a string and store in the list
        all_words = [''.join(map(str, combination)) for combination in all_combinations]
        
        return all_words
    
    def bag_to_hist_DAA(self,bags):
        n_instances = len(bags)

        word_length = self.word_length

        if isinstance(self.alphabet_size, list):
            possible_words = int(np.prod(self.alphabet_size))
        else:
            possible_words = int(self.alphabet_size ** word_length)

        if self.word_to_num is None:
            word_to_num = self.combination_words_DAA(self.alphabet_size)
            self.word_to_num = word_to_num

        all_win_words = np.zeros((n_instances,possible_words))

        # print(word_to_num)

        for j in range(n_instances):
            bag = bags[j]
            # print(bag)

            for key in bag.keys():
                v = bag[key]

                n = self.word_to_num.index(key)

                all_win_words[j,n] = v

        return all_win_words

    def dynamic_alphabet_allocation(self, total_bit, EV, lamda=0.5):

        def regularization_term(x, ev_value, avg_bit, lamda=0.5, pos=1.0):
            
            return -lamda * (x-avg_bit)**2 * ev_value


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

        # non-recursive
        for i in range(1, K+1):
            for j in range(0, N+1):
                
                max_reward = -1e9

                for x in range(min_bit, max_bit+1):

                    if j - x >= 0 and x <= alloc[i-1][j-x]:  
                        
                        current_reward = DP[i-1][j-x] + x*EV[i-1] + regularization_term(x, EV[i-1], A, lamda, i/K)

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

