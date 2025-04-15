<p align="center">
<img src="doc/fig/spartan.png" width="300"/>
</p>

<h1 align="center">SPARTAN</h1>
<h2 align="center">🔠 Data-Adaptive Symbolic Time-Series Approximation</h2>

Main Recent Update:

+ [Feb. 3, 2025] Paper accepted to ACM SIGMOD! The full paper can be found at [paper link]().

If you find our work helpful, please consider citing and giving a star :-)

<details>
<summary>"SPARTAN: Data-Adaptive Symbolic Time-Series Approximation" ACM SIGMOD 2025.</summary>

```bibtex
@inproceedings{yang2025spartan,
  title={SPARTAN: Data-Adaptive Symbolic Time-Series Approximation},
  author={Yang, Fan and Paparrizos, John},
  booktitle={Proceedings of the 2025 ACM SIGMOD international conference on management of data},
  year={2025}
}
```

</details>

## Table of Contents

- [📄 Overview](#overview)
- [⚙️ Get Started](#start)
- [🏄‍♂️ Dive into Symbolic Representation Benchmark Study](#symb)
- [✉️ Contact](#contact)
- [🎉 Acknowledgement](#ack)

<h2 id="overview"> 📄 Overview </h2>

Symbolic approximations are dimensionality reduction techniques that convert time series into sequences of discrete symbols, enhancing interpretability while reducing computational and storage costs. To construct symbolic representations, first numeric representations approximate and capture properties of raw time series, followed by a discretization step that converts these numeric dimensions into symbols. Despite decades of development, existing approaches have several key limitations that often result in unsatisfactory performance: they (i) rely on data-agnostic numeric approximations, disregarding intrinsic properties of the time series; (ii) decompose dimensions into equal-sized subspaces, assuming independence among dimensions; and (iii) allocate a uniform encoding budget for discretizing each dimension or subspace, assuming balanced importance. To address these shortcomings, we propose SPARTAN, a novel data-adaptive symbolic approximation method that intelligently allocates the encoding budget according to the importance of the constructed uncorrelated dimensions. Specifically, SPARTAN (i) leverages intrinsic dimensionality reduction properties to derive non-overlapping, uncorrelated latent dimensions; (ii) adaptively distributes the budget based on the importance of each dimension by solving a constrained optimization problem; and (iii) prevents false dismissals in similarity search by ensuring a lower bound on the true distance in the original space. To demonstrate SPARTAN’s robustness, we conduct the most comprehensive study to date, comparing SPARTAN with seven state-of-the-art symbolic methods across four tasks: classification, clustering, indexing, and anomaly detection. Rigorous statistical analysis across hundreds of datasets shows that SPARTAN outperforms competing methods significantly on all tasks in terms of downstream accuracy, given the same budget. Notably, SPARTAN achieves up to a 2x speedup compared to the most accurate rival. Overall, SPARTAN effectively improves the symbolic representation quality without storage or runtime overheads, paving the way for future advancements.

<h2 id="start"> ⚙️ Get Started </h2>

To install SPARTAN you will need to follow

**Step 1**: Install the required dependencies

```shell
# create your environment, and then
pip install -r requirements.txt
```

**Step 2**: Preparing the datasets for time-series analytical tasks:

+ Download [the UCR Archive](http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip) for classification, clustering, and the tightness of lower bound (TLB) tasks.
+ Download the [TSB-UAD Archive](https://github.com/TheDatumOrg/TSB-UAD) for time-series anomaly detection.

<h2 id="symb"> 🏄‍♂️ Dive into Symbolic Representation Benchmark Study </h2>

### #1 Example Usage

```shell
import numpy as np
from TSB_Symbolic.symbolic import SPARTAN

# random data
data = np.random.rand(100,50)

# initialize SPARTAN with alphabet_size=4 and word_length=8
model = SPARTAN(
    alphabet_size=4,
    word_length = 8
    )

# fit and transform the data
spartan_repr = model.fit_transform(data) # shape: (100, 8)
print("SPARTAN Transformed Data: \n", spartan_repr[:5]) 

# map it to readable symbols (using 'abcd' as an example)
mapping = {0: 'a', 1: 'b', 2: 'c', 3: 'd'}
symbol_map = np.vectorize(lambda x: mapping[int(x)])
spartan_repr = symbol_map(spartan_repr)
print("Map to readable symbols: \n", spartan_repr[:5]) 
```

### #2 Evaluation

Here, we present code examples for four downstream tasks. Config examples can be find in `benchmark/configs`.

#### Task-I: 1NN Classification

To test SPARTAN classification accuracy on a single dataset:

```shell
python -m benchmark.eval_classfication --classifier spartan --data /path/to/your/data --problem DatasetName --config ./path/to/model_params/
```

To test SPARTAN accuracy on all UCR datasets:

```shell
python main.py --eval_task classification --classifier spartan --data /path/to/your/data --config ./path/to/model_params/
```

#### Task-II: Clustering

To test SPARTAN clustering performance on a single dataset:

```shell
python -m benchmark.eval_clustering --classifier spartan --data /path/to/your/data --problem DatasetName --config ./path/to/model_params/
```

To test SPARTAN performance on all UCR datasets:

```shell
python main.py --eval_task clustering --classifier spartan --data /path/to/your/data --config ./path/to/model_params/
```

#### Task-III: Anomaly Detection

To test SPARTAN anomaly detection performance on a single time series:

```shell
python -m benchmark.eval_anomaly --data /path/to/your/data --classifier spartan --problem FileName --config ./path/to/model_params/
```

To test SPARTAN performance on all time series:

```shell
python main.py --eval_task anomaly --classifier spartan --data /path/to/your/data --config ./path/to/model_params/
```

#### Task-IV: Tightness of Lower Bound (TLB)

To test the tlb performance (SPARTAN, SAX, and SFA) on a single UCR dataset:

```shell
python -m benchmark.eval_tlb --data /path/to/your/data --problem DatasetName -x DatasetID --alpha_max MaxAlphabetSize --alpha_min MinAlphabetSize --wordlen_max MaxWordLen --wordlen_min MinWordLen
```

To test the tlb performance (SPARTAN, SAX, and SFA) on all UCR datasets:

```shell
python main.py --eval_task tlb --data /path/to/your/data --alpha_max MaxAlphabetSize --alpha_min MinAlphabetSize --wordlen_max MaxWordLen --wordlen_min MinWordLen
```

<h2 id="contact"> ✉️ Contact </h2>

If you have any questions or suggestions, feel free to contact:

* Fan Yang (yang.7007@osu.edu)
* John Paparrizos (paparrizos.1@osu.edu)

Or describe it in Issues.

<h2 id="ack"> 🎉 Acknowledgement </h2>

We would like to acknowledge Ryan DeMilt (demilt.4@osu.edu) for the valuable contributions to this work. Also appreciate the following github repos a lot for their valuable code base:

* https://github.com/tslearn-team/tslearn/
* https://github.com/sktime/sktime
* https://github.com/aeon-toolkit/aeon
