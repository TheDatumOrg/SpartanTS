<p align="center">
<img src="doc/fig/spartan.png" width="300"/>
</p>

<h1 align="center">SPARTAN</h1>
<h2 align="center">🔠 Data-Adaptive Symbolic Time-Series Approximation</h2>


## #1 Getting Started

To install SPARTAN you will need to follow

**Step 1**: Install the required dependencies

```shell
# create your environment, and then
pip install -r requirements.txt
```

**Step 2**: Preparing the datasets for time-series analytical tasks:
+ Download [the UCR Archive](http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip) for classification, clustering, and the tightness of lower bound (TLB) tasks.
+ Download the [TSB-UAD Archive](https://github.com/TheDatumOrg/TSB-UAD) for time-series anomaly detection.

## #2 Example
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

## #3 Evaluation

Here, we present code examples for four downstream tasks. Config examples can be find in `benchmark/configs`.

### Task-I: 1NN Classification
To test SPARTAN classification accuracy on a single dataset:

```shell
python -m benchmark.eval_classfication --classifier spartan --data /path/to/your/data --problem DatasetName --config ./path/to/model_params/
```

To test SPARTAN accuracy on all UCR datasets:
```shell
python main.py --eval_task classification --classifier spartan --data /path/to/your/data --config ./path/to/model_params/
```

### Task-II: Clustering
To test SPARTAN clustering performance on a single dataset:

```shell
python -m benchmark.eval_clustering --classifier spartan --data /path/to/your/data --problem DatasetName --config ./path/to/model_params/
```

To test SPARTAN performance on all UCR datasets:
```shell
python main.py --eval_task clustering --classifier spartan --data /path/to/your/data --config ./path/to/model_params/
```

### Task-III: Anomaly Detection
To test SPARTAN anomaly detection performance on a single time series:

```shell
python -m benchmark.eval_anomaly --data /path/to/your/data --classifier spartan --problem FileName --config ./path/to/model_params/
```

To test SPARTAN performance on all time series:
```shell
python main.py --eval_task anomaly --classifier spartan --data /path/to/your/data --config ./path/to/model_params/
```

### Task-IV: Tightness of Lower Bound (TLB)
To test the tlb performance (SPARTAN, SAX, and SFA) on a single UCR dataset:

```shell
python -m benchmark.eval_tlb --data /path/to/your/data --problem DatasetName -x DatasetID --alpha_max MaxAlphabetSize --alpha_min MinAlphabetSize --wordlen_max MaxWordLen --wordlen_min MinWordLen
```

To test the tlb performance (SPARTAN, SAX, and SFA) on all UCR datasets:
```shell
python main.py --eval_task tlb --data /path/to/your/data --alpha_max MaxAlphabetSize --alpha_min MinAlphabetSize --wordlen_max MaxWordLen --wordlen_min MinWordLen
```