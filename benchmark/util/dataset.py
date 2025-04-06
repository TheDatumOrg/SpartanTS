import os
import re
import shutil
import tempfile
import urllib
import zipfile
from datetime import datetime
from distutils.util import strtobool
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample_poly

from sklearn.model_selection import train_test_split

name = "DataLoader"



def resample_length(X,len_resample=None):
    if len_resample is None:
        max_len = X.shape[2]
    else:
        max_len = len_resample
    tmp = []
    for i in range(len(X)):
        _x = X[i,:]

        _y = []
        for y in _x:
            if np.isnan(y).any():
                y = resample_poly(y[~np.isnan(y)],max_len,len(y[~np.isnan(y)]))
            _y.append(y)
        tmp.append(_y)
    X_transform = np.array(tmp)
    return X_transform
def fill_nontrailing_missing(X):
    tmp = []
    for i in range(len(X)):
        _x = X[i,:]

        _y = []
        for y in _x:
            if np.isnan(y).any():
                #y_end = np.isnan(y).nonzero().max()
                y_series = pd.Series(y)
                #y_series = y_series.interpolate(method='linear',limit_direction='both')
                y_series = y_series.ffill()
                y_series = y_series.bfill()
                y = np.array(np.transpose(y_series))
            _y.append(y)
        tmp.append(_y)
    X_transform = np.array(tmp)
    return X_transform


def _load_header_info(file):
    """Load the meta data from a .ts file and advance file to the data.

    Parameters
    ----------
    file : stream.
        input file to read header from, assumed to be just opened

    Returns
    -------
    meta_data : dict.
        dictionary with the data characteristics stored in the header.
    """
    meta_data = {
        "problemname": "none",
        "timestamps": False,
        "missing": False,
        "univariate": True,
        "equallength": True,
        "classlabel": True,
        "targetlabel": False,
        "class_values": [],
    }
    boolean_keys = ["timestamps", "missing", "univariate", "equallength", "targetlabel"]
    for line in file:
        line = line.strip().lower()
        line = re.sub(r"\s+", " ", line)
        if line and not line.startswith("#"):
            tokens = line.split(" ")
            token_len = len(tokens)
            key = tokens[0][1:]
            if key == "data":
                if line != "@data":
                    raise IOError("data tag should not have an associated value")
                return meta_data
            if key in meta_data.keys():
                if key in boolean_keys:
                    if token_len != 2:
                        raise IOError(f"{tokens[0]} tag requires a boolean value")
                    if tokens[1] == "true":
                        meta_data[key] = True
                    elif tokens[1] == "false":
                        meta_data[key] = False
                elif key == "problemname":
                    meta_data[key] = tokens[1]
                elif key == "classlabel":
                    if tokens[1] == "true":
                        meta_data["classlabel"] = True
                        if token_len == 2:
                            raise IOError(
                                "if the classlabel tag is true then class values "
                                "must be supplied"
                            )
                    elif tokens[1] == "false":
                        meta_data["classlabel"] = False
                    else:
                        raise IOError("invalid class label value")
                    meta_data["class_values"] = [token.strip() for token in tokens[2:]]
        if meta_data["targetlabel"]:
            meta_data["classlabel"] = False
    return meta_data


def _get_channel_strings(line, target, missing):
    """Split a string with timestamps into separate csv strings."""
    channel_strings = re.sub(r"\s", "", line)
    channel_strings = channel_strings.split("):")
    c = len(channel_strings)
    if target:
        c = c - 1
    for i in range(c):
        channel_strings[i] = channel_strings[i] + ")"
        numbers = re.findall(r"\d+\.\d+|" + missing, channel_strings[i])
        channel_strings[i] = ",".join(numbers)
    return channel_strings


def _load_data(file, meta_data, replace_missing_vals_with="NaN"):
    """Load data from a file with no header.

    this assumes each time series has the same number of channels, but allows unequal
    length series between cases.

    Parameters
    ----------
    file : stream, input file to read data from, assume no comments or header info
    meta_data : dict.
        with meta data in the file header loaded with _load_header_info

    Returns
    -------
    data: list[np.ndarray].
        list of numpy arrays of floats: the time series
    y_values : np.ndarray.
        numpy array of strings: the class/target variable values
    meta_data :  dict.
        dictionary of characteristics enhanced with number of channels and series length
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    """
    data = []
    n_cases = 0
    n_channels = 0  # Assumed the same for all
    current_channels = 0
    series_length = 0
    y_values = []
    target = False
    if meta_data["classlabel"] or meta_data["targetlabel"]:
        target = True
    for line in file:
        line = line.strip().lower()
        line = line.replace("nan", replace_missing_vals_with)
        line = line.replace("?", replace_missing_vals_with)
        if "timestamps" in meta_data and meta_data["timestamps"]:
            channels = _get_channel_strings(line, target, replace_missing_vals_with)
        else:
            channels = line.split(":")
        n_cases += 1
        current_channels = len(channels)
        if target:
            current_channels -= 1
        if n_cases == 1:  # Find n_channels and length  from first if not unequal
            n_channels = current_channels
            if meta_data["equallength"]:
                series_length = len(channels[0].split(","))
        else:
            if current_channels != n_channels:
                raise IOError(
                    f"Inconsistent number of dimensions in case {n_cases}. "
                    f"Expecting {n_channels} but have read {current_channels}"
                )
            if meta_data["univariate"]:
                if current_channels > 1:
                    raise IOError(
                        f"Seen {current_channels} in case {n_cases}."
                        f"Expecting univariate from meta data"
                    )
        if meta_data["equallength"]:
            current_length = series_length
            max_length = current_length
        else:
            current_length = len(channels[0].split(","))
            max_length = current_length
            for i in range(0,n_channels):
                single_channel = channels[i].strip()
                data_series = single_channel.split(',')
                if max_length < len(data_series):
                    max_length = len(data_series)
        np_case = np.zeros(shape=(n_channels, max_length))

        for i in range(0, n_channels):
            single_channel = channels[i].strip()
            data_series = single_channel.split(",")
            data_series = [float(x) for x in data_series]
            if len(data_series) < max_length:
                data_series = np.pad(data_series,(0,(max_length - len(data_series))),'constant',constant_values=np.nan)
            # if len(data_series) != current_length:
            #     equal_length = meta_data["equallength"]
            #     raise IOError(
            #         f"channel {i} in case {n_cases} has a different number of "
            #         f"observations to the other channels. "
            #         f"Saw {current_length} in the first channel but"
            #         f" {len(data_series)} in the channel {i}. The meta data "
            #         f"specifies equal length == {equal_length}. But even if series "
            #         f"length are unequal, all channels for a single case must be the "
            #         f"same length"
            #     )
            np_case[i] = np.array(data_series)
        data.append(np_case)
        if target:
            y_values.append(channels[n_channels])
    if meta_data["equallength"]:
        data = np.array(data)
    else:
        max_len = max([len(arr[0]) for arr in data])
        padded_data = []
        for arr in data:
            padded_data.append([np.pad(arr[0],(0,(max_len-len(arr[0]))),'constant',constant_values=np.nan)])
            #print(arr)
        data = np.array(padded_data)
    return data, np.asarray(y_values), meta_data


def load_from_tsfile(
    full_file_path_and_name,
    replace_missing_vals_with="NaN",
    return_meta_data=False,
    return_type="auto",
):
    """Load time series .ts file into X and (optionally) y.

    Parameters
    ----------
    full_file_path_and_name : string
        full path of the file to load, .ts extension is assumed.
    replace_missing_vals_with : string, default="NaN"
        issing values in the file are replaces with this value
    return_meta_data : boolean, default=False
        return a dictionary with the meta data loaded from the file
    return_type : string, default = "auto"
        data type to convert to.
        If "auto", returns numpy3D for equal length and list of numpy2D for unequal.
        If "numpy2D", will squash a univariate equal length into a numpy2D (n_cases,
        n_timepoints). Other options are available but not supported medium term.

    Returns
    -------
    data: Union[np.ndarray,list]
        time series data, np.ndarray (n_cases, n_channels, series_length) if equal
        length time series, list of [n_cases] np.ndarray (n_channels, n_timepoints)
        if unequal length series.
    y : target variable, np.ndarray of string or int
    meta_data : dict (optional).
        dictionary of characteristics, with keys
        "problemname" (string), booleans: "timestamps", "missing", "univariate",
        "equallength", "classlabel", "targetlabel" and "class_values": [],

    Raises
    ------
    IOError if the load fails.
    """
    # Check file ends in .ts, if not, insert
    if not full_file_path_and_name.endswith(".ts"):
        full_file_path_and_name = full_file_path_and_name + ".ts"
    # Open file
    with open(full_file_path_and_name, "r", encoding="utf-8") as file:
        # Read in headers
        meta_data = _load_header_info(file)
        # load into list of numpy
        data, y, meta_data = _load_data(file, meta_data)

    # if equal load to 3D numpy
    if meta_data["equallength"]:
        data = np.array(data)
        if return_type == "numpy2D" and meta_data["univariate"]:
            data = data.squeeze()
    # If regression problem, convert y to float
    if meta_data["targetlabel"]:
        y = y.astype(float)
    if return_meta_data:
        return data, y, meta_data
    return data, y


def create_numpy_dataset(
        name = 'ArrowHead',
        path='data/ucr/Univariate_ts/',
        return_meta_data =False,
        equalize_length=True,
        fill_missing = True,
        resample=False,
        test_size=0.3,
        random_state=0
    ):
    train_dataset_path = os.path.join(path,'{}/{}_{}.ts'.format(name, name,'TRAIN'))
    X_train,y_train,train_meta_data = load_from_tsfile(train_dataset_path,return_meta_data=True)

    test_dataset_path = os.path.join(path,'{}/{}_{}.ts'.format(name, name,'TEST'))
    test_dataset_path = path +'/{}/{}_{}.ts'.format(name, name,'TEST')
    X_test,y_test,test_meta_data = load_from_tsfile(test_dataset_path,return_meta_data=True)
        
    if (not (train_meta_data['equallength'] and test_meta_data['equallength'])) and equalize_length:
        max_len = max([X_train.shape[2],X_test.shape[2]])

        extend_tmp = []
        if(X_train.shape[2] < X_test.shape[2]):
            X_train = np.pad(X_train, [(0,0),(0,0),(0,max_len - X_train.shape[2])],mode = 'constant',constant_values=np.nan)
        elif(X_train.shape[2] > X_test.shape[2]):
            X_test = np.pad(X_test, [(0,0),(0,0),(0,max_len - X_test.shape[2])],mode = 'constant',constant_values=np.nan)

        X_train = resample_length(X_train,max_len)
        X_test = resample_length(X_test,max_len)
    if (train_meta_data['missing'] or test_meta_data['missing']) and fill_missing:
        X_train = fill_nontrailing_missing(X_train)

        X_test = fill_nontrailing_missing(X_test)

    if resample:
        X_full = np.concatenate([X_train,X_test],axis=0)
        y_full = np.concatenate([y_train,y_test],axis=0)

        X_train,X_test,y_train,y_test = train_test_split(X_full,y_full,test_size=test_size)
    
    return X_train,y_train,X_test,y_test   
