# @Time   : 2022.03.03
# @Author : Darrius Lei
# @Email  : darrius.lei@outlook.com
import os
import pydash as ps
import datetime, time
from matplotlib import pyplot as plt

def set_attr(obj, dict_, keys = None, except_type = None):
    '''Quickly assign properties to objects

    Parameters:
    -----------
    obj: object of a python class
        Assign values to the properties of the object

    dict_: dict
        A dictionary for assigning values to object attributes, 
        whose key is the attribute, and the corresponding value is the value
    
    keys: list, optional
        defalue: None
        Select the corresponding key in the dictionary to assign a value to the object

    '''
    if keys is not None:
        dict_ = ps.pick(dict, keys);
    for key, value in dict_.items():
        if type(value) == except_type:
            continue;
        setattr(obj, key, value);

def get_attr(obj, keys):
    '''Quick access to object properties

    Parameters:
    -----------

    obj: object of a python class
        Get the properties of the object
    
    keys: list
        A list of properties of the object that require
    
    Returns:
    --------

    dict: dict
        The required attribute is the value corresponding to the keys of the dictionary
    

    '''
    dict = dict();
    for key in keys:
        if hasattr(obj, key):
            dict[key] = getattr(obj, key);
    return dict;

def find_files(root_dir, end_with = '.txt', decorate = '"', ignores = None):
    '''Find the file with the corresponding extension from the root directory.

    Parameters:
    -----------
    root_dir: the root directory

    end_with: File extension

    decorate:string decoration

    Example:
    --------
    >>> #find all files end with '.txt' 
    >>> from lib.util import find_files
    >>> files = find_files(
        root_dir = '/home/data', 
        end_with = '.txt',
        decorate = '');
    '''
    ignores = [] if ignores is None else ignores;
    f_files = [];
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(end_with) and file not in ignores:
                f_files.append(decorate + os.path.join(root, file) + decorate);
    return f_files;

def set_seed(seed):
    '''
    '''
    import torch
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 

def get_date(separator = '_'):
    '''Quickly get formated date
    '''
    return str(datetime.date.today()).replace('-', separator);

def get_time(separator = '_'):
    '''Quickly get formated time
    '''
    now = datetime.datetime.now();
    t = f'{now.hour}-{now.minute}-{now.second}';
    return t.replace('-', separator);

def s2hms(s):
    '''Convert s to hms
    '''
    return time.strftime("%H hour - %M min - %S s", time.gmtime(s));

def clear_output():
    print('\033[F\033[J', end='')

def single_plot(x_data, y_data, x_label, y_label, path):
    plt.figure(figsize = (10, 6));
    plt.plot(x_data, y_data);
    plt.xlabel(x_label);
    plt.ylabel(y_label);
    plt.savefig(path, dpi = 400);