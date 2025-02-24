import numpy as np
import glob
from sklearn.model_selection import KFold

def load_dataset(dataset_path):
    """
    Parameters
    --------------------
    :param dataset_path:
    :param n_folds:
    :return: list
        List contains split datasets for K-Fold cross-validation
    """

    # load train datapath from path
    path0 = glob.glob(dataset_path+'/0/*')
    path1 = glob.glob(dataset_path+'/1/*')
    path2 = glob.glob(dataset_path+'/2/*')
    path3 = glob.glob(dataset_path+'/3/*')
    path4 = glob.glob(dataset_path+'/4/*')

    all_path = path0+path1+path2+path3+path4


    return all_path