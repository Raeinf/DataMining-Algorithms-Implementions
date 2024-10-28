import pandas as pd

def read_glass(filePath):
    data = pd.read_csv(filePath, index_col=0,header=None)
    data.reset_index(drop=True, inplace=True)
    data.columns = range(data.shape[1])
    return data

def read_aggregation(filePath):
    data = pd.read_csv( filePath, header=None, sep="\s+")
    return data

def read_iris(filePath):
    data = pd.read_csv(filePath, header=None)
    items = data[4].unique()
    replacement_mapping = {item: letter for item, letter in zip(items, [1, 2, 3])}
    data[4] = data[4].map(replacement_mapping)
    return data

def read_D31(filePath):
    data = pd.read_csv(filePath, header=None, sep = "\t")
    return data

def read_R15(filePath):
    data = pd.read_csv(filePath, header=None, sep = "\t")
    return data




