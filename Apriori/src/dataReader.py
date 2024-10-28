# import pandas as pd
# import numpy as np


def dataReader(filePath: str):
    with open(filePath, "r") as file:
        # Read all lines from the file
        lines = file.readlines()
    # spliting data into array of ints
    data = []
    for line in lines:

        row = set([int(x) for x in line.split()])
        data.append(row)

    # converting list to numpy int arrays
    return data
