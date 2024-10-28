import os
from src.dataReader import *
from src.algorithms import run_kmeans_multiple_times, run_kmedian_multiple_times

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


"********** Hw2  *****************"



if __name__ == "__main__":

    "-----------------Aggregation.data-------------------------"
    filePath = os.path.join(__location__, "data", "Aggregation.data")
    data = read_aggregation(filePath)
    k = data[data.shape[1] - 1].nunique()
    result, purity = run_kmeans_multiple_times(data, k)
    print("K-means of Aggregation is done !\n \tnumber of iteration: ", result[2])
    print("\tpurity: ",purity)
    result, purity = run_kmedian_multiple_times(data, k)
    print("K-median of Aggregation is done !\n \tnumber of iteration: ", result[2])
    print("\tpurity: ",purity)


    "-----------------D31.data-------------------------"
    filePath = os.path.join(__location__,"data", "D31.data")
    data = read_D31(filePath)
    k = data[data.shape[1] - 1].nunique()
    result, purity = run_kmeans_multiple_times(data, k)
    print("K-means of D31 is done !\n \tnumber of iteration: ", result[2])
    print("\tpurity: ",purity)
    result, purity = run_kmedian_multiple_times(data, k)
    print("K-median of D31 is done !\n \tnumber of iteration: ", result[2])
    print("\tpurity: ",purity)

    "-----------------R15.data-------------------------"
    filePath = os.path.join(__location__,"data", "R15.data")
    data = read_R15(filePath)
    k = data[data.shape[1] - 1].nunique()
    result, purity = run_kmeans_multiple_times(data, k)
    print("K-means of R15 is done !\n \tnumber of iteration: ", result[2])
    print("\tpurity: ",purity)
    result, purity = run_kmedian_multiple_times(data, k)
    print("K-median of R15 is done !\n \tnumber of iteration: ", result[2])
    print("\tpurity: ",purity)


    "-----------------glass.data-------------------------"
    filePath = os.path.join(__location__,"data", "glass.data")
    data = read_glass(filePath)
    k = data[data.shape[1] - 1].nunique()
    result, purity = run_kmeans_multiple_times(data, k)
    print("K-means of glass is done !\n \tnumber of iteration: ", result[2])
    print("\tpurity: ",purity)
    result, purity = run_kmedian_multiple_times(data, k)
    print("K-median of glass is done !\n \tnumber of iteration: ", result[2])
    print("\tpurity: ",purity)

    "-----------------iris.data-------------------------"
    filePath = os.path.join(__location__,"data", "iris.data")
    data = read_iris(filePath)
    k = data[data.shape[1] - 1].nunique()
    result, purity = run_kmeans_multiple_times(data, k)
    print("K-means of iris is done !\n \tnumber of iteration: ", result[2])
    print("\tpurity: ",purity)
    result, purity = run_kmedian_multiple_times(data, k)
    print("K-median of iris is done !\n \tnumber of iteration: ", result[2])
    print("\tpurity: ",purity)

    "********************************************************"


    "-------------------Home Work Part2---------------------"
    from src.algorithms import k_means
    import numpy as np
    import pandas as pd
    data = {
        0: np.array([1, 1.5, 3, 5, 3.5, 4.5, 3.5]),
        1: np.array([1, 2, 4, 7, 5, 5, 4.5])
    }
    centroids = {
            0: np.array([1, 7]),
            1: np.array([1, 5])
        }

    data = pd.DataFrame(data)
    centroids = pd.DataFrame(centroids)
    k = 2
    result= k_means(data, 2, centroids, max_iterations=2,lable = False)
    print("\nPart 2 of Home Work")
    print("result is:\n", result[0])









