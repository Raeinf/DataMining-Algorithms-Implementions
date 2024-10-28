from src.dataReader import dataReader
from src.Algorithms import apriori
from src.Algorithms import associationRule
import os
import sys

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


"********** A Garbage fast Apriori code  *****************"


def print_result(result):

    print("frequent itemsets are:")
    for i in result:
        for key, value in sorted(result[i].items()):
            print(key, value, sep=":")


def print_result2(result2):
    print("Association rules are:")
    for item in sorted(result2):
        print(item[0], "-->", item[1])


if __name__ == "__main__":
    filePath = os.path.join(__location__, "data", "retail.dat")
    data = dataReader(filePath)
    print("* This is fast implimentation of aipriori algorithm *")
    if len(sys.argv) == 3:
        minsup = int(sys.argv[1])
        minconf = float(sys.argv[2])
        print("The minsup is: ", minsup)
        print("The algorithm is running....")
        result1 = apriori(data, minsup)
        result2 = associationRule(result1, minconf)
        print_result(result1)
        print_result2(result2)

    elif len(sys.argv) == 1:
        print("The algorithm will use default minsup(300)")
        print("the algorithm is running....")
        result1 = apriori(data, 300)
        result2 = associationRule(result1, 0.2)
        print_result(result1)
        print_result2(result2)
    else:
        print("!!!!!!!!!!Wrong Usage of algorithm Please read Readme.txt!!!!!!!!!")
