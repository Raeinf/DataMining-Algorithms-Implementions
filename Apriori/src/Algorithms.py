from itertools import combinations
import collections


"A Garbage fast Apriori code"


# this will create CK+1 out of Lk
def generateNewCandidate(LK):
    newLK = set()
    tmpLK = set(LK.keys())
    for item in LK:
        if item not in tmpLK:
            continue
        shouldRemove = set()
        *firstItems, lastItem = item
        tmpLK.remove(item)
        matched = {lastItem}
        for item2 in tmpLK:
            (
                *firstItems2,
                lastItem2,
            ) = item2

            if firstItems == firstItems2:
                matched.add(lastItem2)
                shouldRemove.add(item2)

        if len(matched) != 0:
            firstItems = tuple(firstItems)
            matched = sorted(matched)
            for a, b in combinations(
                matched, 2
            ):  # This was sorted befor but not any more !
                # result = tuple(sorted(firstItems + (a,) + (b,)))
                result = firstItems + (a,) + (b,)
                newLK.add(result)

        for item in shouldRemove:
            tmpLK.remove(item)

        # an alternative
        # tmpLK.difference(shouldRemove)

    return newLK


# This will Break Down a Ck to its subset for see if it subset are frequent or not ! and will prune them
def prune(CK, LK):
    result = set()
    for item in CK:
        flag = True
        for i in range(len(item)):
            if (item[:i] + item[i + 1 :]) not in LK:
                flag = False
                break
        if flag:
            result.add(item)

    return result


# filter will cound the support and will create Lk out of Ck
def filterr(ck, itemIndices, minsup):
    LK = dict()
    for item in ck:
        result = itemIndices[item[0]]
        for i in range(1, len(item)):
            result = result.intersection(itemIndices[item[i]])
        if len(result) >= minsup:
            LK[item] = len(result)
    return LK


# ..............................................................
def apriori(dataset, minsup=300):
    result = dict()
    itemIndices = collections.defaultdict(set)
    # finding the transaction indices contain every item --> something like eclat thing to make algorithm faster
    for i, transaction in enumerate(dataset):
        for item in transaction:
            itemIndices[item].add(i)

    # ..........................................................

    # Counting the support for every item and adding L1 to results
    itemIndices = {
        key: value for key, value in itemIndices.items() if len(value) >= minsup
    }
    result[1] = {(key,): len(value) for key, value in itemIndices.items()}

    # Creating C2 and filter them
    CK = sorted(combinations(itemIndices.keys(), 2))
    LK = dict()
    for item in CK:
        intersection = set.intersection(itemIndices[item[0]], itemIndices[item[1]])
        if len(intersection) >= minsup:
            LK[tuple(item)] = len(intersection)
    result[2] = LK
    # ..........................................................

    # run algorithm to run till finish from L2 to end
    count = 3
    while len(LK) != 0:
        CK = generateNewCandidate(LK)
        CK = prune(CK, LK)
        LK = filterr(CK, itemIndices, minsup)
        result[count] = LK
        count += 1

    return result


# .........................................................................


# This function will simply return the frequency of item
def find_count(item, frequentItemset):
    return frequentItemset[len(item)][item]


def associationRule(frequentItemset, minconf=0.2):
    result = set()
    for item in frequentItemset:
        for frequnt in frequentItemset[item]:
            freqCount = find_count(frequnt, frequentItemset)
            for i in range(1, len(frequnt)):
                item1 = set(combinations(frequnt, i))
                for item2 in item1:
                    freqCopy = set(frequnt)
                    for item3 in item2:
                        freqCopy.remove(item3)
                    freqCopy = tuple(freqCopy)
                    if freqCount / find_count(item2, frequentItemset) >= minconf:
                        result.add((item2, freqCopy))
    return result
