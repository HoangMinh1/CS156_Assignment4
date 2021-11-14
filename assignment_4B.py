''' 
Assignment 4B for CS 156
@authors: Hoang Nguyen, David Truong, Brooke Dalton

To run program please input two arguments that are the positive and negative files
Ex: python3 assignment_4B.py data/pos/ data/neg/

The output should be a .txt file called "Output.txt" that contains the top 5 positive 
and top 5 negative evidences

'''

import sys
import glob
import string
import collections
import numpy as np
import random

# Step 1: Create a command argument that allows passing paths of the pos and neg directories as input
# example: python3 filename <posPath> <negPath>

posPath = sys.argv[1] # positive reviews path
negPath = sys.argv[2] # negative reviews path

posFiles = glob.glob(posPath + "*.txt")  # pattern matching files with .txt
negFiles = glob.glob(negPath + "*.txt")

ratio = 0.85 # ratio for spliting 85/15

trainingPos = random.sample(posFiles, int(ratio * len(posFiles))) # split into training data
trainingNeg = random.sample(negFiles, int(ratio * len(negFiles)))

testPost = list(set(posFiles) - set(trainingPos)) # get the differences btw posFile and training to make test data
testNeg = list(set(negFiles) - set(trainingNeg))


# Step 2: Automatically read the whole file and convert them into a form that
# can be trainable (bag of words)
# ex: ["hello": 1, "my": 2 ...] where 1 & 2 is frequency of the word "hello" & "my"

posBagOfWords = []
negBagOfWords = []

counterBag = collections.Counter({})  # used for calculation
posCounterBag = collections.Counter({})
negCounterBag = collections.Counter({})


for filePath in trainingPos:
    with open(filePath, 'r') as file:
        content = file.read()
        lower_content = content.lower()  # to lower case
        # remove punctuation
        no_punct = lower_content.translate(
            str.maketrans('', '', string.punctuation))
        # remove '\n' from string
        no_newLine = no_punct.replace('\n', '')
        # split the content into words
        final_content = no_newLine.split(' ')
        # remove empty string in the content
        final_content = list(filter(None, final_content))
        # remove duplicate
        final_content = list(set(final_content))
        # add final result
        posBagOfWords.append(final_content)
        # create a Counter
        counter = collections.Counter(final_content)
        # add to counterBag and posAllWords
        counterBag += counter
        posCounterBag += counter

    file.close()


for filePath in trainingNeg:
    with open(filePath, 'r') as file:
        content = file.read()
        lower_content = content.lower()  # to lower case
        # remove punctuation
        no_punct = lower_content.translate(
            str.maketrans('', '', string.punctuation))
        # remove '\n' from string
        no_newLine = no_punct.replace('\n', '')
        # split the content into words
        final_content = no_newLine.split(' ')
        # remove empty string in the content
        final_content = list(filter(None, final_content))
        # remove duplicate
        final_content = list(set(final_content))
        # add final result
        negBagOfWords.append(final_content)
        # create a Counter
        counter = collections.Counter(final_content)
        # add to counterBag and negAllWords
        counterBag += counter
        negCounterBag += counter

    file.close()

revAllWords = counterBag.keys()

# Step 3: Calculate features which has highest I(feature | pos) and I(feature | neg)
# Print out top 5 (Assignment 4B)

# combines all the dictionaries in the list
posAllWords = dict(posCounterBag)
negAllWords = dict(negCounterBag)

posLength = len(posBagOfWords)                        # should be 800
negLength = len(negBagOfWords)                        # should be 800
totalLength = len(posBagOfWords+negBagOfWords)        # should be 1600

listOfPosIndicator = {}
for i in revAllWords:
    if i in posAllWords:
        if i in negAllWords:
            negCount = negAllWords[i]
        else:
            negCount = 0
        posCount = posAllWords[i]
        totalCount = posCount + negCount
        posIndicator = np.log2(
            (posCount * totalLength) / (totalCount * posLength))
    else:
        posIndicator = 0
    if posCount != 1 and negCount != 0:
        listOfPosIndicator[i] = posIndicator

listOfNegIndicator = {}
for i in revAllWords:
    if i in negAllWords:
        if i in posAllWords:
            posCount = posAllWords[i]
        else:
            posCount = 0
        negCount = negAllWords[i]
        totalCount = posCount + negCount
        negIndicator = np.log2(
            (negCount * totalLength) / (totalCount * negLength))
    else:
        negIndicator = 0
    if negCount != 1 and posCount != 0:
        listOfNegIndicator[i] = negIndicator

#usefulnessIndicator = {}
# for i in revAllWords:
#useful = abs(listOfPosIndicator[i] - listOfNegIndicator[i])
#usefulnessIndicator[i] = useful

# print to text file
with open("Output.txt", "w") as file:
    most_pos = dict(collections.Counter(listOfPosIndicator).most_common(5))
    file.write("Top 5 Positive: \n")
    for i in most_pos:
        file.write(" {0} {1}\n".format(i, most_pos[i]))

    file.write("Top 5 Negative: \n")
    most_neg = dict(collections.Counter(listOfNegIndicator).most_common(5))
    for i in most_neg:
        file.write(" {0} {1}\n".format(i, most_neg[i]))
file.close()
