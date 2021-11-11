import sys
import glob
import string
import collections

# Step 1: Create a command argument that allows passing path of the directory as input
# example: python3 filename <inputPath>
inputPath = sys.argv[1]

posPath = inputPath + "/pos/"
negPath = inputPath + "/neg/"

posFiles = glob.glob(posPath + "*.txt") # pattern matching files with .txt 
negFiles = glob.glob(negPath + "*.txt")


# Step 2: Automatically read the whole file and convert them into a form that
# can be trainable (bag of words)
# ex: ["hello": 1, "my": 2 ...] where 1 & 2 is frequency of the word "hello" & "my"

posBagOfWords = []
negBagOfWords = []

counterBag = collections.Counter({}) # used for calculation
posCounterBag = collections.Counter({})
negCounterBag = collections.Counter({})


for filePath in posFiles:
    with open(filePath, 'r') as file:
        content = file.read()
        lower_content = content.lower() # to lower case
        # remove punctuation
        no_punct = lower_content.translate(str.maketrans('','',string.punctuation))
        # remove '\n' from string
        no_newLine = no_punct.replace('\n', '');
        # split the content into words
        final_content = no_newLine.split(' ')
        # remove empty string in the content
        final_content = list(filter(None, final_content))
        # add final result
        posBagOfWords.append(final_content)
        # create a Counter
        counter = collections.Counter(final_content)
        # add to counterBag and posAllWords
        counterBag += counter
        posCounterBag += counter
        
    file.close()




for filePath in negFiles:
    with open(filePath, 'r') as file:
        content = file.read()
        lower_content = content.lower() # to lower case
        # remove punctuation
        no_punct = lower_content.translate(str.maketrans('','',string.punctuation))
        # remove '\n' from string
        no_newLine = no_punct.replace('\n', '');
        # split the content into words
        final_content = no_newLine.split(' ')
        # remove empty string in the content
        final_content = list(filter(None, final_content))
        # add final result
        negBagOfWords.append(final_content)
        # create a Counter
        counter = collections.Counter(final_content)
        # add to counterBag and negAllWords
        counterBag += counter
        negCounterBag += counter

    file.close()
    
revAllWords = counterBag.keys()

#print(posBagOfWords)
#print(negBagOfWords)
#print(counterBag)
print("PosCounterBag is :", posCounterBag)
print("NegCounterBag is :", negCounterBag)
print(revAllWords)





# Step 3: Calculate features which has highest P(feature | pos) and P(feature | neg)
# Print out top 5 (Assignment 4B)

# Step 4: Selecting features
# calculate the usefuleness of each word using the formula on slide 18 p11-12
# Select the data which have the highest features


# Step 5: Write a Bayes classification
