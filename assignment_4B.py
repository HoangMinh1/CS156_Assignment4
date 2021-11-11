import sys
import glob
import string

# Step 1: Create a command argument that allows passing path of the directory as input
# example: python3 filename <inputPath>
inputPath = sys.argv[1]

posPath = inputPath + "/pos/"
negPath = inputPath + "/neg/"

posFiles = glob.glob(posPath + "*.txt")  # pattern matching files with .txt
negFiles = glob.glob(negPath + "*.txt")


# Step 2: Automatically read the whole file and convert them into a form that
# can be trainable (bag of words)
# ex: ["hello": 1, "my": 2 ...] where 1 & 2 is frequency of the word "hello" & "my"
posBagOfWords = []
negBagOfWords = []

# positive reviews

for filePath in posFiles:
    with open(filePath, 'r') as file:
        content = file.read().split()

        freq = {}
        for word in content:
            # to lower case
            lower_content = word.lower()
            # remove punctuation
            no_punct = lower_content.translate(
                str.maketrans('', '', string.punctuation))
            # remove '\n' from string
            no_newLine = no_punct.replace('\n', '')
            # count frequency of words
            count = freq.setdefault(no_newLine, 0)
            count += 1
            freq[no_newLine] += 1

        # delete empty string
        del freq['']
        # add dictionaries to list
        posBagOfWords.append(freq)

    file.close()


# negative reviews

for filePath in negFiles:
    with open(filePath, 'r') as file:
        content = file.read().split()

        freq = {}
        for word in content:
            # to lower case
            lower_content = word.lower()
            # remove punctuation
            no_punct = lower_content.translate(
                str.maketrans('', '', string.punctuation))
            # remove '\n' from string
            no_newLine = no_punct.replace('\n', '')
            # count frequency of words
            count = freq.setdefault(no_newLine, 0)
            count += 1
            freq[no_newLine] += 1

        # delete emtpy string
        del freq['']
        # add dictionaries to list
        negBagOfWords.append(freq)

    file.close()

print(posBagOfWords)
print(negBagOfWords)

totalPos = []
totalNeg = []
totalRev = []

# Step 3: Calculate features which has highest P(feature | pos) and P(feature | neg)
# Print out top 5 (Assignment 4B)

# Step 4: Selecting features
# calculate the usefuleness of each word using the formula on slide 18 p11-12
# Select the data which have the highest features


# Step 5: Write a Bayes classification
