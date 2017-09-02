'''
Model for Words:

Let's say global dictionary we make ends up being:
[chair, table, spoon, television]

We get sentence:
"I pulled the chair up to the table"

So converted to a vector that'd be:
[1, 1, 0, 0]
'''

import nltk
from nltk.tokenize import word_tokenize #splits up a sentence into a vector of words
from nltk.stem import WordNetLemmatizer #removes ings, ed, etc etc
#stemming a word may not produce a valid word --> running to runn
#lemmatizing will make a valid word --> running to run
import numpy as np 
import random
import pickle
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000 #if you get MemoryError, you ran out of ram

def create_lexicon(pos, neg): #lexicon is list of words you know
    lexicon = []
    for fi in [pos, neg]:
        with open(fi, 'r') as f: #opening file with the intention to read
            contents = f.readlines() 
            for l in contents[:hm_lines]: #for each line up to how many lines we're reading
                all_words = word_tokenize(l.lower())
                lexicon += list(all_words) #add words to lexicon list
    
    lexicon = [lemmatizer.lemmatize(i) for i in lexicon] #lemmatizes all of them
    w_counts = Counter(lexicon) #Counter basically does this: w_counts {'the':51321, 'and':2343} 

    l2 = []
    for w in w_counts:
        if 5000 > w_counts[w] > 50: #gets rid of words like "the" and also rare words
            l2.append(w)
    print('Length of L2:', len(l2))
    return l2

def sample_handling(sample, lexicon, classification):
    feature_set = [] #going to be list of lists:
    '''
    [
        [ [0 1 0 1 1 0], [1 0] ], 
        #first list is list for the sentence (features)
        #second one says if positive or negative (label) ([1 0] = positive, [0 1] = negative)
        []
    ]
    '''
    with open(sample, 'r') as f:
        contents = f.readlines()
        for l in contents[:hm_lines]:
            current_words = word_tokenize(l.lower())
            current_words = [lemmatizer.lemmatize(i) for i in current_words]
            features = np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value = lexicon.index(word.lower()) #search for word.lower, and find index
                    features[index_value] += 1 #if it was a paragraph, += makes sense. small sentence, so probably doesnt matter
            
            features = list(features)
            feature_set.append([features, classification])

    return feature_set

def create_feature_sets_and_labels(pos, neg, test_size=0.1):
    lexicon = create_lexicon(pos, neg)
    features = []
    features += sample_handling(pos, lexicon, [1, 0]) #last parameter defines positive as [1, 0]
    features += sample_handling(neg, lexicon, [0, 1])
    random.shuffle(features) #important to shuffle for neural network

    testing_size = int(test_size*len(features))
    features = np.array(features)

    train_x = list(features[:,0][:-testing_size]) 
    train_y = list(features[:,1][:-testing_size]) 
    #this gets all features: [[5, 8], [7, 9]] --> [:,0] means you want [5, 7] (all 0th elements)
    #[:-test_size] --> gets all of them until the last 10% (to save up testing size)

    test_x = list(features[:,0][-testing_size:]) 
    test_y = list(features[:,1][-testing_size:]) #getlast 10%

    return train_x, train_y, test_x, test_y

if __name__ == '__main__': #allows us to call this script by itself
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels('positive_examples.txt', 'negative_examples.txt')
    with open('sentiment_set.pickle', 'wb') as f: #write to file
        pickle.dump([train_x, train_y, test_x, test_y], f)