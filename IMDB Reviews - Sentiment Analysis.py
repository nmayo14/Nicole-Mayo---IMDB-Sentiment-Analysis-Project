# -*- coding: utf-8 -*-
"""
Created on August 12, 2018
Author: Nicole Mayo
"""
import numpy as np    # increases efficiency of matrix operations
import pandas as pd   # reads in data files of mixed data types
import re             # regular expressions to find/replace strings
import nltk           # natural language toolkit
from nltk.corpus import stopwords   # get list of stopwords to filter
                                    # out non-sentiment filler words
                                    
stop_words = set(stopwords.words('english')) # make the stopword list a set
                                             # to increase speed of comparisons


train = pd.read_csv("trainingData.txt", header=0, delimiter="\t", quoting=3)    
# read the training data stored in trainingData.txt
test = pd.read_csv("testData.txt", header=0, delimiter="\t", quoting=3)     
# read the test data stored in testData.txt
# note: data files are tab delimited
         

                                       
""" clean_my_text(): cleans the data with several replacements/deletions,
    tokenizes the text, and removes stopwords
    input: string data
    output: cleaned string data ready for sentiment analysis
"""
def clean_my_text(text):
    text = re.sub(r"<.*?>", "", text)      # quick removal of HTML tags
    text = re.sub("[^a-zA-Z]", " ", text)  # strip out all non-alpha chars
    text = text.strip().lower()            # convert all text to lowercase
    text = re.sub(" s ", " ", text)        # remove isolated s chars that 
                                           # result from cleaning possessives

    tokenizer = nltk.tokenize.TreebankWordTokenizer()  # tokenizes text using
                                                       # smart divisions
    tokens = tokenizer.tokenize(text)      # store results in tokens
    

    unstopped = []                         # holds the cleaned data string
    for word in tokens:
        if word not in stop_words:         # removes stopwords
            unstopped.append(word)         # adds word to unstopped string
            stemmer = nltk.stem.WordNetLemmatizer()   # consolidates different
                                                      # word forms
            cleanText = " ".join(stemmer.lemmatize(token) for token in unstopped)
                # joins final clean tokens into a string
    return cleanText



""" clean_my_data() calls clean_my_text for each line of text in a dataset
    category  
    input: data file containing raw text  
    output: data file containing cleaned text entries
"""
def clean_my_data(myDataSet):
    print("Cleaning all of the data")
    i = 0
    for textEntry in myDataSet.review:              # reads line of text under 
                                                    # review category
        cleanElement = clean_my_text(textEntry)     # cleans line of text
        myDataSet.loc[i, "review"] = cleanElement   # stores cleaned text
                                                    # in original data file
        i = i + 1
        if (i%50 == 0):
            print("Cleaning review number", i, "out of", len(myDataSet.index))
    print("Finished cleaning all of the data\n")
    return myDataSet


print("Operating on training data...\n")
cleanTrainingData = clean_my_data(train)            # cleans the training data



""" create_bag_of_words() generates the bag of words used to evaluate sentiment
    input: cleaned dataset
    output: tf-idf weighted sparse matrix
"""
def create_bag_of_words(X):
    from sklearn.feature_extraction.text import CountVectorizer
        # use scikit-learn for vectorization
    
    print ('Generating bag of words...')
    
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 ngram_range = (1,2), \
                                 max_features = 10000)
        # generates vectorization for ngrams of up to 2 words in length
        # this will greatly increase feature size, but gives more accurate
        # sentiment analysis since some word combinations have large
        # impact on sentiment ie: ("not good", "very fast")
                                                         
    train_data_features = vectorizer.fit_transform(X)
        # vectorizes sparse matrix
    
    train_data_features = train_data_features.toarray()
        # convert to a NumPy array for efficient matrix operations
    from sklearn.feature_extraction.text import TfidfTransformer
    tfidf = TfidfTransformer()
    tfidf_features = tfidf.fit_transform(train_data_features).toarray()
        # use tf-idf to weight features - places highest sentiment value on
        # low-frequency ngrams that are not too uncommon 
    return vectorizer, tfidf_features, tfidf



vectorizer, tfidf_features, tfidf  = (create_bag_of_words(cleanTrainingData["review"]))   
        # stores the sparse matrix of the tf-idf weighted features



""" train_logistic_regression() uses logistic regression model to
    evaluate sentiment
    options: C sets how strong regularization will be: large C = small amount
    input: tf-idf matrix and the sentiment attached to the training example
    output: the trained logistic regression model
"""
def train_logistic_regression(features, label):
    print ("Training the logistic regression model...")
    from sklearn.linear_model import LogisticRegression
    ml_model = LogisticRegression(C = 100, random_state = 0)
    ml_model.fit(features, label)
    print ('Finished training the model\n')
    return ml_model



ml_model = train_logistic_regression(tfidf_features, cleanTrainingData["sentiment"])
    # holds the trained model
    
print("Operating on test data...\n")
cleanTestData = clean_my_data(test)   
    # cleans the test data for accuracy evaluation

test_data_features = vectorizer.transform(cleanTestData["review"])
test_data_features = test_data_features.toarray()
    # vectorizes the test data

test_data_tfidf_features = tfidf.fit_transform(test_data_features)
test_data_tfidf_features = test_data_tfidf_features.toarray()
    # tf-idf of test data ngrams

predicted_y = ml_model.predict(test_data_tfidf_features)
    # uses the trained logistic regression model to assign sentiment to each
    # test data example

correctly_identified_y = predicted_y == cleanTestData["sentiment"]
accuracy = np.mean(correctly_identified_y) * 100
print ('The accuracy of the model in predicting movie review sentiment is %.0f%%' %accuracy)
    # compares the predicted sentiment (predicted_y) vs the actual 
    # value stored in "sentiment"