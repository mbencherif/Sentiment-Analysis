

import pandas as pd
from bs4 import BeautifulSoup  
import re

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer


def pre_process(text):
    text=BeautifulSoup(text, features="html.parser").get_text()
    text=re.sub("[^a-zA-Z]"," ", text.lower())
    stemmer = PorterStemmer()
    stemmed_words = list(map(stemmer.stem, text.split()))
    lemmer = WordNetLemmatizer()
    stemmed_words = [lemmer.lemmatize(w) for w in stemmed_words]
    # text=[w for w in text.split() if not w in stop_w]
    text = [w for w in stemmed_words if not w in stop_w]
    text = " ".join(text)
    return text

stop_w= set(stopwords.words("english"))

if __name__ == '__main__':

    labeled_data = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    labeled_data['clean_review']=labeled_data['review'].apply(lambda r: pre_process(r))
    test_size=int(0.1*labeled_data.shape[0])
    train=labeled_data.iloc[test_size:,:]
    test=labeled_data.iloc[:test_size,:]
    vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None,max_features = 5000)
    train_data_features = vectorizer.fit_transform(list(train.clean_review)).toarray()
    test_data_features = vectorizer.transform(list(test.clean_review)).toarray()
    # vocab = vectorizer.get_feature_names()
    # print(vocab)
    # # Sum up the counts of each vocabulary word
    # dist = np.sum(train_data_features, axis=0)
    # # For each, print the vocabulary word and the number of times it
    # # appears in the training set
    # for tag, count in sorted(zip(vocab, dist), key=lambda x: x[1], reverse=True):
    #     print(count, tag)
    forest = RandomForestClassifier(n_estimators = 100) 
    forest = forest.fit( train_data_features, train["sentiment"] )
    values=forest.predict(test_data_features)
    test=test.assign(predict=pd.Series(values,index=test.index))

    
    compare=zip(list(test.sentiment),list(test.predict))
    correct=0
    total=0
    for (a,b) in compare:
        total+=1
        if a==b:
            correct+=1 
        else:
            pass
    accuracy=correct/total

    print("Using random forest, the accuracy is {}".format(accuracy))








