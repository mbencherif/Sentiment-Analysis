import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
import nltk
from collections import Counter
import sys

# the method that read the review and remove non-character and stopwords
def process_review(review):
    review_text = BeautifulSoup(review).get_text()
    letters_only_review = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only_review.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return (" ".join(meaningful_words))

# read each review, "clean" the review, and append the review in the original file
# with a column called "clean_review"
def process_file(fileName):
    labeled_data = pd.read_csv(fileName, header=0, delimiter="\t", quoting=3)
    clean_review=[]
    nums = labeled_data.shape[0]
    for i in range(0, nums):
        clean_review.append(process_review(labeled_data["review"][i]))
    labeled_data = labeled_data.assign(clean_review=clean_review)
    return labeled_data

#################################################
# the method that calculate the probability of prediction based on the word in the review sentence
def predict_by_calculate(test, train):
    nums = train.shape[0]
    positive = 0
    negative = 0
    dict_positive={}
    dict_negative={}
    for i in range(0, nums):
        words = train["clean_review"][i].split()
        if(train["sentiment"][i]==1):
            positive+=1
            for word in words:
                add_dict(word, dict_positive)
        else:
            negative+=1
            for word in words:
                add_dict(word, dict_negative)

    positive_prob = positive/nums
    negative_prob = negative/nums
    total_positive_words = 0
    total_negative_words = 0
    for word in dict_positive.keys():
        total_positive_words+=dict_positive[word]
    for word in dict_negative.keys():
        total_negative_words+=dict_negative[word]

    predicts=[]
    test_nums = test.shape[0]

    for i in range(0, test_nums):
        text = list(test["clean_review"])[i].split()
        positive_prediction = make_class_prediction(text, dict_positive, positive_prob, total_positive_words)
        negative_prediction = make_class_prediction(text, dict_negative, negative_prob, total_negative_words)
        if positive_prediction>negative_prediction:
            predicts.append(1)
        else:
            predicts.append(0)
    return test.assign(predict=predicts)

def make_class_prediction(text, dict, class_prob,total_class_words):
    prediction = class_prob
    text_counts = Counter(text)
    for word in text_counts:
        if word not in dict:
            count=1
        else:
            count = dict[word]+1
        prediction *= (text_counts[word]*count / total_class_words)
  # Now we multiply by the probability of the class existing in the documents.
    return prediction


# help method that add an word to an dictionary
def add_dict(word, dict):
    if word not in dict:
        dict[word]=1
    else:
        dict[word]+=1

#########################################
# this method do the sentiment prediction using nltk package

def total_words_sets(train):
    nums = train.shape[0]
    total_words = []
    for i in range(0, nums):
        words = train["clean_review"][i].split()
        for word in words:
            total_words.append(word)
    return [ key for (key, count) in Counter(total_words).most_common(5000)]

# dict_total store all the words exit in the train set and their frequency.
# input review is a list of word in the cleaned review
# the returned dict contains all the keys in the input dict_total, and calculate the word frequency in the review list
def wsd_features(review, total_words_sets):
    dict={}
    for word in total_words_sets:
        dict[word]=0
    for w in review:
        if w in dict:
            dict[w]+=1
    return dict

# create_feature_set using train set
def create_feature_set(train,total_words_sets):
    feature_set = []
    nums = train.shape[0]
    for index in range(0, nums):
        words = list(train["clean_review"])[index].split()
        curr_featuresets=[(wsd_features(words, total_words_sets), list(train["sentiment"])[index])]
        feature_set+=curr_featuresets
    return feature_set


# train classifier using training set
def train_classifier(training_set):
    # create the classifier
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    return classifier

# evaluate classifer using test_set
def evaluate_classifier(classifier, test_set):
    # get the accuracy and print it
    print(nltk.classify.accuracy(classifier, test_set))

# run classifier and return the result
def run_classifier(classifier, test, total_words_sets):
    result = []
    for review in test["clean_review"]:
        label = classifier.classify(wsd_features(review.split(), total_words_sets))
        result.append(label)
    return test.assign(predict=result)

################################################
# this method using the python machine learning library Scikit-learn
# the counts of text was generated using a vectorizer
def scikit_learn(test, train):
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    train_data_features = vectorizer.fit_transform([s for s in train["clean_review"]])
    train_data_features = train_data_features.toarray()
    test_data_features = vectorizer.transform(s for s in test["clean_review"]).toarray()

    # try different naive bayes model to train the data and do the prediction
    # inclulding multinomialNB, BernoulliNB, and GaussianNB
    nb1 = MultinomialNB()
    nb1.fit(train_data_features,train["sentiment"])
    multinomial_predicts = nb1.predict(test_data_features)
    multinomial_result = test.assign(predict=multinomial_predicts)

    nb2 = BernoulliNB()
    nb2.fit(train_data_features,train["sentiment"])
    bernoulli_predicts = nb2.predict(test_data_features)
    bernoulli_result = test.assign(predict=bernoulli_predicts)

    nb3 = GaussianNB()
    nb3.fit(train_data_features, train["sentiment"])
    gaussian_predicts = nb3.predict(test_data_features)
    gaussian_result = test.assign(predict=gaussian_predicts)
    return multinomial_result,bernoulli_result,gaussian_result


# calculate the accuracy of the prediction
def calculate_accuracy(result, type):
    correct = 0
    total = 0
    for (a, b) in zip(list(result.predict), list(result.sentiment)):
        if(a==b):
            correct+=1
        total+=1
    accuracy = correct/total
    print("Using, {}, the accuracy is {}".format(type, accuracy))

def analyze_result(file1, file2, file3, file4, file5):
    predict1 = pd.read_csv(file1)
    predict2 = pd.read_csv(file2)
    predict3 = pd.read_csv(file3)
    predict4 = pd.read_csv(file4)
    predict5 = pd.read_csv(file5)

    result=[0,0,0,0,0,0]
    for (a,b,c,d,e) in zip(list(predict1.sentiment), list(predict2.sentiment), list(predict3.sentiment), list(predict4.sentiment), list(predict5.sentiment)):
        counter = Counter([a,b,c,d,e])
        result[counter[0]]+=1
    print ("There are {} reviews are identical positive". format(result[0]))
    for i in range(1,5):
        print ("There are {} reviews are identified as positive in {} of the 5 algorithms".format(result[i], 5-i))
    print ("There are {} reviews are identical negative". format(result[5]))


if __name__ == '__main__':
    choice = input("Please choose the action from the following options or enter 'q' to quit: \n"
                   "1. Accuracy of Mathmatic Naive Bayes\n"
                   "2. Accuracy of NLTK Naive Bayes\n"
                   "3. Accuracy of Scikit-learn Naive Bayes\n"
                   "4. Predcition of Mathmatic Naive Bayes\n"
                   "5. Predcition of NLTK Naive Bayes\n"
                   "6. Predcition of Scikit-learn Naive Bayes\n"
                   "7. Analyze prediction result\n")


    while choice != 'q':
        choice = int(choice)
        if choice in [1,2,3]:
            print("Processing files...")
            labeledData = process_file("labeledTrainData.tsv")
            nums = labeledData.shape[0]
            devide = int(0.9 * nums)
            test = labeledData.iloc[devide:, :]
            train = labeledData.iloc[:devide, :]
            if choice == 1:
                print("Calculating accuracy...")
                result = predict_by_calculate(test, train)
                calculate_accuracy(result, "self_calculate naive bayes")
            elif choice == 2:
                print("Calculating accuracy...")
                print("This process takes long time, please be patient...")
                total_words = total_words_sets(train)
                training_set = create_feature_set(train, total_words)
                test_set = create_feature_set(test, total_words)
                classifier = train_classifier(training_set)
                evaluate_classifier(classifier, test_set)
                result5 = run_classifier(classifier, test, total_words)
                calculate_accuracy(result5, "nltk naive bayes")
            elif choice == 3:
                print("Calculating accuracy...")
                multinomial_result, bernoulli_result, gaussian_result = scikit_learn(test, train)
                calculate_accuracy(multinomial_result, "multinomial naive bayes")
                calculate_accuracy(bernoulli_result, "bernoulli naive bayes")
                calculate_accuracy(gaussian_result, "gaussian naive bayes")

        elif choice in [4,5,6]:
            print("Processing files...")
            labeledData = process_file("labeledTrainData.tsv")
            train = process_file("labeledTrainData.tsv")
            test = process_file("testData.tsv")

            if choice==4:
                print("Printing out file \"Bag_of_Words_calculate.csv\"...")
                result = predict_by_calculate(test, train)
                output1 = pd.DataFrame(data={"id": result["id"], "sentiment": result["predict"]})
                output1.to_csv("Bag_of_Words_calculate.csv", index=False, quoting=3)

            elif choice==5:
                print("Processing...")
                print("This process takes long time, please be patient!")
                total_words = total_words_sets(train)
                training_set = create_feature_set(train, total_words)
                classifier = train_classifier(training_set)
                result5 = run_classifier(classifier, test, total_words)
                print("Printing out file \"Bag_of_Words_nltk.csv\"...")
                output5 = pd.DataFrame(data={"id": result5["id"], "sentiment": result5["predict"]})
                output5.to_csv("Bag_of_Words_nltk.csv", index=False, quoting=3)

            elif choice==6:
                print("Processing...")
                multinomial_result, bernoulli_result, gaussian_result = scikit_learn(test, train)
                print("Printing out file \"Bag_of_Words_multinomial.csv\"...")
                output2 = pd.DataFrame(data={"id": multinomial_result["id"], "sentiment": multinomial_result["predict"]})
                output2.to_csv("Bag_of_Words_multinomial.csv", index=False, quoting=3)
                print("Printing out file \"Bag_of_Words_bernoulli.csv\"...")
                output3 = pd.DataFrame(data={"id": bernoulli_result["id"], "sentiment": bernoulli_result["predict"]})
                output3.to_csv("Bag_of_Words_bernoulli.csv", index=False, quoting=3)
                print("Printing out file \"Bag_of_Words_gaussian.csv\"...")
                output4 = pd.DataFrame(data={"id": gaussian_result["id"], "sentiment": gaussian_result["predict"]})
                output4.to_csv("Bag_of_Words_gaussian.csv", index=False, quoting=3)

        elif choice ==7:
            analyze_result("Bag_of_Words_calculate.csv","Bag_of_Words_nltk.csv","Bag_of_Words_multinomial.csv","Bag_of_Words_bernoulli.csv", "Bag_of_Words_gaussian.csv")

        print("Please choose from 1 to 6 or enter 'q' to quit!")
        choice = input("Please choose the action from the following options: \n"
                       "1. Accuracy of Mathmatic Naive Bayes\n"
                       "2. Accuracy of NLTK Naive Bayes\n"
                       "3. Accuracy of Scikit-learn Naive Bayes\n"
                       "4. Predcition of Mathmatic Naive Bayes\n"
                       "5. Predcition of NLTK Naive Bayes\n"
                       "6. Predcition of Scikit-learn Naive Bayes\n"
                       "7. Analyze prediction result\n")




