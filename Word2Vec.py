import gensim
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
import logging
from gensim.models import word2vec
from sklearn.cluster import KMeans
import time
from sklearn.ensemble import RandomForestClassifier


def read_data():
    train = pd.read_csv("datasets/labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
    test = pd.read_csv("datasets/testData.tsv", header=0, delimiter="\t", quoting=3)
    unlabeled_train = pd.read_csv("datasets/unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

    # Verify the number of reviews that were read (100,000 in total)
    print("Read %d labeled train reviews, %d labeled test reviews, " \
          "and %d unlabeled reviews\n" % (train["review"].size,
                                          test["review"].size, unlabeled_train["review"].size))
    return train, test, unlabeled_train


def prepare_sentences():
    train, test, unlabeled_train = read_data()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = []
    print("Parsing sentences from training set")
    for review in train["review"]:
        sentences += review_to_sentences(review, tokenizer)
    print("Parsing sentences from unlabeled set")
    for review in unlabeled_train["review"]:
        sentences += review_to_sentences(review, tokenizer)
    return sentences


def review_to_wordlist(review, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review, features="html.parser").get_text()
    #
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return words


# Define a function to split a review into parsed sentences
def review_to_sentences(review, tokenizer, remove_stopwords=False):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append(review_to_wordlist(raw_sentence, \
                                                remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences


def get_clean_reviews(num_features):
    train, test, unlabeled_train = read_data()
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_wordlist(review, \
                                                      remove_stopwords=True))
        clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_wordlist(review, \
                                                     remove_stopwords=True))
    return clean_train_reviews, clean_test_reviews


# Initialize and train the model (this will take some time)
def train_models():
    # Set values for various parameters
    num_features = 300  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    sentences = read_data()
    print("Training model...")
    model = word2vec.Word2Vec(sentences, workers=num_workers, \
                              size=num_features, min_count=min_word_count, \
                              window=context, sample=downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "300features_40minwords_10context"
    model.save(model_name)


def create_bag_of_centroids(wordlist, word_centroid_map):
    #
    # The number of clusters is equal to the highest cluster index
    # in the word / centroid map
    num_centroids = max(word_centroid_map.values()) + 1
    #
    # Pre-allocate the bag of centroids vector (for speed)
    bag_of_centroids = np.zeros(num_centroids, dtype="float32")
    #
    # Loop over the words in the review. If the word is in the vocabulary,
    # find which cluster it belongs to, and increment that cluster count
    # by one
    for word in wordlist:
        if word in word_centroid_map:
            index = word_centroid_map[word]
            bag_of_centroids[index] += 1
    #
    # Return the "bag of centroids"
    return bag_of_centroids


def prepare_bag_of_centroids():
    train, test, unlabeled_train = read_data()
    clean_train_reviews, clean_test_reviews = get_clean_reviews(num_features=300)
    # Pre-allocate an array for the training set bags of centroids (for speed)
    train_centroids = np.zeros((train["review"].size, num_clusters), \
                               dtype="float32")

    # Transform the training set reviews into bags of centroids
    counter = 0
    for review in clean_train_reviews:
        train_centroids[counter] = create_bag_of_centroids(review, \
                                                           word_centroid_map)
        counter += 1

    # Repeat for test reviews
    test_centroids = np.zeros((test["review"].size, num_clusters), \
                              dtype="float32")

    counter = 0
    for review in clean_test_reviews:
        test_centroids[counter] = create_bag_of_centroids(review, \
                                                          word_centroid_map)
        counter += 1
    return train_centroids, test_centroids


def predict():
    train, test, unlabeled_train = read_data()
    train_centroids, test_centroids = prepare_bag_of_centroids()
    # Fit a random forest and extract predictions
    forest = RandomForestClassifier(n_estimators=100)

    # Fitting the forest may take a few minutes
    print("Fitting a random forest to labeled training data...")
    forest = forest.fit(train_centroids, train["sentiment"])
    result = forest.predict(test_centroids)

    # Write the test results
    output = pd.DataFrame(data={"id": test["id"], "sentiment": result})
    output.to_csv("BagOfCentroids.csv", index=False, quoting=3)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', \
                        level=logging.INFO)
    # train_models()
    model = gensim.models.Word2Vec.load("300features_40minwords_10context")
    start = time.time()  # Start time
    # Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
    # average of 5 words per cluster
    word_vectors = model.wv.vectors
    num_clusters = int(word_vectors.shape[0] / 5)

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans(n_clusters=num_clusters)
    idx = kmeans_clustering.fit_predict(word_vectors)

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print("Time taken for K Means clustering: ", elapsed, "seconds.")
    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip(model.wv.index2word, idx))
    # For the first 10 clusters
    for cluster in range(0, 10):
        #
        # Print the cluster number
        print("\nCluster %d" % cluster)
        #
        # Find all of the words for that cluster number, and print them out
        words = []
        for i in range(0, len(word_centroid_map.values())):
            if list(word_centroid_map.values())[i] == cluster:
                words.append(list(word_centroid_map.keys())[i])
        print(words)
    predict()
