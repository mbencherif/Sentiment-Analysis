import re

from bs4 import BeautifulSoup
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords

from Models.Word2Vec import read_data
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer  # Importing Vectorizer


# TFIDF and SVM Classifier

def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review, "html.parser").get_text()  # remove html
    letters = re.sub("[^a-zA-Z0-9!?'-]", " ", review_text)  # passing only alphabets, numbers and some few punctuations
    lemma = WordNetLemmatizer()
    words_arr = [lemma.lemmatize(w) for w in word_tokenize(str(letters).lower())]  # Lammetize and tokenize
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words_arr if not w in stops]  # removing common english words
    return " ".join(meaningful_words)


def get_clean_reviews():
    train, test, unlabeled_train = read_data()
    clean_train_reviews = []
    for review in train["review"]:
        clean_train_reviews.append(review_to_words(review))
    clean_test_reviews = []
    for review in test["review"]:
        clean_test_reviews.append(review_to_words(review))
    return clean_train_reviews, clean_test_reviews


def predict():
    train, test, unlabeled_train = read_data()
    clean_train_reviews, clean_test_reviews = get_clean_reviews()
    vectorizer = CountVectorizer(analyzer='word', max_features=2500)

    train_data_features = vectorizer.fit_transform(clean_train_reviews)  # Vectorizing training Data
    train_data_features = train_data_features.toarray()

    test_data_features = vectorizer.transform(clean_test_reviews)  # Vectorize Test Data
    test_data_features = test_data_features.toarray()

    tfidf_transformer = TfidfTransformer().fit(train_data_features)  # TFIDF
    messages_tfidf = tfidf_transformer.transform(train_data_features)
    test_tfidf = tfidf_transformer.transform(test_data_features)

    linear_svc = LinearSVC()
    linear_svc.fit(messages_tfidf, train['sentiment'])  # SVM
    pred = linear_svc.predict(test_tfidf)

    acc_linear_svc = round(linear_svc.score(messages_tfidf, train['sentiment']) * 100, 2)
    print(acc_linear_svc)


if __name__ == '__main__':
    predict()
