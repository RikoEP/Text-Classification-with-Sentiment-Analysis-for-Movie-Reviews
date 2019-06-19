import random
import os

from nltk import word_tokenize as wt
from nltk.tag import pos_tag as pt
from nltk.probability import FreqDist
from nltk.classify import NaiveBayesClassifier as naive
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC, NuSVC

import pickle


def find_features_train(document):
    words = wt(document)
    features = {}
    for w in word_features_train:
        features[w] = (w in words)

    return features


def find_features_test(document):
    words = wt(document)
    features = {}
    for w in word_features_test:
        features[w] = (w in words)

    return features


# Training Dataset
pos_review_path_train = 'IMDB_movie_reviews/train/pos'
neg_review_path_train = 'IMDB_movie_reviews/train/neg'

pos_files_train = os.listdir(pos_review_path_train)
neg_files_train = os.listdir(neg_review_path_train)

all_words_train = []
documents_train = []

# Test Dataset
pos_review_path_test = 'IMDB_movie_reviews/test/pos'
neg_review_path_test = 'IMDB_movie_reviews/test/neg'

pos_files_test = os.listdir(pos_review_path_test)
neg_files_test = os.listdir(neg_review_path_test)

all_words_test = []
documents_test = []

allowed_word_types = ['J']

# Train Documents
# Positive
for review in pos_files_train:
    open_file = open('IMDB_movie_reviews/train/pos' + '/' + review, encoding='utf8').read()
    documents_train.append((open_file, 'pos'))
    words = wt(open_file)
    pos_tag = pt(words)
    for w in pos_tag:
        if w[1][0] in allowed_word_types:
            all_words_train.append(w[0].lower())

# Negative
for review in neg_files_train:
    open_file = open('IMDB_movie_reviews/train/neg' + '/' + review, encoding='utf8').read()
    documents_train.append((open_file, 'neg'))
    words = wt(open_file)
    pos_tag = pt(words)
    for w in pos_tag:
        if w[1][0] in allowed_word_types:
            all_words_train.append(w[0].lower())

save_documents_train = open('pickled_algos_train/documents_train.pickle', 'wb')
pickle.dump(documents_train, save_documents_train)
save_documents_train.close()

# Test Documents
# Positive
for review in pos_files_test:
    open_file = open('IMDB_movie_reviews/test/pos' + '/' + review, encoding='utf8').read()
    documents_test.append((open_file, 'pos'))
    words = wt(open_file)
    pos_tag = pt(words)
    for w in pos_tag:
        if w[1][0] in allowed_word_types:
            all_words_test.append(w[0].lower())

# Negative
for review in neg_files_test:
    open_file = open('IMDB_movie_reviews/test/neg' + '/' + review, encoding='utf8').read()
    documents_test.append((open_file, 'neg'))
    words = wt(open_file)
    pos_tag = pt(words)
    for w in pos_tag:
        if w[1][0] in allowed_word_types:
            all_words_test.append(w[0].lower())

save_documents_test = open('pickled_algos_test/documents_test.pickle', 'wb')
pickle.dump(documents_test, save_documents_test)
save_documents_test.close()

# Train Word Features
all_words_train = FreqDist(all_words_train)
word_features_train = list(all_words_train.keys())[:5000]

save_word_features_train = open('pickled_algos_train/word_features_train.pickle', 'wb')
pickle.dump(word_features_train, save_word_features_train)
save_word_features_train.close()

# Test Word Features
all_words_test = FreqDist(all_words_test)
word_features_test = list(all_words_test.keys())[:5000]

save_word_features_test = open('pickled_algos_test/word_features_test.pickle', 'wb')
pickle.dump(word_features_test, save_word_features_test)
save_word_features_test.close()

# Train Feature Sets
featureSets_train = [(find_features_train(rev), category) for (rev, category) in documents_train]

save_featureSets_train = open('pickled_algos_train/featureSets_train.pickle', 'wb')
pickle.dump(featureSets_train, save_featureSets_train)
save_featureSets_train.close()

# Test Feature Sets
featureSets_test = [(find_features_test(rev), category) for (rev, category) in documents_test]

save_featureSets_test = open('pickled_algos_test/featureSets_test.pickle', 'wb')
pickle.dump(featureSets_test, save_featureSets_test)
save_featureSets_test.close()

# Training Set
random.shuffle(featureSets_train)
training_set = featureSets_train[:17500]

# Original Naive Bayes
classifier = naive.train(training_set)

save_classifier = open('pickled_algos_train/originalNaiveBayes.pickle', 'wb')
pickle.dump(classifier, save_classifier)
save_classifier.close()

# MultinomialNB Classifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

save_classifier = open('pickled_algos_train/MNB_classifier.pickle', 'wb')
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

# BernoulliNB Classifier
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)

save_classifier = open('pickled_algos_train/BernoulliNB_classifier.pickle', 'wb')
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

# LogisticRegression Classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)

save_classifier = open('pickled_algos_train/LogisticRegression_classifier.pickle', 'wb')
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

# SGD Classifier
SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)

save_classifier = open('pickled_algos_train/SGDClassifier_classifier.pickle', 'wb')
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()

# LinearSVC Classifier
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

save_classifier = open('pickled_algos_train/LinearSVC_classifier.pickle', 'wb')
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

# NuSVC Classifier
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

save_classifier = open('pickled_algos_train/NuSVC_classifier.pickle', 'wb')
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

print("Training done")
