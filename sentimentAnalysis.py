import random

from nltk import word_tokenize as wt
from nltk.classify import ClassifierI
from statistics import mode

import pickle


def find_features(document):
    words = wt(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features


def sentiment(text):
    features = find_features(text)

    return voted_classifier.classify(features), voted_classifier.confidence(features)


class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)

        return conf


# Documents
documents_load = open('pickled_algos_train/documents_train.pickle', 'rb')
documents = pickle.load(documents_load)
documents_load.close()

# Word Features
word_features_load = open('pickled_algos_train/word_features_train.pickle', 'rb')
word_features = pickle.load(word_features_load)
word_features_load.close()

# Feature Sets
featureSets_load = open('pickled_algos_train/featureSets_train.pickle', 'rb')
featureSets = pickle.load(featureSets_load)
featureSets_load.close()

# Training Set
random.shuffle(featureSets)
training_set = featureSets[:17500]

# Original Naive Bayes
classifier_load = open('pickled_algos_train/originalNaiveBayes.pickle', 'rb')
classifier = pickle.load(classifier_load)
classifier_load.close()

# MultinomialNB Classifier
MNB_classifier_load = open('pickled_algos_train/MNB_classifier.pickle', 'rb')
MNB_classifier = pickle.load(MNB_classifier_load)
MNB_classifier_load.close()

# BernoulliNB Classifier
BernoulliNB_classifier_load = open('pickled_algos_train/BernoulliNB_classifier.pickle', 'rb')
BernoulliNB_classifier = pickle.load(BernoulliNB_classifier_load)
BernoulliNB_classifier_load.close()

# LogisticRegression Classifier
LogisticRegression_classifier_load = open('pickled_algos_train/LogisticRegression_classifier.pickle', 'rb')
LogisticRegression_classifier = pickle.load(LogisticRegression_classifier_load)
LogisticRegression_classifier_load.close()

# SGD Classifier
SGDClassifier_classifier_load = open('pickled_algos_train/SGDClassifier_classifier.pickle', 'rb')
SGDClassifier_classifier = pickle.load(SGDClassifier_classifier_load)
SGDClassifier_classifier_load.close()

# LinearSVC Classifier
LinearSVC_classifier_load = open('pickled_algos_train/LinearSVC_classifier.pickle', 'rb')
LinearSVC_classifier = pickle.load(LinearSVC_classifier_load)
LinearSVC_classifier_load.close()

# NuSVC Classifier
NuSVC_classifier_load = open('pickled_algos_train/NuSVC_classifier.pickle', 'rb')
NuSVC_classifier = pickle.load(NuSVC_classifier_load)
NuSVC_classifier_load.close()

voted_classifier = VoteClassifier(classifier,
                                  MNB_classifier,
                                  BernoulliNB_classifier,
                                  LogisticRegression_classifier,
                                  SGDClassifier_classifier,
                                  LinearSVC_classifier,
                                  NuSVC_classifier)
