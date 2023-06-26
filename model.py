import nltk
import re
from collections import defaultdict
from nltk import NaiveBayesClassifier


class GamePredictor(object):
    def __init__(self):
        self.game_keywords = []
        self.classifier = None

    def append_rel_data(self, game_keywords):
        self.game_keywords = game_keywords

    def clean_text(self, text):
        # Remove emojis
        text = re.sub(r'\\u\w+', '', text.encode('unicode-escape').decode('utf-8'))
        # Remove special characters and symbols
        text = re.sub(r'[^\w\s]', '', text)
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\b\w{25,}\b', '', text)
        return text

    def extract_relevant_data(self, chat_logs):
        relevant_data = []
        for log in chat_logs:
            clean_log = self.clean_text(log)
            for keyword in self.game_keywords:
                if keyword.lower() in clean_log.lower():
                    relevant_data.append(clean_log)
                    break
        return relevant_data

    def create_game_keywords(self, logs):
        cleaned_words = []
        for log in logs:
            words = nltk.tokenize.word_tokenize(self.clean_text(log))
            if words:
                cleaned_words += words
        freq_dist = nltk.FreqDist(cleaned_words)
        self.game_keywords = [name for (name, count) in freq_dist.most_common(1000)]

    def create_feature_set(self, chat_logs):
        tokens = []
        relevant_data = self.extract_relevant_data(chat_logs)
        for log in relevant_data:
            words = nltk.word_tokenize(log)
            tokens.extend(words)
        feature_set = nltk.FreqDist(tokens)
        return feature_set

    def train_classifier(self, training_data):
        feature_sets = []
        for data in training_data:
            features = self.create_feature_set(data['Message'])
            feature_sets.append((features, str(data['stream_game_id'].iloc[0])))
        self.classifier = NaiveBayesClassifier.train(feature_sets, estimator=nltk.probability.MLEProbDist)

    def evaluate(self, test_set):
        feature_sets = []
        for data in test_set:
            features = self.create_feature_set(data['Message'])
            feature_sets.append((features, str(data['stream_game_id'].iloc[0])))
        return nltk.classify.accuracy(self.classifier, feature_sets)

    def predict_game(self, chat_log):
        features = self.create_feature_set(chat_log)
        return self.classifier.classify(features)
