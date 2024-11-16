import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

import gensim
from torch import nn

from env import SECRET
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

columns = ["target", "ids", "date", "flag", "user", "text"]
data = pd.read_csv(SECRET, names=columns, encoding = "ISO-8859-1", nrows=20)
print(data)

# Tokenization
nltk.download("stopwords")

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

data.text.apply(lambda x: preprocess(x))
train, test = train_test_split(data, test_size = 0.2, shuffle=True)

documents = [text.split() for text in train.text]
print(documents)
# w2v = gensim.models.word2vec.Word2Vec(vector_size=W2V_SIZE,window=W2V_WINDOW, epochs=W2V_EPOCH, min_count=W2V_MIN_COUNT)
# w2v.build_vocab()