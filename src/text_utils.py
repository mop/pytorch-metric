import data
import gensim
import gensim.downloader as gensim_api
import re
import nltk
import logging
from sklearn import feature_extraction
from sklearn.preprocessing import normalize
import numpy as np


def preprocess_text(text: str, stem:bool =True, lemm: bool = True, stopwords: list=None):
     ## clean (convert to lowercase and remove punctuations and   
    #characters and then strip
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
            
    ## Tokenize (convert from string to list)
    lst_text = text.split()
    ## remove Stopwords
    if stopwords is None:
        stopwords = nltk.corpus.stopwords.words('english')
    if stopwords is not None:
        lst_text = [word for word in lst_text if word not in
                    stopwords]
                
    ## Stemming (remove -ing, -ly, ...)
    if stem:
        ps = nltk.stem.porter.PorterStemmer()
        lst_text = [ps.stem(word) for word in lst_text]
                
    ## Lemmatisation (convert the word into root word)
    if lemm:
        lem = nltk.stem.wordnet.WordNetLemmatizer()
        lst_text = [lem.lemmatize(word) for word in lst_text]
            
    ## back to string from list
    text = " ".join(lst_text)
    return text

def train_text_embeddings(dataset: data.DMLDataset, embedding_size: int = 128):
    all_texts = []
    for batch in dataset:
        all_texts += [preprocess_text(t) for t in batch['text']]

    logging.info('training word2vec model')
    model = gensim.models.Word2Vec(all_texts, vector_size=embedding_size, window=5, min_count=1)
    logging.info('trained word2vec model')

    return model

def encode_text(model, dataset: data.DMLDataset) -> (np.ndarray, np.ndarray):
    all_texts = []
    all_labels = []
    for batch in dataset:
        all_texts += [preprocess_text(t) for t in batch['text']]
        all_labels.append(batch['label'].detach().cpu().numpy())
    all_labels = np.concatenate(all_labels)

    all_fvecs = []
    for txt in all_texts:
        fvec = np.mean([model.wv[word] for word in txt if word in model.wv], axis=0)
        all_fvecs.append(fvec)

    all_fvecs = np.vstack(all_fvecs)
    all_fvecs = all_fvecs / np.maximum(1e-5, np.linalg.norm(all_fvecs, axis=-1, keepdims=True))
    return all_fvecs, all_labels


def train_text_tfidf(dataset: data.DMLDataset, max_features: int = 1000, ngram_range=(1, 1)):
    stopwords = set(nltk.corpus.stopwords.words('english'))
    #cv = CountVectorizer(max_df=0.8, stop_words=set(stopwords), max_features=max_features, ngram_range=ngram_range)

    all_texts = []
    all_labels = []
    for batch in dataset:
        all_texts += [preprocess_text(t) for t in batch['text']]
        all_labels.append(batch['label'].detach().cpu().numpy())
    all_labels = np.concatenate(all_labels)

    tfidf = feature_extraction.text.TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    tfidf.fit(all_texts)

    return tfidf

def encode_text_tfidf(model: feature_extraction.text.TfidfVectorizer,
                      dataset: data.DMLDataset) -> (np.ndarray, np.ndarray):
    all_texts = []
    all_labels = []
    for batch in dataset:
        all_texts += [preprocess_text(t) for t in batch['text']]
        all_labels.append(batch['label'].detach().cpu().numpy())
    all_labels = np.concatenate(all_labels)

    all_fvecs = model.transform(all_texts)
    all_fvecs = normalize(all_fvecs, norm='l2', axis=1)

    #all_fvecs = all_fvecs / np.maximum(1e-5, np.linalg.norm(all_fvecs, axis=-1, keepdims=True))
    return all_fvecs, all_labels

