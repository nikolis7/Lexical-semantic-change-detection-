from string import punctuation
from nltk.corpus import stopwords
import os
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from collections import OrderedDict
import unidecode
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
import pandas as pd
from numpy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity
from numpy import dot
from numpy.linalg import norm
from gensim.models.word2vec import PathLineSentences
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import multiprocessing


cores = multiprocessing.cpu_count() # Count the number of cores in a computer

punctuation = list(punctuation)
stop_words = stopwords.words('english')


class Data():

    def __getitem__(self, path=None):
        if path != None:
            with open(path, 'r') as f:
                self.doc = f.read()
            return self.doc

    def _preprocess(self, targets, corpus):
        self.index = []
        self.t_index = OrderedDict()
        for target in targets:

            for _, item in enumerate(corpus):
                if target in item:

                    #count_target = item.count(target)
                    #   Avoiding the sentences with multiple occurrences of the target term for the time being###
                   # if count_target == 1:
                    # if target not in self.t_index.keys():
                    #self.t_index[target] = [_]
                    # else:
                    self.t_index[target].append(_)
                    self.index.append(_)
        return self.index, self.t_index


'''
LOAD & EXTRACT DATA
'''
root_dir = os.getcwd()


p1 = os.path.join(root_dir, 'ccoha1.txt')
p2 = os.path.join(root_dir, 'ccoha2.txt')
t = os.path.join(root_dir, 'targets.txt')
r = os.path.join(root_dir, 'targets_results_.txt')


datasets = Data()  # initialization
doc1 = datasets.__getitem__(p1).split('\n')
doc2 = datasets.__getitem__(p2).split('\n')
t1 = datasets.__getitem__(t).split('\n')
results = datasets.__getitem__(r).split('\n')
results
target_act = [x for x in t1 if len(x) > 1]
t1 = [x.lower() for x in t1 if len(x) > 1]
index1 = datasets._preprocess(t1, doc1)
index2 = datasets._preprocess(t1, doc2)
index_t1 = index1[1]
index_t2 = index2[1]
print('The target words are:', t1)
target_words = t1

# conversions
target_uni = [unidecode.unidecode(m) for m in t1]

target_uni


'''

def initialize_word2vec_model(path_corpus, algorithm, hs, neg_samples, sampl_threshold, dim,
                              window_size, min_count, iterations, workers):
    # Initialize model
    model = gensim.models.Word2Vec(sg=algorithm,  # skipgram
                                   hs=hs,  # negative sampling
                                   negative=neg_samples,  # number of negative samples
                                   sample=sampl_threshold,
                                   # threshold for subsampling, if None, no subsampling is performed
                                   size=dim, window=window_size, min_count=min_count, iter=iterations, workers=workers)

    # Initialize vocabulary
    vocab_sentences = PathLineSentences(path_corpus)
    logging.getLogger('gensim').setLevel(logging.ERROR)
    # build vocabulary
    model.build_vocab(vocab_sentences)
    total_examples = model.corpus_count
    # retrieve sentences
    sentences = PathLineSentences(path_corpus)

    return model, total_examples, sentences


def train_word2vec_model(pretrained_matrix, embeddings_to_load, apply_incremental, apply_twec, dim, word2vec_model, total_examples,
                         sentences):
    # check if pretrained embeddings exist
    if not pretrained_matrix:
        word2vec_model.train(sentences, total_examples=total_examples, epochs=word2vec_model.epochs)
    elif (pretrained_matrix in list_of_pretrained) or apply_incremental or apply_twec:
        # check if exists pretrained embedding with given dimensions
        if dim in pretrained_dim:
            encoding ='utf-8'
            # change encoding to fix " 'utf-8' codec can't decode byte 0xc3 in position 97: invalid continuation byte" error
            if pretrained_matrix == 'sweconll17':
                encoding = 'latin-1'

            # initialize embeddings
            model_wv = KeyedVectors.load_word2vec_format(embeddings_to_load, encoding=encoding,binary=False)
            word2vec_model.build_vocab([list(model_wv.vocab.keys())], update=True)
            word2vec_model.intersect_word2vec_format(embeddings_to_load, encoding=encoding,binary=False, lockf=1.0)
            word2vec_model.train(sentences, total_examples=total_examples, epochs=word2vec_model.epochs)

    return word2vec_model

'''


def listcorpus(doc):
    corpus = []
    for sentence in doc:
        corpus.append([token for token in word_tokenize(sentence)
                          if token not in stop_words and token not in punctuation])
    return corpus



type(doc1[0])

word_tokenize(doc1[0])

[token for token in word_tokenize(
    doc1[0]) if token not in stop_words and token not in punctuation]


corpus1 = listcorpus(doc1)
corpus2 = listcorpus(doc2)


print(len(doc1))

# Initialiaze SGNS model
model_1 = Word2Vec(vector_size=300, window=3, hs=0, min_alpha=0.0007, alpha=0.03,
                   min_count=1, workers=cores-1, sg=1, negative=20)
# build vocabulary
model_1.build_vocab(corpus1, progress_per=1000)

# save the model for later use
model_1.save("word2vec_SGNS_1.model")
# Load the model
model_1 = Word2Vec.load("word2vec_SGNS_1.model")
# Train the model
model_1.train(corpus1, total_examples=model_1.corpus_count, epochs=100)
# save after train
model_1.save("word2vec_SGNS_1.model")




# Initialiaze SGNS model
model_2 = Word2Vec(vector_size=100, window=3, hs=0, min_alpha=0.0007, alpha=0.03,
                   min_count=1, workers=cores-1, sg=1, negative=20)
# build vocabulary
model_2.build_vocab(corpus2, progress_per=1000)

# save the model for later use
model_2.save("word2vec_SGNS_2.model")
# Load the model
model_2 = Word2Vec.load("word2vec_SGNS_2.model")
# Train the model
model_2.train(corpus2, total_examples=model_2.corpus_count, epochs=100)
# save after train
model_2.save("word2vec_SGNS_2.model")


# Create a matrix with the word embeddings
def matrix(words, model_1):
    x = {}
    for word in words:
        print(word)
        x[word] = model_1.wv[word]

    p = pd.DataFrame(x)
    return p


embeddings_corpus_1 = matrix(target_uni, model_1)

embeddings_corpus_2 = matrix(target_uni, model_2)


def OP(embeddings_corpus_1, embeddings_corpus_2):

    # Orthogonal Procrustes
    A = embeddings_corpus_1.to_numpy()
    B = embeddings_corpus_2.to_numpy()

    # Orthogonal Procrustes is used to find a matrix R which will allow for a mapping from A to B
    M = np.dot(B, A.transpose())
    u, s, vh = svd(M)
    R = np.dot(u, vh)

    # Transform A using R to map it to B. The transformed matrix A_new = RA
    # Mapped matrix
    new_A = np.dot(R, A)

    return new_A, B


new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)


def cosine_similarity(target_words, new_A, B):

    t = target_words
    output = {}

    for i, word in enumerate(t):
        output[word] = dot(new_A[:, i].transpose(), B[:, i].transpose(
        ))/(norm(new_A[:, i].transpose())*norm(B[:, i].transpose()))

    return output


output = cosine_similarity(target_words, new_A, B)


output


def classify(output):
    s = []
    for i, j in output.items():
        if j > 0.5:
            s.append(0)
        else:
            s.append(1)
    return s


s = classify(output)


def accuracy(s, results):
    count = 0
    for i, word in enumerate(output):
        if s[i] == int(results[i]):
            count += 1
    acc = count/len(results)

    return acc


accuracy(s, results)






