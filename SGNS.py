import os
import re
from string import punctuation
from collections import OrderedDict
import unidecode
import numpy as np
import logging
import matplotlib.pyplot as plt
import gensim
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from gensim.models import KeyedVectors
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
import pandas as pd
from numpy.linalg import svd
import sklearn
from sklearn.metrics.pairwise import cosine_similarity as cos 
from numpy import dot
from numpy.linalg import norm
from gensim.models.word2vec import PathLineSentences
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import punkt 
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import multiprocessing
import gensim.downloader as api
from scipy.linalg import orthogonal_procrustes
from gensim.models import Word2Vec
from gensim.similarities import WordEmbeddingSimilarityIndex
from gensim.similarities import SparseTermSimilarityMatrix
from scipy.spatial.distance import jensenshannon
from gensim.similarities import WmdSimilarity
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score












#############  START   #################################


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
                    count_target = item.count(target)
                    # Avoiding the sentences with multiple occurrences of the target term for the time being###
                    if count_target == 1:
                        if target not in self.t_index.keys():
                            self.t_index[target] = [_]
                        else:
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


# Clean corpus 
def listcorpus(doc):
    corpus = []
    for sentence in doc:
        corpus.append([token for token in word_tokenize(sentence)
                          if token not in stop_words and token not in punctuation])
    return corpus


corpus1 = listcorpus(doc1)
corpus2 = listcorpus(doc2)



#############################
#############################
#Clean dataset from POS

def remove_pos(corpus): 
    clean_corpus = []
    for sentence in corpus: 
        sent = [] 
        for word in sentence: 
            if word.endswith('_nn') or word.endswith('_vb'):
                sent.append(word.split('_')[0])
            else:
                sent.append(word) 
        clean_corpus.append(sent)
    return clean_corpus
   

corpus1_ = remove_pos(corpus1)
corpus2_ = remove_pos(corpus2)


#Clean dataset from POS
def clean_targets(target_words):
    targets = []
    
    for word in target_words: 
        targets.append(word.split('_')[0])
    return targets 

targets = clean_targets(target_words)

########################
# Useful function 

# Create a matrix with the word embeddings
def matrix(words, model):
    x = {}
    for word in words:
        print(word)
        x[word] = model.wv[word]

    p = pd.DataFrame(x)
    return p


# Orthogonal Procrustes 
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


# Orthogonal Procrustes scipy 
def OP_2(embeddings_corpus_1, embeddings_corpus_2):

    # Orthogonal Procrustes
    A = embeddings_corpus_1.to_numpy()
    B = embeddings_corpus_2.to_numpy()

    R, sca = orthogonal_procrustes(A, B)
    
    new_A = np.dot(A, R.T)
    
    return new_A, B



# Cosine Similarity 
def cosine_similarity(target_words, new_A, B):

    output = {}

    for i, word in enumerate(target_words):
        output[word] = dot(new_A[:, i].transpose(), B[:, i].transpose(
        ))/(norm(new_A[:, i].transpose())*norm(B[:, i].transpose()))

    return output


def classify(output):
    s = []
    for i, j in output.items():
        if j > 0.92:
            s.append(0)
        else:
            s.append(1)
    return s


def accuracy(s, results):
    count = 0
    for i, word in enumerate(output):
        if s[i] == int(results[i]):
            count += 1
    acc = count/len(results)

    return acc


# Find common words in two corpuses 

def common(s0, s1):
    s0 = set().union(*s0)
    s1 = set().union(*s1)
    return len(list(set(s0)&set(s1))), list(set(s0)&set(s1))


num, common_words = common(corpus1_,corpus2_)


#######################################################################################
#######################################################################################
# PIPELINE 


def sgns_pipeline(targets=targets, pretrained=False, alignment= "OP", epochs=10, plot=True):
    
    if pretrained==False:
        # Initialiaze SGNS model
            model_1 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                            sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=5)
            # build vocabulary
            model_1.build_vocab(corpus1_, progress_per=10000)
            # Train the model
            model_1.train(corpus1_, total_examples=model_1.corpus_count, epochs=epochs)
            # Get the embeddings 
            embeddings_corpus_1 = matrix(targets, model_1)
                
            
            if alignment == "OP": 
                # Initialiaze SGNS model
                model_2 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                                sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=5)
                # build vocabulary
                model_2.build_vocab(corpus2_, progress_per=10000)
                # Train the model
                model_2.train(corpus2_, total_examples=model_2.corpus_count, epochs=epochs)
                # Get the embeddings 
                embeddings_corpus_2 = matrix(targets, model_2)
                
                
                # Alignment 
                new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
                output = cosine_similarity(targets, new_A, B)
                #Results 
                s = classify(output)
                accuracy(s, results)
                
                if plot==True: 
                    fig, ax = plt.subplots()  # Create a figure containing a single axes.
                    ax.scatter(output.values(),output.values())
                    # add labels to all points
                    for (x, y) in output.items():
                        ax.text(y, y, x, va='bottom', ha='center')
                    ax.set_xlabel('Cosine_similarity')
                    ax.set_ylabel('Words')
                    #hide y-axis 
                    ax.get_yaxis().set_visible(False)
                    ax.legend()  
            
            else: 
                # Incremental learning 
                
                # Build vocabulary from your data for corpus 2 
                model_1.build_vocab(corpus2_, progress_per=10000, update=True)
                # Fine-tune model on your data
                model_1.train(corpus2_, total_examples=model_1.corpus_count, epochs=epochs)    
                # get embeddings 
                embeddings_corpus_2 = matrix(targets, model_1)
                # results without alignment 
                output = cosine_similarity(targets, embeddings_corpus_1.to_numpy(), embeddings_corpus_2.to_numpy())
                s = classify(output)
                accuracy(s, results)
                
                if plot==True: 
                    fig, ax = plt.subplots()  # Create a figure containing a single axes.
                    ax.scatter(output.values(),output.values())
                    # add labels to all points
                    for (x, y) in output.items():
                        ax.text(y, y, x, va='bottom', ha='center')
                    ax.set_xlabel('Cosine_similarity')
                    ax.set_ylabel('Words')
                    #hide y-axis 
                    ax.get_yaxis().set_visible(False)
                    ax.legend()           
    else: 
            # pre-trained 
                    
            # Load pre-trained word2vec model
            pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
            # Initialize new word2vec model with same parameters as pre-trained model
            model = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                            sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=5)
            # Build vocabulary from your data
            model.build_vocab(corpus1_, progress_per=10000)
            # Set weights of new model to pre-trained model
            model.wv.vectors = pretrained_model.vectors
            # Fine-tune model on your data
            model.train(corpus1_, total_examples=model.corpus_count, epochs=epochs)    
            # Get the embeddings 
            embeddings_corpus_1 = matrix(targets, model) 
            
            if alignment == "OP": 
                # Initialiaze SGNS model
                model_2 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                                sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=5)
                # build vocabulary
                model_2.build_vocab(corpus2_, progress_per=10)
                # Set weights of new model to pre-trained model
                model_2.wv.vectors = pretrained_model.vectors
                # Train the model
                model_2.train(corpus2_, total_examples=model_2.corpus_count, epochs=epochs)
                # Get the embeddings 
                embeddings_corpus_2 = matrix(targets, model_2)
                
                # Alignment 
                new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
                output = cosine_similarity(targets, new_A, B)
                #Results 
                s = classify(output)
                accuracy(s, results)
                
                if plot==True: 
                    fig, ax = plt.subplots()  # Create a figure containing a single axes.
                    ax.scatter(output.values(),output.values())
                    # add labels to all points
                    for (x, y) in output.items():
                        ax.text(y, y, x, va='bottom', ha='center')
                    ax.set_xlabel('Cosine_similarity')
                    ax.set_ylabel('Words')
                    #hide y-axis 
                    ax.get_yaxis().set_visible(False)
                    ax.legend()  
            
            else: 
                # Incremental learning 
                
                # Build vocabulary from your data for corpus 2 
                model.build_vocab(corpus2_, progress_per=10000, update=True)
                # Fine-tune model on your data
                model.train(corpus2_, total_examples=model_1.corpus_count, epochs=epochs)    
                # get embeddings 
                embeddings_corpus_2 = matrix(targets, model)
                # results without alignment 
                output = cosine_similarity(targets, embeddings_corpus_1.to_numpy(), embeddings_corpus_2.to_numpy())
                s = classify(output)
                accuracy(s, results)
                
                if plot==True: 
                    fig, ax = plt.subplots()  # Create a figure containing a single axes.
                    ax.scatter(output.values(),output.values())
                    # add labels to all points
                    for (x, y) in output.items():
                        ax.text(y, y, x, va='bottom', ha='center')
                    ax.set_xlabel('Cosine_similarity')
                    ax.set_ylabel('Words')
                    #hide y-axis 
                    ax.get_yaxis().set_visible(False)
                    ax.legend()       
                                            
    return print('run completed!') 

#######################################################################################
#######################################################################################



#######################################################################################
# Try pipeline 
sgns_pipeline(targets=targets, pretrained=False, alignment="OP", epochs=2, plot=True)
#######################################################################################



########################
# 1) Run only with own train model with alignment 

# Initialiaze SGNS model
model_1 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=5)
# build vocabulary
model_1.build_vocab(corpus1_, progress_per=10000)

# save the model for later use
#model_1.save("word2vec_SGNS_1.model")
# Load the model
#model_1 = Word2Vec.load("word2vec_SGNS_1.model")
# Train the model
model_1.train(corpus1_, total_examples=model_1.corpus_count, epochs=100)

# save after train
model_1.save("word2vec_SGNS_1.model")

# Load the model
model_1 = Word2Vec.load("word2vec_SGNS_1.model")


# Extract embeddings 
embeddings_corpus_1 = matrix(targets, model_1)

# embeddings_corpus_all_1 = matrix(common_words,model_1)


model_2 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=5)
# build vocabulary
model_2.build_vocab(corpus2_, progress_per=10000)

# Train the model
model_2.train(corpus1_, total_examples=model_2.corpus_count, epochs=100)

# save after train
model_2.save("word2vec_SGNS_2.model")

# Load the model
model_2 = Word2Vec.load("word2vec_SGNS_2.model")


# Extract embeddings 
embeddings_corpus_2 = matrix(targets, model_2)

#embeddings_corpus_all_2 = matrix(common_words,model_2)

# Alignment 
new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(targets, new_A, B)
output
#Results 
s = classify(output)
accuracy(s, results)

# Alignment  for all common words 
new_A, B = OP(embeddings_corpus_all_1, embeddings_corpus_all_2)
output_all = cosine_similarity(targets, new_A, B)


# Visualize words and cosine similarity 

#1
fig, ax = plt.subplots()  # Create a figure containing a single axes.

ax.scatter(output.values(),output.values())
# add labels to all points
for (x, y) in output.items():
    ax.text(y, y, x, va='bottom', ha='center')
ax.set_xlabel('Cosine_similarity')
ax.set_ylabel('Words')
#hide y-axis 
ax.get_yaxis().set_visible(False)
ax.legend()


#2
fig = plt.figure(figsize = (14, 5))
 
# creating the bar plot
plt.bar(targets, output.values(), color ='maroon',
        width = 0.4)
plt.xticks(rotation=45, ha='right')
plt.xlabel("Words")
plt.ylabel("Cosine similarity")
#plt.title("Students enrolled in different courses")
plt.show()


#### Run with scipy alignment 

new_A, B = OP_2(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(targets, new_A, B)
output



########################
# 2) Run only with own train model  with incremental learning & alignment 

model_1 = Word2Vec(vector_size=300, window=10, hs=0, min_alpha=0.0007, alpha=0.03,
                   sample=1e-3, min_count=1, workers=cores-1, sg=1, negative=3)
# build vocabulary
model_1.build_vocab(corpus1_, progress_per=1000)

# Train the model
model_1.train(corpus1_, total_examples=model_1.corpus_count, epochs=100)
# save after train
model_1.save("word2vec_SGNS_1.model")

#Load the model
model_1 = Word2Vec.load("word2vec_SGNS_1.model")


embeddings_corpus_1 = matrix(targets, model_1)

#embeddings_corpus_all_1 = matrix(common_words,model_1)


# Build vocabulary from your data for corpus 2 
model_1.build_vocab(corpus2_, progress_per=10000, update=True)

# Fine-tune model on your data
model_1.train(corpus2_, total_examples=model_1.corpus_count, epochs=100)

# Save fine-tuned model
model_1.save('word2vec_SGNS_2_incremental.model')

#Load the model
model_1 = Word2Vec.load("word2vec_SGNS_2_incremental.model")

embeddings_corpus_2 = matrix(targets, model_1)


# Alignment 
new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(targets, new_A, B)
output
#Results 
s = classify(output)
accuracy(s, results)




######################################
######################################
# 3) Use pretrained model with incremental learning & Alignment 

# Load pre-trained word2vec model
pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Initialize new word2vec model with same parameters as pre-trained model
model = Word2Vec(vector_size=300, window=10,min_count=1, workers=cores-1, sg=1, negative=3)

# Set weights of model to pre-trained vectors
#model.wv.add_vectors(pretrained_model.index_to_key, pretrained_model.vectors)
model.wv.vectors = pretrained_model.vectors

# Build vocabulary from your data
model.build_vocab(corpus1_, progress_per=10000, min_count=1)

# Fine-tune model on your data
model.train(corpus1_, total_examples=model.corpus_count, epochs=100)

# Save fine-tuned model
model.save('pre_trained_fine_tuned_model_1.model')

#Load the model
model = Word2Vec.load("pre_trained_fine_tuned_model_1.model")

embeddings_corpus_1 = matrix(targets, model)


# incremental

# Build vocabulary from your data for corpus 2 
model.build_vocab(corpus2_, progress_per=10000, min_count=1, update=True)

#model.wv.add_vectors(corpus2_, ignore_missing=True)

# Fine-tune model on your data
model.train(corpus2_, total_examples=model.corpus_count, epochs=100)

# Save fine-tuned model
model.save('pre_trained_fine_tuned_model_2_incremental.model')

# Load the model
model = Word2Vec.load("pre_trained_fine_tuned_model_2_incremental.model")

embeddings_corpus_2 = matrix(targets, model)


# Alignment 
new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(targets, new_A, B)
output
#Results 
s = classify(output)
accuracy(s, results)


######################################


######################################
######################################
# 4) Pre-trained model with alignment 

# Load the model
model_1 = Word2Vec.load("pre_trained_fine_tuned_model_1.model")

embeddings_corpus_1 = matrix(targets, model_1)

#####
#train independently on corpus_2

# Load pre-trained word2vec model
pretrained_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Initialize new word2vec model with same parameters as pre-trained model
model_2 = Word2Vec(vector_size=300, window=10, min_count=1, workers=cores-1, sg=1, negative=3)

# Build vocabulary from your data
model_2.build_vocab(corpus2_, progress_per=1000)

# Set weights of new model to pre-trained model
#model.intersect_word2vec_format('pretrained_model.bin', binary=True, lockf=1.0)
model_2.wv.vectors = pretrained_model.vectors

# Fine-tune model on your data
model_2.train(corpus2_, total_examples=model_2.corpus_count, epochs=100)

embeddings_corpus_2 = matrix(targets, model_2)

# Alignment 
new_A, B = OP(embeddings_corpus_1, embeddings_corpus_2)
output = cosine_similarity(targets, new_A, B)

#Results 
s = classify(output)
accuracy(s, results)

output


# Find k-nearest words of selected words based on similarity 




# Preprocess your corpus into a list of list of words
#corpus = [['word', 'list', 'document', ...], ['another', 'document', 'in', ...], ...]

# Train a Word2Vec model on your preprocessed corpus
#model = Word2Vec(corpus, size=100, window=5, min_count=1, workers=4)

# Select target words for analysis
#target_words = ['selected', 'words', 'to', 'analyze']

# Retrieve word vectors for target words for corpus 1 
word_vectors_1 = [model_1.wv[word] for word in targets]

# Create a similarity index for the Word2Vec model
similarity_index = WordEmbeddingSimilarityIndex(model_1.wv)

# Define the number of nearest words to retrieve
k = 5

# Derive the k nearest words for each target word

knn_words = {}

for i, word in enumerate(targets):
    similarities = similarity_index.most_similar(word, topn=k)
    nearest_words = sorted(similarities, key=lambda item: -item[1])[:k]
    knn_words[word] = nearest_words
    print(f"Nearest words to '{word}': {nearest_words}")
    


targets 

knn_words['word']







# Retrieve word vectors for target words in both models
word_vectors1 = [model_1.wv[word] for word in targets]
word_vectors2 = [model_2.wv[word] for word in targets]

# Create a similarity index for the first Word2Vec model
similarity_index1 = WordEmbeddingSimilarityIndex(model_1.wv)

# Create a similarity index for the second Word2Vec model
similarity_index2 = WordEmbeddingSimilarityIndex(model_2.wv)

# Define the number of nearest words to retrieve
k = 5

# Calculate JSD score for each target word
jsd_scores = []
for i, word in enumerate(targets):
    # Retrieve k nearest words for the first corpus
    similarities1 = similarity_index1.most_similar(word, topn=k)
    nearest_words1 = [(w, sim) for w, sim in similarities1 if w in model_1.wv.vocab]

    # Retrieve k nearest words for the second corpus
    similarities2 = similarity_index2.most_similar(word, topn=k)
    nearest_words2 = [(w, sim) for w, sim in similarities2 if w in model_2.wv.vocab]

    # Extract the nearest word vectors
    nearest_word_vectors1 = np.array([model_1.wv[word] for word, _ in nearest_words1])
    nearest_word_vectors2 = np.array([model_2.wv[word] for word, _ in nearest_words2])

    # Calculate the probability distributions for the nearest words
    distribution1 = np.array([sim for _, sim in nearest_words1])
    distribution2 = np.array([sim for _, sim in nearest_words2])

    # Normalize the probability distributions
    distribution1 /= np.sum(distribution1)
    distribution2 /= np.sum(distribution2)

    # Calculate the JSD score between the distributions
    jsd = jensenshannon(distribution1, distribution2)

    jsd_scores.append(jsd)

# Print the JSD scores for each target word
for word, score in zip(targets, jsd_scores):
    print(f"JSD score for '{word}': {score}")
    
    
    
    
    
    
    
    
    
    
    
  
# Retrieve word vectors for target words in both models
word_vectors1 = [model_1.wv[word] for word in targets]
word_vectors2 = [model_2.wv[word] for word in targets]

# Define the number of nearest words to retrieve
k = 5

# Calculate JSD score for each target word
jsd_scores = []
for i, word in enumerate(targets):
    # Retrieve k nearest words for the first corpus
    nearest_words1 = model_1.wv.most_similar(word, topn=k)

    # Retrieve k nearest words for the second corpus
    nearest_words2 = model_2.wv.most_similar(word, topn=k)

    # Extract the nearest word vectors
    nearest_word_vectors1 = np.array([model_1.wv[word] for word, _ in nearest_words1])
    nearest_word_vectors2 = np.array([model_2.wv[word] for word, _ in nearest_words2])

    # Calculate the probability distributions for the nearest words
    distribution1 = np.array([sim for _, sim in nearest_words1])
    distribution2 = np.array([sim for _, sim in nearest_words2])

    # Normalize the probability distributions
    distribution1 /= np.sum(distribution1)
    distribution2 /= np.sum(distribution2)

    # Calculate the JSD score between the distributions
    jsd = jensenshannon(distribution1, distribution2)

    jsd_scores.append(jsd)

# Print the JSD scores for each target word
for word, score in zip(targets, jsd_scores):
    print(f"JSD score for '{word}': {score}")
    
    
    
    
    
    
    
    
    
    
    
    
# Calculate JSD score for each target word
jsd_scores = []
for word in targets:
    # Retrieve nearest word embeddings for the first corpus
    nearest_word_embeddings1 = model_1.wv.most_similar(word, topn=k, restrict_vocab=None)

    # Retrieve nearest word embeddings for the second corpus
    nearest_word_embeddings2 = model_2.wv.most_similar(word, topn=k, restrict_vocab=None)

    # Extract the word vectors
    word_vectors1 = np.array([emb for _, emb in nearest_word_embeddings1])
    word_vectors2 = np.array([emb for _, emb in nearest_word_embeddings2])

    # Calculate the probability distributions for the word vectors
    distribution1 = np.mean(word_vectors1, axis=0)
    distribution2 = np.mean(word_vectors2, axis=0)

    # Normalize the probability distributions
    distribution1 /= np.sum(distribution1)
    distribution2 /= np.sum(distribution2)

    # Calculate the JSD score between the distributions
    jsd = jensenshannon(distribution1, distribution2)

    jsd_scores.append(jsd)

# Print the JSD scores for each target word
for word, score in zip(targets, jsd_scores):
    print(f"JSD score for '{word}': {score}")
    
    
    
    
    
    
    
    
    
    
    
    
    
#### SELECT THE THRESHOLD FOR COMMON WORDS BASED ON GROUND TRUTH  ####

#### The methodology is to take the average cosine similarity of the proposed semantic change #######

def threshold(similarity_dict, annotation_list):
    """
    Calculate the average cosine similarity of words identified as having undergone semantic change.
    
    Parameters:
    similarity_dict (dict): Dictionary with words as keys and their cosine similarities as values.
    annotation_list (list): List of binary annotations where '1' indicates semantic change.
    
    Returns:
    float: The average cosine similarity of the words identified as having semantic change.
    """
    # Ensure both the dictionary and the annotation list have the same length
    if len(similarity_dict) != len(annotation_list):
        raise ValueError("The length of similarity dictionary and annotation list must match.")
    
    # List of cosine similarities for words marked as having semantic change
    changed_word_similarities = [
        sim for i, (word, sim) in enumerate(similarity_dict.items()) if annotation_list[i] == '1'
    ]
    
    # Calculate the average cosine similarity for the changed words
    if changed_word_similarities:
        average_similarity = sum(changed_word_similarities) / len(changed_word_similarities)
    else:
        raise ValueError("No words were annotated as having semantic change.")
    
    return average_similarity



threshold(output, results)


######## Propose words for annotation #############

def extract_align_and_filter_words(model1, model2, threshold):
    """
    Extract embeddings for common words from two Word2Vec models, align them using Orthogonal Procrustes, 
    calculate cosine similarity, and return words that have cosine similarity below the threshold.
    
    Parameters:
    model1: Trained Word2Vec model for corpus1.
    model2: Trained Word2Vec model for corpus2.
    threshold (float): The cosine similarity threshold for identifying potential words for annotation.
    
    Returns:
    list: A list of words with cosine similarity below the threshold (proposed for annotation).
    dict: A dictionary with common words and their cosine similarities.
    """
    
    # Step 1: Find the common words between the two models
    common_words = list(set(model1.wv.index_to_key).intersection(set(model2.wv.index_to_key)))
    
    if not common_words:
        raise ValueError("No common words found between the two models.")
    
    # Step 2: Extract embeddings for the common words
    embeddings1 = np.array([model1.wv[word] for word in common_words])
    embeddings2 = np.array([model2.wv[word] for word in common_words])
    
    # Step 3: Align the embeddings using Orthogonal Procrustes
    R, _ = orthogonal_procrustes(embeddings1, embeddings2)
    aligned_embeddings1 = np.dot(embeddings1, R)
    
    # Step 4: Calculate cosine similarity between the aligned embeddings
    similarity_dict = {}
    for i, word in enumerate(common_words):
        cos_sim = dot(aligned_embeddings1[i], embeddings2[i]) / (norm(aligned_embeddings1[i]) * norm(embeddings2[i]))
        similarity_dict[word] = cos_sim
    
    # Step 5: Filter words whose cosine similarity is below the threshold
    proposed_words = [word for word, sim in similarity_dict.items() if sim < threshold]
    
    return proposed_words, similarity_dict


# Example usage:

# Assuming `model1` is the Word2Vec model trained on corpus1 and `model2` is trained on corpus2
# Example initialization (replace with your actual models)
# from gensim.models import Word2Vec
# model1 = Word2Vec(corpus1, vector_size=300, window=5, min_count=1, workers=4, sg=1)
# model2 = Word2Vec(corpus2, vector_size=300, window=5, min_count=1, workers=4, sg=1)

# Step 6: Define the threshold
threshold_value = 0.765  # Example threshold

# Step 7: Extract, align, and filter words based on cosine similarity threshold
proposed_words, similarity_dict = extract_align_and_filter_words(model_1, model_2, threshold_value)




######## Create a csv file for the annotators  #############



def find_word_mapping_to_sentences(corpus, words):
    """
    Create a mapping of words to the indices of the sentences in which they appear.
    
    Parameters:
    corpus (list): List of preprocessed sentences (tokenized) from the corpus.
    words (list): List of words to track in the corpus.
    
    Returns:
    dict: A dictionary where each word maps to a list of sentence indices in the corpus.
    """
    word_to_sentences = defaultdict(list)
    
    # Check each sentence for the occurrence of target words
    for idx, tokens in enumerate(corpus):
        for word in words:
            if word.lower() in [t.lower() for t in tokens]:  # Check if the word exists in the tokenized sentence
                word_to_sentences[word].append(idx)
    
    return word_to_sentences


def select_representative_sentence(word, corpus, word_index, original_corpus):
    """
    Select the representative sentence for a given word from the original corpus based on TF-IDF scores.
    
    Parameters:
    word (str): The target word.
    corpus (list): List of preprocessed sentences from the corpus.
    word_index (dict): Dictionary mapping words to sentence indices in the preprocessed corpus.
    original_corpus (list): The original corpus with raw sentences (unprocessed).
    
    Returns:
    str: The representative sentence from the original corpus for the word.
    """
    # Extract sentences where the word appears (from the original corpus)
    sentence_indices = word_index[word]
    sentences = [original_corpus[idx] for idx in sentence_indices]
    
    # Preprocess these sentences again for TF-IDF
    preprocessed_sentences = [' '.join(sentence) if isinstance(sentence, list) else sentence for sentence in sentences]
    
    # Apply TF-IDF vectorizer to the sentences
    vectorizer = TfidfVectorizer(vocabulary=[word])
    tfidf_matrix = vectorizer.fit_transform(preprocessed_sentences)
    
    # Calculate the TF-IDF score for the word in each sentence
    tfidf_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    
    # Select the sentence with the highest TF-IDF score (from the original corpus)
    representative_sentence = sentences[np.argmax(tfidf_scores)]
    
    return representative_sentence


def create_annotation_csv(proposed_words, corpus1_, corpus2_, doc1, doc2, output_file="annotation_task.csv"):
    """
    Create a CSV file with proposed words and representative sentences (from the original corpus) from both corpora for annotation.
    
    Parameters:
    proposed_words (list): List of words proposed for annotation (those that meet the threshold).
    corpus1_ (list): List of preprocessed, tokenized sentences from the first corpus.
    corpus2_ (list): List of preprocessed, tokenized sentences from the second corpus.
    doc1 (list): Original sentences from the first corpus (unprocessed).
    doc2 (list): Original sentences from the second corpus (unprocessed).
    output_file (str): Name of the output CSV file.
    
    Returns:
    None
    """
    
    # Step 1: Find word-to-sentence mappings in both preprocessed corpora
    word_index1 = find_word_mapping_to_sentences(corpus1_, proposed_words)
    word_index2 = find_word_mapping_to_sentences(corpus2_, proposed_words)
    
    # Open the CSV file for writing
    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['Word', 'Sentence in Corpus 1', 'Sentence in Corpus 2', 'Annotation (Semantic Change: Yes/No)'])
        
        # Iterate through the proposed words
        for word in proposed_words:
            # Fetch representative sentences from both corpora using the TF-IDF method (from original corpora)
            if word in word_index1 and word in word_index2:
                sentence1 = select_representative_sentence(word, corpus1_, word_index1, doc1)
                sentence2 = select_representative_sentence(word, corpus2_, word_index2, doc2)
                
                # Write the word and its sentences from both corpora to the CSV
                writer.writerow([word, sentence1, sentence2, ''])  # Empty annotation column
                
    print(f"CSV file '{output_file}' created successfully!")
    
    
    
    
    
    
    
    
# Create the CSV file for annotation with representative sentences from original corpora
create_annotation_csv(proposed_words, corpus1_, corpus2_, doc1, doc2)



########## Finalize the preprocessed dataset ##################



def majority_vote_annotation(csv_files, output_file="final_annotation.csv"):
    """
    Create a final CSV file based on majority voting from multiple annotation CSV files.
    
    Parameters:
    csv_files (list): List of CSV file names (paths) to be merged.
    output_file (str): Name of the final CSV file with majority-voted annotations.
    
    Returns:
    None
    """
    # Step 1: Load all CSV files into a list of DataFrames
    dfs = [pd.read_csv(file) for file in csv_files]
    
    # Assuming the CSVs have columns: ['Word', 'Sentence in Corpus 1', 'Sentence in Corpus 2', 'Annotation (Semantic Change: Yes/No)']
    
    # Step 2: Initialize a DataFrame for the final output (based on the first file)
    final_df = dfs[0][['Word', 'Sentence in Corpus 1', 'Sentence in Corpus 2']].copy()
    
    # Step 3: For each word, apply majority voting on annotations
    annotations = []
    for i in range(len(final_df)):
        word_annotations = [df.iloc[i]['Annotation (Semantic Change: Yes/No)'] for df in dfs]  # Get annotations from each file for the current word
        
        # Apply majority vote using Counter
        majority_annotation = Counter(word_annotations).most_common(1)[0][0]
        annotations.append(majority_annotation)
    
    # Step 4: Add the final annotations to the DataFrame
    final_df['Annotation (Semantic Change: Yes/No)'] = annotations
    
    # Step 5: Write the final DataFrame to a new CSV file
    final_df.to_csv(output_file, index=False)
    
    print(f"Final annotated CSV file '{output_file}' created successfully!")

# Example usage:
csv_files = ["annotation_file1.csv", "annotation_file2.csv", "annotation_file3.csv"]
majority_vote_annotation(csv_files, output_file="final_annotation.csv")




### Creating the new features #####


# Load stopwords
stop_words = set(stopwords.words('english'))

# Load the annotated dataset
df = pd.read_csv('final_annotation.csv')  # Ensure this file has columns for 'Sentence in Corpus 1' and 'Sentence in Corpus 2'

# Preprocess text data (lowercasing, tokenizing, and removing stopwords)
def preprocess_text(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(tokens)

# Apply preprocessing to both sentences in Corpus 1 and Corpus 2
df['Sentence in Corpus 1'] = df['Sentence in Corpus 1'].apply(preprocess_text)
df['Sentence in Corpus 2'] = df['Sentence in Corpus 2'].apply(preprocess_text)

# Feature 1: TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=300)
tfidf_corpus1 = tfidf_vectorizer.fit_transform(df['Sentence in Corpus 1']).toarray()
tfidf_corpus2 = tfidf_vectorizer.fit_transform(df['Sentence in Corpus 2']).toarray()

# Feature 2: Cosine similarity between TF-IDF vectors
cosine_similarities_tfidf = np.array([cosine_similarity([tfidf_corpus1[i]], [tfidf_corpus2[i]])[0][0]
                                      for i in range(len(df))]).reshape(-1, 1)

# Feature 3: Sentence length difference
sentence_len_diff = np.abs(df['Sentence in Corpus 1'].apply(len) - df['Sentence in Corpus 2'].apply(len)).values.reshape(-1, 1)

# Feature 4: Word frequency change
word_freq_corpus1 = df['Sentence in Corpus 1'].apply(lambda x: len(x.split()))
word_freq_corpus2 = df['Sentence in Corpus 2'].apply(lambda x: len(x.split()))
word_freq_diff = np.abs(word_freq_corpus1 - word_freq_corpus2).values.reshape(-1, 1)

# Feature 5: BERT Sentence Embeddings and Cosine Similarity
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Use a pre-trained model from sentence-transformers
df['Embedding Corpus 1'] = df['Sentence in Corpus 1'].apply(lambda x: model.encode(x))
df['Embedding Corpus 2'] = df['Sentence in Corpus 2'].apply(lambda x: model.encode(x))

cosine_sim_embeddings = np.array([cosine_similarity([df['Embedding Corpus 1'][i]], 
                                                    [df['Embedding Corpus 2'][i]])[0][0]
                                  for i in range(len(df))]).reshape(-1, 1)

# Feature 6: POS Tagging Difference
nlp = spacy.load('en_core_web_sm')

def pos_tag_distribution(sentence):
    doc = nlp(sentence)
    pos_counts = doc.count_by(spacy.attrs.POS)
    total = sum(pos_counts.values())
    pos_dist = {k: v / total for k, v in pos_counts.items()}
    return pos_dist

df['POS Corpus 1'] = df['Sentence in Corpus 1'].apply(pos_tag_distribution)
df['POS Corpus 2'] = df['Sentence in Corpus 2'].apply(pos_tag_distribution)

# Create a feature based on the difference between POS tag distributions
def pos_difference(pos1, pos2):
    all_tags = set(pos1.keys()).union(set(pos2.keys()))
    diff = sum(abs(pos1.get(tag, 0) - pos2.get(tag, 0)) for tag in all_tags)
    return diff

df['POS Difference'] = df.apply(lambda row: pos_difference(row['POS Corpus 1'], row['POS Corpus 2']), axis=1)
pos_diff = df['POS Difference'].values.reshape(-1, 1)

# Combine all features into a single feature matrix (X)
X = np.hstack([tfidf_corpus1, tfidf_corpus2, cosine_similarities_tfidf, sentence_len_diff, word_freq_diff, cosine_sim_embeddings, pos_diff])

# Target variable (Annotation)
y = df['Annotation (Semantic Change: Yes/No)'].apply(lambda x: 1 if x == '1' else 0).values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Save the feature matrix and the model for later use
pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]).to_csv('features_with_annotations.csv', index=False)
















# ================= Supervised Learning Approaches ================= #

# Function to evaluate models
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model.__class__.__name__}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))
    print("ROC AUC:", roc_auc_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Model 1: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
evaluate_model(rf, X_train, X_test, y_train, y_test)

# Model 2: Support Vector Machine (SVM)
svm = SVC(kernel='linear', probability=True, random_state=42)
evaluate_model(svm, X_train, X_test, y_train, y_test)

# Model 3: Logistic Regression
log_reg = LogisticRegression(random_state=42)
evaluate_model(log_reg, X_train, X_test, y_train, y_test)

# ================= Fine-Tuning with GridSearchCV ================= #

# Example: Fine-tuning Random Forest with GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Fine-tuning Random Forest using GridSearchCV
rf_grid = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='f1', verbose=2, n_jobs=-1)
rf_grid.fit(X_train, y_train)

# Print best parameters from grid search
print("Best parameters found by GridSearchCV for Random Forest:")
print(rf_grid.best_params_)

# Evaluate the fine-tuned model
evaluate_model(rf_grid.best_estimator_, X_train, X_test, y_train, y_test)

# Fine-tune other models (SVM, Logistic Regression, etc.) similarly if needed.

# ================= Cross-Validation for Robustness ================= #

# Perform cross-validation to ensure robustness
rf_cv_scores = cross_val_score(rf_grid.best_estimator_, X, y, cv=10, scoring='accuracy')
print(f"10-fold cross-validation accuracy for Random Forest: {np.mean(rf_cv_scores):.3f}")

# ================= Save the Features and Model ================= #

# Save the final feature matrix and the model for later use
pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]).to_csv('features_with_annotations.csv', index=False)









