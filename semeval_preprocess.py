import os
import re
from string import punctuation
from collections import OrderedDict
import numpy as np
import logging
import matplotlib.pyplot as plt
import logging  # Setting up the loggings to monitor gensim
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)
from nltk.tokenize import word_tokenize
import nltk
from nltk.tokenize import punkt 
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
import multiprocessing
import gensim.downloader as api
import csv
from collections import defaultdict
from collections import Counter
from nltk.corpus import stopwords
from collections import Counter







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
# t = os.path.join(root_dir, 'targets.txt')
# r = os.path.join(root_dir, 'targets_results_.txt')


datasets = Data()  # initialization
doc1 = datasets.__getitem__(p1).split('\n')
doc2 = datasets.__getitem__(p2).split('\n')


# remove stopwords and punctuation 

def remove_stopwords_and_punctuation(corpus):
    cleaned_corpus = []
    for sentence in corpus:
        tokens = word_tokenize(sentence.lower())  # Tokenize and convert to lowercase
        cleaned_sentence = [word for word in tokens if word not in stop_words and word not in punctuation]
        cleaned_corpus.append(' '.join(cleaned_sentence))
    return cleaned_corpus

# Apply stopword and punctuation removal
cleaned_doc1 = remove_stopwords_and_punctuation(doc1)
cleaned_doc2 = remove_stopwords_and_punctuation(doc2)







# Clean corpus 
def listcorpus(doc):
    corpus = []
    for sentence in doc:
        corpus.append([token for token in word_tokenize(sentence)
                          if token not in stop_words and token not in punctuation])
    return corpus





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





###### Exploratory Data Analysis ####### 

# 1. Basic Corpus Overview

# Number of sentences in each corpus
num_sentences_corpus1 = len(doc1)
num_sentences_corpus2 = len(doc2)

# Word count per sentence
word_count_corpus1 = sum(len(sentence.split()) for sentence in doc1)
word_count_corpus2 = sum(len(sentence.split()) for sentence in doc2)

# Average sentence length
avg_sentence_length_corpus1 = word_count_corpus1 / num_sentences_corpus1
avg_sentence_length_corpus2 = word_count_corpus2 / num_sentences_corpus2

print(f"Corpus 1: {num_sentences_corpus1} sentences, {word_count_corpus1} words, Avg. sentence length: {avg_sentence_length_corpus1:.2f}")
print(f"Corpus 2: {num_sentences_corpus2} sentences, {word_count_corpus2} words, Avg. sentence length: {avg_sentence_length_corpus2:.2f}")


# 2. Vocabulary and Word Frequencies
# Compare the vocabulary and word frequencies between the two corpora. 
# We have excluded the stopwords and punctuatiom 


# i)   Unique words in each corpus.
# ii)  Top 10 most frequent words in each corpus.
# iii) Common words between both corpora.

from collections import Counter
from nltk.tokenize import word_tokenize

# Tokenize the sentences into words
words_corpus1 = [word_tokenize(sentence.lower()) for sentence in cleaned_doc1]
words_corpus2 = [word_tokenize(sentence.lower()) for sentence in cleaned_doc2]

# Flatten the lists of words
flat_words_corpus1 = [word for sublist in words_corpus1 for word in sublist]
flat_words_corpus2 = [word for sublist in words_corpus2 for word in sublist]

# Word frequencies
word_freq_corpus1 = Counter(flat_words_corpus1)
word_freq_corpus2 = Counter(flat_words_corpus2)

# Top 10 most frequent words
print("Top 10 words in Corpus 1:", word_freq_corpus1.most_common(10))
print("Top 10 words in Corpus 2:", word_freq_corpus2.most_common(10))

# Unique words
unique_words_corpus1 = set(flat_words_corpus1)
unique_words_corpus2 = set(flat_words_corpus2)

# Common words between both corpora
common_words = unique_words_corpus1.intersection(unique_words_corpus2)
print(f"Number of unique words in Corpus 1: {len(unique_words_corpus1)}")
print(f"Number of unique words in Corpus 2: {len(unique_words_corpus2)}")
print(f"Number of common words between corpora: {len(common_words)}")



# 3. Word Length and Sentence Length Distribution

# analyze the distribution of word lengths and sentence lengths to see if there is a shift in complexity or style between the two time periods.

import matplotlib.pyplot as plt

# Word lengths for both corpora
word_lengths_corpus1 = [len(word) for word in flat_words_corpus1]
word_lengths_corpus2 = [len(word) for word in flat_words_corpus2]

# Plot histograms
plt.hist(word_lengths_corpus1, bins=range(1, 20), alpha=0.7, label='Corpus 1')
plt.hist(word_lengths_corpus2, bins=range(1, 20), alpha=0.7, label='Corpus 2')
plt.xlabel('Word Length')
plt.ylabel('Frequency')
plt.legend()
plt.title('Word Length Distribution')
plt.show()



# Sentence lengths (in words) for both corpora

import matplotlib.pyplot as plt
import string

# Function to clean and tokenize each sentence (remove punctuation, lowercase)
def clean_and_tokenize(sentence):
    # Remove punctuation using string.punctuation and tokenize
    tokens = sentence.translate(str.maketrans('', '', string.punctuation)).lower().split()
    return tokens

# Clean and tokenize sentences from both corpora
tokenized_sentences_corpus1 = [clean_and_tokenize(sentence) for sentence in doc1]
tokenized_sentences_corpus2 = [clean_and_tokenize(sentence) for sentence in doc2]

# Recalculate sentence lengths after cleaning and tokenization
sentence_lengths_corpus1 = [len(tokens) for tokens in tokenized_sentences_corpus1]
sentence_lengths_corpus2 = [len(tokens) for tokens in tokenized_sentences_corpus2]

# Plot histograms for sentence lengths in both corpora
plt.hist(sentence_lengths_corpus1, bins=range(1, 100), alpha=0.7, label='Corpus 1')
plt.hist(sentence_lengths_corpus2, bins=range(1, 100), alpha=0.7, label='Corpus 2')
plt.xlabel('Sentence Length (Words)')
plt.ylabel('Frequency')
plt.legend()
plt.title('Sentence Length Distribution')
plt.show()








# 4. N-gram Analysis
#Perform bigram (two-word) and trigram (three-word) analysis to uncover common word sequences or phrases in each corpus.

import nltk
from collections import Counter
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
import string

# Function to clean and tokenize each sentence (remove punctuation, lowercase)
def clean_and_tokenize(sentence):
    # Remove punctuation using string.punctuation and tokenize
    sentence_clean = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
    tokens = word_tokenize(sentence_clean)
    return tokens

# Function to get n-grams and their frequencies
def get_ngrams(corpus, n=2):
    tokens = [clean_and_tokenize(sentence) for sentence in corpus]
    flat_tokens = [word for sublist in tokens for word in sublist]
    return Counter(ngrams(flat_tokens, n))

# Get top bigrams and trigrams for both corpora
bigrams_corpus1 = get_ngrams(doc1, 2)
bigrams_corpus2 = get_ngrams(doc2, 2)

# Print the top 10 bigrams
print("Top 10 bigrams in Corpus 1:", bigrams_corpus1.most_common(10))
print("Top 10 bigrams in Corpus 2:", bigrams_corpus2.most_common(10))

# Trigrams (n=3)
trigrams_corpus1 = get_ngrams(doc1, 3)
trigrams_corpus2 = get_ngrams(doc2, 3)

# Print the top 10 trigrams
print("Top 10 trigrams in Corpus 1:", trigrams_corpus1.most_common(10))
print("Top 10 trigrams in Corpus 2:", trigrams_corpus2.most_common(10))





# 5. Part-of-Speech (POS) Tagging
# Analyze the distribution of POS tags in each corpus to see if there are shifts in the types of words used (e.g., more nouns, fewer verbs).


# Download necessary resources for NLTK
nltk.data.path.append('C:/Users/Tsaro/anaconda3/envs/sgns_pipeline/nltk_data')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')  # Ensure 'punkt' is also downloaded for tokenization


# Function to tokenize and get POS distribution using nltk
def pos_distribution_nltk(corpus):
    pos_counts = Counter()
    for sentence in corpus:
        # Tokenize the sentence
        tokens = nltk.word_tokenize(sentence)
        # Get POS tags
        pos_tags = nltk.pos_tag(tokens)
        # Extract POS tags and update counts
        pos_counts.update([tag for word, tag in pos_tags])
    return pos_counts

# POS distribution for both corpora (assuming doc1 and doc2 are defined)
pos_corpus1 = pos_distribution_nltk(doc1)
pos_corpus2 = pos_distribution_nltk(doc2)

print("POS distribution in Corpus 1:", pos_corpus1)
print("POS distribution in Corpus 2:", pos_corpus2)












# 6. Word Embedding Similarity
# To explore semantic change, you could calculate the similarity of word embeddings (e.g., Word2Vec) between the two corpora.

# Train a Word2Vec model on each corpus.
# Calculate the cosine similarity between embeddings of the same word in both corpora.

#from gensim.models import Word2Vec
#from sklearn.metrics.pairwise import cosine_similarity
#import numpy as np

# Train Word2Vec models
#model_corpus1 = Word2Vec([sentence.split() for sentence in doc1], vector_size=100, window=5, min_count=1, workers=4)
#model_corpus2 = Word2Vec([sentence.split() for sentence in doc2], vector_size=100, window=5, min_count=1, workers=4)

# Get word embeddings for a common word
#word = 'example'  # Replace 'example' with a word common in both corpora
#embedding_corpus1 = model_corpus1.wv[word].reshape(1, -1)
#embedding_corpus2 = model_corpus2.wv[word].reshape(1, -1)

# Calculate cosine similarity
#similarity = cosine_similarity(embedding_corpus1, embedding_corpus2)
#print(f"Cosine similarity between embeddings of '{word}': {similarity[0][0]}")





# 7. TF-IDF Analysis
# Use TF-IDF (Term Frequency-Inverse Document Frequency) to identify distinctive words for each corpus.

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=50)

# Combine corpora into one and label them
corpus_combined = doc1 + doc2

# Fit the vectorizer and transform the corpus
tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_combined)

# Get the top TF-IDF words in each corpus
tfidf_words = tfidf_vectorizer.get_feature_names_out()
print("Top TF-IDF words:", tfidf_words)






