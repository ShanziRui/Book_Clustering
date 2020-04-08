from gensim.models import Phrases
from gensim.models import LdaModel
import logging
import pprint
import load
import glob
import os
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re
import random
import gensim
from gensim import corpora
from gensim import models
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from gensim.similarities.docsim import MatrixSimilarity
from sklearn import metrics
from gensim.models.coherencemodel import CoherenceModel


txt_files = glob.glob("*.txt")


# load each book from plain text
books = []
for txt in txt_files:
    books.append(load.Book(txt, False,
                           False, os.getcwd()))


print("*"*12+" Top 7 most frequently downloaded books "+"*"*12)
for book in books:
    print(book.title, book.author, sep=': ')
print()

# build dataframe each stands for one chapter
stemmer = SnowballStemmer('english')
words = set(stopwords.words('english'))
population = pd.DataFrame([], columns=['Title', 'Author',
                                       'NumChapters', 'Contents', 'Clean', 'Tokens', 'NumDoc', 'Documents'])

print("*"*12+" Population size of each book "+"*"*12)

author = []
size = []

for book in books:
    index = 1
    total = 0
    for chapter in book.chapters:
        Title = book.title
        Author = book.author
        NumChapters = index
        Contents = " ".join(chapter)
        # remove stopwords and stemmer
        Clean = " ".join([stemmer.stem(i) for i in re.sub(
            "[^a-zA-Z]", " ", Contents).split() if i not in words]).lower()
        Tokens = Clean.split(' ')
        NumDoc = len(Tokens)//150
        if NumDoc == 0:
            Documents = []
        else:
            Documents = np.array_split(Tokens, NumDoc)
        total = total + NumDoc
        df2 = pd.DataFrame([[Title, Author, NumChapters, Contents, Clean, Tokens, NumDoc, Documents]], columns=[
                           'Title', 'Author', 'NumChapters', 'Contents', 'Clean', 'Tokens', 'NumDoc', 'Documents'])
        population = population.append(df2, ignore_index=True)
        index = index + 1
    print(book.title, total)
    author.append(book.author)
    size.append(total)

print()

# till now, Documents column stores # documents of each chapter

training = pd.DataFrame([], columns=['Author', 'Title', 'Document', 'Heirarchical'])

train_docs = []

for book in books:
    Title = book.title
    Author = book.author
    bk = population[population['Author'] == Author]
    bk = bk[['Author', 'Title', 'Documents']]
    Documents = []
    for index, row in bk.iterrows():
        for docs in row['Documents']:
            #doc = ' '.join(docs)
            Documents.append(docs)
    Samples = random.sample(Documents, 200)
    for Document in Samples:
        train_docs.append(Document.tolist())
        df3 = pd.DataFrame([[Author, Title, Document]], columns=[
            'Author', 'Title', 'Document'])
        training = training.append(df3, ignore_index=True)

# till now, each row in training stands for one document, in total 1400 documents from 7 books


# Compute bigrams
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(train_docs, min_count=20)
for idx in range(len(train_docs)):
    for token in bigram[train_docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            train_docs[idx].append(token)


# Feature Engineering
# Create a dictionary representation of the documents.
dictionary = corpora.Dictionary(train_docs)
# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)


# Bag-of-words representation of the documents
bow_corpus = [dictionary.doc2bow(text) for text in train_docs]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(bow_corpus))

# TF-IDF representation of the documents
tfidf = models.TfidfModel(bow_corpus)
tfidf_corpus = tfidf[bow_corpus]


# LDA
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


chunksize = 2000
passes = 20
iterations = 400
eval_every = None

    # Make a index to word dictionary.
temp = dictionary[0]  # "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
        corpus=bow_corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha='auto',
        eta='auto',
        iterations=iterations,
        num_topics=6,
        passes=passes,
        eval_every=eval_every
    )

lda_corpus = model[bow_corpus]

# Train LDA model.
# Set training parameters.
num_topics = 10
chunksize = 2000
passes = 20
iterations = 400
eval_every = None

# Make a index to word dictionary.
temp = dictionary[0]  # "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=bow_corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

top_topics = model.top_topics(bow_corpus)  # , num_words=20)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)


#pyLDAvis.enable_notebook()
#vis = pyLDAvis.gensim.prepare(model, bow_corpus, dictionary=dictionary)
#pyLDAvis.save_html(vis, 'lda.html')

all_topics_csr = gensim.matutils.corpus2csc(tfidf_corpus)
all_topics_numpy = all_topics_csr.T.toarray()

# sklearn_pca = PCA(n_components = 2)
# Y_sklearn = sklearn_pca.fit_transform(all_topics_numpy)
# gmm = GaussianMixture(n_components=8, covariance_type='full').fit(Y_sklearn)
# prediction_gmm = gmm.predict(Y_sklearn)
# probs = gmm.predict_proba(Y_sklearn)

# centers = np.zeros((8,2))
# for i in range(8):
#     density = mvn(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(Y_sklearn)
#     centers[i, :] = Y_sklearn[np.argmax(density)]

plt.figure(figsize = (12,8))
# plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=prediction_gmm ,s=50, cmap='viridis')
# plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=1);

plt.title("Dendogram")
dend = shc.dendrogram(shc.linkage(all_topics_numpy, method='ward'))
#plt.axhline(y=45, color='r', linestyle='--')

cluster = AgglomerativeClustering(n_clusters=8, affinity='euclidean', linkage='ward')
print(len(cluster.fit_predict(all_topics_numpy)))


# plt.figure(figsize=(10, 7))
# plt.scatter(all_topics_numpy[:,0], all_topics_numpy[:,1], c=cluster.labels_, cmap='rainbow')



def matrix(corpus):

    matsim = MatrixSimilarity(corpus, num_features=len(dictionary))
    similarity_matrix = matsim[corpus]
    return similarity_matrix

# evaluation of different inputs
input_set = [bow_corpus, tfidf_corpus, lda_corpus]
out_evaluation = {}
out_evaluation['input'] = ['Bag of Word', 'TF-IDF', 'LDA Model']
out_evaluation['silhouette'] = []
for input in input_set:
    similarity_matrix = matrix(input)
    cluster.fit(similarity_matrix)
    clusters = cluster.labels_.tolist()
    silhouette_score = metrics.silhouette_score(similarity_matrix, clusters, metric='euclidean')
    out_evaluation['silhouette'].append(silhouette_score)
pd.DataFrame(out_evaluation).to_csv('Heirarchical.csv', index=False)

# champion input
matsim = MatrixSimilarity(lda_corpus, num_features=len(dictionary))
similarity_matrix = matsim[lda_corpus]
cluster.fit(similarity_matrix)
clusters = cluster.labels_.tolist()
training['Heirarchical'] = clusters

pd.DataFrame(training).to_csv('training_Heirarchical.csv', index=False)
