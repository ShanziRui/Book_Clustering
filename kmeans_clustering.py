from sklearn.manifold import MDS
import matplotlib.path as mpath
from gensim.models.coherencemodel import CoherenceModel
from sklearn import metrics
from gensim.matutils import corpus2dense, corpus2csc
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from gensim.models import Phrases
import pyLDAvis.gensim
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
from gensim.similarities.docsim import MatrixSimilarity


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

training = pd.DataFrame([], columns=['Author', 'Title', 'Document'])

train_docs = []

for book in books:
    Title = book.title
    Author = book.author
    bk = population[population['Author'] == Author]
    bk = bk[['Author', 'Title', 'Documents']]
    Documents = []
    for index, row in bk.iterrows():
        for docs in row['Documents']:
            # doc = ' '.join(docs)
            Documents.append(docs)
    Samples = random.sample(Documents, 200)
    for Document in Samples:
        train_docs.append(Document.tolist())
        df3 = pd.DataFrame([[Author, Title, Document]], columns=[
            'Author', 'Title', 'Document'])
        training = training.append(df3, ignore_index=True)

# till now, each row in training stands for one document, in total 1400 documents from 7 books

# remove words that appear only once

all_tokens = sum(train_docs, [])
tokens_once = set(word for word in set(all_tokens)
                  if all_tokens.count(word) == 1)
train_docs = [[word for word in text if word not in tokens_once]
              for text in train_docs]


# Compute bigrams
# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(train_docs)  # , min_count=20)
for idx in range(len(train_docs)):
    for token in bigram[train_docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            train_docs[idx].append(token)


# Feature Engineering
# Create a dictionary representation of the documents.
dictionary = corpora.Dictionary(train_docs)
# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.7)


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

# Train LDA model.
# Set training parameters.
# num_topics = 10


def compute_coherence_values(k):
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
        num_topics=k,
        passes=passes,
        eval_every=eval_every
    )

    top_topics = model.top_topics(bow_corpus)  # , num_words=20)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    # avg_topic_coherence = sum([t[1] for t in top_topics]) / k
    # print('Average topic coherence: %.4f.' % avg_topic_coherence)

    # cm = CoherenceModel(model=model, corpus=bow_corpus, coherence='u_mass')
    cv = CoherenceModel(model=model, texts=train_docs,
                        dictionary=dictionary, coherence='c_v')
    coherence = cv.get_coherence()

    return model, top_topics, coherence


'''
# optimal num_topic
# Topics range
min_topics = 2
max_topics = 20
step_size = 2
topics_range = range(min_topics, max_topics, step_size)


# Validation sets
num_of_docs = len(bow_corpus)
model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Coherence': []
                 }


# iterate through number of topics
for k in topics_range:
    # iterate through alpha values
    # get the coherence score for the given parameters
    model, top_topics, cv = compute_coherence_values(k=k)
    # Save the model results
    model_results['Validation_Set'].append('BOW_Corpus')
    model_results['Topics'].append(k)
    model_results['Coherence'].append(cv)

pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
opt_topic = pd.DataFrame(model_results)

star = mpath.Path.unit_regular_star(6)
circle = mpath.Path.unit_circle()
# concatenate the circle with an internal cutout of the star
verts = np.concatenate([circle.vertices, star.vertices[::-1, ...]])
codes = np.concatenate([circle.codes, star.codes])
cut_star = mpath.Path(verts, codes)

x = range(2, 20, 2)
plt.plot(x, opt_topic['Coherence'], '--m', marker=cut_star, markersize=10)
# plt.xlim(list(range(2, 20, 2)))
plt.title('Topic Coherence: Determing Optimal number of topics')
plt.xlabel('Number of topics')
plt.ylabel('Coherence C_v Score')
plt.savefig('Cohenrence.png', dpi=300, bbox_inches='tight')
plt.clf()
'''

# num_topics = 6
model, top_topics, cv = compute_coherence_values(k=6)
lda_corpus = model[bow_corpus]


'''
# pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(model, bow_corpus, dictionary=dictionary)
pyLDAvis.save_html(vis, 'lda.html')
'''


# K-Means
num_docs = dictionary.num_docs  # 1400
# print('number of documents: ', num_docs)
num_terms = len(dictionary.keys())
# corpus_bow_dense = corpus2dense(bow_corpus, num_terms, num_docs=num_docs)
# corpus_tfidf_dense = corpus2dense(tfidf_corpus, num_terms, num_docs=num_docs)
# print('Number of documents from tfidf: %d' % len(tfidf_corpus))


def matrix(corpus):

    matsim = MatrixSimilarity(corpus, num_features=len(dictionary))
    similarity_matrix = matsim[corpus]
    return similarity_matrix


# corpus_tfidf_sparse = gensim.matutils.corpus2csc(tfidf_corpus, num_terms, num_docs)
'''
silhouette = []
wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(corpus_tfidf_dense)
    wcss.append(kmeans.inertia_)
    labels = kmeans.labels_
    silhouette_score = metrics.silhouette_score(
        corpus_tfidf_dense, labels, metric='euclidean')
    silhouette.append(silhouette_score)


# plot Elbow
plt.plot(range(1, 15), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Square Distance')
plt.savefig('Elbow.png', dpi=300, bbox_inches='tight')
plt.clf()


# plot Silhouette
plt.plot(range(2, 15), silhouette)
plt.title('Silhouette Method')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.savefig('Silhouette.png', dpi=300, bbox_inches='tight')
plt.clf()
'''


# best k = 8
num_clusters = 8
kmeans = KMeans(n_clusters=8, init='k-means++',
                max_iter=300, n_init=10, random_state=0)

# evaluation of different inputs
input_set = [bow_corpus, tfidf_corpus, lda_corpus]
out_evaluation = {}
out_evaluation['input'] = ['Bag of Word', 'TF-IDF', 'LDA Model']
out_evaluation['silhouette'] = []
for input in input_set:
    similarity_matrix = matrix(input)
    kmeans.fit(similarity_matrix)
    clusters = kmeans.labels_.tolist()
    silhouette_score = metrics.silhouette_score(
        similarity_matrix, clusters, metric='euclidean')
    out_evaluation['silhouette'].append(silhouette_score)
pd.DataFrame(out_evaluation).to_csv(
    'kmean_evaluation_results.csv', index=False)

df = pd.DataFrame(out_evaluation)
# Draw plot
fig, ax = plt.subplots(figsize=(14, 10), dpi=80)
ax.vlines(x=df.index, ymin=0, ymax=df.silhouette,
          color='firebrick', alpha=0.7, linewidth=2)
ax.scatter(x=df.index, y=df.silhouette, s=75, color='firebrick', alpha=0.7)

# Title, Label, Ticks and Ylim
ax.set_title('Evaluation Chart for Text Clustering', fontdict={'size': 22})
ax.set_ylabel('Evaluation Score', fontdict={'size': 10})
ax.set_xticks(df.index)
ax.set_xticklabels(df.input.str.upper(), rotation=60, fontdict={
                   'horizontalalignment': 'right', 'size': 10})
ax.set_ylim(0, 1)

plt.savefig('Evaluation.png', dpi=300, bbox_inches='tight')
plt.clf()


# champion input
matsim = MatrixSimilarity(lda_corpus, num_features=len(dictionary))
similarity_matrix = matsim[lda_corpus]
kmeans.fit(similarity_matrix)
clusters = kmeans.labels_.tolist()
training['kmean_cluster'] = clusters


order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
# terms = dictionary.keys()

print("Top terms per cluster:")
print()

for i in range(num_clusters):
    print("Cluster %d words:" % i, end='')

    for ind in order_centroids[i, :10]:  # replace 6 with n words per cluster
        print(' %s' % dictionary[ind].encode('utf-8', 'ignore'), end=',')
    print()  # add whitespace
    print()  # add whitespace

    print("Cluster %d authors:" % i, end='')
    for author in set(training[training['kmean_cluster'] == i]['Author'].values.tolist()):
        print(' %s,' % author, end='')
    print()  # add whitespace
    print()  # add whitespace


# visualization documents clusters
matsim = MatrixSimilarity(tfidf_corpus, num_features=len(dictionary))
similarity_matrix = matsim[tfidf_corpus]

# Multidimensional scaling
MDS()
# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

pos = mds.fit_transform(similarity_matrix)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]
print()
print()


# set up colors per clusters using a dict
cluster_colors = {0: '#1b9e77', 1: '#d95f02',
                  2: '#7570b3', 3: '#e7298a', 4: '#66a61e', 5: '#85a297', 6: '#debcb0', 7: '#00688b'}

cluster_names = {0: 'Cluster 1',
                 1: 'Cluster 2',
                 2: 'Cluster 3',
                 3: 'Cluster 4',
                 4: 'Cluster 5',
                 5: 'Cluster 6',
                 6: 'Cluster 7',
                 7: 'Cluster 8'}

# group by cluster
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters,
                       title=training['Author'].tolist()))

groups = df.groupby('label')

# set up plot
fig, ax = plt.subplots(figsize=(17, 9))  # set size
ax.margins(0.05)

# iterate through groups to layer the plot
# note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=12,
            label=cluster_names[name], color=cluster_colors[name],
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='off')
    ax.tick_params(
        axis='y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='off')

ax.legend(numpoints=1)  # show legend with only 1 point

# add label in x,y position with the label as the film title
# for i in range(len(df)):
#     ax.text(df.ix[i]['x'], df.ix[i]['y'], df.ix[i]['title'], size=8)


plt.savefig('Visualization of clustering.png', dpi=300,
            bbox_inches='tight')  # show the plot
