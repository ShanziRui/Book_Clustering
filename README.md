# Book_Clustering
Use digital books from Gutenberg Library to train clustering models based on book contents



The Operation Flow
===================


Setting up
------------

Download the submission folder on BrightSpace which contains 1 report document, 1 README file and 3 folders.

- Open the README file
- Unzip the "python_files" folder which contains 7 plain txt files and 4 python files.
- Open terminal, locate the directory of the submission folder, assume the directory is on your Desktop.
	cd /Users/catherinerui/Desktop/python_files
- Make sure "python_files" contains contains "load.py" together with other three .py files
- Run kmeans_clustering.py using Python3
	python3 kmeans_clustering.py
- The main body of our process is in kmeans_clustering.py file, this file is the main file we use to prepare data and run three algorithms

- The em_clustering.py file and heirarchical_clustering.py file expand the modelling part for EM algorithm and hierarchical algorithm with further visualization and evaluation steps

- Kappa folder contains materials used for calculating kappa score, the "training_result.csv" contains all clustering results from three algorithms and it is one of the output file by running "kmeans_clustering.py"


Output Results
-----------------

The whole program takes a couple minutes to finish, here is a list of file output you will get.

- 7 folders, named by the corresponding book title, each contains the separated txt file of all chapters in that book.
	e.g. "pride-and-prejudice" contains 61 chapters in total, the program creates 61 txt files for each chapter, stores in the folder named "pride-and-prejudice-chapters"

- A visualization figure of kmeans clustering result with lda as input (lda has the best performance) named "pca_visualization.png"
	To obtain the visualization of other inputs, you can change line 348 to similarity_matrix = matrix(tfidf_corpus) or similarity_matrix = matrix(bow_corpus)

- A visualization figure of evaluation result for kmeans in terms of silhouette score named "Evaluation.png"

- A csv file of kmeans evaluation result in terms of silhouette score named "kmean_evaluation_results.csv"

- A csv file of training dataframe with three columns of assigned cluster number from each algorithm result named "training_result.csv"



Hidden Results
-----------------

Parts of the codes are commented out in order to make result clear.

- line 195 to 240 was used for determining the optimal number of topics for LDA topic modelling, the result figure is in the visualization folder

- line 247 to 251 was used for visualizing LDA modelling result in terms of a interactive page, the result is in the visualization folder named "lda.html"

- line 271 to 301 was used for determining the best k for clustering algorithms, the result figure is in the visualization folder

- line 412 to 477 was used for multidimensional scaling of clustering results, the visualization result is in the visualization folder



Program prints
-----------------

The output information printed in order:


1.

************ Top 7 most frequently downloaded books ************

A Tale of Two Cities: Charles Dickens
The Adventures of Sherlock Holmes: Arthur Conan Doyle
Pride and Prejudice: Jane Austen
Frankenstein: Mary Wollstonecraft (Godwin) Shelley
Little Women: Louisa May Alcott
Moby Dick; or The Whale: Herman Melville
Adventures of Huckleberry Finn: Mark Twain (Samuel Clemens)



2.

************ Population size of each book ************
A Tale of Two Cities 454
The Adventures of Sherlock Holmes 355
Pride and Prejudice 376
Frankenstein 251
Little Women 643
Moby Dick; or The Whale 724
Adventures of Huckleberry Finn 362


3.

Number of unique tokens: 1812
Number of documents: 1400


4.

Top terms per cluster:

Cluster 0 words: b'heard', b'north', b'histori', b'part', b'rose', b'fli', b'could_help', b'share', b'condemn', b'possess',

Cluster 0 authors: Arthur Conan Doyle, Charles Dickens, Herman Melville, Louisa May Alcott,

Cluster 1 words: b'her', b'fellow', b'lorri', b'woman', b'face', b'deni', b'shook', b'a', b'seen', b'let',

Cluster 1 authors: Mary Wollstonecraft (Godwin) Shelley, Charles Dickens, Arthur Conan Doyle, Jane Austen, Louisa May Alcott,

Cluster 2 words: b'tide', b'home', b'vain', b'success', b'spite', b'notic', b'toil', b'command', b'consent', b'lead',

Cluster 2 authors: Herman Melville, Charles Dickens, Arthur Conan Doyle, Louisa May Alcott, Mark Twain (Samuel Clemens),

Cluster 3 words: b'rate', b'chain', b'therefor', b'aspect', b'oh', b'burst', b'deserv', b'consol', b'shine', b'opinion',

Cluster 3 authors: Arthur Conan Doyle, Charles Dickens, Herman Melville,

Cluster 4 words: b'remark', b'like', b'express', b'absolut', b'natur', b'this', b'pari', b'unusu', b'save', b'effect',

Cluster 4 authors: Arthur Conan Doyle, Charles Dickens, Jane Austen,

Cluster 5 words: b'dog', b'have', b'told', b'short', b'smoke', b'hide', b'shot', b'everybodi', b'wretch', b'explain',

Cluster 5 authors: Arthur Conan Doyle, Charles Dickens, Mary Wollstonecraft (Godwin) Shelley, Louisa May Alcott,

Cluster 6 words: b'feet', b'head', b'way', b'mean', b'stern', b'dear', b'nevertheless', b'almost', b'care', b'eye',

Cluster 6 authors: Arthur Conan Doyle, Charles Dickens, Mary Wollstonecraft (Godwin) Shelley, Louisa May Alcott,

Cluster 7 words: b'of', b'steadi', b'certain', b'least', b'citi', b'probabl', b'flower', b'minut', b'prison', b'quiet',

Cluster 7 authors: Herman Melville, Mary Wollstonecraft (Godwin) Shelley, Charles Dickens, Arthur Conan Doyle, Louisa May Alcott,

