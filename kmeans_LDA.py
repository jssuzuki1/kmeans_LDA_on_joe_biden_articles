#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: by Jeff Suzuki on 8/25/2023, adapted from paulhuynh's original code
"""

import re,string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.manifold import MDS


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.model_selection import train_test_split, KFold

import pandas as pd
import os

from gensim.models import Word2Vec,LdaMulticore, TfidfModel
from gensim import corpora


from gensim.models.doc2vec import Doc2Vec, TaggedDocument


import numpy as np


# In[3]:


###############################################################################
### Function to process documents
###############################################################################
def clean_doc(doc): 
    #split document into individual words
    tokens=doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 4]
    #lowercase all words
    tokens = [word.lower() for word in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))

    # Add other words to the set of stopwords
    stop_words.add('would')
    stop_words.add('could')
    
    tokens = [w for w in tokens if not w in stop_words]         
    return tokens


# In[53]:


###############################################################################
### Processing text into lists
###############################################################################

# Get the current working directory
current_directory = os.getcwd()

# Construct the file path by joining the current directory with the file name
file_name = "2023_Summer_453_Class_Corpus.csv"
file_path = os.path.join(current_directory, file_name)

data = pd.read_csv(file_path)

#create empty list to store text documents titles
titles=[]

#for loop which appends the DSI title to the titles list
for i in range(0,len(data)):
    temp_text=data['DSI_Title'].iloc[i]
    titles.append(temp_text)


# In[7]:


## Adding this for merging purposes in the LDA step. 
data['Document'] = data['Unnamed: 0']


# In[8]:


titles


# In[9]:


#create empty list to store text documents
text_body=[]

#for loop which appends the text to the text_body list
for i in range(0,len(data)):
    temp_text=data['Text'].iloc[i]
    text_body.append(temp_text)


# In[10]:


#Note: the text_body is the unprocessed list of documents read directly form 
#the csv.
text_body


# In[11]:


#empty list to store processed documents
processed_text=[]
#for loop to process the text to the processed_text list
for i in text_body:
    text=clean_doc(i)
    processed_text.append(text)


# In[12]:


#Note: the processed_text is the PROCESSED list of documents read directly form 
#the csv.  Note the list of words is separated by commas.
processed_text


# In[13]:


#stitch back together individual words to reform body of text
final_processed_text=[]

for i in processed_text:
    temp_DSI=i[0]
    for k in range(1,len(i)):
        temp_DSI=temp_DSI+' '+i[k]
    final_processed_text.append(temp_DSI)


# In[16]:


###############################################################################
### Sklearn TFIDF 
###############################################################################
#note the ngram_range will allow you to include multiple words within the TFIDF matrix
#Call Tfidf Vectorizer
Tfidf=TfidfVectorizer(ngram_range=(1,1))

#fit the vectorizer using final processed documents.  The vectorizer requires the 
#stiched back together document.

TFIDF_matrix=Tfidf.fit_transform(final_processed_text)     

#creating datafram from TFIDF Matrix
matrix=pd.DataFrame(TFIDF_matrix.toarray(), columns=Tfidf.get_feature_names_out(), index=titles)


# In[17]:


###############################################################################
### Explore TFIDF Values
###############################################################################

average_TFIDF={}
for i in matrix.columns:
    average_TFIDF[i]=np.mean(matrix[i])

average_TFIDF_DF=pd.DataFrame(average_TFIDF,index=[0]).transpose()

average_TFIDF_DF.columns=['TFIDF']

#calculate Q1 and Q3 range
Q1=np.percentile(average_TFIDF_DF, 25)
Q3=np.percentile(average_TFIDF_DF, 75)
IQR = Q3 - Q1
outlier=Q3+(1.5*IQR)


#words that exceed the Q3+IQR*1.5
outlier_list=average_TFIDF_DF[average_TFIDF_DF['TFIDF']>=outlier]


# In[20]:


###############################################################################
### Doc2Vec
###############################################################################
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(final_processed_text)]
model = Doc2Vec(documents, vector_size=100, window=2, min_count=1, workers=4)

doc2vec_df=pd.DataFrame()
for i in range(0,len(processed_text)):
    vector=pd.DataFrame(model.infer_vector(processed_text[i])).transpose()
    doc2vec_df=pd.concat([doc2vec_df,vector], axis=0)

doc2vec_df=doc2vec_df.reset_index()

doc_titles={'title': titles}
t=pd.DataFrame(doc_titles)

## I reversed the order of this to make more sense to me.
doc2vec_df=pd.concat([t, doc2vec_df], axis=1)

doc2vec_df=doc2vec_df.drop('index', axis=1)


# In[21]:


###############################################################################
### Gensim Word2vec 
###############################################################################

#Note, there are opportunities to use the word2vec matrix to determine words 
#which are similar.  Similar words can be used to create equivalent classes.  
#k-means is not used to group individual words using the Word2Vec output.

#word to vec on the processed text
model_w2v = Word2Vec(processed_text, vector_size=100, window=5, min_count=1, workers=4)

#join all processed DSI words into single list
processed_text_w2v=[]
for i in processed_text:
    for k in i:
        processed_text_w2v.append(k)
        
w2v_words=list(set(processed_text_w2v))


# In[23]:


#empty dictionary to store words with vectors
w2v_vectors={}

#for loop to obtain weights for each word
for i in w2v_words:
    temp_vec=model_w2v.wv[i]
    w2v_vectors[i]=temp_vec


# In[24]:


#create a final dataframe to view word vectors
w2v_df=pd.DataFrame(w2v_vectors).transpose()

w2v_df

#the following section runs applies the k-means algorithm on the TFIDF matrix.


# In[27]:


###############################################################################
### K Means Clustering - TFIDF
###############################################################################
k = 6
n_init = 100

km = KMeans(n_clusters=k, random_state=89, n_init=n_init)
km.fit(TFIDF_matrix)
clusters = km.labels_.tolist()

print('TFIDF clusters:', clusters)


# In[29]:


terms = Tfidf.get_feature_names_out()
Dictionary={'Doc Name':titles, 'Cluster':clusters,  'Text': final_processed_text}
frame=pd.DataFrame(Dictionary, columns=['Cluster', 'Doc Name','Text']).sort_values(by=['Cluster', 'Doc Name'])


# In[31]:


# print("Top terms per cluster:")
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 


# In[32]:


## Initialize terms_dict
terms_dict=[]

#save the terms for each cluster and document to dictionaries.  To be used later
#for plotting output.

#dictionary to store terms and titles
cluster_terms={}
cluster_title={}


# In[35]:


## Loop to append the cluster terms
for i in range(k):
    print("TFIDF Cluster Terms %d:" % i),
    temp_terms=[]
    temp_titles=[]
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind])
        terms_dict.append(terms[ind])
        temp_terms.append(terms[ind])
    cluster_terms[i]=temp_terms
    
    print("TFIDF Cluster %d titles:" % i, end='')
    temp=frame[frame['Cluster']==i]
    for title in temp['Doc Name']:
        print(' %s,' % title, end='')
        temp_titles.append(title)
    cluster_title[i]=temp_titles


# In[40]:


# Create a list of dictionaries for DataFrame
data = [{'Cluster Number': cluster_num, 'Terms': ', '.join(terms)} for cluster_num, terms in cluster_terms.items()]

# Create a DataFrame from the list of dictionaries
cluster_terms_df = pd.DataFrame(data)

# Export the DataFrame to CSV
cluster_terms_df.to_csv(file_path, index=False)


# In[41]:


###############################################################################
### Plotting
###############################################################################

# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

dist = 1 - cosine_similarity(TFIDF_matrix)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]


# In[42]:


#set up colors per clusters using a dict.  number of colors must correspond to K
cluster_colors = {
    0: 'blue',         # Blue
    1: 'green',        # Green
    2: 'purple',       # Purple
    3: 'orange',       # Orange
    4: 'red',          # Red
    5: 'cyan',         # Cyan
}

#set up cluster names using a dict.  
cluster_dict=cluster_title

#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters, title=range(0,len(clusters)))) 

#group by cluster
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(15, 15)) # set size
# gs = gridspec.GridSpec(2, 1, height_ratios=[20, 1])
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=15,
            label=cluster_dict[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='on')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='on')
    
# Set up the legend with adjusted font size
legend = ax.legend(loc='lower center', bbox_to_anchor=(2, -0.2), labelspacing=5)

# Loop through legend labels and adjust font size
for label in legend.get_texts():
    label.set_text('\n'.join(label.get_text().split()))
    label.set_fontsize(30)  # Adjust the font size here
    
# Increase the font size of the axes labels
ax.tick_params(axis='both', which='major', labelsize=40)  # Adjust the font size here

# Save the plot as a high-resolution JPG image
plt.savefig('tfidf_cluster_plot.jpg', dpi=500, bbox_inches='tight') 

# Show the plot
plt.show()


# In[44]:


#The following section of code is to run the k-means algorithm on the doc2vec outputs.
#note the differences in document clusters compared to the TFIDF matrix.
###############################################################################
### K Means Clustering Doc2Vec
###############################################################################
doc2vec_k_means=doc2vec_df.drop('title', axis=1)

n_init = 100
k = 6
km = KMeans(n_clusters=k, random_state =89, n_init = n_init)
km.fit(doc2vec_k_means)
clusters_d2v = km.labels_.tolist()

Dictionary={'Doc Name':titles, 'Cluster':clusters_d2v,  'Text': final_processed_text}
frame=pd.DataFrame(Dictionary, columns=['Cluster', 'Doc Name','Text']).sort_values(by=['Cluster', 'Doc Name'])

frame


# In[45]:


frame['Cluster']


# In[46]:


#note doc2vec clusters will not have individual words due to the vector representation
#is based on the entire document not indvidual words. As a result, there won't be individual
#word outputs from each cluster.   
for i in range(k):
    temp=frame[frame['Cluster']==i]
    temp_title_list=[]
    for title in temp['Doc Name']:
        temp_title_list.append(title)
    cluster_title[i]=temp_title_list

cluster_title


# In[474]:


###############################################################################
### Plotting Doc2vec
###############################################################################
# convert two components as we're plotting points in a two-dimensional plane
# "precomputed" because we provide a distance matrix
# we will also specify `random_state` so the plot is reproducible.

## set parameter
mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)

## ? what ?
dist = 1 - cosine_similarity(doc2vec_k_means)

pos = mds.fit_transform(dist)  # shape (n_components, n_samples)

xs, ys = pos[:, 0], pos[:, 1]

#set up cluster names using a dict.  
cluster_dict=cluster_title    

cluster_dict


# In[475]:


#create data frame that has the result of the MDS plus the cluster numbers and titles
df = pd.DataFrame(dict(x=xs, y=ys, label=clusters_d2v, title=range(0,len(clusters)))) 

df


# In[477]:


#set up colors per clusters using a dict.  number of colors must correspond to K
cluster_colors = {
    0: 'blue',         # Blue
    1: 'green',        # Green
    2: 'purple',       # Purple
    3: 'orange',       # Orange
    4: 'red',          # Red
    5: 'cyan',         # Cyan
}

#group by cluster
groups = df.groupby('label')

fig, ax = plt.subplots(figsize=(15, 15)) # set size
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling

#iterate through groups to layer the plot
#note that I use the cluster_name and cluster_color dicts with the 'name' lookup to return the appropriate color/label
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=15,
            label=cluster_dict[name], color=cluster_colors[name], 
            mec='none')
    ax.set_aspect('auto')
    ax.tick_params(\
        axis= 'x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelbottom='on')
    ax.tick_params(\
        axis= 'y',         # changes apply to the y-axis
        which='both',      # both major and minor ticks are affected
        left='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        labelleft='on')
    
# Set up the legend with adjusted font size
legend = ax.legend(loc='lower center', bbox_to_anchor=(2, -0.2), labelspacing=5)

# Loop through legend labels and adjust font size
for label in legend.get_texts():
    label.set_text('\n'.join(label.get_text().split()))
    label.set_fontsize(30)  # Adjust the font size here
    
# Increase the font size of the axes labels
ax.tick_params(axis='both', which='major', labelsize=40)  # Adjust the font size here

# Save the plot as a high-resolution JPG image
plt.savefig('Doc2Vec.jpg', dpi=500, bbox_inches='tight') 

# Show the plot
plt.show()


# In[47]:


###############################################################################
###  LDA Code
###############################################################################

#LDA using bag of words
dictionary = corpora.Dictionary(processed_text)
corpus = [dictionary.doc2bow(doc) for doc in processed_text]

ldamodel = LdaMulticore(corpus, num_topics=6, id2word=dictionary, passes=2, workers=2)    

for idx, topic in ldamodel.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[35]:


# #LDA using TFIDF
# tfidf = TfidfModel(corpus)
# corpus_tfidf = tfidf[corpus]

# ldamodel = LdaMulticore(corpus_tfidf, num_topics=8, id2word=dictionary, passes=100, workers=2)    

# for idx, topic in ldamodel.print_topics(-1):
#     print('Topic: {} \nWords: {}'.format(idx, topic))


# In[48]:


# Assuming you have already trained your LDA model and named it 'ldamodel'
# Assuming 'num_words' is the number of words you want to display for each topic

# Get the top words for each topic
topics_words = ldamodel.show_topics(num_topics=-1, formatted=False)

topics_words


# In[52]:


## This is what I was actually looking for. 
keywords = pd.DataFrame(topics_words, columns= ['Topic Number', 'Keywords'])

topic_labels = {
    0: "COVID",
    1: "Immigration",
    2: "Debt Limit",
    3: "Age",
    4: "Russian Sanctions",
    5: "Climate"}

keywords['Topic'] = topic_labels

new_order = ['Topic Number', 'Topic', 'Keywords']
keywords = keywords[new_order]

# Export the DataFrame to CSV
keywords.to_csv(file_path, index=False)


# In[426]:


# Get topic distribution for each document
document_topics = [ldamodel.get_document_topics(doc) for doc in corpus]

# Print the topics assigned to each document
for doc_id, topics in enumerate(document_topics):
    print(f"Document {doc_id}:")
    for topic_id, topic_prob in topics:
        print(f"Topic {topic_id}: Probability {topic_prob:.4f}")
    print()


# In[427]:


for idx, topic in ldamodel.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))


# In[428]:


import pandas as pd

document_topics = [ldamodel.get_document_topics(doc) for doc in corpus]
a = []

for doc_id, topics in enumerate(document_topics):
    for topic_id, topic_prob in topics:
        a.append([doc_id, f"Topic {topic_id}", f"Probability {topic_prob:.4f}"])
        
result = pd.merge(LDA_doc_assignments, data['Document'], left_index=True, right_index=True)
        
LDA_doc_assignments = pd.DataFrame(a, columns=["Document", "Topic", "Probability"])
# LDA_doc_assignments['DSI_Title'] = titles
# LDA_doc_assignments
# LDA_doc_assignments.sort_values(by=['Topic', 'DSI_Title'])

result = pd.merge(LDA_doc_assignments, data[['Document', 'DSI_Title']], on='Document').sort_values(by=['Topic Number', 'DSI_Title'])


# In[431]:


## Show the duplicate entries to show the 'soft' aspect of topic mapping. 
duplicates = result[result.duplicated('DSI_Title', keep=False)]


# In[60]:


# Export the DataFrame to CSV
duplicates.to_csv(file_path, index=False)

