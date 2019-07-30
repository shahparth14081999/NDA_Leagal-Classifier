"""
Created on Wed Feb 13 15:00:02 2018

@author: niravshah and jaydeepthik
"""
# Import all the required libraries / packages 
#=============================================

# common libraries imports
import pandas as pd
import numpy as np
from collections import Counter
import re
from gensim.models.doc2vec import TaggedDocument
# languange processing imports
import nltk
from gensim.corpora import Dictionary
# model imports
import gensim
from gensim.models.ldamulticore import LdaMulticore
from gensim.models import LdaModel
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import base64
import io,os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

# get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()  # defines the style of the plots to be seaborn style

# read the data into Python Environment at this step 
# Expected input file is a CSV file with 3 columns 
#      Column 1 : Document Id                [Doc_ID]
#      Column 2 : Document Text / content    [Doc_Text]
#      column 3 : Type of Document           [Doc_Type]
#======================================================

# Step 1 - read the Input files for training data

train_data = {}
Doc_Type=[]
Doc_Text=[]
Doc_ID = []

doc_types = ["NDA","Legal"]

sufficient_data = True
iD=0
for i in doc_types:
    files = os.listdir("training_data/"+i)
    no=0
    for j in files:
        if(j[-4:]!=".txt"):
            continue
        no+=1
        Doc_ID.append('id'+str(iD))
        iD+=1
        Doc_Type.append(i)
        with open ("training_data/"+i+"/"+j, "r") as myfile:
            data=myfile.readlines()
        Doc_Text.append(" ".join(data))
    if(no < 1):
        sufficient_data=False
        
if(sufficient_data):
        
    train_data = pd.DataFrame({'Doc_ID':Doc_ID,'Doc_Text':Doc_Text,'Doc_Type':Doc_Type})
    #train_data=pd.read_excel("master_train.xls")
    # get the dimension of the Dataset { # rows , # Cols }
    # Show the top 3 records / observations
    train_data.head(3)
    
    # Feature Inspection 
    #====================
    
    # check if there's missing data , below expression will return count of Missing values
    train_data.isnull().sum()
    
    # check how many different Doc_Types we have [ are they really 3 types ]
    train_data.Doc_Type.value_counts().index
    train_data = train_data.dropna()
    print(train_data.shape)
    
    # Visualization of Data 
    #=======================
    
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    
    Doc_Type_vc = train_data.Doc_Type.value_counts()
    
    ax.bar(range(len(Doc_Type_vc)), Doc_Type_vc)
    ax.set_xticks(range(len(Doc_Type_vc)))
    ax.set_xticklabels(Doc_Type_vc.index, fontsize=16)
    
    for rect, c, value in zip(ax.patches, ['b', 'r', 'g'], Doc_Type_vc.values):
        rect.set_color(c)
        height = rect.get_height()
        width = rect.get_width()
        x_loc = rect.get_x()
        ax.text(x_loc + width/2, 0.9*height, value, ha='center', va='center', fontsize=18, color='white')
    
    # Lets now inspect the 'Doc_Text' variable
    #==========================================
    
    # Split the strings of texts into individual words 
    document_lengths = np.array(list(map(len, train_data.Doc_Text.str.split(' '))))
    
    print("The average number of words in a document is: {}.".format(np.mean(document_lengths)))
    print("The minimum number of words in a document is: {}.".format(min(document_lengths)))
    print("The maximum number of words in a document is: {}.".format(max(document_lengths)))
    
    # Visualization of the 'Words Ditribution in Documents'
    fig, ax = plt.subplots(figsize=(15,6))
    
    ax.set_title("Distribution of number of words", fontsize=16)
    ax.set_xlabel("Number of words")
    sns.distplot(document_lengths, bins=50, ax=ax);
    
    
    # Feature Creation
    #==================
    
    # Tokenize the Text
    
    # find and remove non-ascii words
    # I stored our special word in a variable for later use
    our_special_word = 'qwerty'
    
    def remove_ascii_words(df):
        """ removes non-ascii characters from the 'texts' column in df.
        It returns the words containig non-ascii characers.
        """
        non_ascii_words = []
        for i in range(len(df)):
            for word in df.loc[i, 'Doc_Text'].split(' '):
                if any([ord(character) >= 128 for character in word]):
                    non_ascii_words.append(word)
                    df.loc[i, 'Doc_Text'] = df.loc[i, 'Doc_Text'].replace(word, our_special_word)
        return non_ascii_words
    
    non_ascii_words = remove_ascii_words(train_data)
    
    print("Replaced {} words with characters with an ordinal >= 128 in the train data.".format(
        len(non_ascii_words)))
    
    
    def get_good_tokens(sentence):
        replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-z!?]+', '', token), sentence))
        removed_punctation = list(filter(lambda token: token, replaced_punctation))
        return removed_punctation
    
    
    # Here we get transform the documents into sentences for the word2vecmodel
    # we made a function such that later on when we make the submission, we don't need to write duplicate code
    
    
    def d2v_preprocessing(df):
        # All the preprocessing steps for word2vec are done in this function.
        #All mutations are done on the dataframe itself. So this function returns
        #nothing.
        
        df['Doc_Text'] = df.Doc_Text.str.lower()
        df['document_sentences'] = df.Doc_Text.str.split('.')  # split texts into individual sentences
        df['tokenized_sentences'] = list(map(lambda sentences:
                                             list(map(nltk.word_tokenize, sentences)),
                                             df.document_sentences))  # tokenize sentences
        df['tokenized_sentences'] = list(map(lambda sentences:
                                             list(map(get_good_tokens, sentences)),
                                             df.tokenized_sentences))  # remove unwanted characters
        df['tokenized_sentences'] = list(map(lambda sentences:
                                             list(filter(lambda lst: lst, sentences)),
                                             df.tokenized_sentences))  # remove empty lists
    
    d2v_preprocessing(train_data)
    
    # Tokenizning text for LDA
    def lda_get_good_tokens(df):
        df['Doc_Text'] = df.Doc_Text.str.lower()
        df['tokenized_text'] = list(map(nltk.word_tokenize, df.Doc_Text))
        df['tokenized_text'] = list(map(get_good_tokens, df.tokenized_text))
    
    lda_get_good_tokens(train_data)
    
    
    tokenized_only_dict = Counter(np.concatenate(train_data.tokenized_text.values))
    tokenized_only_df = pd.DataFrame.from_dict(tokenized_only_dict, orient='index')
    tokenized_only_df.rename(columns={0: 'count'}, inplace=True)
    tokenized_only_df.sort_values('count', ascending=False, inplace=True)
    
    
    # We made a function out of this since we will use it again later on 
    def word_frequency_barplot(df, nr_top_words=50):
        """ df should have a column named count.
        """
        fig, ax = plt.subplots(1,1,figsize=(20,5))
    
        sns.barplot(list(range(nr_top_words)), df['count'].values[:nr_top_words], palette='hls', ax=ax)
    
        ax.set_xticks(list(range(nr_top_words)))
        ax.set_xticklabels(df.index[:nr_top_words], fontsize=14, rotation=90)
        return ax
        
    ax = word_frequency_barplot(tokenized_only_df)
    ax.set_title("Word Frequencies", fontsize=16);
    
    # Remove Words with little or no meaning
    
    def remove_stopwords(df):
        """ Removes stopwords based on a known set of stopwords
        available in the nltk package. In addition, we include our
        made up word in here.
        """
        # Luckily nltk already has a set of stopwords that we can remove from the texts.
        stopwords = nltk.corpus.stopwords.words('english')
        # we'll add our own special word in here 'qwerty'
        stopwords.append(our_special_word)
    
        df['stopwords_removed'] = list(map(lambda doc:
                                           [word for word in doc if word not in stopwords],
                                           df['tokenized_text']))
    
    remove_stopwords(train_data)
    
    # Stemming
    
    def stem_words(df):
        lemm = nltk.stem.WordNetLemmatizer()
        df['lemmatized_text'] = list(map(lambda sentence:
                                         list(map(lemm.lemmatize, sentence)),
                                         df.stopwords_removed))
    
        p_stemmer = nltk.stem.porter.PorterStemmer()
        df['stemmed_text'] = list(map(lambda sentence:
                                      list(map(p_stemmer.stem, sentence)),
                                      df.lemmatized_text))
    
    stem_words(train_data)
    
    # Vectorize words
    
    dictionary = Dictionary(documents=train_data.stemmed_text.values)
    dictionary.save('model/dictionary.txtdic')
    print("Found {} words.".format(len(dictionary.values())))
    
    #dictionary.filter_extremes(no_above=0.8, no_below=3)
    
    dictionary.compactify()  # Reindexes the remaining words after filtering
    print("Left with {} words.".format(len(dictionary.values())))
    
    
    #Make a BOW ( Bag of Words ) for every document
    def document_to_bow(df):
        df['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df.stemmed_text))
        
    document_to_bow(train_data)
    
    
    # we make a function such that later on when we make the submission, we don't need to write duplicate code
    def lda_preprocessing(df):
        """ All the preprocessing steps for LDA are combined in this function.
        All mutations are done on the dataframe itself. So this function returns
        nothing.
        """
        lda_get_good_tokens(df)
        remove_stopwords(df)
        stem_words(df)
        document_to_bow(df)
    
    cleansed_words_df = pd.DataFrame.from_dict(dictionary.token2id, orient='index')
    cleansed_words_df.rename(columns={0: 'id'}, inplace=True)
    
    cleansed_words_df['count'] = list(map(lambda id_: dictionary.dfs.get(id_), cleansed_words_df.id))
    del cleansed_words_df['id']
    
    cleansed_words_df.sort_values('count', ascending=False, inplace=True)
    
    
    ax = word_frequency_barplot(cleansed_words_df)
    ax.set_title("Document Frequencies (Number of documents a word appears in)", fontsize=16);
    
    
    # Model Training
    #================
    
    # LDA Training
    corpus = train_data.bow
    
    num_topics = 150
    #A multicore approach to decrease training time
    LDAmodel = LdaModel(corpus=corpus,
                            id2word=dictionary,
                            num_topics=num_topics,
                            chunksize=4000,
                            passes=7,
                            alpha='asymmetric')
    
    def document_to_lda_features(lda_model, document):
        """ Transforms a bag of words document to features.
        It returns the proportion of how much each topic was
        present in the document.
        """
        topic_importances = LDAmodel.get_document_topics(document, minimum_probability=0)
        topic_importances = np.array(topic_importances)
        return topic_importances[:,1]
    
    train_data['lda_features'] = list(map(lambda doc:
                                          document_to_lda_features(LDAmodel, doc),
                                          train_data.bow))
    
    def get_topic_top_words(lda_model, topic_id, nr_top_words=5):
        """ Returns the top words for topic_id from lda_model.
        """
        id_tuples = lda_model.get_topic_terms(topic_id, topn=nr_top_words)
        word_ids = np.array(id_tuples)[:,0]
        words = map(lambda id_: lda_model.id2word[id_], word_ids)
        return words
        
    for Type in zip(list(train_data.Doc_Type.unique())):
        print("Looking up top words from top topics from {}.".format(Type))
        for x in sorted(np.argsort(train_data.loc[train_data.Doc_Type == Type, 'lda_features'].mean())[-5:]):
            top_words = get_topic_top_words(LDAmodel, x)
            print("For topic {}, the top words are: {}.".format(x, ", ".join(top_words)))
        print("")
    
    
    def show_image(base64_encoded_image):
        """ Decodes a base64 encoded image and plots it.
        """
        fig, ax = plt.subplots(figsize=(10,10))
    
        decoded_image = base64.b64decode(base64_encoded_image)
        img = io.BytesIO(decoded_image)
        img = mpimg.imread(img, format='PNG')
    
        ax.imshow(img)
        ax.axis('off');
    
    
    # Word2Vec Training 
    
    sentences = []
    for sentence_group in train_data.tokenized_sentences:
        sentences.extend(sentence_group)
    
    print("Number of sentences: {}.".format(len(sentences)))
    print("Number of texts: {}.".format(len(train_data)))
    
    
    # Function to get doc2vec embeddings from the document list
    def train_model_doc2vec(doc_list):
        tagged = []
        i=0
        for doc in doc_list:
            tagged.append(TaggedDocument(words=doc, tags=[i]))
            i+=1
            
        model = gensim.models.Doc2Vec(dm=0, alpha=0.025, vector_size=25, min_count=1, epochs=80)
        model.build_vocab(tagged)
        model.train(tagged, total_examples=model.corpus_count, epochs = model.epochs)
        return model   
        
    doc2vec_input = [sentence[0] for sentence in train_data.tokenized_sentences]
    doc2vec_model = train_model_doc2vec(doc2vec_input)
    
    train_data['d2v_features'] = [doc2vec_model[i] for i in range(len(train_data))]
    
    X_train_lda = np.array(list(map(np.array, train_data.lda_features)))    #LDA Features
    X_train_d2v = np.array(list(map(np.array, train_data.d2v_features)))    #Doc2Vec Features
    X_train_combined = np.append(X_train_lda, X_train_d2v, axis=1)          #Combined Features
    
    #Preprocessing Labels
    
    y_data = train_data.Doc_Type
    hots=y_data.unique()
    hots_num= np.array(range(len(hots)))
    
    for i in range(len(hots)): 
      y_data=y_data.replace(hots[i],hots_num[i])
        
    y_data = np_utils.to_categorical(y_data, len(hots))

    X_train, X_test, y_train, y_test = train_test_split(X_train_combined, y_data, test_size=0.2, random_state=38)  #Splitting Data into Training and Testing Data
    
    ##############Neural Network Model To pRedict ##############
    tf.keras.backend.clear_session()
    tf.reset_default_graph()
    nn_model = keras.Sequential([
        keras.layers.Dense(50,activation=tf.nn.relu, input_shape=[X_train.shape[1]]),
        keras.layers.Dense(20,activation=tf.nn.relu),
        keras.layers.Dense(len(hots), activation=tf.nn.softmax)])
    
    nn_model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    nn_model.fit(X_train, y_train,epochs=20)
    
    ##############Testing Trained Model######################
    test_score = nn_model.evaluate(X_test,y_test)
    print("Test Score Based on 30% testing data is ",test_score[1])
    
    model_json = nn_model.to_json()
    with open("model/model.json", "w+") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    nn_model.save_weights("model/model.h5")
    
    print("Saved Trained Model Successfully")
else:
    print("Please add 1 text sample of each for training(recommended atleast 50 samples each to get good accuracy")