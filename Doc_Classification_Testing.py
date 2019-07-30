"""
Created on Wed Feb 15 08:26:02 2018

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
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import model_from_json


test_data = {}
Doc_Text=[]
Doc_ID = []
Doc_name = []
files = os.listdir("testing_data")
iD=0
doc_types = ["NDA","Legal","OTH"]

if(len(files) > 0):
    #Reading all the files from testing data
    for i in files:
        if(i[-4:]!=".txt"):
            continue
        Doc_ID.append('id'+str(iD))
        Doc_name.append(i)
        iD+=1
        with open ("testing_data/"+i, "r",encoding="utf8") as myfile:
            data=myfile.readlines()
        Doc_Text.append(" ".join(data))
        
    
        
    test_data = pd.DataFrame({'Doc_ID':Doc_ID,'Doc_Text':Doc_Text,'Doc_name':Doc_name})
    
    
    # get the dimension of the Dataset { # rows , # Cols }
    # Show the top 3 records / observations
    test_data.head(3)
    
    # Feature Inspection 
    #====================
    
    # check if there's missing data , below expression will return count of Missing values
    test_data.isnull().sum()
    
    # check how many different Doc_Types we have [ are they really 3 types ]
    test_data = test_data.dropna()
    print(test_data.shape)
    
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
    
    non_ascii_words = remove_ascii_words(test_data)
    
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
    
    d2v_preprocessing(test_data)
    
    # Tokenizning text for LDA
    def lda_get_good_tokens(df):
        df['Doc_Text'] = df.Doc_Text.str.lower()
        df['tokenized_text'] = list(map(nltk.word_tokenize, df.Doc_Text))
        df['tokenized_text'] = list(map(get_good_tokens, df.tokenized_text))
    
    lda_get_good_tokens(test_data)
    
    
    tokenized_only_dict = Counter(np.concatenate(test_data.tokenized_text.values))
    tokenized_only_df = pd.DataFrame.from_dict(tokenized_only_dict, orient='index')
    tokenized_only_df.rename(columns={0: 'count'}, inplace=True)
    tokenized_only_df.sort_values('count', ascending=False, inplace=True)
    
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
    
    remove_stopwords(test_data)
    
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
    
    stem_words(test_data)
    
    # Vectorize words
    
    dictionary=Dictionary.load('model/dictionary.txtdic')
    dictionary.add_documents(test_data.stemmed_text.values)
    print("Found {} words.".format(len(dictionary.values())))
    
    #dictionary.filter_extremes(no_above=0.8, no_below=3)
    
    dictionary.compactify()  # Reindexes the remaining words after filtering
    print("Left with {} words.".format(len(dictionary.values())))
    
    
    #Make a BOW ( Bag of Words ) for every document
    def document_to_bow(df):
        df['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df.stemmed_text))
        
    document_to_bow(test_data)
    
    
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
    
    # Model Training
    #================
    
    # LDA Training
    corpus = test_data.bow
    
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
    
    test_data['lda_features'] = list(map(lambda doc:
                                          document_to_lda_features(LDAmodel, doc),
                                          test_data.bow))
    
    def get_topic_top_words(lda_model, topic_id, nr_top_words=5):
        """ Returns the top words for topic_id from lda_model.
        """
        id_tuples = lda_model.get_topic_terms(topic_id, topn=nr_top_words)
        word_ids = np.array(id_tuples)[:,0]
        words = map(lambda id_: lda_model.id2word[id_], word_ids)
        return words
    
    # Word2Vec Training 
    
    sentences = []
    for sentence_group in test_data.tokenized_sentences:
        sentences.extend(sentence_group)
    
    print("Number of sentences: {}.".format(len(sentences)))
    print("Number of texts: {}.".format(len(test_data)))
    
    
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
        
    doc2vec_input = [sentence[0] for sentence in test_data.tokenized_sentences]
    doc2vec_model = train_model_doc2vec(doc2vec_input)
    
    test_data['d2v_features'] = [doc2vec_model[i] for i in range(len(test_data))]    
    
    X_test_lda = np.array(list(map(np.array, test_data.lda_features)))    #LDA Features
    X_test_d2v = np.array(list(map(np.array, test_data.d2v_features)))    #Doc2Vec Features
    X_test_combined = np.append(X_test_lda, X_test_d2v, axis=1)          #Combined Features
    
    #Preprocessing Label
    
        
    ##############Neural Network Model To pRedict ##############
    tf.keras.backend.clear_session()
    tf.reset_default_graph()
    
    json_file = open('model/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    nn_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    nn_model.load_weights("model/model.h5")
    print("Loaded model from disk")
    
    nn_model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])    
    
    for ind in range(len(X_test_combined)):
        p = list(nn_model.predict(np.array([X_test_combined[ind]]))[0])
        doctype = p.index(max(p))
        print("\n\nFile Name :- ",test_data.Doc_name[ind])
        print("Predicted Scores are",p)
        print("Predicted Label is",doc_types[doctype])
        
else:
    print("Please add atleast 1 testing sample")