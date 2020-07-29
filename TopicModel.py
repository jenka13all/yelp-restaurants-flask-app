#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:06:47 2020

@author: jennifer
"""
import re
from wordcloud import WordCloud

# Gensim
import gensim
import spacy
import gensim.corpora as corpora
from gensim.utils import simple_preprocess

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(
    ['food', 'restaurant', 'staff', 'place', 'order', 'table',
     'meal', 'ive', 'weve', 'server', 'waiter', 'waitress']
)


class TopicModel():
    def sent_to_words(sentences):
        for sent in sentences:
            sent = re.sub('\s+', ' ', sent)  # remove newline chars
            sent = re.sub("\'", "", sent)  # remove single quotes
            sent = gensim.utils.simple_preprocess(str(sent), deacc=True) 
            yield(sent)
            
    def process_words(texts, stop_words=stop_words, allowed_postags=['NOUN', 'ADJ']):
        #bigram model
        bigram = gensim.models.Phrases(texts, min_count=5, threshold=100) # higher threshold fewer phrases.
        bigram_mod = gensim.models.phrases.Phraser(bigram)

        #remove stopwords
        texts = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
    
        #add bigrams
        texts = [bigram_mod[doc] for doc in texts]
    
        #lemmatize
        texts_out = []
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
        for sent in texts:
            doc = nlp(" ".join(sent)) 
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        
        # remove stopwords once more after lemmatization
        texts_out = [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts_out]    
        return texts_out
    
    def make_lda_model(self, data_df, business_id):
        num_topics = 2
        data = data_df.text.values.tolist()
    
        #create tokens
        data_words = list(TopicModel.sent_to_words(data))
    
        #process tokens
        data_ready = TopicModel.process_words(data_words, stop_words=stop_words) 
        
        # Create Dictionary
        id2word = corpora.Dictionary(data_ready)
    
        #filter out words that appear in less than 5 documents - get rid of misspellings
        no_below = 5
        id2word.filter_extremes(no_below=no_below)
    
        # Create Corpus: Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in data_ready]
        
        #build LDA model
        lda_model = gensim.models.ldamodel.LdaModel(
            corpus=corpus,
            id2word=id2word,
            num_topics=num_topics, 
            random_state=100,
            update_every=1,
            chunksize=10,
            passes=10,
            alpha='symmetric',
            iterations=100,
            per_word_topics=True)
        
        return lda_model
    
    def make_wordclouds(self, business_id, lda_model):
        num_topics = 2
        
        filenames = [
            'static/images/cloud_' + business_id + '_0.png',
            'static/images/cloud_' + business_id + '_1.png',
        ]
        
        cloud = WordCloud(
            background_color='black',
            width=300,
            height=150,
            colormap='Set2',
            prefer_horizontal=1.0
        )
    
        topics = lda_model.show_topics(formatted=False)
        
        for i in range(0, num_topics):
            partial_filename = filenames[i]
            topic_words = dict(topics[i][1])
            cloud.generate_from_frequencies(topic_words).to_file(partial_filename)
    
        
        return filenames