##
#	An adaptation of Ahmed Besbes' Sentiment analysis on Twitter using word2vec and keras
##

##
#	Environment set-up 
#
##
import pandas as pd #provides sql-like data manipulation tools
pd.options.mode.chained_assignment = None

import numpy as np #high dimensional vector computing library
from copy import deepcopy
from string import punctuation
from random import shuffle

import gensim
from gensim.models.word2vec import Word2Vec # the world2vec model gensim class
LabeledSentence = gensim.models.doc2vec.LabeledSentence

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer # a tweet tokenizer from nltk
tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from keras.models import Sequential
from keras.layers import Activation, Dense

##
#	Environment set-up and data preparation
#
##

def ingest(filename):
	#reads in csv files of tweets and their respective sentiment values
	#
	data = pd.read_csv(filename, engine='python')
	data.drop(['ItemID', 'Date','Blank','SentimentSource'], axis=1, inplace=True)
	data = data[data.Sentiment.isnull() == False]
	data['Sentiment'] = data['Sentiment'].map({4:1,0:0})
	data = data[data['SentimentText'].isnull() == False]
	data.reset_index(inplace=True)
	data.drop('index', axis=1, inplace=True)
	print ('dataset loaded with shape',data.shape)
	return data

def tokenize(tweet):
	try:
	  #tweet = tweet.lower().split()
	  #tokens = tweet.lower().split()
	  #tweet = tweet.decode('utf-8').lower()
	  #tweet = tweet.lower()
	  tokens = tweet.lower().split()
	  tokens = tokenizer.tokenize(tweet)
	  #print(tokens)
	  tokens = list(filter(lambda t: not t.startswith('@'), tokens))
	  tokens = list(filter(lambda t: not t.startswith('#'), tokens))
	  tokens = list(filter(lambda t: not t.startswith('http'), tokens))
	  return tokens
	except:
	  return 'NC'


def postprocess(data, n=120000):
	data = data.head(n)
	#print('just called head n')
	#print(data)
	data['tokens'] = data['SentimentText'].progress_map(tokenize)##progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
	#print('just called tokenize')
	#print(data)
	data = data[data.tokens != 'NC']
	print('just pruned errors')
	#print(data)
	data.reset_index(inplace=True)
	#print('just called reset index')
	#print(data)
	data.drop('index', inplace=True, axis=1)
	#print(data)
	return data

##
# Building the word2vec model 
#
##

def labelizeTweets(tweets, label_type):
	labelized = []
	for i,v in tqdm(enumerate(tweets)):
	  label = '%s_%s'%(label_type,i)
	  labelized.append(LabeledSentence(v, [label]))
	return labelized

def buildWordVector(tokens, size):
    #vec = np.zeros(size).reshape((1, size))
    vec2 = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            #vec += tweet_w2v[word].reshape((1, size)) * tfidf[word]
            vec2 += tweet_w2v2[word].reshape((1, size)) * tfidf[word]
            count += 1
	    #count += 1
        except KeyError: # handling the case where the token is not
                         # in the corpus. useful for testing.
            continue
    if count != 0:
        #vec /= count
        vec2 /= count
    #return vec
    return vec2


#### ~~~~~~~~~~ ####


#### ~~~~~~~~~~ ####


#### ~~~~~~~~~~ ####


#### Main Progr ####
if __name__ == '__main__':

	n_dim = 200
	n = 120000
	min_count = 10

	filename2 = './tweets2.csv'


	data2 = ingest(filename2)
	data2 = postprocess(data2, n)
	#x_train, x_test, y_train, y_test, = train_test_split(np.array(data.head(n).tokens), np.array(data.head(n).Sentiment), test_size = 0.3)


	x_train2, x_test2, y_train2, y_test2, = train_test_split(np.array(data2.head(n).tokens), np.array(data2.head(n).Sentiment), test_size = 0.3)

	#x_train = labelizeTweets(x_train, 'TRAIN')
	#x_test = labelizeTweets(x_test, 'TEST')

	
	x_train2 = labelizeTweets(x_train2, 'TRAIN')
	x_test2 = labelizeTweets(x_test2, 'TEST')

	
#	tweet_w2v = Word2Vec(size = n_dim, min_count=10)
#	tweet_w2v.build_vocab([x.words for x in tqdm(x_train)])
#	tweet_w2v.train([x.words for x in tqdm(x_train)],total_examples=tweet_w2v.corpus_count,epochs=tweet_w2v.iter)
	
	tweet_w2v2 = Word2Vec(size = n_dim, min_count=5)
	tweet_w2v2.build_vocab([x.words for x in tqdm(x_train2)])
	tweet_w2v2.train([x.words for x in tqdm(x_train2)],total_examples=tweet_w2v2.corpus_count,epochs=tweet_w2v2.iter)
	
	#try:
	#print(tweet_w2v['the'])
	#print(tweet_w2v.most_similar('gay'))
	#print(tweet_w2v.most_similar('queer'))
	#print(tweet_w2v.most_similar('homosexual'))
	#print(tweet_w2v.most_similar('fag'))
	#print(tweet_w2v.most_similar('homo'))
	#print(tweet_w2v.most_similar('lesbian'))
	#print(tweet_w2v.most_similar('dyke'))
	#ABprint(tweet_w2v.most_similar('transgender'))
	#print(tweet_w2v.most_similar('trans'))
	#print(tweet_w2v.most_similar('lgbt'))
	#tweet_w2v.most_similar('bar')
	#tweet_w2v.most_similar('the')
	
	import bokeh.plotting as bp
	from bokeh.models import HoverTool, BoxSelectTool
	from bokeh.plotting import figure, show, output_file, save

	#defining the chart

	#plot_tfdif = bp.figure(plot_width=700, plot_height=600, title="A map of 1000 word vectors", tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",x_axis_type=None, y_axis_type=None, min_border=1)
	#plot_tfidf = bp.figure(plot_width=800, plot_height=800, title="A map of \d word vectors"%len(word_vectors),tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",x_axis_type=None, y_axis_type=None, min_border=1)

	#getting a list of word vectors. limit to 10000. each is of 200 dimension
	word_vectors = [tweet_w2v2[w] for w in tweet_w2v2.wv.vocab.keys()]
	print(len(word_vectors))
	plot_tfidf = bp.figure(plot_width=800, plot_height=800, title="A map of 11000 word vectors",tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",x_axis_type=None, y_axis_type=None, min_border=1)

	#dimensionality reduction. converting the vectors to 2d vectors
	from sklearn.manifold import TSNE
	tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
	tsne_w2v = tsne_model.fit_transform(word_vectors)

	#putting everything in a dataframe
	tsne_df = pd.DataFrame(tsne_w2v, columns=['x','y'])
	tsne_df['words'] = tweet_w2v2.wv.vocab.keys()

	#plotting the corresponding word appearts when you hover on the data point.
	plot_tfidf.scatter(x='x', y='y', source=tsne_df)
	hover = plot_tfidf.select(dict(type=HoverTool))
	hover.tooltips={"word": "@words"}
	output_file("twitAB.html")
	save(plot_tfidf)
	
	if True:

		print ('building tf-idf matrix ...')
		vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
		matrix = vectorizer.fit_transform([x.words for x in x_train2])
		tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
		print ('vocab size :', len(tfidf))

		from sklearn.preprocessing import scale
		train_vecs_w2v2 = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train2))])
		train_vecs_w2v2 = scale(train_vecs_w2v2)
		test_vecs_w2v2 = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test2))])
		test_vecs_w2v2 = scale(test_vecs_w2v2)

		model = Sequential()
		model.add(Dense(32, activation='relu', input_dim=200))
		model.add(Dense(1, activation='sigmoid'))
		model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

		model.fit(train_vecs_w2v2, y_train2, epochs=9, batch_size=32, verbose=2)

		score = model.evaluate(test_vecs_w2v2, y_test2, batch_size=64, verbose=2)
		print (score[1])
