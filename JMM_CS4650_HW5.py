import spacy
import en_core_web_lg
from newsapi import NewsApiClient
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import pickle
import pandas as pd
import string
from collections import Counter
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from matplotlib import pyplot as plt

nlp_eng = en_core_web_lg.load()
newsapi = NewsApiClient (api_key = '1203eebe322c485ca95f054f4d07d395')

articles = newsapi.get_everything(q='coronavirus', language='en', from_param='2021-10-04', to='2021-10-26', sort_by='relevancy', page=1, page_size=100) 

filename = 'articlesCOVID.pckl'
pickle.dump(articles, open(filename, 'wb'))

filename = 'articlesCOVID.pckl'
loadedModel = pickle.load(open(filename, 'rb'))

filepath = 'C:/CS4650_HW5/articlesCOVID.pckl'
pickle.dump(loadedModel, open(filepath, 'wb'))

dados=[]
for i, article in enumerate(articles):
	for x in articles['articles']:
		title = x['title']
		description = x['description']
		content = x['content']
		date = x["publishedAt"]
		dados.append({'title': title, 'date':date, 'desc': description, 'content':content, 'date':date})

df = pd.DataFrame(dados)
df = df.dropna()
df.head()

word_tags = ['VERB', 'NOUN', 'PROPN']

def getKeyWordsEng(artcl):
	doc = nlp_eng(artcl)
	result = []
	punctuation = string.punctuation
	for token in doc:
		if(token.pos_ in word_tags):
			result.append(token.text)
	return result

results = []
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
for content in df.content.values:
	results.append([('#' + x[0]) for x in Counter(getKeyWordsEng(content)).most_common(5)])

df['keywords'] = results
df.to_csv(r'C:/CS4650_HW5/articlesCOVID.pckl.csv')

text = str(results)
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color='white').generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()