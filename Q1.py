import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize,wordpunct_tokenize,sent_tokenize
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.util import ngrams
from nltk import ne_chunk
import requests

url = "https://en.wikipedia.org/wiki/Google"
source = requests.get(url)
html_text = source.text
soup = BeautifulSoup(html_text, "html.parser")
text=soup.get_text()
# print (text)
with open('input.txt','w',encoding='utf-8') as output:
    output.write(text)

#Tokenization
print("this is for Tokenization")
input_tokenizer = nltk.data.load('input.txt')
sentences = sent_tokenize(input_tokenizer)
print(sentences)
print('\n')
words = word_tokenize(input_tokenizer)
print(words)
print('\n')


#
# # # Stemming
print("this is for Stemming")
stemmer = SnowballStemmer("english")
output_stemming = stemmer.stem(text)
print(output_stemming)
print('\n')

# # # #POS
print("this is for POS")

output_POS = []
output_POS.append(nltk.pos_tag(words))
print (output_POS)
print('\n')


# # #
# # #Lemmatization
print("this is for Lemmatization")
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
lemmatization=[]
for word in words:
    lemmatization.append(lemmatizer.lemmatize(word))
# output_lemmatization = lemmatizer.lemmatize(words)
print (lemmatization)
print('\n')

# #
# #Trigram
print("this is for Trigram")
trigrams = list(ngrams(words,3))
print(trigrams)
print('\n')

# # # #Named Entity Recognition
print("this is for Named Entity Recognition(NER)")

sentence = "The Spring Web MVC framework provides Model View Controller architecture and ready components that can be used to develop flexible and loosely coupled web applications"
print(ne_chunk(pos_tag(wordpunct_tokenize(sentence))))
