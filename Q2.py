import ssl

ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.datasets import fetch_20newsgroups

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)

tfidf_Vect = TfidfVectorizer()
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score1= metrics.accuracy_score(twenty_test.target, predicted)
print(score1)

#knn
clf1 = KNeighborsClassifier(n_neighbors=3)
clf1.fit(X_train_tfidf, twenty_train.target)
y_pred = clf1.predict(X_test_tfidf)
score2= metrics.accuracy_score(twenty_test.target, y_pred)
print(score2)

#use bigram
tfidf_Vect1=TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf1 = tfidf_Vect1.fit_transform(twenty_train.data)
clf1.fit(X_train_tfidf1,twenty_train.target)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf1 = tfidf_Vect1.transform(twenty_test.data)
predicted1= clf1.predict(X_test_tfidf1)
score3= metrics.accuracy_score(twenty_test.target, predicted1)
print(score3)

#stopwords,uninformative in representing the content of a text
tfidf_Vect = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf_Vect.fit_transform(twenty_train.data)
# print(tfidf_Vect.vocabulary_)
clf = MultinomialNB()
clf.fit(X_train_tfidf, twenty_train.target)

twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
X_test_tfidf = tfidf_Vect.transform(twenty_test.data)

predicted = clf.predict(X_test_tfidf)

score4= metrics.accuracy_score(twenty_test.target, predicted)
print(score4)