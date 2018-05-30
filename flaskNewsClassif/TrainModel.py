from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import pickle

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med', 'sci.space']

twenty_train = fetch_20newsgroups(categories=categories, shuffle=True)

count_vect = CountVectorizer()
X_new_counts = count_vect.fit_transform(twenty_train.data)

tfidf_transformer = TfidfTransformer()
X_new_tfidf = tfidf_transformer.fit_transform(X_new_counts)

text_clf = SGDClassifier(loss='hinge', penalty='l2',
                         alpha=1e-3, random_state=42,
                         max_iter=5, tol=None)
text_clf.fit(X_new_tfidf, twenty_train.target)

filename = 'finalized_model.sav'
cv = 'vectorizer'
tfidf = 'transformer'
pickle.dump(text_clf, open(filename, 'wb'))
pickle.dump(count_vect, open(cv, 'wb'))
pickle.dump(tfidf_transformer, open(tfidf, 'wb'))
