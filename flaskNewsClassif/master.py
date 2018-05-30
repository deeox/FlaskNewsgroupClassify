from flask import Flask, render_template, redirect, request

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
import pickle

app = Flask(__name__)


@app.route('/results')
def correct(result):
    return render_template("results.html", result=result)


@app.route('/')
def home():
    return render_template("homepage.html")


@app.route('/', methods=['POST'])
def homepage():
    text_clf = pickle.load(open('finalized_model.sav', 'rb'))
    count_vect = pickle.load(open('vectorizer', 'rb'))
    tfidf = pickle.load(open('transformer', 'rb'))

    str1 = str(request.form['article'])
    article = list()
    article.append(str1)

    X_new_counts = count_vect.transform(article)

    X_new_tfidf = tfidf.transform(X_new_counts)
    print("1")

    predicted = text_clf.predict(X_new_tfidf)

    categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med', 'sci.space']
    twenty_train = fetch_20newsgroups(categories=categories, shuffle=True)
    for doc, category in zip(article, predicted):
        result = twenty_train.target_names[category]
        print(result)
    redirect('/results')

    return correct(result)





