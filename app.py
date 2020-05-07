from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # df = pd.read_csv("train.csv")
    # X = df['text']
    # y = df['label']
    #
    # # Extract Feature With CountVectorizer
    # cv = CountVectorizer(max_features=5000, ngram_range=(1, 3))
    # X = cv.fit_transform(X)  # Fit the Data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # clf = MultinomialNB()
    # clf.fit(X_train,y_train)
    # print(clf.score(X_test,y_test))
    #
    # joblib.dump(clf, 'NB_spam_model.pkl')
    # pickle.dump(cv, open("vector.pickel", "wb"))
    cv = pickle.load(open("vector.pickel", "rb"))

    NB_spam_model = open('NB_spam_model.pkl', 'rb')
    clf = joblib.load(NB_spam_model)

    if request.method == 'POST':
        message = request.form['message'].lower()
        data = [message]
        vect = cv.transform(data).toarray()
        prediction = clf.predict(vect)
    return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)
