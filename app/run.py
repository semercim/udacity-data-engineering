import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sklearn.preprocessing import FunctionTransformer
from sqlalchemy import create_engine

app = Flask(__name__)


def count_words(X):
    return X.sum(axis=1)


count_word_transformer = FunctionTransformer(count_words)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('CleanData', engine)

# load model
model = joblib.load("../models/classifier_testing.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    multiple_classes = df.loc[:, ~df.columns.isin(['id', 'message', 'original', 'genre'])]
    class_counts = multiple_classes.sum(axis=1).value_counts()

    languages_equals = pd.DataFrame(data=(df['message'] == df['original']), columns=['SameLanguage'])

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=class_counts.index,
                    y=class_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Multilabel Messages',
                'yaxis': {
                    'title': "Counts of Messages"
                },
                'xaxis': {
                    'title': "Counts of Labels"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=['Original', 'Translation'],
                    y=[sum(languages_equals['SameLanguage']),
                       len(languages_equals['SameLanguage']) - sum(languages_equals['SameLanguage'])]
                )
            ],

            'layout': {
                'title': 'Translation or Original',
                'yaxis': {
                    'title': "Language"
                },
                'xaxis': {
                    'title': "Counts"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict(pd.DataFrame(data=[query], columns=['message']))[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
