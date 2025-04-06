import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk import word_tokenize, sent_tokenize
import re
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV
import pickle
from joblib import parallel_backend


def load_data(database_filepath):
    """
    :param database_filepath:
    :return:
    """
    engine = create_engine('sqlite:///' + database_filepath)
    with engine.connect() as conn, conn.begin():
        df = pd.read_sql_table("CleanData", conn)
    X = df['message']
    Y = df[df.columns.difference(['id', 'message', 'original', 'genre'])]
    category_names = df.columns.difference(['id', 'message', 'original', 'genre'])
    return X, Y, category_names


def tokenize(text):
    """
    :param text:
    :return:
    """
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    cleaned_tokens = [WordNetLemmatizer().lemmatize(w) for w in words]
    return cleaned_tokens


def build_model():
    """
    :return:
    """
    pipeline = Pipeline(
        [
            ('features', CountVectorizer(tokenizer=tokenize)),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ]
    )
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    :param model:
    :param X_test:
    :param Y_test:
    :param category_names:
    :return:
    """
    Y_pred = model.predict(X_test)
    Y_pred_np = np.array(Y_pred)
    Y_test_np = Y_test.values
    for i in range(len(category_names)):
        print(classification_report(Y_test_np[:, i], Y_pred_np[:, i]))
    return


def save_model(model, model_filepath):
    """
    :param model:
    :param model_filepath:
    :return:
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    return


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        with parallel_backend('threading', n_jobs=6):
            model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        best_pipeline = model.best_estimator_
        save_model(best_pipeline, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
