import sys

from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sqlalchemy import create_engine
import pandas as pd
from nltk import word_tokenize
import re
from sklearn.pipeline import Pipeline, FeatureUnion
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
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
    :param text: the message to be tokenized
    :return: a list of lists containing the tokenized messages
    """
    text = text.lower().strip()
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    cleaned_tokens = [WordNetLemmatizer().lemmatize(w) for w in words]
    return cleaned_tokens


def count_words(X):
    """
    :param X: the matrix whose the number unique tokens in each row is to be counted
    :return: a vector with the token counts in each row
    """
    return (X>0).sum(axis=1)


count_word_transformer = FunctionTransformer(count_words)


def build_model():
    """
    :return: a pipelined cv classifier
    """

    # Include the step to extract the features as a pipeline
    features = Pipeline(
        steps=[
            ('vectorizer', TfidfVectorizer(tokenizer=tokenize, norm=None)),
            (
                'united_features', FeatureUnion(
                    [
                        ('pass', 'passthrough'),
                        ('text_length', count_word_transformer)
                    ]
                )
            ),
            ('features_scaler', StandardScaler(with_mean=False)),  # Scale the values
        ],
        verbose=True
    )

    # Combined the feature pipeline with the classifier
    pipeline = Pipeline(
        [
            ('features', features),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ],
        verbose=True
    )

    # the example set os parameters to be search
    parameters = {
        'clf__estimator__n_estimators': [25, 50, 100]
    }

    # create the cross validator with corresponding grid search parameters.
    cv = GridSearchCV(pipeline, param_grid=parameters, pre_dispatch='8*n_jobs')

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    :param model: model to be tested
    :param X_test: the training data
    :param Y_test: the labels of the training data
    :param category_names: the names of the labels
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
    :param model: model to be saved
    :param model_filepath: the path to save the model
    :return:
    """
    pickle.dump(model, open(model_filepath, 'wb'))
    return


def main():
    """
    The main steps to follow
    :return:
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
        print('Best parameter set: \n', model.best_params_)

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
