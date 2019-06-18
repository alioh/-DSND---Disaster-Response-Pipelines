import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import pickle
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

nltk.download(['punkt', 'wordnet'])


def load_data(database_filepath):
    """loads data

    Args:
    database_filepath: string. location of the database file.

    Returns:
    X = features
    Y = targets
    category_names = categories/targets in a list.
    """

    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(database_filepath.split('/')[-1], engine)
    # split the data to X= Features = messages
    #                   Y= Targets
    X = df.message
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    # Get all column names
    category_names = []
    for i in Y.columns:
        category_names.append(i)
    return X, Y, category_names


def tokenize(text):
    """tokenize text

    Args:
    text: string. text to tokenize.

    Returns:
    final_token = list of a text after tokenize.
    """

    # Regex: find anything that is not character or number
    # reference: https://stackoverflow.com/a/27772817
    text_regex = '[^0-9a-zA-Z]+'
    find_regex = re.findall(text_regex, text)
    # also remove all empty items in the list.
    final_regex = [x for x in find_regex if x != ' ']
    for rgx in find_regex:
        text = text.replace(rgx, ' ')
    # tokenize and lemmatize text
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    # final output after lowercasing/stripping
    final_token = []
    for t in tokens:
        clean = lemmatizer.lemmatize(t).lower().strip()
        final_token.append(clean)

    return final_token


def build_model():
    """build model

    Args:
    no args

    Returns:
    cv = the result of building and finding the best parameters
    """

    # pipeline, use randomforest model
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # use grid search to find the best parameters
    parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            'vect__max_features': (None, 5000, 1000),
            'tfidf__use_idf': (True, False),
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    # best partams: (cv.best_params_)
    #       {'tfidf__use_idf': False,
    #       'vect__max_df': 0.75,
    #       'vect__max_features': 1000}
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """evaluate my model

    Args:
    model: object. model that we want to evaluate.
    X_test: dataframe. features.
    Y_test: dataframe. targets.
    category_names: list. targets in a list.

    Returns:
    cv = the result of building and finding the best parameters
    """

    y_pred = model.predict(X_test)

    # using classification_report to find
    # precision / recall / f1-score / support
    # for each column
    for col in range(0, len(category_names)):
        print(category_names[col])
        print(classification_report(Y_test[category_names[col]], y_pred[:, col]))


def save_model(model, model_filepath):
    """save the model

    Args:
    model: object. model that we want to save.
    model_filepath: string. where we want to save the model.

    Returns:
    no return.
    """

    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
