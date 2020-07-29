import sys
import libraries
from sqlalchemy import create_engine
import os
import pandas as pd
import re
import numpy as np
import timeit
from pickle import dump

# NLP imports
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

#scikit learn imports
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

# may need to download certain ntlk packages
nltk.download(['punkt','stopwords','wordnet'])

def load_data(database_filepath):
    '''
    Function to load data from the sqllite database
    and return X and Y for model training
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    # create dataframe using table in db
    df = pd.read_sql_table('disaster_resp',engine) 
    # split into dependent and independent variables
    X = df['message']
    # target variable is all columns that have category data
    Y = df.iloc[:,4:]

    

def tokenize(text):
    '''
    Function to tokenize text
    '''

    # normalize: lowercase and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize using words
    tokens = word_tokenize(text)
    
    # remove stop words
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    
    #lemmatize words
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(tok).strip() for tok in tokens]
    
    return clean_tokens


def build_model():
    '''
    Build a classification model. Contains a pipeline
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(SGDClassifier()))
    ])
    
    params = {  'clf__estimator__bootstrap': [True, False],
            'clf__estimator__n_estimators': [10, 20], 
            'clf__estimator__max_features': ['log2','auto'],
            'clf__estimator__criterion': ['entropy', 'gini'], 
             }
    cv = GridSearchCV(pipeline, param_grid=params, scoring='precision_samples', cv = None)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Using test data from data partition, evaluate model using 
    micro f1 scores due to imbalance in dataset
    '''
    
    Y_pred = model.predict(X_test)

    
    


def save_model(model, model_filepath):
    dump(model, open(model_filepath, 'wb'))


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()