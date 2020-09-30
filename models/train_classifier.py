import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV


import pickle


def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("DisasterResponse",engine)
    X = df["message"]
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names


def tokenize(text):
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    url_place_holder = "urlplaceholder"
    
    # use english dictionary for stopwords
    stop_words = stopwords.words("english")
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    
    # Replace url with a url placeholder string
    for detected_url in detected_urls:
        text = text.replace(detected_url, url_place_holder)

    # Extract the word tokens from the provided text
    tokens = nltk.word_tokenize(text)
    
    #Lemmanitizer to remove inflectional and derivationally related forms of a word
    lemmatizer = nltk.WordNetLemmatizer()

    # List of clean tokens without any stop words
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens if w not in stop_words]
    return clean_tokens

def build_model():
    vect = CountVectorizer(tokenizer=tokenize)
    tfidf = TfidfVectorizer(smooth_idf=False)

    pipeline  = Pipeline([
        ('features', FeatureUnion([
            ('vect', vect),
            ('tfidf', tfidf)])
        ),
        ('clf', MultiOutputClassifier(RandomForestClassifier())),
        ])
    
    # some possible hyperparameters, most of them not used due to the long run time
    parameters = {'features__vect__ngram_range': ((1, 1), (1, 2)),
    #'vect__max_df': (0.75, 1.0),
    #'vect__max_features': (None, 5000),
    #'tfidf__use_idf': (True, False),
    'clf__estimator__n_estimators': [50, 75],#,100],
    'clf__estimator__min_samples_split': [2, 4]}
    
    # use gridsearch to find best hyperparameters
    cv = GridSearchCV(pipeline, param_grid=parameters,verbose = 2, n_jobs = -1, cv=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    y_pred_pd = pd.DataFrame(y_pred, columns = Y_test.columns)
    for column in Y_test.columns:
        print('********\n')
        print('FEATURE: {}\n'.format(column))
        print(classification_report(Y_test[column],y_pred_pd[column]))


def save_model(model, model_filepath):
    file = open(model_filepath,'wb')
    pickle.dump(model, file)


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