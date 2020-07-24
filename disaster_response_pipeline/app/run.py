import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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
df = pd.read_sql_table('disaster_resp', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
 
    
    
    # these variables are to get a count per category visual
    # in order to sort by category count, need to get lists for the
    # category and the counts per category
    # then create a dataframe using these lists so we can sort and
    # still have values that align with each category 
    # extract each column of df as a list and save into variables
    # tha will be used in chart
    cat_names = list(df.columns[4:])
    cat_sum = [df[i].sum() for i in cat_names]

    test = pd.DataFrame({'category': cat_names,\
            'count': cat_sum})\
            .sort_values(by=['count'], ascending = False)

    cat_names = list(test['category'])
    cat_sum = list(test['count'])
    
    # Top 10 categories and their % of the total
    top10_names = list(test.iloc[0:10]['category'])
    top10_count = list(test.iloc[0:10]['count'])
    top10_count.append((sum(test.iloc[10:]['count'])))
    top10_names.append('Other')

    top10 = pd.DataFrame({'category': top10_names,\
            'count': top10_count})
    
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
                {"hole": 0.6,
                 "type": "pie",
                 "labels": genre_names,
                 "values": genre_counts
                }
            ],

            'layout': {
                'title': 'Distribution of Message Genres by %',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_names,
                    y=cat_sum
                )
            ],

            'layout': {
                'title': 'Number of Messages per Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                {"hole": 0.6,
                 "type": "pie",
                 "labels": top10_names,
                 "values": top10_count
                }
            ],
            'layout': {
                'title': 'Number of Messages per Category by % (Top 10)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
                
            }
        },
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
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()