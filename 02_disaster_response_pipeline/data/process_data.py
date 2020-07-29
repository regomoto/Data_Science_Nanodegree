import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function to load message data and category data 
    from csv files. Merge the data in a dataframe
    
    Return a merged dataframe
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.join(categories.set_index('id'), on='id')
    return df


def clean_data(df):
    '''
    Function to transform raw data so
    that it can be used to train a multi output
    classifier. 
    '''
    # make dataframe with only values from the categories column
    # convert result from string to int
    # get the right most value from string
    # since either 1 or 0
    df_wip = df.categories.str.split(';', expand = True)
    for col in df_wip:
        df_wip[col] = df_wip[col].str[-1].astype('int')

    # get column names from category column
    # split each item in the list on '-'
    # left of dash is column name, right is 1 or 0
    # split outputs to a list with length = 2
    # so take first item in list as column name
    cols = [i.split('-')[0] for i in df['categories'][0].split(';')]

    # assign new column names
    df_wip.columns = cols

    # append work in progress dataframe and the df
    df = pd.concat([df, df_wip], axis = 1)

    # drop categories column
    df.drop(columns = ['categories'], inplace = True)

    #drop dupliacted rows
    df.drop_duplicates(inplace = True)

    # replace values in 'related' column so there is
    # binary output instead of 3 labels
    df['related'].replace({2: 0}, inplace = True)

    # drop the child_alone column, since it did not have
    # binary values. Only has 0's in training set
    
    df = df.drop('child_alone', axis = 1)

    return df

def save_data(df, database_filename):
    '''
    Function to write transformed data 
    to a database that can be used for model
    training
    '''
    # write to sql database
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_resp', engine, if_exists = 'replace', index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()