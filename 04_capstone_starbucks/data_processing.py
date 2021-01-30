'''
Purpose: Create an data processing script that takes in all of the 
JSON files for the starbucks data and automate the cleaning steps.
Cleaning steps based on exploratory data analysis that was initially
performed during first run of this project.


Input:
portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
profile.json - demographic data for each customer
transcript.json - records for transactions, offers received, offers viewed, and offers completed


Output:

portfolio.json - transformed portfolio file

profile.json - transformed profile file

transcript.json - transformed transcript file


'''
import pandas as pd
import numpy as np
import math
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, date
from sklearn.preprocessing import MultiLabelBinarizer


def portfolio_transform(portfolio):
    '''
    transform the portfolio file dataframe and output a 
    new dataframe with changes needed to complete analysis
    
    INPUT
    portfolio - raw portfolio file
    
    OUTPUT: 
    portfolio - the portfolio file with data transformations
    '''
    # hot encode offer_type
    offer_temp = pd.get_dummies(portfolio['offer_type'])
    portfolio = pd.concat([portfolio, offer_temp], axis = 1, sort = False)
    
    # convert duration to hours for consistency with other datasets
    portfolio['duration'] = portfolio['duration']*24
    
    # hot encode channels
    mlb = MultiLabelBinarizer(sparse_output=True)
    portfolio = portfolio.join(
                pd.DataFrame.sparse.from_spmatrix(
                    mlb.fit_transform(portfolio.pop('channels')),
                    index=portfolio.index,
                    columns=mlb.classes_))
    
    return portfolio


def profile_transform(profile):
    '''
    transform the profile file dataframe and output a 
    new dataframe with changes needed to complete analysis
    
    INPUT
    profile - raw portfolio file
    
    OUTPUT: 
    profile - the portfolio file with data transformations
    '''
    # change people with age = 118 to age = 200 since most likely incorrect age entered
    profile['age'] = profile['age'].apply(lambda x: np.NaN if x == 118 else x)

    # make a generation column based on age
    # generation calculations
    year_curr = datetime.today().year
    boomer_age_upper = year_curr - 1946
    boomer_age_lower = year_curr - 1964
    genx_age_upper = year_curr - 1965
    genx_age_lower= year_curr -1980
    millennial_age_upper = year_curr - 1981
    millennial_age_lower = year_curr - 1996
    genz_age_upper = year_curr - 1997
    genz_age_lower = year_curr - year_curr
    silent_age_upper = year_curr - 1928
    silent_age_lower = year_curr - 1945

    # using date of birth cutoffs, create a new column that states generations
    profile.loc[profile['age'] >= silent_age_lower, 'Age_Gen'] = 'Silent'
    profile.loc[((profile['age'] >= boomer_age_lower) & \
                  (profile['age'] <= boomer_age_upper)), 'Age_Gen'] = 'Boomers'
    profile.loc[((profile['age'] >= genx_age_lower) & \
                  (profile['age'] <= genx_age_upper)), 'Age_Gen'] = 'Generation X'
    profile.loc[((profile['age'] >= millennial_age_lower) & \
                  (profile['age'] <= millennial_age_upper)), 'Age_Gen'] = 'Millenials'
    profile.loc[((profile['age'] >= genz_age_lower) & \
                  (profile['age'] <= genz_age_upper)), 'Age_Gen'] = 'Generation Z'



    # change column to date time. make new column
    profile['new_became_member_on'] = pd.to_datetime(profile['became_member_on'], format='%Y%m%d')
    
    # create a new column that contains how long a person has been a customer
    profile['member_len'] = (datetime.now() - profile['new_became_member_on']).dt.days
    
    return profile


def transcript_transform(transcript, portfolio):
    '''
    transform the transcript file dataframe and output a 
    new dataframe with changes needed to complete analysis
    
    INPUT
    transcript - raw portfolio file
    portfolio - transformed portfolio to join with transcript
    
    OUTPUT: 
    transcript - a transformed transcript log file with corresponding offer data from portfolio data
    '''
   
    # execute self join using tested transformation from previous code block
    # to make the dictionary in the value column into individual columns
    transcript = transcript.join(transcript['value'].apply(pd.Series), how = 'outer')
    # there are two similar columns: 'offer id' and 'offer_id'
    # combine the two columns to have only one column with these values
    transcript['offer_id2'] = transcript['offer_id']
    transcript['offer_id2'].fillna(transcript['offer id'], inplace = True)
    transcript.drop(['offer id', 'offer_id'], axis = 1, inplace = True)
    # rename column to 'offer_id'
    transcript.rename(columns = {'offer_id2': 'offer_id'}, inplace = True)
    
    # merge transcript and portfolio so each offer's info is also in dataset
    transcript = transcript.merge(portfolio, left_on = 'offer_id', right_on = 'id', how = 'left')
    # drop redundant columns
    transcript = transcript.drop(['value', 'id', 'offer_type'], axis = 1)

    return transcript