#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pandas as pd
import numpy as np
import re
import time

from sqlalchemy import create_engine
from config import db_password


# In[2]:


file_dir = '/Users/admin/Desktop/Data_Analytics_BootCamp/Challenge8_Movies-ETL'


# In[3]:


with open(f'{file_dir}/wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)

len(wiki_movies_raw)


# In[4]:


kaggle_metadata = pd.read_csv(f'{file_dir}/movies_metadata.csv', low_memory=False)
ratings = pd.read_csv(f'{file_dir}/ratings.csv')


# In[5]:


wiki_movies_df = pd.DataFrame(wiki_movies_raw)
wiki_movies_df.head()


# In[6]:


wiki_movies_df.columns.tolist()


# In[7]:


# using list comprehensions for filter data
wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie 
                   and 'No. of episodes' not in movie]
len(wiki_movies)


# In[8]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune–Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles
    # merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')

    return movie        

clean_movies = [clean_movie(movie) for movie in wiki_movies]


# In[9]:


wiki_movies_df[wiki_movies_df['Arabic'].notnull()]


# In[10]:


wiki_movies_df[wiki_movies_df['Arabic'].notnull()]['url']


# In[11]:


wiki_movies_df = pd.DataFrame(clean_movies)
sorted(wiki_movies_df.columns.tolist())


# In[12]:


clean_movies = [clean_movie(movie) for movie in wiki_movies]
wiki_movies_df = pd.DataFrame(clean_movies)

len(wiki_movies_df)


# In[13]:


#Removing duplicated data

wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))

wiki_movies_df.head()


# In[14]:


# finding the null columns
null_values = [[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]

null_values


# In[15]:


# Remove mostly null columns
wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]
wiki_movies_df.columns


# In[16]:


# make a data series that drops missing values
box_office = wiki_movies_df['Box office'].dropna() 
box_office


# In[17]:


def is_not_a_string(x):
    return type(x) != str

box_office[box_office.map(lambda x: type(x) != str)]


# In[18]:


box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)
box_office


# In[19]:


form_one = r'\$\d+\.?\d*\s*[mb]illi?on'
box_office.str.contains(form_one, flags=re.IGNORECASE).sum()


# In[20]:


form_two = r'\$\d{1,3}(?:,\d{3})+'
box_office.str.contains(form_two, flags=re.IGNORECASE).sum()


# In[21]:


matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)


# In[22]:


box_office[~matches_form_one & ~matches_form_two]


# In[23]:


form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
box_office.str.extract(f'({form_one}|{form_two})')


# In[24]:


def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan


# In[25]:


# extract the values from box_office using str.extract
wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', 
                                                      flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[26]:


wiki_movies_df.drop('Box office', axis=1, inplace=True)


# In[27]:


budget = wiki_movies_df['Budget']


# In[28]:


# Parse Budget Data
budget = wiki_movies_df['Budget'].dropna()
budget


# In[29]:


# Convert any lists to strings
budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)


# In[30]:


# remove any values between a dollar sign and a hyphen 
budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[31]:


# Parse box budget data
matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matches_form_one & ~matches_form_two]


# In[32]:


wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[33]:


wiki_movies_df.drop('Budget', axis=1, inplace=True)


# In[34]:


# Parse Release Date
release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[35]:


date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'


# In[36]:


wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], 
                                                infer_datetime_format=True)
wiki_movies_df


# In[37]:


# Parse Running Time
running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
running_time


# In[38]:


running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE).sum()


# In[39]:


running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]


# In[40]:


running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()


# In[41]:


running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]


# In[42]:


running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')


# In[43]:


running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)


# In[44]:


wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 
                                                            else row[2], 
                                                            axis=1)


# In[45]:


wiki_movies_df.drop('Running time', axis=1, inplace=True)


# In[46]:


kaggle_metadata.dtypes


# In[47]:


kaggle_metadata['adult'].value_counts()


# In[48]:


# Remove Bad Data
kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]


# In[49]:


kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')


# In[50]:


kaggle_metadata['video'].value_counts()


# In[51]:


kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'


# In[52]:


kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')


# In[53]:


kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])


# In[54]:


# rating data
ratings.info(null_counts=True)


# In[55]:


pd.to_datetime(ratings['timestamp'], unit='s')


# In[56]:


ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')


# In[57]:


ratings['rating'].plot(kind='hist')
ratings['rating'].describe()


# In[58]:


# merge Wikipedia and Kaggle Metadata
movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])


# In[59]:


# merge title
movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]


# In[60]:


# merge running_time
movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')


# In[61]:


# budget
movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')


# In[62]:


# Box Office
movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')


# In[63]:


# Release Date
movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')


# In[64]:


movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]


# In[65]:


movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') 
                                     & (movies_df['release_date_kaggle'] < '1965-01-01')].index)


# In[66]:


movies_df[movies_df['release_date_wiki'].isnull()]


# In[67]:


# Language
movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)


# In[68]:


movies_df['original_language'].value_counts(dropna=False)


# In[69]:


movies_df[['Production company(s)','production_companies']]


# In[70]:


# drop the title_wiki, release_date_wiki, Language, and Production company(s) columns
movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)


# In[71]:


def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)


# In[72]:


fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df


# In[73]:


# check the merged data if there's any columns with only one value. 
for col in movies_df.columns:
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 1:
        print(col)


# In[74]:


movies_df['video'].value_counts(dropna=False)


# In[75]:


# reorder the columns
movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection',
                              'url','imdb_link','runtime','budget_kaggle','revenue','release_date_kaggle',
                              'popularity','vote_average','vote_count','genres','original_language','overview',
                              'spoken_languages','Country','production_companies','production_countries',
                              'Distributor','Producer(s)','Director','Starring','Cinematography','Editor(s)',
                              'Writer(s)','Composer(s)','Based on']
                         ]


# In[76]:


# rename the columns
movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)


# In[77]:


# creating pivot that movieId becomes the index
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)                 .pivot(index='movieId',columns='rating', values='count')


# In[78]:


# prepend rating_ to each column with a list comprehension
rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]


# In[79]:


# using left merge/join for movie_df and rating_counts
movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')


# In[80]:


# fill-in 0 for movies that got zero rating
movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)


# In[81]:


# connection to local server
db_string = f"postgres://postgres:{db_password}@127.0.0.1:5433/movie_data_challenge"


# In[82]:


# create the database engine
engine = create_engine(db_string)


# In[83]:


# import movie data
movies_df.to_sql(name='movies', con=engine)


# In[ ]:


# import the Ratings Data using chunsize= parameter in read_csv()
# create a variable for the number of rows imported
rows_imported = 0

# get the start_time from time.time()
start_time = time.time()

for data in pd.read_csv(f'{file_dir}/ratings.csv', chunksize=1000000):

    # print out the range of rows that are being imported
    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    
    data.to_sql(name='ratings', con=engine, if_exists='append')

    # increment the number of rows imported by the chunksize
    rows_imported += len(data)
    
    # print that the rows have finished importing
    print(f'Done. {time.time() - start_time} total seconds elapsed')
