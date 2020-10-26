import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ast 
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
import warnings; warnings.simplefilter('ignore')


%matplotlib inline

movies_df = pd.read_csv('movies_metadata.csv')
ratings = pd.read_csv('ratings_small.csv')
movies_df.columns

movies_df = movies_df.drop(['belongs_to_collection','budget','homepage','original_language','release_date','revenue','runtime','spoken_languages','status','video','poster_path','production_companies','production_countries'], axis = 1)
# movies_df.head()

# ratings.head()

movies_df['genres'] = movies_df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
# print (movies_df['genres'].head())

#vote_count
V = movies_df[movies_df['vote_count'].notnull()]['vote_count'].astype('float')

#vote_counts
R = movies_df[movies_df['vote_average'].notnull()]['vote_average'].astype('float')

# this is C
C = R.mean()
M = V.quantile(0.95)
df = pd.DataFrame()
df = movies_df[(movies_df['vote_count'] >= M) & (movies_df['vote_average'].notnull())][['title','vote_count','vote_average','popularity','genres','overview']]
df.shape

df['Weighted_average'] = ((R*V) + (C*M))/(V+M)
recm_movies = df.sort_values('Weighted_average', ascending=False).head(500)
# print (recm_movies.head())

import matplotlib.pyplot as plt
import seaborn as sns

weight_average=recm_movies.sort_values('Weighted_average',ascending=False)
plt.figure(figsize=(12,6))
axis1=sns.barplot(x=recm_movies['Weighted_average'].head(10), y=recm_movies['title'].head(10), data=recm_movies)
plt.xlim(4, 10)
plt.title('Best Movies by weighted_average', weight='bold')
plt.xlabel('Weighted Average Score', weight='bold')
plt.ylabel('Movie Title', weight='bold')

popular = pd.DataFrame()
popular = recm_movies.copy()
popular['popularity'] = recm_movies[recm_movies['popularity'].notnull()]['popularity'].astype('float')
popular = popular.sort_values('popularity',ascending = False)

# print (popular.head())

plt.figure(figsize=(12,6))
axis1=sns.barplot(x=popular['popularity'].head(10), y=popular['title'].head(10), data=popular)
plt.xlim(4, 350)
plt.title('Best Movies by Popularity', weight='bold')
plt.xlabel('Popularity', weight='bold')
plt.ylabel('Movie Title', weight='bold')

s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'genre'
gen_movies = recm_movies.drop('genres', axis=1).join(s)
# print (gen_movies.head(10))
#gen_movies.columns

df_w = gen_movies[ (gen_movies['genre'] == 'Action') & (gen_movies['vote_count'] >= M)]
df_w.sort_values('Weighted_average', ascending = False).head(10)

df_w = df_w.sort_values('Weighted_average', ascending = False)
plt.figure(figsize=(12,6))
axis1=sns.barplot(x=df_w['Weighted_average'].head(10), y=df_w['title'].head(10), data=df_w)
plt.xlim(4, 10)
plt.title('Best Action Movies by weighted average', weight='bold')
plt.xlabel('Weighted Average Score', weight='bold')
plt.ylabel('Action Movie Title', weight='bold')

gen_md = gen_movies
df_w = gen_md[ (gen_md['genre'] == 'Drama') & (gen_md['vote_count'] >= M)]

df_w = df_w.sort_values('Weighted_average', ascending = False)
plt.figure(figsize=(12,6))
axis1=sns.barplot(x=df_w['Weighted_average'].head(10), y=df_w['title'].head(10), data=df_w)
plt.xlim(4, 10)
plt.title('Best Drama Movies by weighted average', weight='bold')
plt.xlabel('Weighted Average Score', weight='bold')
plt.ylabel('Drama Movie Title', weight='bold')

cont_recm = recm_movies.copy()
cont_recm.head()

from sklearn.feature_extraction.text import TfidfVectorizer

tfv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), stop_words = 'english')

cont_recm['overview'] = cont_recm['overview'].fillna('')

# Fitting the TF-IDF on the 'overview' text
tfv_matrix = tfv.fit_transform(cont_recm['overview'])
#Finding Cosine_similarity
cos_sim = linear_kernel(tfv_matrix, tfv_matrix)
cont_recm = cont_recm.reset_index()
indices = pd.Series(cont_recm.index, index=cont_recm['title'])
#print (indices.head(20))


def sugg_recm(title):

    # Get the index corresponding to original_title
    idx = indices[title]    
    # Get the pairwsie similarity scores 
    sim_scores = list(enumerate(cos_sim[idx]))
     # Sort the movies 
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    movies = indices.iloc[movie_indices]
    return movies.head(10)

sugg_recm('Star Wars')


