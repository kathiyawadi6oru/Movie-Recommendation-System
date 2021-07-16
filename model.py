import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("movie_metadata.csv")

df['title_year'].fillna(0,inplace=True)

df['title_year'] = df['title_year'].apply(np.int64)

#Sort movies based on score calculated above
df2 = df.sort_values('imdb_score', ascending=False)

#Print the top 20 movies
df2[['movie_title', 'title_year', 'director_name', 'genres', 'language', 'imdb_score']].head(20)

dataset= df[['director_name','genres','title_year','movie_title','actor_1_name','actor_2_name','actor_3_name']]

## clean genres--- remove | between generes
dataset['genres'] = dataset['genres'].apply(lambda a: str(a).replace('|', ' '))

dataset['movie_title'][0]

dataset['movie_title'] = dataset['movie_title'].apply(lambda a:a[:-1])
dataset['movie_title'][0]

## combined features on which we will calculate cosine similarity

dataset['director_genre_actors'] = dataset['director_name']+' '+dataset['actor_1_name']+' '+' '+dataset['actor_2_name']+' '+dataset['actor_3_name']+' '+dataset['genres']


dataset.fillna('', inplace=True)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vec = CountVectorizer()
vec_matrix = vec.fit_transform(dataset['director_genre_actors'])

similarity = cosine_similarity(vec_matrix)

def recommend_movie(movie):
    if movie not in dataset['movie_title'].unique():
        return()
    else:
        i = dataset.loc[dataset['movie_title']==movie].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        
        l = []
        year=[]
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(dataset['movie_title'][a])
            year.append(dataset['title_year'][a])
            
        df2 = pd.DataFrame({'Movies Recommended':l, 'Year':year})
        #df2.drop_duplicates
        j=0
        m=[]
        for j in range(len(lst)):
            m.append( l[j] + " :: " + str(year[j]))
            #m.append(year[j][1])
        
        return m

        
import pickle
pickle.dump(similarity,open("model.pkl","wb"))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
#print(recommend_movie('The Kids Are All Right'))

