import pandas as pd
import nltk
import re
from nltk.stem.porter import PorterStemmer
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
from sklearn.feature_extraction.text import CountVectorizer
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html
from sklearn.metrics.pairwise import cosine_similarity

k_drama_df= pd.read_csv("./data/koreanTV.csv")

k_df=k_drama_df
newShortStory = k_df['Short Story']

ps = PorterStemmer()
def remove_splChar_normalizeWords(ss_line):
    word_list = []
    ss_line = re.sub('[^A-Za-z0-9]',' ', ss_line)
    #ss_line.replace('\n','')
    for word in ss_line.split():
        word_list.append(ps.stem(word))
        
    return  " ".join(word_list)


newShortStory = k_df['Short Story']
newShortStory = newShortStory.apply(remove_splChar_normalizeWords)

k_df['Modified Short Story'] = newShortStory

k_df=k_df.drop(['Votes:', 'Time','Short Story'],axis=1)
#k_df.columns.values

k_df['Tags'] =  [g.replace(',','') for g in k_df['Genre']]

k_df['Tags'] = k_df['Tags']+" "+ [g.replace(',','') for g in k_df['Stars']] +" "+k_df['Modified Short Story']

k_df['Tags'] = [word.lower() for word in k_df['Tags']]
k_df['Title_low']=[title.lower() for title in k_df['Title']]

# creating vectorizer (with stopwords as well)
cv = CountVectorizer(max_features = 7000, stop_words = "english")
vect_mat = cv.fit_transform(k_df["Tags"]).toarray()

cv.get_feature_names_out()

similarity = cosine_similarity(vect_mat)

print(similarity)

#get input for recommondation
def genre_recomm(input_title):
    movieList = []

    if(input_title.lower() not in k_df['Title_low'].values):
        return 'The k-drama you like is not in the data base try another name'
    
    series_index = k_df[k_df["Title_low"] == input_title.lower()].index[0]
    
    # Calculate similarity
    distances = similarity[series_index]
   
    series_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
    
    for i in series_list:      
        movieList.append([k_df.iloc[i[0]]["Title"],k_df.iloc[i[0]]["Rating"]])  
        #print(i)
    #print((ml[0] + '\t' + ml[1]).expandtabs(30) for ml in movieList)
    return movieList


def getAllTitlesAvailable():
    return list(k_df['Title'].values)