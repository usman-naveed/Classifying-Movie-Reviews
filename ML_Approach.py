from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import nltk
import string
import re
import numpy as np
from sklearn.metrics import confusion_matrix

nltk.download('stopwords')
nltk.download('wordnet')

df1 = pd.read_csv("./imdb_labelled.txt", sep='\t', names=['sentence', 'sentiment'])
important_words = []
tr = str.maketrans("", "", string.punctuation)

# removing stop words, punctuation, and lemmatizing words
for i in range(748):
    sentences = (df1.loc[[i], ['sentence']].to_string().lower())
    sentences = re.sub(r'\d+', '', sentences)
    sentences = re.sub(r'sentence', '', sentences)
    sentences = sentences.translate(tr)
    tokens = nltk.word_tokenize(sentences)
    important_words.append([t for t in tokens if t not in stopwords.words('english')])

# putting modified data back into dataframe for model
for i in range(748):
    results = ' '.join(important_words[i])
    df1.iloc[i, df1.columns.get_loc('sentence')] = results

print(df1)

# vectorization of corpus using TF-IDF
vectorization = TfidfVectorizer()
tf_idf = vectorization.fit_transform(df1['sentence'])
print(tf_idf)

# training and fitting model
X_train, X_test, y_train, y_test = train_test_split(tf_idf.toarray(), df1['sentiment'], test_size=0.2, random_state=42)
rm = RandomForestClassifier().fit(X_train, y_train)
nb = MultinomialNB().fit(X_train, y_train)
knn = KNeighborsClassifier(n_neighbors=11).fit(X_train, y_train)
print('Confusion Matrix for KNN: \n', confusion_matrix(y_test, knn.predict(X_test)))
print('Confusion Matrix for NB: \n', confusion_matrix(y_test, nb.predict(X_test)))
print('Actual\n', y_test)
print('Predicted\n', nb.predict(X_test))
print('KNN (no cv): ', knn.score(X_test, y_test))
print('Naive Bayes (no cv): ', nb.score(X_test, y_test))
print('Random Forest (no cv): ', rm.score(X_test, y_test))
scores = cross_val_score(nb, tf_idf.toarray(), df1['sentiment'], cv=5)
scores2 = cross_val_score(rm, tf_idf.toarray(), df1['sentiment'], cv=5)
scores3 = cross_val_score(knn, tf_idf.toarray(), df1['sentiment'], cv=5)
print('CV NB: ', np.mean(scores))
print('CV RF: ', np.mean(scores2))
print('CV KNN: ', np.mean(scores3))
