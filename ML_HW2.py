import pandas as pd

#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import CountVectorizer

#dataset = [
#    "I enjoy reading about Machine Learning and Machine Learning is my PhD subject",
#    "I would enjoy a walk in the park",
#    "I was reading in the library"
#]

#ds = pd.read_csv('assignment_2_dataset.csv')

#

import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score




df = pd.read_csv('assignment_2_dataset.csv')

y = df.polarity

X = df.body

vec = TfidfVectorizer()
X = vec.fit_transform(X)
print(X)
print(' \n')
svd = TruncatedSVD(n_components=100, random_state=0)
X = svd.fit_transform(X)
print(X)
print(' \n')
clf = LogisticRegressionCV(cv=5, random_state=0, max_iter=250)

clf = clf.fit(X, y)
print(clf.score(X, y))
print(' \n')

mlp = MLPClassifier(activation='logistic', solver='sgd', random_state=0, max_iter=5000, n_iter_no_change=500)
mlp = mlp.fit(X, y)
print(mlp.score(X, y))
print(' \n')

accuracy = cross_val_score(mlp, X, y, cv=5)
print(accuracy.mean())
