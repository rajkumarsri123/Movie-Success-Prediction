import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics

# Importing the data set
dataset = pd.read_csv("datasetfinal (2).csv")

# Split dataset into X (independent variables) and y (dependent variable)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values

# Encoding categorical data
nameencoder = LabelEncoder()
actor1encoder = LabelEncoder()
actor2encoder = LabelEncoder()
actor3encoder = LabelEncoder()
genresencoder = LabelEncoder()

# Encoding each categorical feature
dataset['director_name'] = nameencoder.fit_transform(dataset['director_name'])
dataset['actor_1_name'] = actor1encoder.fit_transform(dataset['actor_1_name'])
dataset['actor_2_name'] = actor2encoder.fit_transform(dataset['actor_2_name'].astype(str))
dataset['actor_3_name'] = actor3encoder.fit_transform(dataset['actor_3_name'].astype(str))
dataset['genres'] = genresencoder.fit_transform(dataset['genres'])

# Selecting features
features = ["director_name", "actor_1_name", "genres", "imdb_score", "budget", "gross", "profit_percent"]

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(dataset[features], y, test_size=0.2, random_state=0)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training set
model.fit(X_train, y_train)

# Predict values using the test data
nb_predict_train = model.predict(X_test)

# Predict Output
actor_name = input("Director Name        : ")
director_name = input("Actor Name           : ")
genre = input("Genre                : ")
imdb_rating = float(input("IMDB Rating          : "))
budget = float(input("Budget               : "))
gross = float(input("Gross                : "))
profit_percent = float(input("Profit Percentage   : "))

predict = [actor_name, director_name, genre, imdb_rating, budget, gross, profit_percent]

predict[0] = nameencoder.transform([predict[0]])
predict[1] = actor1encoder.transform([predict[1]])
predict[2] = genresencoder.transform([predict[2]])

# Scale or normalize the data
predict = scaler.transform([predict])
prediction = model.predict(predict)

if prediction == 1:
    print("                           HIT")
else:
    print("                           FLOP")

# Accuracy
print("                             ACCURACY: {0:.4f}".format(metrics.accuracy_score(y_test, nb_predict_train)))
