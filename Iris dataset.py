import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# reading the dataset
iris = pd.read_csv('Iris Dataset Classification (KNN)/Iris.csv')

print(iris.shape)

# dropping id column (not required)
iris.drop('Id', axis = 1, inplace = True)
print(iris.head())

# # summary statistics
# print(iris.describe())

# # number of each species (distribution across categories)
# print(iris['Species'].value_counts())

# # univariate plots
# iris.hist()
# plt.show()
# # hence, sepal length and sepal width follow a normal dist., petal length follows a bimodal dist.

# dict mapping species to an integer code
inv_name_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
# integer colour code
colours = [inv_name_dict[item] for item in iris['Species']]
# # scatter plot (sepal)
# scatter = plt.scatter(iris['SepalLengthCm'], iris['SepalWidthCm'], c = colours)
# plt.xlabel('Sepal Length (cm)')
# plt.ylabel('Sepal Width (cm)')
# # add legend
# plt.legend(handles = scatter.legend_elements()[0], labels = inv_name_dict.keys())
# plt.show()
# # hence, it is hard to separate versicolor from virginica vs setosa

# # scatter plot (petal)
# scatter = plt.scatter(iris['PetalLengthCm'], iris['PetalWidthCm'], c = colours)
# plt.xlabel('Petal Length (cm)')
# plt.ylabel('Petal Width (cm)')
# # add legend
# plt.legend(handles = scatter.legend_elements()[0], labels = inv_name_dict.keys())
# plt.show()
# # hence, petal length and width are highly correlated and the distinction between versicolor and virginica is easier to make than with sepal features

# data preparation
X = iris[['PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

# building the model
knn = KNeighborsClassifier()
# create a dict of all the values to be tested for n_neighbors
param_grid = {'n_neighbors': np.arange(2, 10)}
# use gridsearch to test all values for n_neighbors
knn_gscv = GridSearchCV(knn, param_grid, cv = 5)
# fit model to data
knn_gscv.fit(X, y)
# check best parameter
print(knn_gscv.best_params_)
# hence, best n_neighbors = 4
# check best score
print("Accuracy: ", knn_gscv.best_score_)
# model evaluation
y_pred = knn_gscv.best_estimator_.predict(X)
print(confusion_matrix(y, y_pred))
# plot_confusion_matrix(knn_gscv.best_estimator_, X, y, cmap=plt.cm.Blues)
# plt.show()

knn_final = KNeighborsClassifier(n_neighbors = knn_gscv.best_params_['n_neighbors'])
knn_final.fit(X, y)
y_pred = knn_final.predict(X)
print(confusion_matrix(y, y_pred))
print(cross_val_score(knn_final, X, y).mean())
