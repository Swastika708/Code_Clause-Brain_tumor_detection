import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

path = r"C:\Users\SWASTIKA\OneDrive\Desktop\Brain_tumor_detection\brain_tumor_dataset"
classes = ["yes", "no"]
data = []
labels = []

for cls in classes:
    pth = os.path.join(path, cls)
    for j in os.listdir(pth):
        img = cv2.imread(os.path.join(pth+'/'+ j), 0)
        img = cv2.resize(img, (200,200))
        data.append(img)
        labels.append(cls)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

X = np.array(data)
Y = np.array(labels)

X_updated = X.reshape(len(X), -1)

np.unique(Y)

pd.Series(Y).value_counts()

X.shape, X_updated.shape

"""# **Visualize data**"""

plt.imshow(X[0], cmap='gray')

"""# **Prepare data**"""

X_updated = X.reshape(len(X), -1)
X_updated.shape

"""# **Split data**"""

xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,
                                               test_size=.20)

xtrain.shape, xtest.shape

"""# **Feature scaling**"""

print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())

"""# **Feature selection**"""

from sklearn.decomposition import PCA

print(xtrain.shape, xtest.shape)

pca = PCA(.98)
# pca_train = pca.fit_transform(xtrain)
# pca_test = pca.transform(xtest)
pca_train = xtrain
pca_test = xtest

"""# **Train Model**"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(xtrain, ytrain)

sv = SVC()
sv.fit(xtrain, ytrain)

"""# **Evaluation**"""

print("Training Score:", lg.score(xtrain, ytrain))
print("Testing Score:", lg.score(xtest, ytest))

print("Training Score:", sv.score(xtrain, ytrain))
print("Testing Score:", sv.score(xtest, ytest))

"""# **Prediction**"""

pred = sv.predict(xtest)

misclassified=np.where(ytest!=pred)
misclassified

print("Total Misclassified Samples: ",len(misclassified[0]))
print(pred[36],ytest[36])