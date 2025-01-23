import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from tqdm import trange
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

data = np.load("knns/swinir/lr_features_6.npy")
label = np.load("knns/swinir/lr_labels.npy")
print(data.shape, label.shape)

scores = []
randoms = [0, 223, 929, 1234, 10086]

for i in trange(5):
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.33, random_state=randoms[i])
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(x_train, y_train)

    score = neigh.score(x_test, y_test)
    y_pred = neigh.predict(x_test)

    print(classification_report(y_test, y_pred))
