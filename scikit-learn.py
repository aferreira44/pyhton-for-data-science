from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

## [height, weight, shoe size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
    [166, 65, 40], [190, 90, 47], [175, 64, 39], [177, 70, 40],
    [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'female', 'female', 'female', 'male', 'male', 'male', 'female', 'male', 'female', 'male']

## Decision Tree

clf = tree.DecisionTreeClassifier()

clf = clf.fit(X,Y)
prediction = clf.predict([[190, 70, 43]])

print('DecisionTreeClassifier: ', prediction)

## Support Vector Classification

clf = svm.SVC()

clf = clf.fit(X,Y)
prediction = clf.predict([[190, 70, 43]])

print('SVC: ', prediction)

### LinearSVC

clf = svm.LinearSVC()

clf = clf.fit(X,Y)
prediction = clf.predict([[190, 70, 43]])

print('LinearSVC: ', prediction)

## Gaussian Naive Bayes

clf = GaussianNB()

clf = clf.fit(X,Y)
prediction = clf.predict([[190, 70, 43]])

print('GaussianNB: ', prediction)

## KNeighbor Classifier

clf = KNeighborsClassifier()

clf = clf.fit(X,Y)
prediction = clf.predict([[190, 70, 43]])

print('KNeighborsClassifier: ', prediction)