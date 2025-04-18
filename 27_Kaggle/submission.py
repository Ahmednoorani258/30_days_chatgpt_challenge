import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# print(train.head())
# print("______________________________________")
# print(train.describe())
# print("______________________________________")

# print(train.isnull().sum())

# Filling missing values
train['Age'] = train['Age'].fillna(train['Age'].mean())

# converting categorical values to numerical
train['Sex'] = train['Sex'].replace({'male': 0, 'female': 1})

# creating new feature FamilySize
train["FamilySize"] = train["SibSp"] + train["Parch"]

from sklearn.model_selection import train_test_split
X = train[['Pclass', 'Sex', 'Age', 'FamilySize']]
y = train['Survived']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

score = model.score(X_val, y_val)
print("Score:", score)


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
print("New Score:", model.score(X_val, y_val))


model = RandomForestClassifier(n_estimators=200, max_depth=5)
model.fit(X_train, y_train)
print("Tuned Score:", model.score(X_val, y_val))


test['Age'] = test['Age'].fillna(test['Age'].mean())
test['Sex'] = test['Sex'].replace({'male': 0, 'female': 1})
test['FamilySize'] = test['SibSp'] + test['Parch']
X_test = test[['Pclass', 'Sex', 'Age', 'FamilySize']]
predictions = model.predict(X_test)
submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': predictions})
submission.to_csv('submission.csv', index=False)