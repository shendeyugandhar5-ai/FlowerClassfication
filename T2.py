import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
# load the  dataset#
iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

print("First 5 rows:")
print(df.head())
#EDA#
print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nClass Distribution:")
print(df['species'].value_counts())
#plot visualization#
sns.pairplot(df, hue="species")
plt.show()
#heatmap#
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
# train test #
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
 # Logistic relation model #
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

log_accuracy = log_model.score(X_test, y_test)

print("\nLogistic Regression Accuracy:", log_accuracy)

# decision tree#
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

tree_accuracy = tree_model.score(X_test, y_test)

print("Decision Tree Accuracy:", tree_accuracy)
# Compare #
print("\nModel Comparison:")
print("Logistic Regression:", log_accuracy)
print("Decision Tree:", tree_accuracy)
# Confusion matrix #
y_pred = log_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
