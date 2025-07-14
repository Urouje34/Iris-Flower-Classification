import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Loading the CSV file
df = pd.read_csv('D:\Internship projects\Iris.csv')

# Step 2
df.drop('Id', axis=1, inplace=True)

# Step 3: Encoding the labels (species names) into numbers
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Step 4: Spliting the data into features (X) and labels (y)
X = df.drop('Species', axis=1)
y = df['Species']

# Step 5: Spliting into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Training a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Making predictions
y_pred = model.predict(X_test)

# Step 8: Evaluating the model
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 9: Visualizing the data
sns.pairplot(df, hue='Species', palette='Set2')
plt.suptitle("Pairplot of Iris Dataset", y=1.02)
plt.show()
