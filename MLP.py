import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 1: Download and Load the Data
# We will use the Iris dataset from UCI Machine Learning Repository for this example.
# The dataset is available at: https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data

# Download the data using requests
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
filename = "wine.data"

response = requests.get(url)
with open(filename, 'wb') as f:
    f.write(response.content)

# Step 2: Load the Data into a Pandas DataFrame
# The dataset has no header, so we manually define the column names.
column_names = ['class', 'Alcohol', 'Malicacid', 'Ash', 'Alcalinity_of_ash',
'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins',
'Color_intensity', 'Hue', '0D280_0D315_of_diluted_wines', 'Proline']
df = pd.read_csv(filename, header=None, names=column_names)

# Step 3: Preprocess the Data
# Convert the species (categorical data) to numerical labels
# df['species'] = df['species'].astype('category').cat.codes

# Step 4: Split the Data into Training and Testing Sets
X = df.drop('class', axis=1)  # Features
y = df['class']  # Target variable (species)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale the Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Create and Train the MLP Classifier
mlp = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Step 7: Make Predictions
y_pred = mlp.predict(X_test_scaled)

# Step 8: Evaluate the Model
# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", accuracy)