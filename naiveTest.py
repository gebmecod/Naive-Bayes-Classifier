import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Step 1: Data Preprocessing
data = pd.read_csv('Traffic_Accidents.csv')
# Convert True/False values to 1/0
data['Property Damage'].fillna(value=False, inplace=True)
data['Property Damage'] = data['Property Damage'].apply(lambda x: 1 if x == True else 0)

# Define a dictionary to map categories to numeric values
weather_dict = {
    '': 0,
    'CLEAR': 1,
    'RAIN': 2,
    'CLOUDY': 3,
    'SlEET, HAIL': 4,
    'UNKNOWN': 5,
    'FOG': 6,
    "OTHER (NARRATIVE)": 7
}
illum_dict = {
    '': 0,
    'DUSK': 1,
    'DAWN': 2,
    'DARK - LIGHTED': 3,
    'DARK - NOT LIGHTED': 4,
    'DARK-Unknown Lighting': 5,
    'UNKNOWN': 6,
    'DAYLIGHT': 7
}

# Map the categories to numeric values
data['Weather Description'] = data['Weather Description'].map(weather_dict)
data['Illumination Description'] = data['Illumination Description'].map(illum_dict)

relevant_columns = ['Weather Description', 'Illumination Description', 'Property Damage', 'Number of Injuries']
data = data[relevant_columns].dropna()  # Remove rows with missing values


# Step 2: Splitting the Data
X = data[['Weather Description', 'Illumination Description', 'Property Damage']]
y = data['Number of Injuries']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Encoding Categorical Variables
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)


# Step 4: Training the Naive Bayes Model
model = GaussianNB()
model.fit(X_train_encoded, y_train)

# Step 5: Making Predictions
y_pred = model.predict(X_test_encoded)
results = pd.DataFrame({'Actual Number of Injuries': y_test, 'Predicted Number of Injuries': y_pred})

# Evaluate the performance of the model
# accuracy = (y_pred == y_test).sum() / len(y_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Plot the predicted values against the actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Number of Injuries')
plt.ylabel('Predicted Number of Injuries')
plt.title('Actual vs. Predicted Number of Injuries')
plt.show()