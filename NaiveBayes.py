import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# Step 1: Data Preprocessing
data = pd.read_csv('Traffic_Accidents.csv')

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

# Convert True/False values to 1/0
data['Hit and Run'] = data['Hit and Run'].apply(lambda x: 1 if x == True else 0)
data['Property Damage'].fillna(value=False, inplace=True)
data['Property Damage'] = data['Property Damage'].apply(lambda x: 1 if x == True else 0)

# Map the categories to numeric values
data['Weather Description'] = data['Weather Description'].map(weather_dict)
data['Illumination Description'] = data['Illumination Description'].map(illum_dict)

# Define a list of relevant columns and select only those columns for the dataframe. Drop rows with missing values
relevant_columns = ['Weather Description', 'Illumination Description', 'Property Damage', 'Number of Injuries', 'Hit and Run', 'Number of Motor Vehicles'] # Indicate Relevant Columns for the Algorithm
data = data[relevant_columns].dropna()


# Step 2: Split data into training and testing sets
X = data[['Weather Description', 'Illumination Description', 'Property Damage', 'Hit and Run', 'Number of Motor Vehicles']] # Features (independent variables)
y = data['Number of Injuries'] # Target Variable (dependent variable)

# 80-20 train-test  = 80% of the data for the training model and 20% for testing the models performance
# Random state is 42 simply for the reproducibility of the results, same value means same results for the tests
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Encoding Categorical Variables - one-hot encode the categorical variables
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
f1 = f1_score(y_pred, y_test, average="weighted")

print("Accuracy:", accuracy)
print("F1 Score:", f1)

# Plot the predicted values against the actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Number of Injuries')
plt.ylabel('Predicted Number of Injuries')
plt.title('Actual vs. Predicted Number of Injuries')
plt.show()