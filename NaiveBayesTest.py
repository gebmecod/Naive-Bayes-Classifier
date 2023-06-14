import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler  

# Step 1: Data Preprocessing
data = pd.read_csv('Traffic_Accidents.csv')

# Define a dictionary to map categories to numeric values
# weather_dict = {
#     '': 0,
#     'CLEAR': 1,
#     'RAIN': 2,
#     'CLOUDY': 3,
#     'SlEET, HAIL': 4,
#     'UNKNOWN': 5,
#     'FOG': 6,
#     'OTHER (NARRATIVE)': 7,
#     'BLOWING SNOW': 8,
#     'SMOG, SMOKE': 9,
#     'SEVERE CROSSWIND': 10,
#     'BLOWING SAND/SOIL/DIRT': 11
# }
# illum_dict = {
#     '': 0,
#     'DUSK': 1,
#     'DAWN': 2,
#     'DARK - LIGHTED': 3,
#     'DARK - NOT LIGHTED': 4,
#     'DARK-Unknown Lighting': 5,
#     'UNKNOWN': 6,
#     'DAYLIGHT': 7,
#     'OTHER': 8
# }
# coll_dict = {
#     '': 0,
#     'NOT COLLISION W/MOTOR VEHICLE-TRANSPORT': 1,
#     'SIDESWIPE - SAME DIRECTION': 2,
#     'Front to Rear': 3,
#     'ANGLE': 4,
#     'OTHER': 5,
#     'HEAD-ON': 6,
#     'UNKNOWN': 7,
#     'Rear to Side': 8,
#     'REAR-TO-REAR': 9
# }

# Convert True/False values to 1/0
# data['Hit and Run'] = data['Hit and Run'].apply(lambda x: 1 if x == True else 0)
data['Property Damage'].fillna(value=False, inplace=True)
data['Property Damage'] = data['Property Damage'].apply(lambda x: 1 if x == True else 0)

# Map the categories to numeric values
# data['Weather Description'] = data['Weather Description'].map(weather_dict)
# data['Illumination Description'] = data['Illumination Description'].map(illum_dict)

# Define a list of relevant columns and select only those columns for the dataframe. Drop rows with missing values
relevant_columns = ['Weather Code', 'Illumination Code', 'Property Damage', 'Number of Injuries', 'Hit and Run', 'Number of Motor Vehicles', 'Collision Type Code'] # Indicate Relevant Columns for the Algorithm
data = data[relevant_columns].dropna()


# Step 2: Split data into training and testing sets
X = data[['Weather Code', 'Illumination Code', 'Property Damage', 'Hit and Run', 'Number of Motor Vehicles', 'Collision Type Code']] # Features (independent variables) 
y = data['Number of Injuries'] # Target Variable (dependent variable)

# 66-33 train-test  = 66% of the data for the training model and 33% for testing the models performance
# Random state is 42 simply for the reproducibility of the results, same value means same results for the tests
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Step 3: Encoding Categorical Variables - one-hot encode the categorical variables
X_train_encoded = pd.get_dummies(X_train)
X_test_encoded = pd.get_dummies(X_test)
X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)


# Define undersampler
undersampler = RandomUnderSampler(random_state=42)

# Fit and transform undersampler on training data
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train_encoded, y_train)

# Train model on resampled data
model = GaussianNB()
model.fit(X_train_resampled, y_train_resampled)

# Make predictions on test data
y_pred = model.predict(X_test_encoded)

# Step 4: Training the Naive Bayes Model
# model = GaussianNB()
# model.fit(X_train_encoded, y_train)

# Step 5: Making Predictions
# y_pred = model.predict(X_test_encoded)
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