import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load and preprocess data
df = pd.read_csv('healthcare-dataset-stroke-data.csv')
df.drop(['id'], axis=1, inplace=True)

# Avoiding FutureWarning by assigning the result to the DataFrame column
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Label Encoding for categorical variables
enc = LabelEncoder()
df['gender'] = enc.fit_transform(df['gender'])
df['smoking_status'] = enc.fit_transform(df['smoking_status'])
df['work_type'] = enc.fit_transform(df['work_type'])
df['Residence_type'] = enc.fit_transform(df['Residence_type'])
df['ever_married'] = enc.fit_transform(df['ever_married'])

# Feature and target selection
X = df.drop('stroke', axis=1)
y = df.stroke

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, train_size=0.2)

# Train Decision Tree model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Save model to a file
with open('dt.pkl', 'wb') as model_file:
    pickle.dump(dt, model_file)
