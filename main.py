import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('predictive_maintenance.csv')
print(f"Columns in dataset: {data.columns}")
print(f"Data types: {data.dtypes}")
data = data.dropna() 
categorical_columns = ['UDI', 'Product ID', 'Type', 'Failure Type']  
label_encoder = LabelEncoder()
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])
print(f"Cleaned Columns: {data.columns}")
X = data.drop('Failure Type', axis=1)
y = data['Failure Type']  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
