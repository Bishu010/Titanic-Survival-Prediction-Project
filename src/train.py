import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

#Data Loading

def load_data(path="data/train.csv"):
  """Loading the Titanic dataset from a CSV file."""
  df = pd.read_csv(path)
  print(f"Loaded dataset with shape {df.shape}")
  return df


#data exploration
def explore_data(df):
  """Printing Basic Things First"""
  print("\n--- First 5 Rows ---")
  print(df.head())

  print("\n Column Info")
  print(df.info())

 print("\n Missing values per column")
 print(df.isnull().sum())

 print("\n Survival rate")
 print(df["survived"].value_counts(normalize=True))



#Cleaning and Preprocessing
def clean_data(df):
  """ Handle Missing Values and Drop Columns that arent useful """
  df = df.copy()
  #filling missing age with median age
  df["Age"] = df["Age"].fillna(df["Age"].median())

 #filling missing embarked with most common value
  df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

 #we drop cabin coz too many missing values also name, ticket and passenger id which dont help in survival 
  columns_to_drop = ["Cabin", "Name", "Ticket", "PassengerId"]
  df = df.drop(columns=[c for c in columns_to_drop if c in df.columns])

 return df
#featuring engineering

def engineer_features(df):
  """ creating some features and converting categorical columns into numbers"""
  df = df.copy()
  # new feature: family size = siblings/spouses + parents/children +self
  df["FamilySize"] = df["Sibsp"] + df["parch"] + 1

 #is the passenger travelling alobe ?
 df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

#converting gender/sex into numbers, male=0 and female=1
df["Sex"] = df["sex"].map({"male":0, "female":1})
 # converting "Embarked" into numbers usinh one hot encoding
 df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

return df

#model training
def train_model(X_train, y_train):
  """ Training a Logistic regression model"""
  #Scaling
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)

 model = LogisticRegression(max_iter=1000)
 model.fit(X_train_scaled,y_train)

 return model, scaler


#Evaluation of model
def evaluate_model(model, scaler, X_test, y_test):
  """ Evaluation of trained model on the test set."""
  X_test_scaled = scaler.transform(X_test)
  predictions = model.predict(X_test_scaled)

 accuracy = accuracy_score(y_testy, predictions)
 print(f"\nModel Accuracy: {accuracy:.2%}")

print("\n Confusion Matrix ")
print(confusion_matrix(y_test, predictions))

print("\n Classification Report ")
print(classification_report(y_test, predictions))

return accuracy 

#main
def main():
  #loading
  df = load_data("data/train.csv")
  #explore
  explore_data(df)
  #clean
  df = clean_data(df)
  #engineer features
  df = engineer_features(df)
  #split features (X) and target (y)
  X = df.drop(columns={"Survived"])
  y = df["Survived"]

#splitting into training and testing sets 80% train and 20 % test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#training 
model, scaler = train_model(X_train, y_train)

#evaluation 
evaluate_model(model, scaler, X_test, y_test)

#saving trained model and scaler
joblib.dump(model, "titanic_model.pkl")
joblib.dump(scaler,"scaler.pkl")
print('\nMODEL SAVED AS 'titanic_model.pkl'")

if __name__ == "__main__":
main()
