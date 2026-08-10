import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# ============================================================
# Load Data
# ============================================================

df = pd.read_csv("data/train.csv")

print(f"Loaded dataset with shape {df.shape}")


# ============================================================
# Data Cleaning
# ============================================================

# Fill missing Age
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Fare
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Fill missing Embarked
df["Embarked"] = df["Embarked"].fillna(
    df["Embarked"].mode()[0]
)


# ============================================================
# Feature Engineering
# ============================================================

# Create HasCabin feature
df["HasCabin"] = df["Cabin"].notna().astype(int)

# Create FamilySize
df["FamilySize"] = (
    df["SibSp"] +
    df["Parch"] +
    1
)

# Create IsAlone
df["IsAlone"] = (
    df["FamilySize"] == 1
).astype(int)

# Convert Sex to numerical values
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

# One-hot encode Embarked
df = pd.get_dummies(
    df,
    columns=["Embarked"],
    drop_first=True,
    dtype=int
)


# ============================================================
# Remove Unnecessary Columns
# ============================================================

columns_to_drop = [
    "Cabin",
    "Name",
    "Ticket",
    "PassengerId"
]

df = df.drop(
    columns=[
        c for c in columns_to_drop
        if c in df.columns
    ]
)


# ============================================================
# Select Features and Target
# ============================================================

X = df.drop(columns=["Survived"])
y = df["Survived"]


print("\nFeatures used:")
print(X.columns.tolist())


# ============================================================
# Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ============================================================
# Scale Data for Logistic Regression
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# Define Models
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=5
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}


# ============================================================
# Train and Evaluate Models
# ============================================================

results = []

for name, model in models.items():

    if name == "Logistic Regression":

        model.fit(
            X_train_scaled,
            y_train
        )

        predictions = model.predict(
            X_test_scaled
        )

    else:

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_test
        )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy * 100
    })


# ============================================================
# Create Comparison Table
# ============================================================

results_df = pd.DataFrame(results)

results_df["Accuracy"] = results_df[
    "Accuracy"
].round(2)


print("\nModel Comparison")
print("================")

print(
    results_df.to_string(index=False)
)


# ============================================================
# Save Results
# ============================================================

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print(
    "\nComparison saved as model_comparison.csv"
)