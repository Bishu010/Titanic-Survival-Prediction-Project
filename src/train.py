import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib


# ============================================================
# Data Loading
# ============================================================

def load_data(path="data/train.csv"):
    """Load the Titanic training dataset."""
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape {df.shape}")
    return df


# ============================================================
# Data Exploration
# ============================================================

def explore_data(df):
    """Display basic information about the dataset."""

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Column Information ---")
    print(df.info())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Survival Rate ---")
    print(df["Survived"].value_counts(normalize=True))


# ============================================================
# Cleaning and Preprocessing
# ============================================================

def clean_data(df):
    """Handle missing values and remove unnecessary columns."""

    df = df.copy()

    # Fill missing Age with median
    df["Age"] = df["Age"].fillna(df["Age"].median())

    # Fill missing Embarked with most common value
    df["Embarked"] = df["Embarked"].fillna(
        df["Embarked"].mode()[0]
    )

    # Fill missing Fare if necessary
    df["Fare"] = df["Fare"].fillna(
        df["Fare"].median()
    )

    # Create HasCabin before dropping Cabin
    df["HasCabin"] = df["Cabin"].notna().astype(int)

    # Drop columns that are not used by the model
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

    return df


# ============================================================
# Feature Engineering
# ============================================================

def engineer_features(df):
    """Create useful features and encode categorical variables."""

    df = df.copy()

    # Family size
    df["FamilySize"] = (
        df["SibSp"] +
        df["Parch"] +
        1
    )

    # Is the passenger travelling alone?
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

    return df


# ============================================================
# Model Training
# ============================================================

def train_model(X_train, y_train):
    """Train Logistic Regression model."""

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    model = LogisticRegression(
        max_iter=1000
    )

    model.fit(
        X_train_scaled,
        y_train
    )

    return model, scaler


# ============================================================
# Model Evaluation
# ============================================================

def evaluate_model(
    model,
    scaler,
    X_test,
    y_test
):
    """Evaluate the trained model."""

    X_test_scaled = scaler.transform(
        X_test
    )

    predictions = model.predict(
        X_test_scaled
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nModel Accuracy: {accuracy:.2%}"
    )

    print("\n--- Confusion Matrix ---")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\n--- Classification Report ---")
    print(
        classification_report(
            y_test,
            predictions
        )
    )

    return accuracy


# ============================================================
# Create Kaggle Submission
# ============================================================

def create_submission(
    model,
    scaler,
    test_path="data/test.csv"
):
    """Generate predictions for Kaggle submission."""

    test_df = pd.read_csv(test_path)

    passenger_ids = test_df["PassengerId"].copy()

    # Apply the same preprocessing
    test_df["Age"] = test_df["Age"].fillna(
        test_df["Age"].median()
    )

    test_df["Fare"] = test_df["Fare"].fillna(
        test_df["Fare"].median()
    )

    test_df["Embarked"] = test_df["Embarked"].fillna(
        test_df["Embarked"].mode()[0]
    )

    # Create HasCabin
    test_df["HasCabin"] = (
        test_df["Cabin"].notna()
    ).astype(int)

    # Create FamilySize
    test_df["FamilySize"] = (
        test_df["SibSp"] +
        test_df["Parch"] +
        1
    )

    # Create IsAlone
    test_df["IsAlone"] = (
        test_df["FamilySize"] == 1
    ).astype(int)

    # Convert Sex
    test_df["Sex"] = test_df["Sex"].map({
        "male": 0,
        "female": 1
    })

    # One-hot encode Embarked
    test_df = pd.get_dummies(
        test_df,
        columns=["Embarked"],
        drop_first=True,
        dtype=int
    )

    # Remove unused columns
    columns_to_drop = [
        "Cabin",
        "Name",
        "Ticket",
        "PassengerId"
    ]

    test_df = test_df.drop(
        columns=[
            c for c in columns_to_drop
            if c in test_df.columns
        ]
    )

    # Make sure test data has exactly the
    # same columns as training data
    expected_features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "HasCabin",
        "FamilySize",
        "IsAlone",
        "Embarked_Q",
        "Embarked_S"
    ]

    for column in expected_features:
        if column not in test_df.columns:
            test_df[column] = 0

    test_df = test_df[
        expected_features
    ]

    # Scale test data
    test_scaled = scaler.transform(
        test_df
    )

    # Make predictions
    predictions = model.predict(
        test_scaled
    )

    # Create Kaggle submission
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions.astype(int)
    })

    submission.to_csv(
        "submission.csv",
        index=False
    )

    print(
        "\nKaggle submission created: "
        "submission.csv"
    )

    print("\nFirst 5 predictions:")
    print(submission.head())


# ============================================================
# Main
# ============================================================

def main():

    # Load data
    df = load_data(
        "data/train.csv"
    )

    # Explore
    explore_data(df)

    # Clean
    df = clean_data(df)

    # Feature engineering
    df = engineer_features(df)

    # Separate features and target
    X = df.drop(
        columns=["Survived"]
    )

    y = df["Survived"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Train
    model, scaler = train_model(
        X_train,
        y_train
    )

    # Evaluate
    evaluate_model(
        model,
        scaler,
        X_test,
        y_test
    )

    # Save model
    joblib.dump(
        model,
        "titanic_model.pkl"
    )

    # Save scaler
    joblib.dump(
        scaler,
        "scaler.pkl"
    )

    print(
        "\nModel saved as titanic_model.pkl"
    )

    # Create Kaggle submission
    create_submission(
        model,
        scaler,
        "data/test.csv"
    )


if __name__ == "__main__":
    main()