import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset
df = pd.read_csv('matches.csv', low_memory=False)

# Ensure scores are numeric
df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')

# Create result (target): 1 for home win, -1 for away win, 0 for draw
df['result'] = df.apply(lambda row: 1 if row['home_score'] > row['away_score']
                        else (-1 if row['home_score'] < row['away_score'] else 0), axis=1)

# Select only pre-match features
features = ['home', 'away', 'year', 'venue', 'league', 'home_formation', 'away_formation']

# Drop rows with missing values
df = df[features + ['result']].dropna()

# Encode categorical features
le = LabelEncoder()
for col in ['home', 'away', 'venue', 'league', 'home_formation', 'away_formation']:
    df[col] = le.fit_transform(df[col])

# Split data
X = df[features]
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)

# MLP Neural Network model
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

# Function to evaluate model performance and plot graphs
def evaluate_model(name, y_test, y_pred, model, X, y):
    print(f"\n=== {name} ===")

    # Accuracy and Classification Report
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Define class labels and ensure correct ordering
    class_labels = ['Away Win (-1)', 'Draw (0)', 'Home Win (1)']
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 0, 1])

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                cbar=False, square=True, linewidths=0.5, linecolor='black')

    plt.title(f'{name} Confusion Matrix', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Cross-validation scores
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean Cross-validation score: {cv_scores.mean():.4f}")

    return accuracy

# Compare model performance
accuracy_logreg = evaluate_model("Logistic Regression", y_test, y_pred_logreg, logreg, X, y)
accuracy_mlp = evaluate_model("Neural Network (MLP)", y_test, y_pred_mlp, mlp, X, y)

# Model Accuracy Comparison Plot
plt.figure(figsize=(8, 5))
models = ['Logistic Regression', 'Neural Network (MLP)']
accuracies = [accuracy_logreg, accuracy_mlp]

plt.bar(models, accuracies, color=['skyblue', 'lightcoral'])
plt.title('Model Accuracy Comparison', fontsize=14)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
