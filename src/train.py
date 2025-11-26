
import os

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


def main():
    # 1. Loading the dataset
    iris = load_iris()
    X = iris.data      # features
    y = iris.target    # labels

    print("Feature names:", iris.feature_names)
    print("Target names:", iris.target_names)
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Training the model
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 4. Making predictions
    y_pred = model.predict(X_test)
    print("Example predictions:", y_pred[:5])
    print("True labels:", y_test[:5])

    # 5. Evaluating the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    # 6. Plot + save confusion matrix image
    os.makedirs("outputs", exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.heatmap(
    cm,
    annot=True,
    cmap="Blues",
    fmt="d",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names,)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 7. Save the trained model
    joblib.dump(model, "outputs/iris_model.joblib")
 
    


if __name__ == "__main__":
    main()




