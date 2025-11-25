Iris Classifier – AI Fundamentals Project

This project implements a simple machine learning classifier using the classic Iris dataset.
It follows the required folder structure, uses a virtual environment, includes a reproducible CLI script, and generates outputs (model + confusion matrix image).

iris-classifier/
│
├── data/
│   └── (empty – iris dataset loads from scikit-learn)
│
├── notebooks/
│   └── iris_model.ipynb
│
├── outputs/
│   ├── confusion_matrix.png
│   └── iris_model.joblib
│
├── src/
│   └── train.py
│
├── tests/
│   └── test_train.py
│
├── venv/
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

2. How to Set Up the Virtual Environment

From the project root folder:
python3 -m venv venv
source venv/bin/activate    # Linux/macOS
# OR
venv\Scripts\activate       # Windows

Install dependencies:
pip install -r requirements.txt
(Your environment must contain the versions required by the professor.)

3. Running the Training Script
Inside the active virtual environment:
python3 src/train.py

This will:

✔ Load the Iris dataset
✔ Train a DecisionTreeClassifier
✔ Print example predictions and accuracy
✔ Generate a confusion matrix
✔ Save:

outputs/confusion_matrix.png
outputs/iris_model.joblib

4. Tests
Run basic tests with:
pytest
The test file provided (tests/test_train.py) checks that the test system runs correctly.

5. Files Generated Automatically
The following files are created when running train.py:
outputs/confusion_matrix.png – confusion matrix plot
outputs/iris_model.joblib – saved model
These files should not be manually edited.

6. Notes
The Iris dataset is loaded directly from scikit-learn, so the data/ folder stays empty.
This project is only meant to demonstrate the standard ML workflow:
data → training → predictions → evaluation → saved model.
Accuracy may vary slightly, but a Decision Tree often achieves 0.96–1.0 with this dataset.

small update test for Git push.




