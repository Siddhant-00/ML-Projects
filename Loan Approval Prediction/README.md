# Loan Approval Predictor

## Project Description
This project focuses on building a machine learning model to predict the approval status of loan applications. It analyzes various applicant features to determine the likelihood of a loan being approved or rejected. The solution includes comprehensive data analysis, model training with hyperparameter tuning, and a **FastAPI backend API** for serving real-time predictions.

## Dataset
* **Name:** `loan_data.csv`
* **Source:** This dataset contains historical loan application details. (If this dataset is from a specific Kaggle competition or public repository, **please replace this line with the direct URL to the dataset.** For example: `https://www.kaggle.com/datasets/XYZ/loan-approval-data`)
* **Key Features Used:** The model utilizes the following features for prediction:
    * `Married`: Marital status of the applicant (e.g., Yes/No).
    * `ApplicantIncome`: Income of the applicant.
    * `Education`: Education level of the applicant (e.g., Graduate/Not Graduate).
    * `LoanAmount`: The amount of loan requested.
    * `Credit_History`: Whether the applicant has a credit history (e.g., 0 for no, 1 for yes).
* **Target Variable:** `Loan_Status` (Categorical: `0 = Rejected`, `1 = Approved`).

## Model Architecture and Evaluation
Several classification models were explored, including Logistic Regression, K-Nearest Neighbors (KNN) Classifier, and Support Vector Classifier (SVC). Hyperparameter tuning using `GridSearchCV` was applied to optimize the KNN and SVC models.

The **Support Vector Classifier (SVC)**, optimized via `GridSearchCV`, was selected as the final model due to its performance and is saved as `model.pkl`. A `StandardScaler` is also used for feature preprocessing and saved as `Scaler.pkl`.

### Performance Metrics:
The models were primarily evaluated using **Accuracy Score** on the test set.

* **Logistic Regression Accuracy:** `0.7742` (or 77.42%)
* **K-Nearest Neighbors (KNN) Classifier Accuracy:** (Please insert the accuracy score you obtained for KNN here, e.g., `0.XX` or `XX%`)
* **Support Vector Classifier (SVC) Accuracy (Deployed Model):** `0.7742` (or 77.42%)

The SVC model achieved an accuracy of approximately 77.42%, indicating its capability to correctly classify loan applications based on the provided features.

## Project Files
* `Loan Approval Prediction.ipynb`:
    This Jupyter Notebook contains the complete machine learning pipeline:
    * Data loading and initial inspection (`.head()`, `.info()`).
    * Extensive Exploratory Data Analysis (EDA) with various plots:
        * `Loan_Status` distribution.
        * Average `LoanAmount` by `Education` and `Self_Employed`.
        * `LoanAmount` distribution by `Property_Area`.
        * `Loan_Status` breakdown by `Gender` and `Married` status.
        * `ApplicantIncome` vs `LoanAmount` scatter plot.
        * Correlation Heatmap of numerical features.
    * Feature selection and target variable preparation (including `LabelEncoder` for `Loan_Status`).
    * Feature scaling using `StandardScaler`.
    * Data splitting into training and testing sets.
    * Training and evaluation of Logistic Regression, KNeighbors Classifier, and Support Vector Classifier models.
    * Hyperparameter tuning using `GridSearchCV` for KNN and SVC.
    * Serialization of the trained `StandardScaler` to `Scaler.pkl` and the best-performing `SVC` model (from `GridSearchCV`) to `model.pkl` using `joblib`.

* `Loan_app.py`:
    This is a **FastAPI backend API** designed to serve real-time loan approval predictions.
    * It loads the pre-trained `Scaler.pkl` and `model.pkl`.
    * It defines an endpoint `/predict/` that accepts a JSON payload with 5 float values (`x1` to `x5`) representing the applicant's features.
    * These inputs are scaled using the loaded `Scaler` and then fed to the `model` for prediction.
    * It returns the prediction (0 for Rejected, 1 for Approved) as a JSON response.

* `model.pkl`:
    This file contains the serialized (pickled) Support Vector Classifier (SVC) model, trained and optimized in the Jupyter Notebook, ready to be loaded by `Loan_app.py` for predictions.

* `Scaler.pkl`:
    This file contains the serialized `StandardScaler` object, fitted on the training data. It is crucial for preprocessing new input features in the `Loan_app.py` before they are fed to the `model.pkl`, ensuring consistency.

* `requirements.txt` (You will create this):
    Lists all the Python libraries and their versions required to run this project.

## Dependencies / Installation
To run this project, you need Python 3.x and the following libraries. It's highly recommended to use a virtual environment for dependency management.

1.  **Clone the `ML-Projects` repository** (if you haven't already and moved your project files into it):
    ```bash
    git clone [https://github.com/Siddhant-00/ML-Projects.git](https://github.com/Siddhant-00/ML-Projects.git)
    cd ML-Projects/Loan\ Approval\ Prediction # Adjust folder name if you renamed it
    ```
    (If you are already in the `ML-Projects` directory and just added the `Loan Approval Prediction` folder, simply `cd Loan\ Approval\ Prediction`.)

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate # On macOS/Linux
    ```
3.  **Install the required packages:**
    (First, ensure you have generated `requirements.txt` in your `Loan Approval Prediction` folder by running `pip freeze > requirements.txt` and cleaning it up).
    ```bash
    pip install -r requirements.txt
    ```
    * If you choose not to create `requirements.txt` (not recommended for good practice), you can manually install the main dependencies:
        `pip install numpy pandas scikit-learn matplotlib seaborn joblib fastapi uvicorn pydantic`

## Usage

### 1. Run the Jupyter Notebook:
Open `Loan Approval Prediction.ipynb` in Jupyter Notebook or JupyterLab to explore the data analysis, model training, and evaluation process.
```bash
jupyter notebook "Loan Approval Prediction.ipynb"