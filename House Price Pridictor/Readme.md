# House Price Predictor

## Project Description
This project develops a machine learning model to predict house prices based on various key features. It includes data exploration, model training with hyperparameter tuning, and a user-friendly Streamlit web application to make real-time predictions. The primary goal is to provide accurate price estimations for houses based on their characteristics, demonstrating a full end-to-end ML project workflow.

## Dataset
* **Name:** `House Price India.csv`
* **Source:** This dataset typically contains historical house sales data. (If this dataset is from a specific Kaggle competition or public repository, **please replace this line with the direct URL to the dataset.** For example: `https://www.kaggle.com/datasets/sukhmanbrar/house-price-india`)
* **Key Features Used:**
    * `number of bedrooms`: Number of bedrooms in the house.
    * `number of bathrooms`: Number of bathrooms in the house.
    * `living area`: The total living area of the house in square feet.
    * `condition of the house`: A numerical rating indicating the house's condition (e.g., 1-5, where 5 is excellent).
    * `Number of schools nearby`: The count of schools in the vicinity of the house.
* **Target Variable:** `Price` (The sale price of the house).

## Model Architecture and Evaluation
This project explored several regression models, including Linear Regression, Decision Tree Regressor, and Random Forest Regressor. Hyperparameter tuning using `GridSearchCV` was applied to optimize the Decision Tree and Random Forest models for better performance.

The **Random Forest Regressor** (optimized via `GridSearchCV`) was selected as the final model due to its robust performance and is saved as `model.pkl`.

### Performance Metrics:
The models were evaluated using Mean Squared Error (MSE) and Mean Absolute Error (MAE) on the test set.

* **Decision Tree Regressor (tuned):**
    * Mean Squared Error (MSE): `66,238,176,157.39`
* **Linear Regression:**
    * Mean Squared Error (MSE): `67,041,615,772.77`
* **Random Forest Regressor (tuned - deployed model):**
    * Mean Absolute Error (MAE): `160,622.10`

The Random Forest model predicts house prices with an average absolute error of approximately `$160,622`, indicating a reasonable level of accuracy for price estimation given the dataset.

## Project Files
* `House Price Prediction.ipynb`:
    This Jupyter Notebook contains the complete machine learning pipeline:
    * Data loading and initial inspection (`.info()`, `.describe()`).
    * Exploratory Data Analysis (EDA), including a bar plot of "Condition of the house vs Price".
    * Feature selection and target variable definition.
    * Data splitting into training and testing sets.
    * Training and evaluation of Linear Regression, Decision Tree Regressor, and Random Forest Regressor models.
    * Hyperparameter tuning using `GridSearchCV` for Decision Tree and Random Forest.
    * Serialization of the best-performing model (`gridfr` - the Random Forest model's GridSearchCV object) to `model.pkl` using `joblib`.

* `app.py`:
    This is a **Streamlit web application** that provides an interactive interface for predicting house prices.
    * It loads the pre-trained `model.pkl`.
    * Allows users to input values for "Number of Bedrooms", "Number of Bathrooms", "Living Area", "Condition", and "Number of Schools Nearby" using interactive widgets.
    * Displays the predicted house price based on the user's input.

* `model.pkl`:
    This file contains the serialized (pickled) Random Forest Regressor model, trained and optimized in the Jupyter Notebook, ready to be loaded and used for predictions by `app.py`.

* `requirements.txt` (You will create this):
    Lists all the Python libraries and their versions required to run this project.

## Dependencies / Installation
To run this project, you need Python 3.x and the following libraries. It's highly recommended to use a virtual environment for dependency management.

1.  **Clone the `ML-Projects` repository** (if you haven't already and moved your project files into it):
    ```bash
    git clone [https://github.com/Siddhant-00/ML-Projects.git](https://github.com/Siddhant-00/ML-Projects.git)
    cd ML-Projects/House\ Price\ Predictor
    ```
    (If you are already in the `ML-Projects` directory and just added the `House Price Predictor` folder, simply `cd House\ Price\ Predictor`.)

2.  **Create a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate # On macOS/Linux
    ```
3.  **Install the required packages:**
    (First, ensure you have generated `requirements.txt` in your `House Price Predictor` folder by running `pip freeze > requirements.txt` and cleaning it up).
    ```bash
    pip install -r requirements.txt
    ```
    * If you choose not to create `requirements.txt` (not recommended for good practice), you can manually install the main dependencies:
        `pip install numpy pandas scikit-learn matplotlib seaborn joblib streamlit`

## Usage

### 1. Run the Jupyter Notebook:
Open `House Price Prediction.ipynb` in Jupyter Notebook or JupyterLab to explore the data analysis, model training, and evaluation process.
```bash
jupyter notebook "House Price Prediction.ipynb"