#  Car Purchase Prediction
An end-to-end Machine Learning project that predicts whether a used car is likely to be a potential purchase based on its characteristics.

The project includes data preprocessing, exploratory data analysis, machine learning model comparison, model evaluation, explainable predictions, and an interactive Streamlit application.

---

##  Project Overview
The goal of this project is to build a Machine Learning system that predicts the `Buy_Decision` of a used car.

The application allows a user to enter car details such as:

- Car Company
- Model
- Manufacturing Year
- KM Driven
- Engine Capacity
- Mileage
- Transmission
- Fuel Type
- Owner Type
- Insurance Validity
- Accident History

The trained model then provides:

1. Purchase decision
2. Purchase probability
3. Probability interpretation
4. Feature contribution explanation

---

## 🛠️ Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook

---

## 📊 Dataset
The dataset contains used-car information with the following features:

- `Car_Company`
- `Model`
- `Year`
- `KM_Driven`
- `Engine_CC`
- `Mileage`
- `Transmission`
- `Fuel_Type`
- `Owner_Type`
- `Insurance_Valid`
- `Accident_History`
- `Buy_Decision`

The target variable is:

`Buy_Decision`

The target contains two classes:

- `Yes`
- `No`

---

## 🔍 Data Preprocessing
The following preprocessing steps were performed:

- Removed duplicate records
- Handled numerical and categorical features appropriately
- Used Ordinal Encoding for ordered categorical variables
- Used One-Hot Encoding for nominal categorical variables
- Applied Standard Scaling to the numerical feature
- Used a stratified train-test split
- Built a reusable Scikit-learn preprocessing pipeline

### Ordinal Features

- `KM_Driven`
- `Engine_CC`
- `Mileage`
- `Owner_Type`

### Categorical Features

- `Car_Company`
- `Model`
- `Transmission`
- `Fuel_Type`
- `Insurance_Valid`
- `Accident_History`

---

## 🤖 Machine Learning Models
The following classification models were evaluated:

- Logistic Regression
- Random Forest
- Tuned Random Forest
- Decision Tree
- Tuned Decision Tree
- Naive Bayes

The models were evaluated using:

- Accuracy
- ROC-AUC

---

## 📈 Model Performance
| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 91.67% | 97.18% |
| Tuned Random Forest | 91.18% | 97.04% |
| Random Forest | 91.01% | 96.63% |
| Tuned Decision Tree | 89.64% | 93.79% |
| Decision Tree | 88.27% | 86.44% |
| Naive Bayes | 83.89% | 90.28% |

### Final Model

Logistic Regression was selected as the final model because it achieved the highest accuracy and ROC-AUC among the evaluated models.

- Accuracy: **91.67%**
- ROC-AUC: **97.18%**


---

## 🧠 Explainable Prediction
The application provides an explanation of the model's prediction using the learned coefficients of the Logistic Regression model.

For each prediction, the application identifies the features with the strongest contributions and shows whether they contributed positively or negatively toward the prediction.

The application also displays:

- Feature contribution
- Positive or negative impact
- Contribution strength
- A contribution chart

These values represent model contributions and should not be interpreted as causal effects.

---

## 🌐 Streamlit Application
The project includes an interactive Streamlit application where users can enter the details of a used car and receive a machine learning prediction.

### The application allows users to:

1. Select a car company
2. Select the corresponding car model
3. Enter or select the remaining vehicle details
4. Generate a purchase prediction
5. View the purchase probability
6. View the probability interpretation
7. Explore the strongest feature contributions behind the prediction

---

## 📁 Project Structure

```text
Car_purchase_prediction/
│
├── data/
│   └── car predict.csv
│
├── models/
│   ├── final_model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── car_purchase_prediction.ipynb
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ How to Run the Project
### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the project folder

```bash
cd Car_purchase_prediction
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment on Windows

```bash
.venv\Scripts\activate
```

### 5. Install the required libraries

```bash
pip install -r requirements.txt
```

### 6. Run the Streamlit application

```bash
python -m streamlit run app.py
```

---

## 🎯 Project Highlights
- End-to-end Machine Learning workflow
- Multiple classification models compared
- Stratified train-test split
- Reusable Scikit-learn preprocessing pipeline
- Saved trained model and preprocessor
- Interactive Streamlit application
- Probability-based predictions
- Explainable model output
- User-friendly dependent dropdowns for car company and model
- GitHub-ready project structure
---

## 🚀 Future Improvements
- Add more vehicle-related features
- Perform more extensive hyperparameter validation
- Improve probability calibration
- Add advanced explainability techniques
- Deploy the application to a cloud platform
- Add additional visual analytics and dashboards
---

## 👤 Author

**Manvi**

B.Tech — Artificial Intelligence & Data Science