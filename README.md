# 🛵 Food Delivery Time Predictor

A Machine Learning project that predicts **food delivery time in minutes** using **Linear Regression** based on factors such as distance, preparation time, courier experience, weather, traffic level, time of day, and vehicle type.

The project covers the complete basic Machine Learning workflow:

**Data Loading → Data Cleaning → Exploratory Data Analysis → Feature Engineering → Train/Test Split → Model Training → Evaluation → Streamlit Deployment**

---

## 📌 Project Overview

Food delivery time depends on multiple factors. This project uses historical food delivery data to understand the relationships between these factors and delivery time, and then builds a **Linear Regression model** to predict the expected delivery duration.

The trained model can also be used through an interactive **Streamlit web application** where users enter delivery details and receive an estimated delivery time.

---

## 🎯 Objectives

* Analyze the factors affecting food delivery time.
* Clean and preprocess the dataset.
* Perform Exploratory Data Analysis (EDA).
* Encode categorical variables using One-Hot Encoding.
* Train a Linear Regression model.
* Evaluate model performance using **R²** and **Adjusted R²**.
* Build an interactive Streamlit interface for predictions.

---

## 📂 Dataset

The project uses the following dataset:

```text
Food_Delivery_Times.csv
```

### Features

| Feature                  | Description                                |
| ------------------------ | ------------------------------------------ |
| `Order_ID`               | Unique identifier for each order           |
| `Distance_km`            | Delivery distance in kilometers            |
| `Weather`                | Weather condition during delivery          |
| `Traffic_Level`          | Traffic level during delivery              |
| `Time_of_Day`            | Time period of the delivery                |
| `Vehicle_Type`           | Vehicle used for delivery                  |
| `Preparation_Time_min`   | Food preparation time in minutes           |
| `Courier_Experience_yrs` | Courier experience in years                |
| `Delivery_Time_min`      | Target variable — delivery time in minutes |

---

## 🧹 Data Preprocessing

The dataset was checked for:

* Dataset shape
* Data types
* Missing values
* Duplicate rows
* Basic statistical information

### Missing Value Handling

Missing values in categorical columns were replaced using the **mode**:

```python
cat_cols = ['Weather', 'Traffic_Level', 'Time_of_Day']

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])
```

Missing values in `Courier_Experience_yrs` were replaced using the **median**:

```python
df['Courier_Experience_yrs'] = df['Courier_Experience_yrs'].fillna(
    df['Courier_Experience_yrs'].median()
)
```

---

## 📊 Exploratory Data Analysis

Several visualizations were used to understand the dataset.

### Numerical Feature Analysis

The project explores:

* Distance vs Delivery Time
* Preparation Time vs Delivery Time
* Courier Experience vs Delivery Time
* Numerical feature distributions

### Categorical Feature Analysis

Delivery time was compared across:

* Weather
* Traffic Level
* Time of Day
* Vehicle Type

Box plots were used to visualize differences between categorical groups.

### Correlation Analysis

A correlation matrix was created to examine linear relationships between numerical variables and the target variable.

### Key Observations

From the EDA:

* Delivery time generally increases as **distance increases**.
* **Preparation time** has a positive relationship with delivery time.
* **Courier experience** shows a weaker relationship with delivery time compared with distance and preparation time.

---

## ⚙️ Feature Engineering

Categorical variables were converted into numerical features using **One-Hot Encoding**:

```python
df_dummies = pd.get_dummies(
    df_copy,
    columns=[
        'Weather',
        'Traffic_Level',
        'Time_of_Day',
        'Vehicle_Type'
    ]
)
```

The `Order_ID` and target column were removed from the feature matrix:

```python
X_dummies = df_dummies.drop(
    ['Delivery_Time_min', 'Order_ID'],
    axis=1
)

y_dummies = df_dummies['Delivery_Time_min']
```

---

## 🤖 Machine Learning Model

The project uses:

### Linear Regression

Linear Regression was selected as the primary model for predicting delivery time.

```python
from sklearn.linear_model import LinearRegression

model_dummies = LinearRegression()

model_dummies.fit(X_train, y_train)
```

### Train/Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_dummies,
    y_dummies,
    test_size=0.2,
    random_state=42
)
```

---

## 📈 Model Evaluation

The trained model was evaluated on unseen test data.

### R² Score

```text
R² = 0.8196
```

### Adjusted R²

```text
Adjusted R² = 0.8159
```

An R² score of approximately **0.82** indicates that the Linear Regression model explains a substantial portion of the variation in delivery time for this dataset.

---

## 🖥️ Streamlit Web Application

An interactive Streamlit interface was created for making predictions.

Users can enter:

* Distance
* Preparation Time
* Courier Experience
* Weather
* Traffic Level
* Time of Day
* Vehicle Type

The application then returns the predicted food delivery time.

### UI Features

* Clean interactive dashboard
* User-friendly input controls
* Model information sidebar
* Delivery time prediction
* Prediction summary
* Display of model input features

---

## 🗂️ Project Structure

```text
Food-Delivery-Time-Predictor/
│
├── Food Delivery Time Predictor(2).ipynb
├── Food_Delivery_Times.csv
├── app.py
├── food_delivery_model.pkl
├── save_model.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Streamlit**
* **Joblib**
* **Jupyter Notebook / Google Colab**

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/food-delivery-time-predictor.git
```

```bash
cd food-delivery-time-predictor
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Save the Trained Model

After training the model, save it using:

```python
import joblib

joblib.dump(model_dummies, "food_delivery_model.pkl")
```

### 4. Run Streamlit

```bash
python -m streamlit run app.py
```

The application will be available at:

```text
http://localhost:8501
```

---

## 🔮 Future Improvements

Some possible improvements for the project include:

* Compare Linear Regression with other regression algorithms.
* Perform hyperparameter tuning for advanced models.
* Add more detailed model evaluation metrics such as MAE and RMSE.
* Improve the Streamlit UI with charts and prediction insights.
* Add model explainability.
* Use a larger and more diverse delivery dataset.

---

## 📚 What I Learned

Through this project, I practiced:

* Data loading and preprocessing
* Handling missing values
* Exploratory Data Analysis
* Correlation analysis
* Categorical variable encoding
* Train/test splitting
* Linear Regression
* Model evaluation
* Building an ML prediction interface with Streamlit

---

## 👨‍💻 Author

**Vishal**

B.Tech — AI & Robotics Engineering

Interested in **Artificial Intelligence, Machine Learning, Data Science, and AI Engineering**.

---

## ⭐ Acknowledgment

This project was built as a hands-on Machine Learning project to understand how regression algorithms can be applied to a real-world prediction problem.
