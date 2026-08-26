# 🛵 Food Delivery Time Predictor

A Machine Learning project that predicts **food delivery time in minutes** using **Linear Regression**.

The model uses delivery-related features such as distance, preparation time, courier experience, weather, traffic level, time of day, and vehicle type.

The project also includes an interactive **Streamlit web application** for making predictions.

## 🚀 Project Overview

The goal of this project is to build a regression model that can estimate how long a food order will take to be delivered.

The project follows this workflow:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
One-Hot Encoding
      ↓
Train/Test Split
      ↓
Linear Regression
      ↓
Model Evaluation
      ↓
Streamlit Deployment
```

## 🎯 Features Used

The model uses the following features:

| Feature                  | Description                      |
| ------------------------ | -------------------------------- |
| `Distance_km`            | Delivery distance in kilometers  |
| `Weather`                | Weather condition                |
| `Traffic_Level`          | Traffic level                    |
| `Time_of_Day`            | Time period of the delivery      |
| `Vehicle_Type`           | Vehicle used for delivery        |
| `Preparation_Time_min`   | Food preparation time in minutes |
| `Courier_Experience_yrs` | Courier experience in years      |

### Target Variable

```text
Delivery_Time_min
```

This represents the predicted delivery time in minutes.

## 🧹 Data Preprocessing

The dataset was checked for:

* Missing values
* Duplicate rows
* Data types
* Statistical information
* Relationships between features

Missing values in categorical columns were handled using the mode, while missing values in `Courier_Experience_yrs` were handled using the median.

Example:

```python
df["Courier_Experience_yrs"] = df["Courier_Experience_yrs"].fillna(
    df["Courier_Experience_yrs"].median()
)
```

Categorical features were converted into numerical features using One-Hot Encoding.

```python
pd.get_dummies(
    df,
    columns=[
        "Weather",
        "Traffic_Level",
        "Time_of_Day",
        "Vehicle_Type"
    ]
)
```

## 📊 Exploratory Data Analysis

The project includes analysis and visualizations for understanding the relationship between delivery time and different features.

The analysis includes:

* Distribution of numerical features
* Correlation analysis
* Distance vs Delivery Time
* Preparation Time vs Delivery Time
* Courier Experience vs Delivery Time
* Delivery Time by Weather
* Delivery Time by Traffic Level
* Delivery Time by Time of Day
* Delivery Time by Vehicle Type

## 🤖 Machine Learning Model

The project uses:

**Linear Regression**

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
```

The dataset is divided into training and testing sets using an **80/20 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

## 📈 Model Performance

The model achieved approximately:

```text
R² Score:        0.8196
Adjusted R²:     0.8159
```

An R² score of approximately 0.82 means the model explains a substantial portion of the variation in delivery time on this dataset.

## 🖥️ Streamlit Application

The project includes a Streamlit-based user interface.

Users can enter:

* Distance
* Preparation Time
* Courier Experience
* Weather
* Traffic Level
* Time of Day
* Vehicle Type

The application then predicts the estimated delivery time.

### Run the application

```bash
python -m streamlit run app.py
```

The application will open at:

```text
http://localhost:8501
```

## 📁 Project Structure

```text
Food-Delivery-Time-Predictor/
│
├── app.py
├── Food Delivery Time Predictor.ipynb
├── Food_Delivery_Times.csv
├── food_delivery_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/food-delivery-time-predictor.git
```

Move into the project directory:

```bash
cd food-delivery-time-predictor
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
python -m streamlit run app.py
```

## 📌 Future Improvements

Possible improvements include:

* Compare Linear Regression with other regression algorithms
* Add MAE and RMSE evaluation
* Perform hyperparameter tuning
* Improve the Streamlit interface
* Add interactive visualizations
* Add model explainability
* Deploy the Streamlit application online
* Train the model using a larger dataset

## 📚 Key Learnings

Through this project, I practiced:

* Data cleaning
* Missing-value handling
* Exploratory Data Analysis
* Data visualization
* One-Hot Encoding
* Train/Test Split
* Linear Regression
* Model evaluation
* Saving trained ML models
* Building a Streamlit ML application
* Preparing an ML project for GitHub

## 👨‍💻 Author

**Vishal**

B.Tech — AI & Robotics Engineering

Interested in Machine Learning, Artificial Intelligence, Data Science, and AI Engineering.

---

⭐ If you found this project useful, consider giving the repository a star!
