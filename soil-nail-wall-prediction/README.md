# Soil Nail Wall Stability Prediction 🏗️

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange)

## 📌 Project Overview
This project leverages Machine Learning to predict the **Factor of Safety (FoS)** for soil nail walls. Soil nailing is a ground reinforcement technique used to create stable retaining walls. Calculating the stability of these walls manually can be complex and time-consuming.



[Image of soil nail wall cross section]


This application uses a **Random Forest Regressor** to predict the stability based on 9 key geotechnical and geometric parameters. It includes a web-based interface built with **Streamlit** to allow engineers to input parameters and get real-time safety assessments.

## 🚀 Key Features
* **Machine Learning Pipeline:** Complete workflow including data preprocessing, scaling, hyperparameter tuning, and model evaluation.
* **High Accuracy:** Uses an optimized Random Forest model with `GridSearchCV` to ensure reliable predictions.
* **Interactive Web App:** A user-friendly interface to input soil and wall parameters.
* **Real-time Feedback:** Instantly calculates the Factor of Safety and categorizes the wall as **Stable**, **Marginal**, or **Unsafe**.

## 📊 Dataset Parameters
The model is trained on a dataset containing the following input features:

| Parameter | Unit | Description |
| :--- | :--- | :--- |
| **Friction Angle** | degrees | Internal friction angle of the soil |
| **Cohesion** | kPa | Soil cohesion strength |
| **Unit Weight** | kN/m³ | Weight of the soil per unit volume |
| **Depth** | m | Total height of the wall |
| **Embedded Depth** | m | Depth of the wall embedded in the ground |
| **Inclination Angle** | degrees | Angle of the wall inclination |
| **Diameter** | mm | Diameter of the soil nail |
| **Length** | m | Length of the soil nail |
| **Number of Nails** | count | Total number of nails used |

**Target Variable:** Factor of Safety (FoS)

## 🛠️ Technologies Used
* **Python:** Core programming language.
* **Pandas & NumPy:** Data manipulation and analysis.
* **Scikit-Learn:** Machine learning (Random Forest, Pipeline, StandardScaler).
* **Streamlit:** Web application framework.
* **Joblib:** Model serialization (saving/loading).
* **Matplotlib:** Data visualization.

## 📂 Project Structure
```text
soil-nail-wall-prediction/
│
├── data/
│   └── DataSet.xlsx          # The dataset used for training
│
├── models/
│   └── model_pipeline.pkl    # The saved trained model (generated after training)
│
├── src/
│   └── train_model.py        # Script to train and save the model
│
├── app.py                    # The Streamlit web application
├── requirements.txt          # List of dependencies
└── README.md                 # Project documentation
```
⚙️ Setup and Installation

Follow these steps to run the project locally on your machine.

1. Clone the Repository
   ```text
   git clone [https://github.com/YOUR_USERNAME/soil-nail-wall-prediction.git](https://github.com/YOUR_USERNAME/soil-nail-wall-prediction.git)
   cd soil-nail-wall-prediction
   ```
2. Install Dependencies
   ```text
   pip install -r requirements.txt
   ```
3. Train the Model
Before running the app, you need to train the model to generate the .pkl file.
  ```text
  python src/train_model.py
  ```
4. Run the Application
  ```text
  streamlit run app.py
  ```
## 🧠 Model Methodology

Preprocessing: The data is split into training (80%) and testing (20%) sets.

Scaling: A StandardScaler is used to normalize the feature values.

Algorithm: A Random Forest Regressor is used. This is an ensemble learning method that constructs multiple decision trees to output the mean prediction.

Optimization: GridSearchCV is employed to find the best hyperparameters (e.g., number of trees, max depth) using 5-fold cross-validation.
