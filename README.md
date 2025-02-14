# Pima Indians Diabetes Prediction

In this project, I used Logistic Regression, SVM, KNN, and SMOTE to predict the onset of diabetes based on diagnostic measures from the Pima Indians Diabetes dataset from [kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). Note that all patients are females at least 21 years old of Pima Indian heritage.

## Libraries Used
![Python](https://img.shields.io/badge/Python-3.11-blue)
![NumPy](https://img.shields.io/badge/NumPy-1.24.3-blue)
![pandas](https://img.shields.io/badge/pandas-2.0.3-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6.1-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-0.12.2-red)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.1-green)
![KNN](https://img.shields.io/badge/KNN-Algorithm-yellow)
![SVM](https://img.shields.io/badge/SVM-Imbalance%20Handling-blue)

  
## Dataset
The dataset contains 768 samples with 8 features (Age, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, glucose level, BMI) and a binary target variable (1 = diabetic, 0 = non-diabetic).

## Steps
1. Data preprocessing missing values (represented as 0) are replaced with NaN.
2. Splitting the data into training and testing sets.
3. Training a Logistic Regression model.
4. Evaluating the model using accuracy, precision, recall, and F1-score.
5. Visualization by Confusion matrices and ROC curves for each model.

## Results

![image](https://github.com/user-attachments/assets/b28895d2-6909-46af-ac6b-6bf01d23d894)


### logistic regression 
![image](https://github.com/user-attachments/assets/a7cd3acf-d991-44e9-8e91-2a26b56d998e)

(TN): 131 (correctly predicted to not have  diabetes) 😄

(FP): 19 (incorrectly predicted to have diabetes) 

(FN): 34 (incorrectly predicted to not have  diabetes)

(TP): 47 (correctly predicted to have diabetes) 😢

### Support Vector Machine (SVM)
![image](https://github.com/user-attachments/assets/93e118e1-15ac-491a-bed1-6984c7693b72)

(TN): 104

(FP): 46

(FN): 14

(TP): 67

### K-Nearest Neighbors (KNN)
![image](https://github.com/user-attachments/assets/cbfb8631-890f-48c9-a8de-74a0f7e252c3)

(TN): 124

(FP): 26

(FN): 26

(TP): 55

### smote 
![image](https://github.com/user-attachments/assets/35d68089-757f-49b9-b7f3-bca0d142ed21)

(TN): 106

(FP): 44

(FN): 13
(TP): 68

### SVM and SMOTE + SVM have the highest precision for non-diabetic cases (88% and 89%) and have the highest recall for diabetic cases (83% and 84%, respectively).
### Logistic Regression has the highest precision for diabetic cases (71%), and has the highest recall for non-diabetic cases (87%).


# So which model is the best? 🤔
For Non-Diabetic Cases (Class 0)
Logistic Regression has the highest recall (87%) and F1-score (83%).
KNN has the highest precision (83%) and matches Logistic Regression in F1-score (83%).

For Diabetic Cases (Class 1):
SMOTE + SVM has the highest recall (84%) and F1-score (70%).
SVM also performs well with a recall of 83% and F1-score of 69%.

# Conclusion
- Best for Diabetic Cases: SMOTE + SVM (highest recall and F1-score for diabetic cases).

- Best for Non-Diabetic Cases: Logistic Regression (highest recall) or KNN (highest precision).

- Overall Balanced Model: KNN (balanced precision, recall, and F1-score for both classes).




# Future improvements could include:
Experimenting with other models like Random Forest or Gradient Boosting.

Performing hyperparameter tuning to optimize model performance.

