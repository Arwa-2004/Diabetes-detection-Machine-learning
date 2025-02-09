# Pima Indians Diabetes Prediction

In this project, I used Logistic Regression to predict the onset of diabetes based on diagnostic measures from the Pima Indians Diabetes dataset from ***kaggle**. Note that all patients are females at least 21 years old of Pima Indian heritage.

## Dataset
The dataset contains 768 samples with 8 features (Age, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, glucose level, BMI) and a binary target variable (1 = diabetic, 0 = non-diabetic).

## Steps
1. Data preprocessing adjusting/removing missing values, scaling etc...
2. Splitting the data into training and testing sets.
3. Training a Logistic Regression model.
4. Evaluating the model using accuracy, precision, recall, and F1-score.

## Results
- Accuracy: 79%
- Precision (Class 1 - Diabetic): 74%
- Recall (Class 1 - Diabetic): 60%
- F1-Score (Class 1 - Diabetic): 67%
  
### Class 0 (Non-Diabetic):

Precision (0.81): 81% of the predicted non-diabetic cases are correct.

Recall (0.89): The model correctly identifies 89% of the actual non-diabetic cases.

F1-Score (0.84): A balanced measure of precision and recall for non-diabetic cases.

### Class 1 (Diabetic):

Precision (0.74): 74% of the predicted diabetic cases are correct.

Recall (0.60): The model correctly identifies 60% of the actual diabetic cases.

F1-Score (0.67): A balanced measure of precision and recall for diabetic cases.

# Overall:

The model achieves an accuracy of 79%, meaning it correctly predicts the diabetes status for 79% of the test cases.
The macro averages (unweighted) and weighted averages (weighted by support) show consistent performance across both classes.

### Visualizations
#### Confusion Matrix
![image](https://github.com/user-attachments/assets/57696c3a-25c0-4981-a141-44e3b8bb5494)



### Key Takeaways

-The model performs well for non-diabetic cases but struggles with diabetic cases, particularly in terms of recall (60%).

-The class imbalance in the dataset (more non-diabetic cases than diabetic cases) likely contributes to this issue.

## Future improvements could include:
Addressing class imbalance using techniques like oversampling or class weights.

Experimenting with other models like Random Forest or Gradient Boosting.

Performing hyperparameter tuning to optimize model performance.

