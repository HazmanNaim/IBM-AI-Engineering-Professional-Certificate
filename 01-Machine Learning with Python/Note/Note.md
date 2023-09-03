# Week 1 : Introduction to Machine Learning
## 1.1 Course Introduction
Applications of Machine Learning
It is used heavily in the self-driving car industry to classify objects that a car might
encounter while driving, for example, people, traffic signs, and other cars.

It is used to detect and prevent attacks like a distributed denial-of-service attack or
suspicious and malicious usage.

Machine learning is also used to find trends and patterns in stock data that can help decide
which stocks to trade or which prices to buy and sell at.

Another use for machine learning is to help identify cancer in patients.
Using an x-ray scan of the target area, machine learning can help detect any potential tumors.

Machine Learning Algorithms
- Regression
- Simple linear regression
- Multiple linear regression
- Regression trees

Classification
- Logistic regression
- KNN
- SVM
- Multiclass Prediction
- Decision Trees

Clustering
- K-means

## 1.2 Welcome
**What machine learning can do?**

In the healthcare industry, data
scientists use Machine Learning to predict whether a human cell that is believed to be at risk of developing cancer is either benign or malignant. As such, Machine Learning can play a key role in determining a person's health and welfare.

A good decision tree from historical data helps doctors to prescribe the
proper medicine for each of their patients.

How bankers use Machine Learning to make
decisions on whether to approve loan applications.

How to use Machine Learning to
do bank customer segmentation, where it is not usually easy to run for huge volumes of
varied data.

How machine learning helps websites such as YouTube, Amazon,
or Netflix develop recommendations to their customers about various products or services,
such as which movies they might be interested in going to see or which books to buy.

What you will get from this course?
Skill:
regression
classification
clustering
scikit-learn
Scipy

Projects:
cancer detection
predicting economic trends
predicting customer churn
recommendation engines

## 1.3 Introduction to Machine Learning
**Machine Learning:**

Machine Learning is a subfield of computer science that allows computers to learn without being explicitly programmed. It involves building models that can analyze data, detect patterns, and make predictions based on past examples.

**Doctor's Task:**

Machine Learning can help in medical diagnosis by analyzing characteristics of human cell samples to predict whether they are benign or malignant, aiding in early cancer detection.

**Formal Definition:**

Machine Learning is the ability of computers to learn from data without being explicitly programmed to perform a specific task.

**Traditional Approach:**

Before Machine Learning, tasks like image recognition required writing rules, which were not generalized enough for accurate predictions.

**4-Year-Old Child Analogy:**

Machine Learning models learn in a way similar to how a 4-year-old child learns by observing and understanding patterns.

**Real-Life Examples:**

Machine Learning impacts various aspects of society, including personalized recommendations in services like Netflix and Amazon, credit risk assessment by banks, customer segmentation in telecommunications, chatbots, face recognition, and more.

**Machine Learning Techniques:**

There are various Machine Learning techniques, including Regression/Estimation, Classification, Clustering, Association, Anomaly Detection, Sequence Mining, Dimension Reduction, and Recommendation Systems.

**Difference Between AI, Machine Learning, and Deep Learning:**

AI is a broader field that aims to make computers intelligent in various domains. Machine Learning is a subset of AI that deals with statistical learning from examples. Deep Learning is a special field within Machine Learning that enables computers to make intelligent decisions on their own through deep neural networks.

**Upcoming Topics:**

Subsequent videos will cover the purpose of Machine Learning in real-world applications, supervised vs. unsupervised learning, model evaluation, and various Machine Learning algorithms.


## 1.4 Python for Machine Learning

**Python for Machine Learning:** 

Python is a popular and powerful programming language, favored by data scientists for its versatility and ease of use in implementing machine learning algorithms.

**NumPy:**  

NumPy is a math library for working with N-dimensional arrays efficiently, making it essential for array manipulation, data types, and image processing in Python.

**SciPy:**

SciPy is a collection of numerical algorithms and domain-specific toolboxes, offering capabilities for scientific and high-performance computation, including signal processing, optimization, and statistics.

**Matplotlib:**

Matplotlib is a widely-used plotting package providing both 2D and 3D plotting capabilities, helping visualize data effectively.

**Pandas:**

Pandas is a high-level Python library designed for easy data manipulation and analysis, particularly for numerical tables and time series data.

**SciKit Learn:**

SciKit Learn is a free and popular Machine Learning library for Python, containing various classification, regression, and clustering algorithms. It works seamlessly with NumPy and SciPy, offering straightforward implementation of machine learning models with concise code.

**Machine Learning with SciKit Learn:**

The entire process of a machine learning task can be accomplished with just a few lines of code using SciKit Learn. It covers tasks like data pre-processing, feature selection, model training, prediction, evaluation, and exporting the model.

**Standardization and Pre-processing:**

Machine learning algorithms benefit from standardized datasets, and SciKit Learn provides utility functions and transformer classes for pre-processing raw feature vectors.

**Model Training and Evaluation:**

SciKit Learn simplifies model training and evaluation. You can split datasets into train and test sets, train models with the fit method, run predictions, and evaluate accuracy using metrics like confusion matrices.

**Ease of Use:**

Using SciKit Learn, complex machine learning tasks can be accomplished much more easily compared to using just NumPy, SciPy, or pure Python programming.

**Continued Learning:**

Further videos will delve into various machine learning topics, explaining the entire process in detail.

## 1.4 Supervised vs Unsupervised

**Supervised Learning:** 

In supervised learning, we observe and direct the execution of a machine learning model by training it with labeled data. The data contains attributes (features) and their corresponding class labels or categories. Supervised learning includes two main techniques: classification (predicting discrete class labels) and regression (predicting continuous values).

**Unsupervised Learning:** 

In unsupervised learning, the model works on its own to discover hidden patterns or information in the data without the use of labeled data. Discover previously unknown information about the dataset. Unsupervised learning techniques include dimension reduction, density estimation, market basket analysis, and clustering. Clustering is commonly used to group data points with similar characteristics.

**Dimension Reduction:** 

Unsupervised learning techniques like dimension reduction and feature selection play a crucial role in simplifying the classification process by reducing redundant features.

**Market Basket Analysis:** 

Market basket analysis is a modeling technique used to identify relationships between items in a dataset. It predicts that if a certain group of items is purchased, another group of items is likely to be purchased as well.

**Density Estimation:** 

Density estimation is used to explore the data and find underlying patterns or structures within it.

**Clustering:** 

Clustering is a popular unsupervised learning technique used for grouping data points or objects with similar characteristics. It has applications in various domains, such as customer segmentation in banking or organizing music preferences for individuals.

**Difference between Supervised and Unsupervised Learning:** 

The main difference is that supervised learning uses labeled data, while unsupervised learning works with unlabeled data. Supervised learning has specific algorithms for classification and regression, while unsupervised learning has fewer models and evaluation methods, making it less controllable as the machine creates outcomes without explicit guidance.

**Note:** 

Supervised learning involves teaching a machine learning model with labeled data, allowing it to predict future instances based on its training. Unsupervised learning, on the other hand, works without labeled data and focuses on discovering patterns and structure within the data. Both approaches have distinct techniques and applications, catering to different types of problems in machine learning.

# Week 2 : Regression

## 2.1 Introduction to Regression

Regression: 

Regression is a supervised learning technique used for predicting a continuous value (dependent variable) based on one or more independent variables. It helps estimate the relationship between the independent variables and the target variable.

Dependent and Independent Variables: 

In regression, the target variable to be predicted is known as the dependent variable (Y), while the variables used to estimate the target are known as independent variables (X).

Types of Regression: 

There are two main types of regression: simple regression and multiple regression. In simple regression, one independent variable is used to estimate the dependent variable, while multiple regression involves using multiple independent variables.

Linearity: 

The relationship between the dependent and independent variables can be either linear or non-linear, depending on the nature of their correlation.

Applications of Regression: 

Regression analysis finds application in various fields, such as sales forecasting, predicting housing prices, estimating employment income, and determining individual satisfaction based on demographic and psychological factors.

Regression Algorithms: 

There are several regression algorithms available, each with specific use cases and conditions where they perform best. Exploring different regression techniques can provide valuable insights for solving various real-world problems.

Note: 

Regression is a powerful technique in supervised learning used to predict continuous values. By analyzing historical data, a regression model can be built to estimate an unknown or new data point's value based on its features. Simple regression involves using one independent variable, while multiple regression considers more than one independent variable. The linearity of the relationship between the variables can determine whether the regression is linear or non-linear. Regression finds wide-ranging applications in diverse domains, and understanding different regression algorithms allows data scientists to leverage them effectively for various problem-solving tasks.

## 2.2 Simple Linear Regression
Introduction:

Linear regression is a technique used to predict a continuous value (dependent variable) based on one or more other variables (independent variables). There are two types of linear regression models: simple regression and multiple regression.

Simple Linear Regression: 

Involves one independent variable to estimate a dependent variable. For example, predicting Co2 emission using the engine size variable.

Multiple Linear Regression: 

Involves more than one independent variable to estimate a dependent variable. For example, predicting Co2 emission using engine size and cylinders of cars.

Fitting the Line:

Linear regression fits a line through the data to model the relationship between variables. The equation for a simple linear regression model is:

$y_{hat} = θ_0 + θ_1 * x_1$

Where:

y_hat is the predicted value

x1 is the independent variable (engine size)

θ0 is the intercept

θ1 is the slope

The line is chosen to approximate the data points, indicating a linear relationship between the variables.

Finding the Best Fit Line:

The objective is to find the line that minimizes the Mean Squared Error (MSE), which is the average of all residual errors (the distance from data points to the fitted regression line). The error is also known as the residual error.

Mathematically, the MSE equation is:

$MSE = Σ(yi - y_hat)^2 / n$

Where:
yi is the actual value of the dependent variable
y_hat is the predicted value of the dependent variable
n is the number of data points
The goal is to find the values of θ0 and θ1 that minimize the MSE, making the line the best fit for the data.

Calculating θ0 and θ1:
In simple linear regression, θ0 and θ1 can be estimated using the equations:

$θ1 = Σ((xi - x̄) * (yi - ȳ)) / Σ((xi - x̄)^2)$
$θ0 = ȳ - θ1 * x̄$

Where:
xi and yi are the data points
x̄ and ȳ are the means of the independent and dependent variables, respectively.

Using the Model for Prediction:
Once θ0 and θ1 are known, predictions can be made for new data points using the linear model equation. For example, to predict Co2 emission (y) based on engine size (x) for a specific car, use the equation:

$Co2Emission = θ0 + θ1 * EngineSize$

Advantages of Linear Regression:
Linear regression is fast, simple to understand, and highly interpretable.
It doesn't require tuning of parameters like other machine learning algorithms.

Note: There are some errors in the values provided in the transcript, which are mentioned in the transcript itself but will be updated in the next version.

## 2.3 Model Evaluation in Regression Models
### Introduction:
Model evaluation is an essential step in regression to assess the accuracy of the model's predictions for unknown cases. Two common evaluation approaches are:
1. Train and Test on the Same Dataset: The model is trained on the entire dataset, and then a portion of the same dataset is used for testing to calculate accuracy. It tends to have high training accuracy but low out-of-sample accuracy, which may lead to overfitting.
2. Train/Test Split: The dataset is split into training and testing sets, which are mutually exclusive. The model is trained on the training set, and accuracy is evaluated on the testing set. This approach provides more realistic out-of-sample accuracy.

### Metrics for Accuracy:
The simplest metric for accuracy is to compare the actual values (y) with the predicted values $(y_hat)$ for the testing set. The error of the model is calculated as the average difference between predicted and actual values for all rows.

### Train and Test on the Same Dataset:
- Training accuracy: The percentage of correct predictions the model makes using the test dataset (same dataset). A high training accuracy may indicate overfitting, where the model captures noise and lacks generalization.
- Out-of-sample accuracy: The percentage of correct predictions the model makes on data it has not been trained on. Train and test on the same dataset usually have low out-of-sample accuracy due to overfitting.

### Train/Test Split:
- The dataset is split into training and testing sets, enabling more realistic out-of-sample testing. The model has no prior knowledge of the outcome of data points in the testing set.
- Ensure not to lose valuable data; after testing, you can train the model on the entire dataset.

### K-Fold Cross-Validation:
- K-Fold cross-validation addresses the variation and dependency issues seen in the previous approaches.
- The dataset is divided into K folds, and the model is trained and tested on different folds in each iteration, with accuracy averaged to produce a more consistent out-of-sample accuracy.
- K-Fold cross-validation enhances model evaluation but is not covered in-depth in this course.

Remember, proper model evaluation is crucial for building accurate and generalized regression models for predicting unknown cases.

2.4 Evaluation Metrics in Regression Models
### Introduction:
Evaluation metrics are essential to understand the performance of a model. In the context of regression, we can compare the actual values and predicted values to calculate the accuracy of the regression model. Different evaluation metrics provide insights into different aspects of model performance.

### Mean Absolute Error (MAE):
- MAE is the mean of the absolute differences between the actual and predicted values.
- It measures the average magnitude of errors without considering their direction.
- MAE is easy to understand and interpret.

### Mean Squared Error (MSE):
- MSE is the mean of the squared differences between the actual and predicted values.
- It is more popular than MAE as it penalizes larger errors more than smaller errors due to the squared term.
- Useful when large errors have a greater impact on the model evaluation.

### Root Mean Squared Error (RMSE):
- RMSE is the square root of the mean squared error.
- One of the most popular evaluation metrics as it is interpretable in the same units as the response variable (Y units).
- Provides a measure of how well the model predicts the actual values.

### Relative Absolute Error and Relative Squared Error:
- Relative Absolute Error (Residual Sum of Squares) normalizes the total absolute error by dividing it by the total absolute error of the simple predictor.
- Relative Squared Error is similar to Relative Absolute Error and is widely used for calculating R-squared.

### R-Squared (Coefficient of Determination):
- R-squared represents how close the data values are to the fitted regression line.
- It is not an error metric but a popular metric for model accuracy.
- R-squared ranges from 0 to 1, with higher values indicating a better fit of the model to the data.

Each of these metrics serves to quantify the accuracy of predictions, and the choice of metric depends on the specific model, data type, and domain knowledge. Understanding and selecting appropriate evaluation metrics are crucial for assessing and improving the performance of regression models.

## 2.5 Multiple Linear Regression
**Introduction to Multiple Linear Regression**

In this video, we'll be covering multiple linear regression, which involves using multiple independent variables to predict a dependent variable. Unlike simple linear regression, which uses only one independent variable, multiple linear regression extends the modeling capabilities to address more complex real-world scenarios.

**Applications of Multiple Linear Regression**

There are two primary applications of multiple linear regression:

1. Identifying Effects: It helps us understand the strength of the effects that independent variables have on the dependent variable. For example, we can explore whether factors like revision time, test anxiety, lecture attendance, and gender influence students' exam performance.

2. Predicting Impact: It enables us to predict how changes in independent variables affect the dependent variable while keeping other factors constant. For instance, we can predict how a person's blood pressure changes with every unit increase or decrease in body mass index, controlling for other factors.

**The Model Equation**

The multiple linear regression model is represented as:

$\[ \hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n \]$

where:
- $\(\hat{y}\)$ is the predicted value,
- $\(\theta_0\) to \(\theta_n\)$ are the coefficients,
- $\(x_1\) to \(x_n\)$ are the independent variables (features).

This equation can be represented in vector form as:

$\[ \hat{y} = \theta^T \cdot x \]$

where:
- $\(\theta\)$ is the vector of coefficients,
- $\(x\)$ is the vector of features.

**Optimizing Parameters**

The goal of multiple linear regression is to find the optimized parameters $(\(\theta\))$ that minimize the mean squared error (MSE) of the predictions. There are different methods to estimate the best coefficients, including ordinary least squares and optimization algorithms like gradient descent.

**Making Predictions**

After obtaining the optimized parameters, we can make predictions for specific sets of feature values by plugging them into the linear regression equation.

**Interpreting Coefficients**

In multiple linear regression, each coefficient represents the impact of its corresponding feature on the dependent variable. For example, a larger coefficient for the "cylinder" feature indicates a stronger influence on predicting CO₂ emission compared to the "engine size."

**Avoiding Overfitting**

Using too many independent variables without theoretical justification can lead to overfitting, where the model becomes overly complex and lacks generalization. It is essential to avoid overfitting by selecting relevant features and using techniques like regularization.

**Handling Categorical Variables**

Categorical independent variables can be incorporated into the regression model by converting them into numerical variables, such as using dummy variables.

**Checking for Linearity**

To ensure the validity of multiple linear regression, it's crucial to check for linear relationships between the dependent variable and each independent variable. Scatter plots can help visualize linearity, and if it's not present, non-linear regression may be more appropriate.

In conclusion, multiple linear regression is a powerful tool for predicting continuous variables and understanding the relationships between multiple independent variables and a dependent variable. It allows us to gain valuable insights and make accurate predictions in various real-world scenarios.

# Week 3 : Classification
K-Nearest Neighbours
## 3.1 Introduction to Classificiation

**Introduction to Classification:**
- Classification is a supervised learning approach used to categorize or classify unknown items into discrete classes.
- It establishes a relationship between a set of feature variables and a categorical target variable.

**How Classification Works:**
- Classification algorithms determine the class label for an unlabeled test case based on a set of labeled training data points.
- Example: Loan default prediction - Using customer information and previous loan default data to predict if a customer is likely to default on a loan.

**Binary and Multi-Class Classification:**
- Binary Classification: Deals with two classes, e.g., predicting whether a customer will default or not.
- Multi-Class Classification: Involves more than two classes, e.g., predicting which drug is suitable for a patient based on response to medications.

**Business Use Cases of Classification:**
- Predicting customer categories.
- Churn detection - Identifying customers likely to switch to another provider or brand.
- Assessing customer response to advertising campaigns.

**Applications of Classification:**
- Email filtering, speech recognition, handwriting recognition, biometric identification, document classification, etc.

**Types of Classification Algorithms:**
1. Decision Trees
2. Naive Bayes
3. Linear Discriminant Analysis
4. k-Nearest Neighbors
5. Logistic Regression
6. Neural Networks
7. Support Vector Machines

Classification is a versatile and widely used tool in machine learning, applicable in various industries for tasks like customer segmentation, prediction, and document classification. By understanding different classification algorithms, we can effectively tackle a wide range of classification problems and make accurate predictions in various domains.

## 3.2 K-Nearest Neighbours
**Introduction to K-Nearest Neighbors (KNN) Algorithm:**
- K-Nearest Neighbors is a classification algorithm used for predicting the class of an unknown data point based on its similarity to known data points.
- It is a type of supervised learning algorithm and falls under the category of instance-based learning.

**Scenario:**
- Suppose a telecommunications provider has segmented its customer base into four groups based on service usage patterns.
- The goal is to predict the group membership of new customers based on demographic data like region, age, and marital status.

**Understanding KNN Intuition:**
- To predict the class of a new customer (unknown case), KNN searches for the K-nearest neighbors from the dataset based on their similarity to the unknown data point.
- The class label of the unknown case is determined by a majority vote among its K-nearest neighbors.

**Finding the K-Nearest Neighbors:**
1. Choose a value for K (the number of neighbors to examine).
2. Calculate the distance from the new case (holdout) to each data point in the training dataset.
3. Search for the K observations in the training data that are nearest to the measurements of the unknown data point.
4. Predict the response of the unknown data point using the most popular response value from the K-nearest neighbors.

**Calculating Similarity or Distance Between Data Points:**
- To determine the similarity (or dissimilarity) between two data points, we use distance metrics.
- For example, the Euclidean distance is commonly used to calculate similarity between two points in a two-dimensional space (e.g., age and income).

**Choosing the Right K:**
- Selecting the appropriate value for K is crucial for the accuracy of the model.
- A low value of K (e.g., K=1) can lead to overfitting, capturing noise in the data and lack of generalization to new cases.
- A high value of K (e.g., K=20) may result in an overly generalized model.
- The best value for K is often found through cross-validation and testing the accuracy of the model on a validation set.

**KNN for Continuous Target Prediction:**
- KNN can also be used for predicting continuous target values (regression).
- For example, predicting the price of a home based on its features (number of rooms, square footage, etc.).
- The average or median target value of the K-nearest neighbors is used to obtain the predicted value for the new case.

**Conclusion:**
- K-Nearest Neighbors is a simple yet powerful classification algorithm for making predictions based on the similarity between data points.
- It is essential to select the appropriate value for K to ensure the model's accuracy and avoid overfitting or underfitting.

## 3.3 Evaluation Metrics in Classification
## Introduction

Discussing evaluation metrics used to assess the performance of classification models. Evaluation metrics play a crucial role in model development as they provide insights into areas that may require improvement. We'll cover three main evaluation metrics for classifiers: Jaccard index, F1-score, and Log Loss.

## Jaccard Index

The Jaccard index, also known as the Jaccard similarity coefficient, is a simple accuracy measurement. Let's assume we have a churn dataset with true labels represented by 'y' and predicted values by the classifier represented by 'y_hat.' The Jaccard index is defined as the size of the intersection divided by the size of the union of the two label sets.

$Jaccard Index (J) = Intersection(y, y_hat) / Union(y, y_hat)$

For example, if we have a test set of size 10 with 8 correct predictions (8 intersections), the accuracy by the Jaccard index would be 0.66. If the entire set of predicted labels for a sample strictly matches the true set of labels, then the subset accuracy is 1.0; otherwise, it is 0.0.

## Confusion Matrix

Another way to assess classifier accuracy is by using a confusion matrix. Consider a test set with 40 rows. The confusion matrix compares corrected and wrong predictions with the actual labels. Each row in the matrix represents the actual (true) labels in the test set, and each column represents the predicted labels by the classifier.

Let's interpret the confusion matrix for a binary classifier:

| Actual/Predicted | Predicted 0 | Predicted 1 |
|------------------|-------------|-------------|
| Actual 0         | True Negative (TN) | False Positive (FP) |
| Actual 1         | False Negative (FN) | True Positive (TP) |

Based on the count of each section, we can calculate the precision and recall of each class.

$Precision = TP / (TP + FP)$

$Recall (True Positive Rate) = TP / (TP + FN)$

## F1-Score

The F1 score is the harmonic average of precision and recall, providing a balanced measure of a classifier's performance. It reaches its best value at 1 (representing perfect precision and recall) and worst at 0.

$F1-Score = 2 * (Precision * Recall) / (Precision + Recall)$

For example, if the F1-score for class 0 (churn=0) is 0.83 and for class 1 (churn=1) is 0.55, the average accuracy for this classifier is the mean of the F1-scores for both labels, which is 0.72 in this case.

Please note that both the Jaccard index and F1-score can be used for multi-class classifiers as well, but that is beyond the scope of this course.

## Log Loss

In some cases, the output of a classifier is the probability of a class label, rather than the label itself. For example, in logistic regression, the output can be the probability of customer churn (yes or equals to 1), which ranges between 0 and 1.

Logarithmic Loss (Log Loss) measures the performance of a classifier when the predicted output is a probability value between 0 and 1. Lower log loss indicates better accuracy.

Log Loss (Logarithmic Loss) is calculated as the negative logarithm of the predicted probability of the correct class:

$Log Loss = - Σ [ y * log(y_hat) + (1 - y) * log(1 - y_hat) ]$

Where:
- y is the true label (0 or 1) of the sample.
- y_hat is the predicted probability of the sample belonging to class 1 (churn=1).

This formula penalizes the model for incorrect predictions and measures how well the predicted probabilities match the true labels. Lower Log Loss values indicate better accuracy and alignment between predicted probabilities and actual labels. The average Log Loss is then calculated across all rows of the test set.

## Conclusion

Evaluation metrics like Jaccard index, F1-score, and Log Loss help assess the accuracy and effectiveness of classifiers. These metrics play a critical role in fine-tuning models and identifying areas for improvement in classification tasks.

Decision Trees
# 3.4 Introduction to Decision Trees
Decision trees are a powerful and popular machine learning algorithm used for classification and regression tasks. In this video, we will introduce decision trees and explore how they can be utilized for classification. Let's dive in!

## What is a Decision Tree?

A decision tree is a tree-like model that represents a series of decisions and their possible outcomes. It is a flowchart-like structure where each internal node represents a test on a specific feature, each branch represents the outcome of the test, and each leaf node represents a class label or a value. Decision trees are widely used for making decisions in a structured and systematic manner.

## Building a Decision Tree

Let's understand how to build a decision tree using a practical medical research scenario. Imagine you are a medical researcher studying a group of patients suffering from the same illness. These patients responded to one of two medications, drug A and drug B. Your task is to build a model that can predict which drug is suitable for a future patient with similar characteristics.

### Dataset Description

The dataset contains features like age, gender, blood pressure, and cholesterol for each patient, and the target variable is the drug that each patient responded to (drug A or drug B).

### Decision Tree Construction

1. **Start with the Root Node:** We begin by considering all the data at the root node, which represents the entire dataset.

2. **Attribute Selection:** We select an attribute from the dataset that we want to use as a decision point. In our example, the attribute "age" is chosen.

3. **Split the Data:** We split the dataset into distinct nodes based on the selected attribute's values. For instance, if the patient is middle-aged, we directly recommend drug B. If the patient is young or senior, we need more information to decide.

4. **Further Decision Variables:** To make a more informed decision for young or senior patients, we can introduce additional decision variables such as cholesterol levels, gender, or blood pressure.

5. **Continue Splitting:** We continue this process by selecting the most relevant attributes and splitting the data accordingly, branching into different paths based on the outcomes of the tests.

6. **Leaf Nodes:** Ultimately, each path leads to a leaf node, where a patient is assigned to a specific class (drug A or drug B) based on their characteristics.

## Conclusion

In summary, decision trees are a valuable tool for making decisions based on a series of tests and their outcomes. They are widely used in various fields, including medicine, finance, and industry. By constructing decision trees from historical data, we can predict the appropriate course of action for new and unknown cases, thus assisting in making informed decisions. In the next video, we will delve into the process of calculating the significance of attributes, a crucial step in building an effective decision tree.

# 3.5 Building Decision Trees
Exploring the process of building decision trees using a practical example of the drug dataset. The goal is to construct a decision tree that can accurately classify patients into two categories: those who respond to drug A and those who respond to drug B. Let's go through the steps of building a decision tree:

## Recursive Partitioning

Decision trees are constructed using a process called recursive partitioning, which involves repeatedly splitting the data into distinct nodes based on the most predictive features.

## Attribute Selection

The first step is to choose the most relevant attribute for splitting the data. This attribute should be the one that best separates the data into distinct categories. To do this, we calculate the **entropy** of each attribute.

## Entropy

Entropy is a measure of the amount of information disorder or randomness in the data. It is calculated for each node in the decision tree and is used to assess the homogeneity of the samples in that node.

- Entropy is 0 when all the data in a node belongs to a specific category (e.g., all patients respond to drug A or drug B).
- Entropy is 1 when the data in a node is equally divided among different categories (e.g., half the patients respond to drug A, and half respond to drug B).

The formula to calculate entropy for a node with 'n' samples, where 'p' samples belong to class A and 'q' samples belong to class B, is:

$\[ \text{Entropy} = - \left( \frac{p}{n} \log_2 \frac{p}{n} + \frac{q}{n} \log_2 \frac{q}{n} \right) \]$

Where:
- $\( p \)$ is the number of samples belonging to class A in the node.
- $\( q \)$ is the number of samples belonging to class B in the node.
- $\( n = p + q \)$ is the total number of samples in the node.

## Information Gain

Information gain is the key concept in building decision trees. It represents the reduction in entropy after splitting the data based on an attribute. The attribute that results in the highest information gain is selected as the splitting attribute.

The formula to calculate information gain for an attribute after splitting the data is:

$\[ \text{Information Gain} = \text{Entropy before split} - \text{Weighted average of entropies after split} \]$

Where:
- **Entropy before split** is the entropy of the original node before splitting.
- **Weighted average of entropies after split** is the sum of entropies of all resulting nodes after splitting, weighted by the proportion of samples in each node.

The attribute that gives the highest information gain is selected as the splitting attribute at each step of building the decision tree.

## Building the Decision Tree

1. **Start at the Root Node:** Begin with the entire dataset at the root node.

2. **Select Splitting Attribute:** Calculate the entropy for each attribute and choose the attribute that provides the highest information gain.

3. **Split the Data:** Divide the data into distinct branches based on the chosen attribute.

4. **Repeat the Process:** For each branch, recursively repeat the attribute selection and data splitting process until the leaves become pure (100% of samples belong to a specific category) or a stopping criterion is reached.

5. **Leaf Nodes:** The final nodes of the tree, where no further splitting is possible, are called leaf nodes. Each leaf node assigns a patient to a specific class (drug A or drug B) based on their characteristics.

## Conclusion

Building decision trees involves selecting the most relevant attribute for splitting the data to achieve the highest information gain and reduce entropy. The process continues recursively until the tree reaches a state where further splitting does not significantly improve the purity of the leaves. By constructing decision trees in this way, we can make informed and accurate predictions for new and unknown cases based on their attributes.

Week 4 : Linear Classification
Logistic Regression
# 4.1 Logistic Regression for Classification

## Introduction
- Logistic Regression is a statistical and machine learning technique used for **classification**.
- This method helps us answer three key questions:
    1. What is logistic regression?
    2. What types of problems can logistic regression solve?
    3. In which situations is logistic regression applicable?

## Logistic Regression Overview
- Logistic regression classifies records in a dataset based on input field values.
- Example scenario: Telecommunication dataset analyzing customer churn (customers leaving the company).
- Logistic regression helps build a predictive model for customer churn based on historical data.
- Features in the dataset: Services, customer account info, demographics, churn status.
- **Churn** column indicates whether a customer left the company (binary outcome).

## Logistic Regression vs. Linear Regression
- Logistic regression predicts a **categorical or discrete** target field (e.g., binary classes).
- Linear regression predicts continuous values (e.g., price, blood pressure).
- Independent variables (features) should be continuous or transformed for logistic regression.
- Logistic regression handles binary and multi-class classification.

## Applications of Logistic Regression
- **Heart Attack Prediction**: Probability of heart attack based on age, sex, BMI.
- **Medical Mortality Prediction**: Likelihood of patient mortality based on injury.
- **Disease Diagnosis**: Predicting diseases like diabetes from patient characteristics.
- **Marketing**: Predicting product purchase likelihood, subscription halt.
- **Process Failure Prediction**: Probability of process, system, or product failure.
- **Mortgage Default Prediction**: Likelihood of homeowner mortgage default.

## When to Use Logistic Regression
- Situations where logistic regression is suitable:
    1. **Binary Target**: When the target field is binary (e.g., yes/no, churn/no churn).
    2. **Probability Needed**: When probability of prediction is required (e.g., customer buying a product).
    3. **Linear Separability**: Data is linearly separable (decision boundary is line/plane/hyperplane).
    4. **Feature Impact Understanding**: When understanding impact of features is important.
- Logistic regression helps identify statistically significant features and their effects on prediction.

## Formalization of the Problem
- **Dataset**: X is a dataset of m features and n records (real numbers).
- **Target**: Y represents the class (0 or 1) we want to predict.
- The goal of logistic regression is to build a model (Y hat) to predict class and probability for each sample (customer).

**Dataset and Target:**
- The dataset X is represented as a matrix of dimensions m x n, where m is the number of features and n is the number of records.
    - $\( X = \begin{bmatrix}
      x_{11} & x_{12} & \cdots & x_{1n} \\
      x_{21} & x_{22} & \cdots & x_{2n} \\
      \vdots & \vdots & \ddots & \vdots \\
      x_{m1} & x_{m2} & \cdots & x_{mn}
    \end{bmatrix} \)$
- The target Y is a binary class variable represented as a column vector of dimensions n x 1.
    - $\( Y = \begin{bmatrix}
      y_1 \\
      y_2 \\
      \vdots \\
      y_n
    \end{bmatrix} \)$

**Logistic Regression Model:**
- The logistic regression model $(\( \hat{Y} \))$ predicts the probability of a sample (customer) belonging to class 1.
    - $\( \hat{Y} = P(Y = 1 | X) \)$

**Probability Calculation:**
- The probability of a sample (customer) belonging to class 0 can be calculated as:
    - $\( P(Y = 0 | X) = 1 - P(Y = 1 | X) \)$

# 4.2 Linear Regression and Logistic Regression
## Introduction
- This video explains the key differences between **linear regression** and **logistic regression**.
- It discusses the limitations of linear regression for binary classification problems and introduces the **sigmoid function** as a crucial component of logistic regression.

## Logistic Regression Goals
- Logistic regression aims to predict the **class** of each customer and the **probability** of each sample belonging to a class.
- It seeks to build a model $(\( \hat{y} \))$ that estimates the class of a customer given its features $(\( x \))$.
- $\( y \)$ represents the actual labels, and $\( \hat{y} \)$ represents predicted values by the model.

## Linear Regression for Binary Classification
- Linear regression cannot be directly used for binary classification.
- Example scenario: Predicting churn (categorical) using age (continuous) as the feature.
- Linear regression fits a line $(\( y = a + bx \))$ through data points for continuous outcomes.

## Sigmoid Function Introduction
- **Sigmoid function** is a key component of logistic regression.
- It maps any input value to a value between 0 and 1.
- It can transform linear regression's continuous output into a probabilistic range.

## Sigmoid Function Details
- The sigmoid function resembles a step function but produces a smooth curve.
- Sigmoid function equation: $\( \text{sigmoid}(z) = \frac{1}{1 + e^{-z}} \)$.
- When $\( z \)$ (input to sigmoid) is large, $\( e^{-z} \)$ approaches 0, and sigmoid value approaches 1.
- When $\( z \)$ is small, $\( e^{-z} \)$ approaches infinity, and sigmoid value approaches 0.

## Logistic Regression Model with Sigmoid
- In logistic regression, the model predicts the **probability** that an input belongs to class 1.
- The probability of class 0 can be calculated as $\( 1 - \text{sigmoid}(\Theta^T x) \)$.

## Training the Logistic Regression Model
- The goal is to find optimal parameter values $(\( \Theta \))$ for the sigmoid function.
- Steps of training process:
    1. Initialize $\( \Theta \)$ randomly.
    2. Calculate model output using sigmoid: $\( \hat{y} = \text{sigmoid}(\Theta^T x) \)$.
    3. Compare $\( \hat{y} \)$ with actual label $\( y \)$ and calculate error.
    4. Calculate total error (cost) for all customers using a cost function.
    5. Adjust $\( \Theta \)$ to minimize the cost using techniques like gradient descent.
    6. Iterate steps 2 to 5 until cost is sufficiently minimized.

## Conclusion
- Logistic regression addresses binary classification problems with a probabilistic approach.
- Sigmoid function transforms linear regression output into probabilities.
- Training process adjusts model parameters for accurate predictions.

## 4.3 Training the Logistic Regression Model

### Objective
The main goal of training a logistic regression model is to find the optimal parameters \( \Theta \) that best estimate the labels of the samples in the dataset.

### Cost Function
The cost function measures the error between the actual labels $\( y \)$ and the predicted outputs $\( \hat{y} \)$ of the logistic regression model. It quantifies how well the model is performing.

The general formula for the cost function $\( J(\Theta) \)$ for logistic regression is given by:
$\[ J(\Theta) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log(\hat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] \]$

Where:
- $\( n \)$ is the number of training samples.
- $\( y^{(i)} \)$ is the actual label for the $\( i \)$-th sample.
- $\( \hat{y}^{(i)} \)$ is the predicted output for the $\( i \)$-th sample.

### Gradient Descent
Gradient descent is an optimization technique used to update the parameter values $\( \Theta \)$ iteratively in order to minimize the cost function. It works by calculating the gradient of the cost function with respect to each parameter and updating the parameters in the opposite direction of the gradient.

The update rule for gradient descent is as follows:
$\[ \Theta_j := \Theta_j - \alpha \frac{\partial}{\partial \Theta_j} J(\Theta) \]$

Where:
- $\( \alpha \)$ is the learning rate, which controls the step size in each iteration.
- $\( \Theta_j \)$ is the $\( j \)$-th parameter of the parameter vector $\( \Theta \)$.
- $\( \frac{\partial}{\partial \Theta_j} J(\Theta) \)$ is the partial derivative of the cost function with respect to $\( \Theta_j \)$, which indicates the direction and magnitude of the update.

Gradient descent proceeds iteratively by updating the parameters until convergence, which is achieved when the cost function reaches a minimum or a predefined stopping criterion is met.

### Summary of Training Algorithm
1. Initialize the parameters $\( \Theta \)$ with random values.
2. Calculate the cost function $\( J(\Theta) \)$ based on the current parameter values.
3. Calculate the gradient of the cost function with respect to each parameter $\( \Theta_j \)$.
4. Update the parameters $\( \Theta_j \)$ using the gradient descent update rule.
5. Repeat steps 2 to 4 until convergence or a maximum number of iterations is reached.

By following this iterative process, the logistic regression model gradually adjusts its parameters to minimize the cost function, resulting in improved predictions and better classification accuracy.


## 4.4 Support Vector Machine
### Introduction to the Problem
Imagine you have a dataset of human cell samples from patients at risk of developing cancer. The dataset contains various characteristics of these cell samples, and you want to use this information to classify cells as benign or malignant.

### Formal Definition of SVM
A Support Vector Machine (SVM) is a supervised algorithm used for classification. SVM works by first transforming the data into a higher-dimensional space, where data points can be categorized even when they are not linearly separable. It then estimates a separator, or hyperplane, in this transformed space to classify new cases.

### Data Transformation and Kernels
Transforming data into a higher-dimensional space is called "kernelling." It involves using a kernel function to map the data into a new space where it becomes linearly separable. Different types of kernel functions, such as linear, polynomial, Radial Basis Function (RBF), and sigmoid, can be used for this purpose. Choosing the right kernel depends on the dataset, but you don't need to know the details of these functions as they are already implemented in data science libraries.

### Finding the Optimized Separator
The main goal of SVM is to find the best hyperplane that separates the data into different classes. This hyperplane should have the maximum margin between the two classes. Support vectors are examples closest to the hyperplane and are crucial for defining the optimal separator. SVM aims to maximize the margin between the hyperplane and the support vectors.

### Optimization and Decision Boundary
The process of finding the optimized hyperplane involves an optimization procedure that maximizes the margin. This optimization problem can be solved using techniques like gradient descent. The output of the algorithm is the values of \( w \) and \( b \) for the line equation, which can be used for classification. By plugging in input values, you can determine whether an unknown point belongs to a certain class based on whether the equation's value is positive or negative.

### Advantages and Disadvantages of SVM
Support Vector Machines have several advantages, including accuracy in high-dimensional spaces and efficiency in memory usage due to the utilization of support vectors. However, they can be prone to overfitting if the number of features is much greater than the number of samples. SVMs also do not directly provide probability estimates and may not be computationally efficient for very large datasets.

### Applications of SVM
SVM is useful for various applications, such as image analysis (classification and digit recognition), text mining (spam detection, sentiment analysis), gene expression data classification, regression, outlier detection, and clustering.

SVM is a versatile algorithm that can handle different types of machine learning problems, particularly those involving high-dimensional data or complex classification tasks.

## 4.5 Multiclass Prediction

# Week 5 : Intro to Clustering
K-Means Clustering
## 5.1 Intro to Clustering
### Introduction to Customer Segmentation
Imagine you have a dataset of customer information and you want to group customers with similar characteristics together. This is called customer segmentation, which helps businesses target specific groups for more effective marketing. Clustering is an analytical approach to deriving these segments from large datasets based on similarities among customers.

### Clustering for Customer Segmentation
Clustering is a technique used for customer segmentation. It groups similar data points together, creating mutually exclusive clusters. For instance, customers with similar demographics could be grouped into clusters. This allows businesses to create profiles for each group and personalize marketing strategies.

### Clustering Process and Applications
Clustering involves finding groups of data points that are similar within a group but dissimilar to points in other groups. Unlike classification, where data is labeled, clustering is an unsupervised process. It has applications beyond customer segmentation, such as recommendation systems, fraud detection, and more.

## Different Clustering Algorithms
Clustering can be done using various algorithms, each with its characteristics. For example:
- Partition-based algorithms like K-Means, K-Medians, and Fuzzy c-Means create spherical clusters and are efficient for medium to large datasets.
- Hierarchical clustering produces tree-like structures and is intuitive for small datasets.
- Density-based algorithms like DBSCAN are suitable for arbitrary-shaped clusters and can handle noise in data.

## 5.2 Intro to k-Means
**Introduction:**
K-Means Clustering is a technique used for customer segmentation and grouping similar data points together based on their characteristics.

**Types of Clustering:**
- Partitioning, Hierarchical, Density-Based
- K-Means falls under partitioning clustering

**K-Means Overview:**
- Divides data into K non-overlapping clusters
- Unsupervised algorithm
- Minimizes intra-cluster distances, maximizes inter-cluster distances

**Distance Metrics:**
- Measure similarity/dissimilarity
- Commonly used: 

   1. **Euclidean Distance:**
      - Formula: $\sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$
      - Measures straight-line distance between points
      
   2. **Minkowski Distance:**
      - Formula: $\left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{\frac{1}{p}}$
      - Generalization of Euclidean and Manhattan distances
      - For p = 2, Minkowski distance reduces to Euclidean distance

**K-Means Process:**

1. **Initialization:**
   - Choose number of clusters (K)
   - Select K initial centroids (cluster centers)

2. **Assignment:**
   - Calculate distance of each data point from centroids
   - Assign each data point to nearest centroid (forms clusters)

3. **Update Centroids:**
   - Calculate mean of data points in each cluster
   - New centroid becomes the mean

4. **Repeat:**
   - Reassign data points to nearest centroids
   - Update centroids based on new cluster memberships
   - Repeat until centroids stabilize (convergence)

**Convergence:**
- Minimizes within-cluster sum of squares error
- Shape clusters to minimize total distance from centroid

**Challenges:**
- Initial centroids impact final result
- Heuristic algorithm, may converge to local optimum

**Solution:**
- Run K-Means multiple times with different starting centroids
- Pick the best outcome among runs

**Conclusion:**
K-Means Clustering is a powerful technique for customer segmentation and other clustering tasks. While it's a heuristic algorithm with no guarantee of finding the best solution, it provides valuable insights for data grouping and analysis.

## 5.3 More on k-Means
A k-Means algorithm works by following these steps:
1. Randomly place k centroids, one for each cluster. The clusters are ideally placed farther apart.
2. Calculate the Euclidean distance of each data point from the centroids. Other distance measurements can also be used.
3. Assign each data point to its closest centroid, forming groups.
4. Recalculate the position of k centroids based on the mean of all points in the group.
5. Repeat the process until the centroids no longer move.

### Evaluating Clustering Accuracy:
- Ground truth comparison is challenging since k-Means is unsupervised and lacks labels.
- A way to assess cluster quality is by evaluating the average distance within a cluster.
- Another metric is the average distance of data points from their cluster centroids, representing clustering error.

### Determining the Number of Clusters (k):
- Choosing the right k is crucial, but challenging due to data distribution.
- The elbow method is a common approach:
  - Plot the metric (e.g., mean distance) as a function of k.
  - Observe the point where the rate of decrease sharply shifts, indicating the elbow point.
  - The elbow point represents a balanced trade-off between reducing error and overfitting.

### Recap of k-Means Characteristics:
- k-Means is partition-based clustering.
- It is efficient for medium to large data sets.
- It produces spherical clusters around centroids.
- Drawback: Requires pre-specification of the number of clusters, which can be challenging.

Overall, k-Means is a powerful clustering algorithm with practical applications, but careful consideration is needed for selecting the appropriate number of clusters.

