# Enhancing Weak Learners to Predict Loan Defaulting
## Abstract

  

Predicting if a loan will default is a necessary skill for all lenders. Without accurate assessments of potential borrowers, lenders can lose potential profits and damage their reputation. In an attempt to model loan defaulting, weak learners were utilized and enhanced through boosting and locally weighted methods. The boosting methods included CatBoost, XGBoost, and LightGBM while the locally weighted method included LogisticRegression. Additionally, considering a loan is more likely to not default, the data analyzed had a large class imbalance. To remedy this, oversampling methods SMOTE and ADASYN were employed with SMOTE proving to work best. The best model created was a XGBoost model trained on SMOTE data and reached an accuracy of around 93%. The final model is not as accurate as desired, likely due to the dataset lacking other crucial information for predicting if a loan will default.

## Introduction

  

The predicting of loan defaulting is an extremely important and complex issue. As a form of active risk management, being able to determine which borrowers are more likely to default on their loans can save lenders tons of money. Without knowledge of a borrower's likelihood to default, lenders may consistently give out defaulting loans, costing the lender financially and reputationally.

  

Considering the large amount of loan applications and previous data available, a model capable of quickly producing accurate results is imperative. With this in mind, weak learners provided the model training speed desired but failed to accurately predict the data. In order to achieve the desired accuracy, these weak learners were enhanced through boosting and locally weighted methods. As a result of these enhancements to the weak learners, the hope is to maintain the training speed seen previously but increase accuracy to provide meaningful insights for lenders.

  

Boosting methods CatBoost, XGBoost, and LightGBM were employed due to their boosting properties. CatBoost, an open source library for gradient boosting decision trees, comes with categorical functionality and offers quick, accurate models [4], making it an obvious choice for the data. XGBoost offers high accuracy boosted models [7] and is likely the most widely used boosting library today, making it a necessary addition to any study investigating the performance of boosted models. Finally, LightGBM is Microsoft's gradient boosting library that boasts faster training speeds and lower memory usage than other boosting methods [3], lending itself to this data as model speed is important. Additionally, locally weighted logistic regression will be employed as it typically offers greater accuracy than standard regression models.

  

To remedy class imbalance in the dataset, Imblearn's SMOTE and ADASYN [2] will be utilized to create synthetic samples for the minority class. By doing this, the model can be exposed to more data of the minority class and hopefully improve its accuracy in predicting it. The remainder of this paper will focus on the dataset analyzed, the specific models employed, and discuss the implications and findings of the results.

## Description of Data

  

The dataset analyzed comes from Kaggle (https://www.kaggle.com/datasets/subhamjain/loan-prediction-based-on-customer-behavior) and was collected by Univ.AI. The data consists of multiple features describing the loan applicant at the time of their loan application and the target variable is binary, 0 for not defaulting on their loan and 1 for defaulting. The dataset contains three tables; training data, testing data, and the correct classifications for the test data. The training data contains 252,000 samples and the test data contains 28,000 samples.

  

There are 11 features:

- **Income:** Yearly income for the loan applicant

- **Age:** Age of the applicant in years

- **Experience:** Total years of applicants work experience

- **Married/Single:** Whether or not the applicant is married or single. Has 2 categories: married or single.

- **House_Ownership:** The applicant's home ownership status. Has 3 categories: rent, own, neither rent nor own

- **Car_Ownership:** Whether or not the applicant has a car. Has 2 categories: yes or no

- **Profession:** The applicant's profession. Has 52 categories

- **CITY:** The applicant's city of residence. Has 318 categories

- **STATE:** The applicant's state of residence. Has 30 categories

- **CURRENT_JOB_YEARS:** Number of years the applicant has worked their current job

- **CURRENT_HOUSE_YEARS:** Number of years the applicant has lived in their current house

  

And one target:

- **Risk_Flag:** 0 if the borrower did not default on their loan, 1 if the borrower did default on their loan

## Methods

  

- **Preprocessing Methods**
  

For preprocessing, sklearn's LabelEncoder [6] was utilized to convert the categorical features (Married/Single, House_Ownership, Car_Ownership, Profession, CITY, and STATE) into numerical values. No other preprocessing had to be done as the dataset did not have any null values or other issues.

  
  

- **Machine Learning Methods**

  

**CatBoost**

  

With default parameters, CatBoostClassifier yielded a cross-validated accuracy of **88.8%**. Utilizing Bayesian Optimization [1], optimal hyperparameters were found and only moderately improved the model's cross-validated accuracy to **89.5%**. Following this, oversampling techniques SMOTE and ADASYN were employed to better train the model on defaulting loans. Despite these techniques typically increasing model performance, the use of SMOTE and ADASYN dropped the model's cross-validated accuracy to **88%** and **63.4%** respectively.

  

**XGBoost**

  

With default parameters, XGBClassifier yielded a cross-validated accuracy of **88.9%**. RandomSearchCV [6] was utilized to find optimal hyperparameters and yielded a cross-validated accuracy of **89.9%**. Unlike CatBoost, the use of an oversampling technique improved the cross-validated accuracy of the model. For SMOTE, the models cross-validated accuracy rose to **91.7%**. However, similarly to CatBoost, ADASYN oversampling caused the accuracy to fall, this time to **73.5%**.

  

**LightGBM**

  

By far the fastest to train, LightGBM yielded an accuracy of **87.7%** with default parameters. Interestingly, the use of Bayesian Optimization and RandomSearchCV gave optimal parameters that did not improve the accuracy of the model. Additionally, the utilization of SMOTE and ADASYN completely ruined the performance of the model. For both SMOTE and ADASYN the cross-validated accuracy was around **44%**, an absolutely terrible predictor at loan defaulting.

  

**Locally Weighted Logistic Regression**

  

By far the longest to fit on the data, Locally Weighted Logistic Regression yielded an accuracy of **68%** after 2 1/2 hours of training. The implementation came from meanxai on YouTube [5] and utilized sklearn's LogisticRegression class [6]. Despite the long training time, it performed considerably worse than the boosting methods. Considering the computation time for Locally Weighted Logistic Regression, SMOTE and ADASYN data could not be fitted with this model as Google Colab runtime consistently disconnected a few hours into training.

  

- **Final Method**

  

Considering the accuracy of XGBoost with SMOTE data, that will be the final model for predictions. Using 10 folds with sklearn's KFold led to a cross-validated accuracy of **92.6%**, by far the best accuracy found by any model.


## Discussion

  

Considering the best model created, XGBoost with SMOTE data, was only able to reach around 93% accuracy, it is likely that the dataset does not include all the necessary data for extremely accurate loan defaulting prediction. The nationwide loan defaulting rate varies greatly depending on a variety of economic factors not included in the dataset. Additionally, the data does not include information about the loans themselves and simply provides information on the loan applicants. Data regarding the loans themselves could provide insights into loan defaulting prediction as certain loan characteristics could increase or decrease the likelihood of a borrower defaulting on their loan.

  

Additionally, the modeling approach taken could be insufficient for this data. Boosting methods proved to be the most accurate models for predicting, yet each boosting method yielded different results. Considering this, a heterogenous boosting method combining multiple different weak learners could prove to be the best predictor for loan defaulting. This was attempted but proved to be too difficult a task to finish by the deadline.

  

Future research into this topic should focus on expanding the dataset and creating more intricate heterogenous boosting models. With more information about the loans themselves and general economic information at the time combined with utilizing the best performing aspects of the boosting models could produce an extremely accurate model capable of determining which applicants will default on their loans with impeccable accuracy.

## References

  

[1] F. Nogueira. Bayesian Optimization: Open source constrained global optimization tool for Python, 2014. https://github.com/bayesian-optimization/BayesianOptimization

  

[2] G. Lematre, F Nogueira, and C. Aridas. Imbalanced-learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets in Machine Learning, 2017. https://jmlr.org/papers/v18/16-365.html

  

[3] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T. Liu. LightGBM: A highly efficient gradient boosting decision tree, 2017. https://dl.acm.org/doi/10.5555/3294996.3295074

  

[4] L. Prokhorenkova, G. Gusev, A. Vorobev, A. Dorogush, and A. Gulin. CatBoost: unbiased boosting with categorical features, 2017. https://arxiv.org/abs/1706.09516

  

[5] meanxai. "[MXML-4-05] Logistic Regression [5/5] - Locally Weighted Logistic Regression (LWLR)." Published April 4, 2024, https://www.youtube.com/watch?v=d1-QS4uTgj8&t=754s

  

[6] Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp.

  

[7] T. Chen and G. Guestrin. XGBoost: A Scalable Tree Boosting System, 2016.https://arxiv.org/abs/1603.02754
