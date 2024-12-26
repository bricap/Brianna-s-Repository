#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: briannacapuano
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import random
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
random.seed(12939850)
np.random.seed(12939850)

#seed the random number generator with your N-number

num_data = pd.read_csv("rmpCapstoneNum.csv")
qual_data = pd.read_csv("rmpCapstoneQual.csv")

# rename columns
col_names = ["avg_rating", "avg_difficulty", "num_of_ratings","received_pepper", "prop_take_again", "online_class_ratings", "male","female"]
num_data.columns = col_names
# clean the data
num_data = num_data[num_data['num_of_ratings']>5]

#1 - is there evidence of a pro-male gender bias in this dataset? use Mann Whitney sig test
male_rat = num_data[num_data['male']==1]['avg_rating'].dropna()
female_rat = num_data[num_data['female']==1]['avg_rating'].dropna()
mann_whitney_test = mannwhitneyu(male_rat, female_rat, alternative='two-sided')
p_val = mann_whitney_test.pvalue
male_avg = male_rat.mean()
female_avg = female_rat.mean()

# boxplot to visualize average ratings by gender
plt.figure(figsize=(10, 6))
plt.boxplot([male_rat, female_rat], labels=["Male", "Female"], showfliers=False)
plt.ylabel('Average Rating', fontsize=12)
plt.title('Average Ratings: Male vs. Female Professors', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# annotate mean values
plt.text(1, male_avg, f"Mean: {male_avg:.2f}", ha='center', va='bottom', fontsize=10, color='blue')
plt.text(2, female_avg, f"Mean: {female_avg:.2f}", ha='center', va='bottom', fontsize=10, color='blue')

plt.show()


#2 - effect of experience on the quality of teaching, sig test
#drop missing vals
clean = num_data[['avg_rating','num_of_ratings']].dropna()
#Spearman corr
spear_results = spearmanr(clean['num_of_ratings'],clean['avg_rating'])


#3 - relationship between average rating and average difficuly, spearmen corr test
#drop missing vals
clean3 = num_data[['avg_rating','avg_difficulty']].dropna()
#Spearman corr
spear_results3 = spearmanr(clean3['avg_rating'],clean3['avg_difficulty'])


#4 - do professors who teach more online classes recieve higher or lower ratings than those who don't - sig test, split the data
online = num_data.dropna(subset=['online_class_ratings','avg_rating'])
#split data
median_online = online['online_class_ratings'].median()
high_online = online[online['online_class_ratings']>median_online]['avg_rating']
low_online = online[online['online_class_ratings']<=median_online]['avg_rating']
#Mann-Whitney U test
online_test = mannwhitneyu(high_online, low_online, alternative = 'two-sided')
online_p = online_test.pvalue
high_mean = high_online.mean()
low_mean = low_online.mean()

# boxplot to visualize the distribution of average ratings
plt.figure(figsize=(10, 6))
plt.boxplot([high_online, low_online], labels=["High # Online", "Low # Online"], showfliers=False)
plt.ylabel('Average Rating', fontsize=12)
plt.title('Average Ratings: High vs. Low Number of Online Class Professors', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# annotate with mean values
plt.text(1, high_mean, f"Mean: {high_mean:.2f}", ha='center', va='bottom', fontsize=10, color='blue')
plt.text(2, low_mean, f"Mean: {low_mean:.2f}", ha='center', va='bottom', fontsize=10, color='blue')

plt.show()


#5 - relationship between the average rating and the proportion of people who would take the class again
data5 = num_data[['prop_take_again','avg_rating']].dropna()
#spearmann corr
spear5 = spearmanr(data5['avg_rating'], data5['prop_take_again'])


#6 - Do professors who are “hot” receive higher ratings than those who are not? - sig test
data6 = num_data.dropna(subset=['avg_rating','received_pepper'])
#split data, received pepper and didn't
pepper = data6[data6['received_pepper']==1]['avg_rating']
no_pepper = data6[data6['received_pepper']==0]['avg_rating']
#mann whitney test
test6 = mannwhitneyu(pepper, no_pepper, alternative = 'two-sided')
p6 = test6.pvalue
pep_mean = pepper.mean()
nopep_mean = no_pepper.mean()

# boxplot to visualize the distribution of average ratings
plt.figure(figsize=(10, 6))
plt.boxplot([pepper, no_pepper], labels=["Received Pepper", "No Pepper"], showfliers=False)
plt.ylabel('Average Rating', fontsize=12)
plt.title('Average Ratings: Hot vs. Not Hot Professors', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# annotate the mean values
plt.text(1, pep_mean, f"Mean: {pep_mean:.2f}", ha='center', va='bottom', fontsize=11, color='blue')
plt.text(2, nopep_mean, f"Mean: {nopep_mean:.2f}", ha='center', va='bottom', fontsize=11, color='blue')

plt.show()


#7 - regression model predicting average rating from difficulty only, include R^2 and RMSE
reg_data = num_data.dropna(subset=['avg_rating', 'avg_difficulty'])
# define predictor (X) and target variable (y)
X = reg_data[['avg_difficulty']].values
y = reg_data['avg_rating'].values
# simple linear regression
reg_model = LinearRegression()
reg_model.fit(X, y)
# predict
y_pred = reg_model.predict(X)
r_squared = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# scatter plot of actual vs. predicted
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, label="Actual Ratings", color='blue')
plt.plot(X, y_pred, color='red', linewidth=2, label="Regression Line")
plt.xlabel('Average Difficulty', fontsize=12)
plt.ylabel('Average Rating', fontsize=12)
plt.title('Regression Model: Predicting Avg Rating from Difficulty', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()


#8 - regression model predicting average rating from all available factors, include R^2 and RMSE
# how does it compare to the difficulty only model? and on indivdual betas?
# address collinearity concerns

multivar_data = num_data.dropna(subset=[
    'avg_rating', 'avg_difficulty', 'num_of_ratings', 'received_pepper',
    'prop_take_again', 'online_class_ratings', 'male', 'female'
])

# define predictor X and target y
X = multivar_data[
    ['avg_difficulty', 'num_of_ratings', 'received_pepper', 
     'prop_take_again', 'online_class_ratings', 'male', 'female']
]
y = multivar_data['avg_rating']

# split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12939850)

# define lasso regression model and hyperparameter grid
lasso = Lasso(max_iter=10000)
lasso_params = {'alpha': [0.01, 0.1, 1.0, 10.0]}

# perform cross-validation to find best alpha
lasso_cv = GridSearchCV(lasso, lasso_params, cv=5, scoring='r2')
lasso_cv.fit(X_train, y_train)

# best Lasso model w/ best alpha
best_lasso = lasso_cv.best_estimator_

# predictions and evaluate for lasso
lasso_pred = best_lasso.predict(X_test)
lasso_r2 = r2_score(y_test, lasso_pred)
lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))

# coefficients of the best lasso model
lasso_coef = pd.DataFrame({
    'Feature': ['avg_difficulty', 'num_of_ratings', 'received_pepper', 
                'prop_take_again', 'online_class_ratings', 'male', 'female'],
    'Coefficient': best_lasso.coef_
})

#print(lasso_r2, lasso_rmse, lasso_cv.best_params_, lasso_coef)

# scatter plot of actual vs. predicted ratings for the lasso model
plt.figure(figsize=(9, 7))
plt.scatter(y_test, lasso_pred, alpha=0.6, label='Predicted Ratings')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label='Perfect Prediction', linewidth=2)
plt.xlabel('Actual Ratings', fontsize=11)
plt.ylabel('Predicted Ratings', fontsize=11)
plt.title('Actual vs. Predicted Ratings (Lasso Model)', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True)
plt.show()


#9 - classification model that predicts whether a professor receives a “pepper” from average rating only
# prepare classification data
class_data = num_data.dropna(subset=['avg_rating', 'received_pepper'])
X = class_data[['avg_rating']].values  # predictor = average rating
y = class_data['received_pepper'].values #target = received pepper

# analyze class imbalance
class_distr = class_data['received_pepper'].value_counts(normalize=True)

# split the data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12939850)

# logistic regression model with class weight adjustment
log_model = LogisticRegression(class_weight='balanced', random_state=12939850)
log_model.fit(X_train, y_train)

# make the predictions and probabilities
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

# evaluate the model
auc = roc_auc_score(y_test, y_prob)
class_report_text = classification_report(y_test, y_pred)

# plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
plt.xlabel('False Positive Rate (Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('ROC Curve', fontsize=14)
plt.legend(fontsize=15)
plt.grid(True)
plt.show()

class_distr, auc, class_report_text

# plot predicted probabilities
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.6, label="Actual")
plt.scatter(X_test, y_prob, alpha=0.6, label="Predicted", color='orange')
plt.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
plt.xlabel('Average Rating', fontsize=12)
plt.ylabel('Probability of Receiving Pepper', fontsize=12)
plt.title('Classification Model: Pepper or Not based on Average Rating', fontsize=14)
plt.legend(fontsize=2)
plt.grid(True)
plt.show()


#10 - classification model that predicts whether a professor receives a “pepper” from all available factors
data10 = num_data.dropna(subset=['avg_rating', 'avg_difficulty', 'num_of_ratings', 'received_pepper',
                                 'prop_take_again', 'online_class_ratings'])

# features and target variable
X_10 = data10[['avg_rating', 'avg_difficulty', 'num_of_ratings',
               'prop_take_again', 'online_class_ratings', 'male', 'female']].values
y_10 = data10['received_pepper'].values

# split the data
X_train_10, X_test_10, y_train_10, y_test_10 = train_test_split(X_10, y_10, test_size=0.2, random_state=12939850)

# Logistic regression with class balancing
log_model_10 = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=12939850)
log_model_10.fit(X_train_10, y_train_10)

# predictions and probabilities
y_pred_10 = log_model_10.predict(X_test_10)
y_prob_10 = log_model_10.predict_proba(X_test_10)[:, 1]

# evaluate the model
auc_score_10 = roc_auc_score(y_test_10, y_prob_10)
classification_rep_10 = classification_report(y_test_10, y_pred_10)

# plot ROC curve
fpr_10, tpr_10, _ = roc_curve(y_test_10, y_prob_10)
plt.figure(figsize=(8, 6))
plt.plot(fpr_10, tpr_10, label=f'AUC = {auc_score_10:.2f}', linewidth=2)
plt.plot([0, 1], [0, 1], 'r--', linewidth=1)
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve (All Factors)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

auc_score_10, classification_rep_10

plt.figure(figsize=(10, 6))
plt.scatter(X_test_10[:, 0], y_test_10, alpha=0.6, label="Actual")
plt.scatter(X_test_10[:, 0], y_prob_10, alpha=0.6, label="Predicted", color='orange')
plt.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
plt.xlabel('Feature 1 (Ex: Avg Rating)', fontsize=12)
plt.ylabel('Probability of Receiving Pepper', fontsize=12)
plt.title('Classification Model: Pepper or Not Based on All Factors', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()


# extra credit
# load the qualitative data file
qual_data_path = 'rmpCapstoneQual.csv'
qual_data = pd.read_csv(qual_data_path, header=None, names=['Major/Field', 'University', 'US State'])

# combine qualitative and numerical data
combined = pd.concat([num_data, qual_data], axis=1)

# drop rows with missing values
combined = combined.dropna(subset=['avg_rating', 'Major/Field'])

# calculate the average rating by major/field
average_rating_by_major = combined.groupby('Major/Field')['avg_rating'].mean().sort_values(ascending=False)

# display the top majors/fields with the highest average ratings
print(average_rating_by_major.head(5))

