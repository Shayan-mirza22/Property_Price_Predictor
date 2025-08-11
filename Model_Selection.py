from sklearn.model_selection import train_test_split, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np 
import joblib
import json
import pickle

X = joblib.load('X.pkl')
Y = joblib.load('Y.pkl')

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)    # test_size gives the % for splitting data into training data and test data. 
lr_clf = LinearRegression()       
lr_clf.fit(X_train, Y_train)
#print(lr_clf.score(X_test, Y_test))  # 85%

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)       # Instead of 1 split of data it does different splits according to the value of n_splits. Talking about split to test data, training data
#print(cross_val_score(LinearRegression(), X, Y, cv=cv))     # 0.82430186 0.77166234 0.85089567 0.80837764 0.83653286 0.77463657 0.84724578 0.84813854 0.84493306 0.85893313


""" Lasso is a regression technique that performs both:
Regularization -- to prevent overfitting
Feature selection -- by shrinking some coefficients to zero (effectively removing them from the model)
It is a type of linear regression that includes a penalty term (L1 penalty). """

def find_best_model_grid_search(X, Y):
    algos = {                        # Algos is a dictionary having different algos to be tested and implemented to select the best model. It has the model name and its specefic params.
        'linear_regression' : {
            'model' : LinearRegression(),
            'params' : {} # No hyperparameters needed here for vanilla LinearRegression. # These parameters are called hyperparameters
        },
        'lasso' : {
            'model' : Lasso(max_iter=1000),       # Increase max_iter to ensure convergence
            'params' : {
                'alpha' : [0.1, 1, 10],         # The Regularization Parameter alpha
                'selection' : ['random', 'cyclic']
            }
            # 'cyclic': updates coefficients one by one in order, 'random': updates coefficients in random order (faster for large data) """
        },
        'decision_tree' : {
            'model' : DecisionTreeRegressor(),
            'params' : {          # criterion: Function to measure split quality
                'criterion' : ['squared_error', 'friedman_mse'],     # 'friedman_mse': More efficient for Gradient Boosting (optional)
                'splitter' : ['best', 'random']       # 'best': chooses best split. 'random': chooses split randomly (adds randomness)
                # splitter: How to split nodes
            }
        }
    }
    scores = []     # List to append scores of different algos
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs = GridSearchCV(config['model'], config['params'], cv=cv, scoring='r2', return_train_score=False)         # Performing Grid Search to find scores of different algos
        gs.fit(X, Y)
        scores.append({
            'model' : algo_name,
            'best_score' : gs.best_score_,
            'best_params' : gs.best_params_
        })

    return pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])     # Returning the scores in a DataFrame

""" # Show all rows
pd.set_option('display.max_rows', None)

# Show all columns
pd.set_option('display.max_columns', None)

# Ensure wide columns are not cut off
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
 """
#print(str(find_best_model_grid_search(X, Y)))

# output:
"""                model  best_score                                         best_params
0  linear_regression    0.818354                                                  {}
1              lasso    0.763862               {'alpha': 0.1, 'selection': 'random'}
2      decision_tree    0.720594  {'criterion': 'squared_error', 'splitter': 'best'} """

# So seeing the output, we can say that linear reg is giving the best results!

def predict_price(location, sqft, bath, bhk):
    # X.columns are the feature names used to train the model.
    """ This line finds the index where the location column matches the given location.
        This is necessary because location was one-hot encoded — converted into many binary (0/1) columns, one for each location.
        np.where(...)[0][0] gives the first matching index. """
    loc_index = np.where(X.columns == location)[0][0]        
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    """ Since the location column is one-hot encoded, we turn on the respective index by setting it to 1.
        For example, if location = "Whitefield", and loc_index = 25, then x[25] = 1.
        All other locations remain 0. """
    return lr_clf.predict(pd.DataFrame([x], columns=X.columns))[0]


#print(predict_price('1st Phase JP Nagar', 1000, 3, 3)) 

#print(dict(zip(X.columns, lr_clf.coef_)))    # prints the coefficients 

#print(predict_price('Indira Nagar', 1000, 3, 3))


####################################################
# For deployment purposes, we do these conversions #
####################################################

columns = {
    'data_columns' : [col.lower() for col in X.columns]
}

with open("columns.json", "w") as f:
    f.write(json.dumps(columns))


with open('Model_Selection.pickle', 'wb') as f:
    pickle.dump(lr_clf, f)