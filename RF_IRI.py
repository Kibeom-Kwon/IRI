# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from time import time
import pprint
import warnings
warnings.filterwarnings('ignore')
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, VerboseCallback
from skopt.space import Integer
from sklearn.ensemble import RandomForestRegressor as RFR

pd.options.display.max_columns = 50

df =pd.read_excel('Data_IRI_final_Me_SX.xlsx')

unwanted = ['Type','No','IRI']

input_features=[ele for ele in df.columns if ele not in unwanted]


X_train, X_val, y_train, y_val=train_test_split(df[input_features], 
                                                df['IRI'],
                                                test_size=0.3,
                                                random_state=0)

### Tunning hyper parameters
def report_perf(optimizer, X, y, title, callbacks=None):
    """
    A wrapper for measuring time and performances of different optmizers
    
    optimizer = a sklearn or a skopt optimizer
    X = the training set 
    y = our target
    title = a string label for the experiment
    """
    start = time()
    
    if callbacks:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
        
    d=pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_
    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
            +u"\u00B1"+" %.3f") % (time() - start, 
                                  len(optimizer.cv_results_['params']),
                                  best_score,
                                  best_score_std))    
    print('Best parameters:')
    pprint.pprint(best_params)
    print()
    return best_params


RMSE = make_scorer(mean_squared_error, greater_is_better=False)

Regressor= RFR(bootstrap=True)

search_spaces = { 
                  'n_estimators': Integer(484,491),
                  'max_depth':Integer(16,20),
                  'min_samples_split' : Integer(2,4),
                  'min_samples_leaf': Integer(1,3),
                }

opt = BayesSearchCV(Regressor,
                    search_spaces,
                    scoring=RMSE,
                    cv=5,
                    n_iter=100,
                    # n_jobs=-1,  # use just 1 job with CatBoost in order to avoid segmentation fault
                    optimizer_kwargs={'base_estimator': 'GP'},
                    random_state=0,
                    )

best_params = report_perf(opt,X_train, y_train, 'Regressor',
                          callbacks=[VerboseCallback(100),
                                      DeadlineStopper(60*10)])

tunned_model = RFR(**best_params)
tunned_model.fit(X_train, y_train)

y_pred_train = tunned_model.predict(X_train)
y_pred_val = tunned_model.predict(X_val)

check_train = y_pred_train - y_train
check_val = y_pred_val - y_val


print('Cross validation scores')
print('Train')
scores = cross_val_score(tunned_model, X_train, y_train, 
                          scoring = 'neg_mean_squared_error',cv=5)
scores = np.sqrt(np.abs(scores))

print('Mean:',scores.mean())
print('Std:',scores.std())

print('Train set:')
print('RMSE:', '{:.4f}'.format(np.sqrt(np.abs(mean_squared_error(y_train, y_pred_train)))))
print('R2:', '{:.4f}'.format(r2_score(y_train, y_pred_train)))

print('Test set:')
print('RMSE:', '{:.4f}'.format(np.sqrt(np.abs(mean_squared_error(y_val, y_pred_val)))))  
print('R2:', '{:.4f}'.format(r2_score(y_val, y_pred_val)))