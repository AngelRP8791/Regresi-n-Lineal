from utils import db_connect
engine = db_connect()

# la misión es corregir el código (si es que está mal)

def scaler(X_train_, X_test_, nums): # def eleva a la categoría de funcion
                                     # "scaler(...)": nombre con el que se convocará la función
                                     # "X_train_, X_text_, nums": variables que serán sustituidas por valores al ejecutar la función

  X_train = X_train_.copy() # Crea una copia de los datos que sustituirán a "X_train_" y lo almacena temporalmente en "X_train"
  X_test = X_test_.copy() # Crea una copia de los datos que sustituirán a "X_test_" y lo almacena temporalmente en "X_test
  scaler = StandardScaler()
  # Train
  X_train_scaled = pd.DataFrame(
             scaler.fit_transform(X_train[nums]),
             columns=scaler.get_feature_names_out(),
             index = X_train.index)
  X_train_scaled = X_train_scaled.join(X_train[list(set(X_train.columns)  - set(nums))])
  # Test
  X_test_scaled = pd.DataFrame(
      scaler.transform(X_test[nums]),
      columns = scaler.get_feature_names_out(),
      index = X_test.index)
  X_test_scaled = X_test_scaled.join(X_test[list(set(X_test.columns)  - set(nums))])
  X_test_scaled = X_test_scaled[X_train_scaled.columns]
  return X_train_scaled, X_test_scaled


def to_binary(X_train_, X_test_, cats):
  X_train =  X_train_.copy()
  X_test = X_test_.copy()
  ohe = OneHotEncoder(handle_unknown='ignore', drop='first')
  # train
  X_train_bin = pd.DataFrame(ohe.fit_transform(
      X_train[cats]).toarray(),
      columns = ohe.get_feature_names_out(),
      index = X_train.index)
  X_train_bin = X_train_bin.join(X_train[list(set(X_train.columns)  - set(cats))])
  # test
  X_test_bin = pd.DataFrame(ohe.transform(X_test[cats]).toarray(),
      columns = ohe.get_feature_names_out(),
      index = X_test.index)
  X_test_bin = X_test_bin.join(X_test[list(set(X_test.columns)  - set(cats))])
  X_test_bin = X_test_bin[X_train_bin.columns]
  return X_train_bin, X_test_bin


def Elastic_gridcv(X_train, y_train):
    model =  ElasticNet(random_state=321)
    hyperparams = {"alpha" :  [0.01, 0.1, 1, 10,50],
                   "l1_ratio" :  np.linspace(0,1,35,50),
                   "max_iter": [ 1,5,10, 30],
                   "selection": ['cyclic', 'random'],
                   "tol": [1e-6, 1e-7, 1e-9, 1e-4],}
    cv = KFold(n_splits=5, shuffle=True, random_state=123) # replicables...
    grid_search = GridSearchCV(estimator=model,
                               param_grid=hyperparams,
                               cv=cv,
                               scoring= 'neg_mean_absolute_error',)
    grid_result = grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

url = "https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv"

df = pd.read_csv(url)
cats = ['sex', 'smoker', 'region', ]
nums = ['bmi','age', 'children']
# You can made automatic process?
target = 'charges'
X,y = df.drop(columns= [target]), df[target]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)
X_train_ohe, X_test_ohe = to_binary(X_train,  X_test, cats)
X_train_ml, X_test_ml = scaler(X_train_ohe, X_test_ohe, nums)

mde=Elastic_gridcv(X_train_ml, y_train)
preds = mde.predict(X_test_ml)
mean_squared_error(y_test, preds)

mde

print(mean_absolute_error(y_test, preds))