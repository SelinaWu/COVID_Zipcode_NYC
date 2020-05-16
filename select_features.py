# X np.array input features
# y np.array target variable
# F List of feature names (list of strings)

from sklearn import linear_model
!pip install sklearn-relief==1.0.0b2
import sklearn_relief as relief

from sklearn.preprocessing import MinMaxScaler

def select_features(X,y,normalize): #normalize is 0 or 1 

  if normalize == 1:
    scaler = MinMaxScaler() 
    scaled_values = scaler.fit_transform(X) 
    X.loc[:,:] = scaled_values

  corr_FS = dict()
  relief_FS = dict()
  lasso_FS = dict()

  ## Correaltion
  correlation = X.corrwith(y, axis = 0).abs()
  corr_FS['ranked_W'] = sorted(list(correlation))[::-1]
  corr_FS['ranked_idx'] = np.argsort(np.array(list(correlation)))[::-1]

  ## rReliefF
  r = relief.RReliefF(n_features=list(X.shape)[1])# Choose the best 3 features
  my_transformed_matrix = r.fit_transform(np.asarray(X), np.asarray(y))
  relief_FS['ranked_idx'] = np.argsort(r.w_)[::-1]
  relief_FS['ranked_W']  = r.w_[np.argsort(r.w_)[::-1]]

  ## Lasso
  clf = linear_model.Lasso(alpha=0.1)
  clf.fit(X, y)
  lasso_FS['ranked_idx'] = np.argsort(clf.coef_)[::-1]
  lasso_FS['ranked_W']  = clf.coef_[np.argsort(clf.coef_)[::-1]]

  return corr_FS, relief_FS, lasso_FS
