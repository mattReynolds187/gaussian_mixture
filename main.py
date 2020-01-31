import numpy as np
import common
import em


X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt('netflix_complete.txt')


#training on X (incomplete) its unsupervised, no labels
#get X_pred by filling the matrix X using the trained model
#compare to complete X_gold using root mean squared error
mixture = common.init(X, 12, 1)[0]
final_mixture = em.run(X, mixture, 0)[0]
X_pred = em.fill_matrix(X, final_mixture)
print(common.rmse(X_gold, X_pred))
