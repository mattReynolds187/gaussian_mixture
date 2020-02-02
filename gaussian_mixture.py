import numpy as np

class GMM:
    def __init__(self, X: np.ndarray, K: int, seed: int):
        np.random.seed(seed)
        self.X = X
        self.mu = X[np.random.choice(X.shape[0], K, replace=False)]
        self.p = np.ones(K)/K
        self.log_likelihood = None

        #vectorized computation of variance vector
        n = X.shape[0]
        d = X.shape[1]
        self.var = np.sum((self.mu*np.ones([n, K, d]) - X.reshape([n, 1, d]))**2, axis=(0,2))/(n*d)

    def logged_gauss_incomplete_p(self):

        n,d = self.X.shape
        k = self.var.shape[0]
        delta = np.where(self.X == 0, 0, 1) #nxd
        delta_reshaped = delta.reshape([delta.shape[0],1,delta.shape[1]]) #nx1xd
        X_reshaped = self.X.reshape([n,1,d])
        u_3d = (self.mu*np.ones([n,k,d]))*delta_reshaped
        sub_stack = u_3d-X_reshaped #nxkxd
        norm_squared = np.sum(sub_stack*sub_stack, axis = 2)#nxk

        exp_factor_logged = -norm_squared/(2*self.var)
        C_u = np.sum(delta, axis = 1, keepdims=True)

        var_2d = self.var*np.ones([n,k])
        first_factor_logged = np.log(self.p) - (C_u/2)*np.log(2*np.pi*var_2d)

        return first_factor_logged + exp_factor_logged

    def estep(self):

        logged_gauss_p = self.logged_gauss_incomplete_p()
        max_vector = np.amax(logged_gauss_p, axis=1, keepdims=True)
        scaled_gauss = np.exp(logged_gauss_p - max_vector)
        denominator_logged = max_vector + np.log(np.sum(scaled_gauss, axis = 1, keepdims=True))
        log_post = logged_gauss_p - denominator_logged
        log_likelihood = np.sum(denominator_logged)

        return np.exp(log_post), log_likelihood

    def mstep(self, post, min_variance=.25):

        n, d = self.X.shape
        k = post.shape[1]
        new_p = np.sum(post , axis = 0)/n

        delta = np.where(self.X == 0, 0, 1) #nxd
        mu_numerator = np.dot(self.X.T, post).T
        mu_denominator = np.dot(delta.T, post).T
        new_mu = np.where(mu_denominator >= 1, mu_numerator/(mu_denominator + 1e-16), self.mu) #kxd

        delta_reshaped = delta.reshape([delta.shape[0],1,delta.shape[1]]) #nx1xd
        X_reshaped = self.X.reshape([n,1,d])
        u_3d = (new_mu*np.ones([n,k,d]))*delta_reshaped
        sub_stack = u_3d-X_reshaped #nxkxd
        norm_squared = np.sum(sub_stack*sub_stack, axis = 2)#nxk

        C_u = np.sum(delta, axis = 1, keepdims=True)

        summation_factor = np.sum(post*norm_squared, axis = 0)
        first_factor = 1/np.sum(C_u*post, axis = 0)
        var_bad = first_factor*summation_factor
        new_var = np.where(var_bad < min_variance, min_variance, var_bad)

        self.mu = new_mu
        self.var = new_var
        self.p = new_p

    def run(self):

        flag = False
        while True:
            post, new_log_likelihood = self.estep()
            if flag and new_log_likelihood - old_log_likelihood <= abs(new_log_likelihood)/(10**6):
                self.log_likelihood = new_log_likelihood
                break
            old_log_likelihood = new_log_likelihood
            self.mstep(post)
            flag = True

    def fill_matrix(self):

        post = self.estep()[0]
        new_values = np.dot(post, self.mu)
        new_X = np.where(self.X==0, new_values, self.X)

        return new_X

    @staticmethod
    def evaluate_rmse(filled_matrix, test_matrix):
        return np.sqrt(np.mean((filled_matrix - test_matrix)**2))
