import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error  


class NMFRecommender:
    
    def __init__(self, random_state = 15, tol = 1e-3, max_iter = 200, rank = 3): 
        """
        The parameter values for the algorithm
        """
        # init params for class 
        self.__dict__.update(locals())  
       
    def initialize_matrices(self, m, n):  
        """
        Initialize the W and H matrices
        """
        self.m, self.n = m, n 
        np.random.seed(self.random_state)   
        # set random seed and return matrices of desired sizes 
        return np.random.rand(m, self.rank), np.random.rand(self.rank, n)  
        
    def compute_loss(self, V, W, H): 
        """
        Computes the loss of the algorithm according to the frobenius norm
        """
        # return the loss for a given iteration in Frobenius norm 
        return np.linalg.norm(V - np.dot(W, H), ord = "fro") 
    
    def update_matrices(self, V, W, H): 
        """
        The multiplicative update step to update W and H
        """ 
        # init numerator, denominator for new H step 
        numer, denom = np.dot(W.T, V), np.dot(W.T, np.dot(W, H))  
        H_new = H * (numer / denom) 
        # init numerator, denominator for new W step 
        numer, denom = np.dot(V, H_new.T), np.dot(W, np.dot(H_new, H_new.T)) 
        W_new = W * (numer / denom) 
        return W_new, H_new 
      
    def fit(self, V): 
        """
        Fits W and H weight matrices according to the multiplicative update 
        algorithm. Return W and H
        """
        m, n = V.shape 
        # init matrices for optimization 
        W, H = self.initialize_matrices(m, n) 
        # iterate in range max_iters 
        for iter in range(self.max_iter):  
            # update matrices at each step  
            W, H = self.update_matrices(V, W, H) 
            # stopping criteria based on self.tol 
            if self.compute_loss(V, W, H) < self.tol: 
                break  
        # save as attributes 
        self.W, self.H = W, H  
        return 
    
    def reconstruct(self, ):
        """
        Reconstructs the V matrix for comparison against the original V matrix 
        """
        # return reconstructed matrix 
        return np.dot(self.W, self.H) 


def prob4():
    """
    Run NMF recommender on the grocery store example
    """
    V = np.array([[0,1,0,1,2,2],
                  [2,3,1,1,2,2],
                  [1,1,1,0,1,1],
                  [0,2,3,4,1,1],
                  [0,0,0,0,1,0]]) 
    # init NMF object and fit on V 
    nmf = NMFRecommender(rank = 2)  
    nmf.fit(V)
    w, h = nmf.W, nmf.H 
    # return w, h, number in component 2 > component 1 
    return w, h, np.sum(h[1] > h[0])   


def prob5():
    """
    Calculate the rank and run NMF
    """
    # read in the data as pandas df 
    data = pd.read_csv("artist_user.csv", index_col = 0)  

    rank = 3  
    V_temp = data.values 
    # init an error condition for stopping criteria 
    error_cond = 0.0001 * np.linalg.norm(V_temp, ord = "fro")  

    while True: 
        # init NMF object and fit it on X to get W, H 
        nmf = NMF(n_components = rank, init = "random", max_iter = 1000, random_state = 0)  
        W = nmf.fit_transform(V_temp) 
        H = nmf.components_  
        # stopping criteria based on error condition in RMSE 
        if np.sqrt(mean_squared_error(V_temp, np.dot(W, H))) < error_cond:    
            return rank, np.dot(W, H) 

        if rank > 1000: 
            return  
        # update rank otherwise 
        rank += 1  



def discover_weekly(user_id, X):  
    """
    Create the recommended weekly 30 list for a given user
    """
    # read in the data as pandas dataframes 
    data = pd.read_csv("artist_user.csv", index_col = 0)    
    names = pd.read_csv("artists.csv") 

    # init user variables from dfs, X matrix from NMF 
    temp = X[user_id - 2]    
    listened_to = data.loc[user_id] 
    artist_names = names.name.to_numpy()    

    # pandas df to sort based on row from X and drop non-zero values of listens 
    df = pd.DataFrame({"ranked": temp, "listens": listened_to}).set_index(artist_names)
    df = df.sort_values(by = ["ranked"], ascending = False)  

    # return top 30 artists the user has not heard before 
    return df[df.listens == 0].index.values.tolist()[: 30]   