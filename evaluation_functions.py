import numpy as np
import scipy as sp
from tqdm import tqdm

from sklearn import linear_model
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from functools import partial
from itertools import combinations
from gensim import models, matutils

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure



def mate_retrieval(l1_vecs, l2_vecs):
    sim = np.dot(normalize(l1_vecs),normalize(l2_vecs).T)
    '''Mate retrieval rate - the rate when the most symmetric document is ones translation.'''
    return sum([sim[i].argmax()==i for i in range(sim.shape[0])])/sim.shape[0]

def rank(val,a):
    if sp.sparse.issparse(a):
        return a[a>=val].shape[1] #if a is sparse
    return len(a[a>=val]) # if a is dense

def reciprocal_rank_hub(l1_vecs, l2_vecs):
    '''Mean reciprocal rank'''
    sim = cosine_similarity(l1_vecs, l2_vecs)

    l2_sims = cosine_similarity(l2_vecs,l2_vecs)
    sorted_l2 = np.sort(l2_sims)[::-1]
    hub_weighting_l2 = np.mean(sorted_l2[:, 2:200], axis=1)

    l1_sims = cosine_similarity(l1_vecs,l1_vecs)
    sorted_l1 = np.sort(l1_sims)[::-1]
    hub_weighting_l1 = np.mean(sorted_l1[:, :200], axis=1)
    
    sim1 = sim - hub_weighting_l1
    sim2 = sim - hub_weighting_l2

    sim = sim1 + sim2.T

    return sum([1/rank(sim[i,i],sim[i]) for i in range(sim.shape[0])])/sim.shape[0]


def reciprocal_rank(l1_vecs, l2_vecs):
    '''Mean reciprocal rank'''
    sim = cosine_similarity(l1_vecs, l2_vecs)

    return sum([1/rank(sim[i,i],sim[i]) for i in range(sim.shape[0])])/sim.shape[0]

def comp_scores(l1_vecs, l2_vecs):
  return [mate_retrieval(l1_vecs, l2_vecs),reciprocal_rank(l1_vecs, l2_vecs),  reciprocal_rank_hub(l1_vecs, l2_vecs)]


def evaluate_baseline_lca_model_ort(X_train_in, X_test_in, Y_train_in, Y_test_in, dimensions, evaluation_function):
  """
  Input: x_train1, x_test1, x_train2, x_test2 | np.darray - matrices, 
         dimensions | list of integers, dimensons which should be tested
         evaluation function | function which evalutates the results

  The columns of each matrix are document vectors. 
  Language 1 (source) : x_train1, x_test1 | Language 2 (target) : x_train2, x_test2

  Output: Results of the evaluation function for each dimension. 

  """
  scores = []

  for dimension in dimensions:
      #Transpose matrices and reduce to given dimensions
      X_train, X_test = X_train_in[: ,:dimension].T,  X_test_in[:,:dimension].T
      Y_train, Y_test = Y_train_in[:,:dimension].T, Y_test_in[:,:dimension].T
      #Compute SVD to obtain best orthogonal mapping by UV.T 
      u, s, vh = np.linalg.svd(np.dot(X_train, Y_train.T))
      B = np.dot(u, vh)
      #Define the obtained mapping and apply it to the test-data
      linear_mapping = lambda x: np.dot(x, B)
      score = evaluation_function(Y_test.T, linear_mapping(X_test.T))
      scores.append(score)
  return scores 

def evaluate_baseline_lca_model(X_train_in, X_test_in, Y_train_in, Y_test_in, dimensions, evaluation_function):

  """
  Input: x_train1, x_test1, x_train2, x_test2 | np.darray - matrices, 
         dimensions | list of integers, dimensons which should be tested
         evaluation function | function which evalutates the results

  The columns of each matrix are document vectors. 
  Language 1 (source) : x_train1, x_test1 | Language 2 (target) : x_train2, x_test2

  Output: Results of the evaluation function for each dimension. 

  """
  
  
  scores = []
  for dimension in dimensions:
      #Transpose matrices and reduce to given dimensions
      X_train, X_test = X_train_in[: ,:dimension].T,  X_test_in[:,:dimension].T
      Y_train, Y_test = Y_train_in[:,:dimension].T, Y_test_in[:,:dimension].T

      B = np.dot(np.linalg.pinv(Y_train.T), X_train.T)
      linear_mapping = lambda x: np.dot( B.T, x)

      score = evaluation_function(X_test.T, linear_mapping(Y_test).T)
      scores.append(score)
  return scores 


def plot_parameter_graph(dimensions, scores, title, xlabel = "Dimensions", ylabel = "Reciprocal Rank", pair_list=None):
  figure(figsize=(18, 6))

  for k, score in enumerate(scores):
    if pair_list == None:
        plt.plot(dimensions, scores[k], alpha=0.8, label="Language Pair: {}".format(k))
    else:
        plt.plot(dimensions, scores[k], alpha=0.8, label="Language Pair: {} -> {}".format(pair_list[k][0], pair_list[k][1]))
  avg = np.mean(np.asarray(scores), axis=0)
  plt.plot(dimensions, avg, c="r", label="Average Score",linewidth=3.0)

  max_ind = np.argmax(avg)

  plt.scatter(dimensions[max_ind], avg[max_ind], c="k")
  plt.text(dimensions[max_ind], avg[max_ind]-0.1, 
          "Dimension: {} \nMean Score: {}".format(dimensions[max_ind],str(avg[max_ind])[:4] ),
          fontsize= 12
              )
  plt.title(title, fontsize=13)
  plt.xlabel("Dimensions")
  plt.ylabel("Reciprocal Rank")
  plt.legend()
  plt.show()

def compute_lcc_scores( en_train_matrix,
                en_test_matrix,
                fr_train_matrix,
                fr_test_matrix,
                dimensions, 
                evaluation_function):

    scores = []


    for dimension in tqdm(dimensions):
        en = en_train_matrix[: ,:dimension] - np.mean(en_train_matrix[:,:dimension], axis=0)
        fr = fr_train_matrix[: ,:dimension] - np.mean(fr_train_matrix[:,:dimension], axis=0)
        sample_size = en.shape[0]
        zero_matrix = np.zeros((sample_size, dimension))
        X1 = np.concatenate((en, zero_matrix), axis = 1)
        X2 = np.concatenate((zero_matrix, fr), axis= 1)
        X = np.concatenate((X1, X2), axis = 0)
        Y1 = np.concatenate((en, fr), axis = 1)
        Y2 = np.concatenate((en, fr), axis = 1)
        Y = np.concatenate((Y1, Y2), axis = 0)

        reg = linear_model.RidgeCV(alphas=[1e-10, 1e-3, 1e-2, 1e-1, 1, 10])
        reg.fit(X,Y)
        pca = PCA(n_components= int(dimension))
        pca.fit(reg.predict(X))
        rrr = lambda X: np.matmul(pca.transform(reg.predict(X)), pca.components_)

        #sample_size = len(en_docs_test)
        en = en_test_matrix[: ,:dimension] - np.mean(en_train_matrix[:,:dimension], axis=0)
        fr = fr_test_matrix[: ,:dimension] - np.mean(en_train_matrix[:,:dimension], axis=0)
        zero_matrix = np.zeros((en_test_matrix.shape[0], dimension))
        X1 = np.concatenate((en, zero_matrix), axis = 1)
        X2 = np.concatenate((zero_matrix, fr), axis= 1)
        X = np.concatenate((X1, X2), axis = 0)
        english_encodings_lcc = rrr(X1)
        french_encodings_lcc = rrr(X2)
        score = evaluation_function(english_encodings_lcc, french_encodings_lcc)
        scores.append(score)

    return scores