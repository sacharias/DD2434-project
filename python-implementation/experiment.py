#%%
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.svm import SVC
from numpy import linalg as LA
from scipy.spatial.distance import cdist
from scipy.linalg import eigh
from sklearn import preprocessing

def make_D_matrix(K):
    K_sum = np.sum(K, axis=1)
    D = np.diag(K_sum)
    return D


def make_L_matrix(K, D):
    D_temp = np.diag(np.diag(D) ** -0.5)
    L = D_temp @ K @ D_temp
    return L


def step_transfer(L, k=2):
    w, v = eigh(L)
    lambda_cut = w[-k]

    w = np.where(w >= lambda_cut, 1, 0)
    L_hat = np.dot(v, np.dot(np.diag(w), v.T))
    D_hat = np.diag(1 / np.diag(L_hat))
    K_hat = D_hat ** (1 / 2) @ L_hat @ D_hat ** (1 / 2)

    return L_hat, D_hat, K_hat


def linear_step_transfer(L, k=2):
    w, v = eigh(L)
    lambda_cut = w[-k]
    w = np.where(w >= lambda_cut, w, 0)

    L_hat = np.dot(v, np.dot(np.diag(w), v.T))
    D_hat = np.diag(1 / np.diag(L_hat))
    K_hat = D_hat ** (1 / 2) @ L_hat @ D_hat ** (1 / 2)

    return L_hat, D_hat, K_hat


def polynomial_transfer(L, D, K, t):
    L_hat = (L ** t) + 0.00001
    D_hat = np.diag(1 / np.diag(L_hat))
    K_hat = (
        D_hat ** (1 / 2)
        @ D ** (1 / 2)
        @ (LA.inv(D) @ K) ** t
        @ D ** (1 / 2)
        @ D_hat ** (1 / 2)
    )
    K_hat = preprocessing.scale(K_hat)

    return L_hat, D_hat, K_hat


def make_kernel(X_train, X_test, X_unlabel, tf_fun, kwargs):
    X = np.concatenate([X_train, X_test, X_unlabel])

    K = rbf_kernel(X)
    np.fill_diagonal(K, 1) # maybe zero?

    D = make_D_matrix(K)
    L = make_L_matrix(K, D)

    L, D, K = apply_transfer_func(L, D, K, kwargs, tf_fun)

    return K


def apply_transfer_func(L, D, K, hyperparams, type="linear"):
    """hyperparams: k for step and linear_step, t for polynomial"""
    if type == "linear":
        return L, D, K
    if type == "step":
        k = hyperparams["k"]
        return step_transfer(L, k)
    if type == "linear_step":
        k = hyperparams["k"]
        return linear_step_transfer(L)
    if type == "polynomial":
        t = hyperparams["t"]
        print("t", t)
        return polynomial_transfer(L, D, K, t)

    raise ValueError("wrong argument")


#%%
from data import load_digits_data

def run_experiment():
    # Hyperparametrar
    kwargs = {"t": 3, "k": 3}
    tf_fun = "linear"
    C = 10
    train_size, test_size, unlabel_size = 40, 200, 200

    # X_train, X_test, X_unlabel, Y_train, Y_test, Y_unlabel = load_digits_data(train_size, test_size, unlabel_size)
    X_train, X_test, X_unlabel, Y_train, Y_test, Y_unlabel = load_ft_data(train_size, test_size, unlabel_size)
    K = make_kernel(X_train, X_test, X_unlabel, tf_fun, kwargs)
    
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    K_train = K[:n_train, :n_train]
    K_test = K[n_train : n_train + n_test, :n_train]

    clf = SVC(kernel="precomputed", C=C)
    clf.fit(K_train, Y_train)

    return 1 - clf.score(K_test, Y_test)

print("error", run_experiment())

# %%


# %%s
