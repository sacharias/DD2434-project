#%%
import numpy as np
import h5py
from sklearn import preprocessing


def load_digits_data(train_size, test_size, unlabel_size):
    # return X_train, X_test, X_unlabel
    # return Y_train, Y_test, Y_unlabel

    with h5py.File("../Dataset/usps.h5", "r") as hf:
        train = hf.get("train")
        X_tr = train.get("data")[:]
        y_tr = train.get("target")[:]

        test = hf.get("test")
        X_te = test.get("data")[:]
        y_te = test.get("target")[:]

    print("train size max:", X_tr.shape[0])

    #np.random.seed(40)
    idx = np.arange(X_tr.shape[0])
    np.random.shuffle(idx)
    X_tr = X_tr[idx]
    y_tr = y_tr[idx]

    y_tr = np.where(y_tr > 4, 1, 0)
    y_te = np.where(y_te > 4, 1, 0)

    X_tr = X_tr[:train_size]
    y_tr = y_tr[:train_size]
    X_te = X_te[:test_size]
    y_te = y_te[:test_size]

    X_unl = X_tr[train_size : unlabel_size + train_size]
    y_unl = y_tr[train_size : unlabel_size + train_size]

    return X_tr, X_te, X_unl, y_tr, y_te, y_unl


# X_tr, X_te, X_unl, y_tr, y_te, y_unl = load_digits_data(10, 10, 10)

#%%


def load_ft_data(train_size, test_size, unlabel_size):
    x_train = np.load('../Embeddings/x_train_trans_FT.npy')
    x_test = np.load('../Embeddings/x_test_trans_FT.npy')

    y_train = np.load('../Embeddings/y_train.npy')
    y_test = np.load('../Embeddings/y_test.npy')

    scaler = preprocessing.StandardScaler()
    x_train = scaler.fit_transform( x_train )
    x_test = scaler.transform( x_test )


    y_train[y_train == 'positive'] = 0.0
    y_train[y_train == 'negative'] = 1.0

    x_train = x_train[y_train != 'neutral']
    y_train = y_train[y_train != 'neutral']
    x_train = x_train[y_train != 'unknown']
    y_train = y_train[y_train != 'unknown']

    y_test[y_test == 'positive'] = 0.0
    y_test[y_test == 'negative'] = 1.0

    x_test = x_test[y_test != 'neutral']
    y_test = y_test[y_test != 'neutral']
    x_test = x_test[y_test != 'unknown']
    y_test = y_test[y_test != 'unknown']

    idx = np.arange(x_train.shape[0])
    np.random.shuffle(idx)
    x_train = x_train[idx].astype(np.float)
    y_train = y_train[idx].astype(np.float)

    x_train = x_train[:train_size]
    y_train = y_train[:train_size]
    
    x_test = x_test[:test_size].astype(np.float)
    y_test = y_test[:test_size].astype(np.float)

    x_unl = x_train[train_size : unlabel_size + train_size].astype(np.float)
    y_unl = y_train[train_size : unlabel_size + train_size].astype(np.float)

    return x_train, x_test, x_unl, y_train, y_test, y_unl



# load_ft_data(10, 10, 10)

# %%
