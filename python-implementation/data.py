#%%
import numpy as np
import h5py
from sklearn import preprocessing


def load_digits_data(train_size, test_size, unlabel_size):

    with h5py.File("../Dataset/usps.h5", "r") as hf:
        train = hf.get("train")
        X_tr = train.get("data")[:]
        y_tr = train.get("target")[:]

        test = hf.get("test")
        X_te = test.get("data")[:]
        y_te = test.get("target")[:]

    # print("train size max:", X_tr.shape[0])

    scaler = preprocessing.StandardScaler()
    X_tr = scaler.fit_transform( X_tr )
    X_te = scaler.transform( X_te )

    stratified = False

    y_tr = np.where(y_tr > 4, 1, 0)
    y_te = np.where(y_te > 4, 1, 0)

    while not stratified:
        idx = np.arange(X_tr.shape[0])
        np.random.shuffle(idx)
        X_tr2 = X_tr[idx]
        y_tr2 = y_tr[idx]

        X_tr2 = X_tr2[:train_size]
        y_tr2 = y_tr2[:train_size]

        X_unl = X_tr2[train_size : unlabel_size + train_size]
        y_unl = y_tr2[train_size : unlabel_size + train_size]

        if y_tr2[y_tr2 == 0].shape[0] > 0 and y_tr2[y_tr2 == 1].shape[0] > 0:
            stratified = True
    
    X_te = X_te[:test_size]
    y_te = y_te[:test_size]

    # print(y_tr2)
    return X_tr2, X_te, X_unl, y_tr2, y_te, y_unl

# load_digits_data(2, 40, 120 - 2);
# X_tr, X_te, X_unl, y_tr, y_te, y_unl = load_digits_data(10, 10, 10)

#%%

# Add stratify check
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

    stratified = False

    while not stratified:
        idx = np.arange(x_train.shape[0])
        np.random.shuffle(idx)
        x_train2 = x_train[idx].astype(np.float)
        y_train2 = y_train[idx].astype(np.float)

        x_train2 = x_train2[:train_size]
        y_train2 = y_train2[:train_size]
        
        x_unl = x_train2[train_size : unlabel_size + train_size].astype(np.float)
        y_unl = y_train2[train_size : unlabel_size + train_size].astype(np.float)

        if y_train2[y_train2 == 0.0].shape[0] > 0 and y_train2[y_train2 == 1.0].shape[0] > 0:
            stratified = True

    x_test = x_test[:test_size].astype(np.float)
    y_test = y_test[:test_size].astype(np.float)

    return x_train2, x_test, x_unl, y_train2, y_test, y_unl




# load_ft_data(2, 40, 120 - 2);

# %%
