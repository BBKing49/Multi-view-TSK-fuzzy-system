import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from until import vec2lab, lab2vec
from GetFuzzyFeat import preproc, fromxtoz
from sklearn.metrics import accuracy_score

def train_TSK(X, labels):
    data = X.values
    labels_vec = lab2vec(labels)
    #    labels_vec = labels_vec.values
    labels = labels.values
    results = np.zeros((5, 1))
    kf = KFold(n_splits=5, shuffle=True)
    auc_best = 0
    for la in range(-5, -1):
        for m in range(2, 8, 2):
            fold = 0

            for train_index, test_index in kf.split(data):
                train_X = data[train_index]
                train_labels = labels_vec[train_index.T]
                test_X = data[test_index.T]
                test_labels = labels[test_index]
                lambda1 = pow(2, la)
                v, b = preproc(pd.DataFrame(train_X), m)
                train_Xg = fromxtoz(pd.DataFrame(train_X), v, b)
                train_Xg = train_Xg.values

                Xg1 = np.dot(train_Xg.T, train_Xg)

                pg = np.dot(np.linalg.pinv(Xg1 + lambda1 * np.identity(Xg1.shape[0])), np.dot(train_Xg.T, train_labels))
                test_Xg = fromxtoz(pd.DataFrame(test_X), v, b)
                test_Xg = test_Xg.values
                Y_te = np.dot(test_Xg, pg)
                lab_te = vec2lab(Y_te)

                # results[fold] = roc_auc_score(test_labels, lab_te)
                results[fold] = accuracy_score(test_labels, lab_te)
                fold = fold + 1

            if auc_best < np.mean(results):
                auc_best = np.mean(results)
                best_pg = pg
                # best_d = d
                best_v = v
                best_b = b

    return best_pg, best_v, best_b