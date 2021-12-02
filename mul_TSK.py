import numpy as np
import pandas as pd
from GetFuzzyFeat import fromxtoz
from until import vec2lab, lab2vec

def train_mul_TSK(mulview_data_cell, model_cell, view_weight, v_cell, b_cell, T, options):
    # optionas 字典

    c = np.size(T, 1)
    view_nums = options['view_nums']
    N = len(mulview_data_cell[0])
    lambda1 = options['lambda1']
    lambda2 = options['lambda2']
    lambda3 = options['lambda3']
    maxiter = options['maxiter']
    #    model_cell_new={}
    for i in range(0, maxiter):
        sum_weight = 0
        for view_num in range(0, view_nums):
            temp_pg = model_cell[view_num]
            temp_x = mulview_data_cell[view_num]
            v = v_cell[view_num].values
            b = b_cell[view_num].values
            temp_x = fromxtoz(temp_x, v, b)
            # temp_x = temp_x.values
            sum_variance = np.linalg.norm(np.dot(temp_x, temp_pg) - T)
            sum_variance = np.exp(-lambda3 * sum_variance)
            sum_weight = sum_weight + sum_variance

        for view_num in range(0, view_nums):
            acc_p = model_cell[view_num]
            x = mulview_data_cell[view_num]
            v = v_cell[view_num].values
            b = b_cell[view_num].values
            x = fromxtoz(x, v, b)
            # x = x.values
            variance = np.linalg.norm(np.dot(x, acc_p) - T)
            acc_w = np.exp(-lambda3 * variance) / sum_weight
            sum_y = np.zeros((N, c))

            for j in range(0, view_nums):
                if j != view_num:
                    temp_pg = model_cell[j]
                    temp_x = mulview_data_cell[j]
                    v = v_cell[j].values
                    b = b_cell[j].values
                    temp_x = fromxtoz(temp_x, v, b)
                    sum_y = sum_y + np.dot(temp_x, temp_pg)

            y_cooperate = sum_y / (view_nums - 1)
            z = acc_w * np.dot(x.T, x)
            part_a = np.linalg.pinv(z + lambda1 * np.identity(z.shape[0]) + lambda2 * np.dot(x.T, x))
            part_b = acc_w * np.dot(x.T, T) + lambda2 * np.dot(x.T, y_cooperate)
            if view_num == 0:
                model_cell_new = tuple([pd.DataFrame(np.dot(part_a, part_b))])
            else:
                model_cell_new = model_cell_new + tuple([pd.DataFrame(np.dot(part_a, part_b))])

            view_weight[view_num] = acc_w

        del model_cell
        model_cell = model_cell_new
        del model_cell_new

    return model_cell, view_weight

def test_mul_TSK(test_data_cell, model_cell, view_weight, v_cell, b_cell, view_nums, c):

    N = len(test_data_cell[0])
    Y_te = np.zeros((N, c))

    for view_num in range(0,view_nums):
        acc_pg = model_cell[view_num]
        acc_w = view_weight[view_num]
        acc_x = test_data_cell[view_num]
        v = v_cell[view_num].values
        b = b_cell[view_num].values
        acc_x = fromxtoz(acc_x, v, b)
        Y_te = Y_te + acc_w*np.dot(acc_x, acc_pg)

    Y_te[Y_te < 0] = 0
    Y_te[Y_te > 1] = 1

    return Y_te