import numpy as np
import pandas as pd
from sklearn import preprocessing
from train_TSK import train_TSK
from mul_TSK import train_mul_TSK, test_mul_TSK
from until import vec2lab, lab2vec, read_nsheet_xlsx, save_model
from sklearn import model_selection
from sklearn.metrics import accuracy_score


min_max_scaler = preprocessing.MinMaxScaler()

view1 = read_nsheet_xlsx('data/TestFeature_AMINO.xlsx')
view1 = view1[0]
view1 = min_max_scaler.fit_transform(view1)
view1 = pd.DataFrame(view1)

view2 = read_nsheet_xlsx('data/TestFeature_CODON.xlsx')
view2 = view2[0]
view2 = min_max_scaler.fit_transform(view2)
view2 = pd.DataFrame(view2)

view3 = read_nsheet_xlsx('data/TestFeature_DNA.xlsx')
view3 = view3[0]
view3 = min_max_scaler.fit_transform(view3)
view3 = pd.DataFrame(view3)

view4 = read_nsheet_xlsx('data/train-view4.xlsx')
view4 = view4[0]
view4 = min_max_scaler.fit_transform(view4)
view4 = pd.DataFrame(view4)
view4 = view4[[0,1]]

label = np.zeros(5521)
label[0:4902] = 1
label[4902:] = 2
tr_ind = list(range(0, 500))
te_ind = list(range(4902, 4902+500))

train_ind_h, valid_ind_h = model_selection.train_test_split(tr_ind, test_size = 0.3)
train_ind_l, valid_ind_l = model_selection.train_test_split(te_ind, test_size = 0.3)

train_ind = train_ind_h + train_ind_l
valid_ind = valid_ind_h + valid_ind_l

train_view1 = view1.iloc[train_ind]
train_view2 = view2.iloc[train_ind]
train_view3 = view3.iloc[train_ind]
train_view4 = view4.iloc[train_ind]
train_labels = label[train_ind]
train_labels = pd.DataFrame(train_labels)

test_view1 = view1.iloc[valid_ind]
test_view2 = view2.iloc[valid_ind]
test_view3 = view3.iloc[valid_ind]
test_view4 = view4.iloc[valid_ind]
test_labels = label[valid_ind]
test_labels = pd.DataFrame(test_labels)

train_data = tuple([train_view1, train_view2, train_view3, train_view4,  train_labels])
test_data = tuple([test_view1, test_view2, test_view3, test_view4,  test_labels])

del train_view1, train_view2, train_view3, train_view4, train_labels,test_view1, test_view2, test_view3, test_view4, test_labels,

view_nums = len(train_data)
train_labels = train_data[view_nums - 1]
train_labels.rename(columns={0: 'label'}, inplace=True)
test_labels = test_data[view_nums - 1]
test_labels.rename(columns={0: 'label'}, inplace=True)
view_nums = view_nums - 1
options = {}
options['view_nums'] = view_nums
options['maxiter'] = 10
c = max(train_labels.values)
view_weight = np.zeros(view_nums)
N = len(train_data[0])
best_acc_te = 0

model_cell = {}
v_cell = {}
b_cell = {}

# 归一化
# min_max_scaler = preprocessing.MinMaxScaler()
# for view_num in range(0, view_nums):
#     train_data = min_max_scaler.fit_transform(train_data)
#     test_data = min_max_scaler.fit_transform(test_data)

# 训练每个视角的TSK
for view_num in range(0, view_nums):

    [pg, v, b] = train_TSK(train_data[view_num], train_labels)
    if view_num == 0:
        model_cell = tuple([pd.DataFrame(pg)])
        v_cell = tuple([pd.DataFrame(v)])
        b_cell = tuple([pd.DataFrame(b)])
    else:
        model_cell = model_cell + tuple([pd.DataFrame(pg)])
        v_cell = v_cell + tuple([pd.DataFrame(v)])
        b_cell = b_cell + tuple([pd.DataFrame(b)])
    view_weight[view_num] = 1 / view_nums


# 多视角TSK 训练
for lambda1 in range(-3, -2):
    for lambda2 in range(-3, -2):
        for lambda3 in range(-3, -2):
            options['lambda1'] = pow(10, lambda1)
            options['lambda2'] = pow(10, lambda2)
            options['lambda3'] = pow(10, lambda3)
            try:
                model_cell_t, view_weight_t = train_mul_TSK(train_data, model_cell, view_weight, v_cell, b_cell, lab2vec(train_labels), options)

                Y_te = test_mul_TSK(test_data, model_cell_t, view_weight_t, v_cell, b_cell, view_nums, int(c[0]))
                labels_te = vec2lab(Y_te)
                acc = accuracy_score(test_labels, labels_te)
            except Exception as e:
                print(e)

            if acc > best_acc_te:

                best_acc_te = acc
                best_acc = acc
                best_model = model_cell_t

                best_Y_te = Y_te
                best_labels_te = labels_te
                best_view = view_weight_t
                best_options = options
                #                print(acc)
                print("ACC: %s " %(acc))

save_model(best_model, "results/best_model.xlsx")
save_model(v_cell, "results/best_v_cell.xlsx")
save_model(b_cell, "results/best_b_cell.xlsx")
writer = pd.ExcelWriter("results/best_view_weight.xlsx")
best_view = pd.DataFrame(best_view)
best_view.to_excel(writer, 'page_1', float_format='%.5f', index=False, header=False)
writer.save()