import numpy as np
import pandas as pd
import xlrd

def vec2lab(vec):
    label = []
    for i in range(0, len(vec)):
        if vec[i, 0] > vec[i, 1]:
            lab = 1
        else:
            lab = 2
        label.append(lab)

    return label


def lab2vec(lab):
    vec = np.zeros((len(lab), 2))
    for i in range(0, len(lab)):
        if lab.iloc[i].values == 1:
            vec[i, 0] = 1
        else:
            vec[i, 1] = 1

    return vec

def read_nsheet_xlsx(name):
    book = xlrd.open_workbook(name)
    num_sheet = book.nsheets

    xls = []
    for j in range(0, num_sheet):
        sheet = book.sheet_by_index(j)
        for i in range(0, sheet.nrows):
            xls.append(sheet.row_values(i))
        xls = pd.DataFrame(xls)
        if j == 0:
            data = tuple([xls])
        else:
            data = data +tuple([xls])
        xls = []

    return data

def save_model(preds, name):
    view_nums = len(preds)
    writer = pd.ExcelWriter(name)
    for v in range(view_nums):
        preds[v].to_excel(writer, 'page_' + str(v), float_format='%.5f', index=False, header=False)  # float_format 控制精度
        # preds[1].to_excel(writer, 'page_2', float_format='%.5f', index=False, header=False)  # float_format 控制精度
    writer.save()