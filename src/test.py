import json
import numpy as np

with open('../data/result.json') as f:
    data = json.load(f)
list = ['DTC','RFC','GB','XGB']
data_full = {}
for key in data:
    data_full[key] = {}


for key in data:
    for i in list:
        for j in data[key][i]:
            data_full[key][j] = []
for key in data:
    for i in list:
        for j in data[key][i]:
            data_full[key][j].append(data[key][i][j])
print(data_full)

# model = []
# for key in keys:
#     data_full[key] = []
#
# for i in data['100']:
#     data_full['Model'].append(i)
# for i in data_full['Model']:
#     for j in data['100'][i]:
#         data_full[j].append(data['100'][i][j])
# train_time_100 = data_full['Train_time']
# test_time_100 = data_full['Testing_time']
# train_acc_100 = data_full['Trained_accuracy']
# test_acc_100 = data_full['Testing_accuracy']
# tpr_100 = data_full['recall']
# tnr_100 = data_full['fnr']
# fpr_100 = data_full['fpr']
# fnr_100 = data_full['fnr']
# pre_100 = data_full['Precision']
# rec_100 = data_full['recall']
# npv_100 = data_full['npv']
# fdr_100 = data_full['fdr']
# for_100 = data_full['for']
# f1_100 = data_full['f1']
#
# for i in data['1000']:
#     data_full['Model'].append(i)
# for i in data_full['Model']:
#     for j in data['1000'][i]:
#         data_full[j].append(data['1000'][i][j])
# train_time_1000 = data_full['Train_time']
# test_time_1000 = data_full['Testing_time']
# train_acc_1000 = data_full['Trained_accuracy']
# test_acc_1000 = data_full['Testing_accuracy']
# tpr_1000 = data_full['recall']
# tnr_1000 = data_full['fnr']
# fpr_1000 = data_full['fpr']
# fnr_1000 = data_full['fnr']
# pre_1000 = data_full['Precision']
# rec_1000 = data_full['recall']
# npv_1000 = data_full['npv']
# fdr_1000 = data_full['fdr']
# for_1000 = data_full['for']
# f1_1000 = data_full['f1']
# print(train_time_1000)

# print(volume)
# volume = {}
# volume_keys = ['100','1000','5000','10000','30000','50000','82332']
# for key in volume_keys:
#     volume[key] = []
# for i in volume:
#     volume[i].append(data[volume_keys[i]])
# print(data['100'])

# for i in
# print(data_full)

