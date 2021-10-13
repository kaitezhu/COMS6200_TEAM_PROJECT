import json
import numpy as np

with open('../data/result.json') as f:
    data = json.load(f)
keys = ['Model', 'Train_time', 'Testing_time', 'Trained_accuracy', 'Testing_accuracy', 'TN', 'FP', 'FN',
        'TP', 'Precision', 'recall', 'fpr', 'fnr', 'tnr', 'npv', 'fdr', 'for', 'f1']
data_full = {}
for key in keys:
    data_full[key] = []

volume = {}
model = []
# 100
train_time_100 = []
test_time_100 = []
train_acc_100 = []
test_acc_100 = []
tpr_100 = []
tnr_100 = []
fpr_100 = []
fnr_100 = []
pre_100 = []
rec_100 = []
npv_100 = []
fdr_100 = []
for_100 = []
f1_100 = []
# 1000
train_time_1000 = []
test_time_1000 = []
train_acc_1000 = []
test_acc_1000 = []
tpr_1000 = []
tnr_1000 = []
fpr_1000 = []
fnr_1000 = []
pre_1000 = []
rec_1000 = []
npv_1000 = []
fdr_1000 = []
for_1000 = []
f1_1000 = []
# 5000
train_time_5000 = []
test_time_5000 = []
train_acc_5000 = []
test_acc_5000 = []
tpr_5000 = []
tnr_5000 = []
fpr_5000 = []
fnr_5000 = []
pre_5000 = []
rec_5000 = []
npv_5000 = []
fdr_5000 = []
for_5000 = []
f1_5000 = []
# 10000
train_time_10000 = []
test_time_10000 = []
train_acc_10000 = []
test_acc_10000 = []
tpr_10000 = []
tnr_10000 = []
fpr_10000 = []
fnr_10000 = []
pre_10000 = []
rec_10000 = []
npv_10000 = []
fdr_10000 = []
for_10000 = []
f1_10000 = []
# 30000
train_time_30000 = []
test_time_30000 = []
train_acc_30000 = []
test_acc_30000 = []
tpr_30000 = []
tnr_30000 = []
fpr_30000 = []
fnr_30000 = []
pre_30000 = []
rec_30000 = []
npv_30000 = []
fdr_30000 = []
for_30000 = []
f1_30000 = []
# 50000
train_time_50000 = []
test_time_50000 = []
train_acc_50000 = []
test_acc_50000 = []
tpr_50000 = []
tnr_50000 = []
fpr_50000 = []
fnr_50000 = []
pre_50000 = []
rec_50000 = []
npv_50000 = []
fdr_50000 = []
for_50000 = []
f1_50000 = []
# 81173
train_time_81173 = []
test_time_81173 = []
train_acc_81173 = []
test_acc_81173 = []
tpr_81173 = []
tnr_81173 = []
fpr_81173 = []
fnr_81173 = []
pre_81173 = []
rec_81173 = []
npv_81173 = []
fdr_81173 = []
for_81173 = []
f1_81173 = []

for i in data['100']:
    data_full['Model'].append(i)
for i in data_full['Model']:
    for j in data['100'][i]:
        data_full[j].append(data['100'][i][j])
train_time_100 = data_full['Train_time']
test_time_100 = data_full['Testing_time']
train_acc_100 = data_full['Trained_accuracy']
test_acc_100 = data_full['Testing_accuracy']
tpr_100 = data_full['recall']
tnr_100 = data_full['fnr']
fpr_100 = data_full['fpr']
fnr_100 = data_full['fnr']
pre_100 = data_full['Precision']
rec_100 = data_full['recall']
npv_100 = data_full['npv']
fdr_100 = data_full['fdr']
for_100 = data_full['for']
f1_100 = data_full['f1']

for i in data['1000']:
    data_full['Model'].append(i)
for i in data_full['Model']:
    for j in data['1000'][i]:
        data_full[j].append(data['1000'][i][j])
train_time_1000 = data_full['Train_time']
test_time_1000 = data_full['Testing_time']
train_acc_1000 = data_full['Trained_accuracy']
test_acc_1000 = data_full['Testing_accuracy']
tpr_1000 = data_full['recall']
tnr_1000 = data_full['fnr']
fpr_1000 = data_full['fpr']
fnr_1000 = data_full['fnr']
pre_1000 = data_full['Precision']
rec_1000 = data_full['recall']
npv_1000 = data_full['npv']
fdr_1000 = data_full['fdr']
for_1000 = data_full['for']
f1_1000 = data_full['f1']
print(train_time_1000)

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

