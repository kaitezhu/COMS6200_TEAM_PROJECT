file_size = [100,1000,5000,10000,30000,50000,81173]
import pandas as pd
import numpy as np
import warnings
import time
from sklearn.metrics import confusion_matrix as cm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import json
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import balanced_accuracy_score

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
def Diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))
data = {}
warnings.filterwarnings("ignore")

file_train= r'../data/UNSW_NB15_training-set.csv'
file_test= r'../data/UNSW_NB15_testing-set.csv'
random_state = 44777717

for i in file_size:
    print(i)
    data[str(i)] = {}
    #load file
    df_train = pd.read_csv(file_train)
    df_test = pd.read_csv(file_test)

    #clean empty data
    df_train = df_train.replace("-", np.nan)
    df_train = df_train.dropna(axis=0)
    df_test = df_test.replace("-", np.nan)
    df_test = df_test.dropna(axis=0)

    # convert object data to int using one hot encoding
    num_col = df_train.select_dtypes(include='number').columns
    cat_col = df_train.columns.difference(num_col)
    cat_col = cat_col[1:]

    data_cat = df_train[cat_col].copy()
    data_cat = pd.get_dummies(data_cat, columns=cat_col)
    df_train = pd.concat([df_train, data_cat], axis=1)
    df_train.drop(columns=cat_col, inplace=True)

    data_cat = df_test[cat_col].copy()
    data_cat = pd.get_dummies(data_cat, columns=cat_col)
    df_test = pd.concat([df_test, data_cat], axis=1)
    df_test.drop(columns=cat_col, inplace=True)

    #column diff remove
    num_col_train = list(df_train.select_dtypes(include='number').columns)
    num_col_train.remove('id')
    num_col_train.remove('label')
    num_col_test = list(df_test.select_dtypes(include='number').columns)
    num_col_test.remove('id')
    num_col_test.remove('label')
    diff = Diff(num_col_train, num_col_test)
    for j in diff:
        if j in num_col_train:
            num_col_train.remove(j)
            df_train = df_train.drop(columns=[j], axis=1)
        else:
            num_col_test.remove(j)
            df_test = df_test.drop(columns=[j], axis=1)
    num_col_train.append('label')
    num_col_test.append('label')

    #correlation
    corr_train_bin = df_train[num_col_train].corr()
    corr_ybin = abs(corr_train_bin['label'])
    highest_corr_bin = corr_ybin[corr_ybin > 0.3]
    bin_cols = highest_corr_bin.index

    #feature selection
    df_train = df_train[bin_cols].copy()
    df_test = df_test[bin_cols].copy()
    diff = Diff(df_test.columns, df_train.columns)

    df_train = df_train.sample(n=i)

    #split
    X_train = df_train.drop(['label'], axis=1)  # drop the column with the target/unnecessary variables
    X_test = df_test.drop(['label', ], axis=1)  # drop the column with the target/unnecessary variables
    y_train = df_train[['label']]  # target
    y_test = df_test[['label']]  # target

    # DTC
    DTC = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=7,
        splitter='best',
        random_state=random_state
    )

    data[str(i)]['DTC'] = {}
    _start_time = time.time()
    DTC.fit(X_train, y_train)
    train_time = round(time.time() - _start_time,4)
    data[str(i)]['DTC']['Train_time'] = train_time
    _start_time = time.time()
    pred_result = DTC.predict(X_test)
    data[str(i)]['DTC']['Testing_time'] = round(time.time() - _start_time,4)
    data[str(i)]['DTC']['Trained_accuracy'] = round(DTC.score(X_train, y_train)*100,2)
    data[str(i)]['DTC']['Testing_accuracy'] = round(DTC.score(X_test, y_test) * 100,2)
    fpr_list, tpr_list, thresholds = roc_curve(y_test, pred_result)
    data[str(i)]['DTC']['fpr_list'] = fpr_list
    data[str(i)]['DTC']['tpr_list'] = tpr_list
    data[str(i)]['DTC']['thresholds'] = thresholds
    data[str(i)]['DTC']['AUC'] = auc(fpr_list, tpr_list)
    data[str(i)]['DTC']['Balance'] = balanced_accuracy_score(y_test, pred_result)
    cf_matrix = cm(y_test, pred_result)
    tn, fp, fn, tp = cm(y_test, pred_result).ravel()
    data[str(i)]['DTC']['TN'] = tn
    data[str(i)]['DTC']['FP'] = fp
    data[str(i)]['DTC']['FN'] = fn
    data[str(i)]['DTC']['TP'] = tp
    if pd.isna(round(tp/(tp+fp)*100, 2)):
        data[str(i)]['DTC']['Precision'] = 0.0
    else:
        data[str(i)]['DTC']['Precision'] = round(tp/(tp+fp)*100, 2)

    if pd.isna(round(tp/(tp+fn)*100, 2)):
        data[str(i)]['DTC']['recall'] = 0.0
    else:
        data[str(i)]['DTC']['recall'] = round(tp/(tp+fn)*100, 2)

    if pd.isna(round(fp/(fp+tn)*100, 2)):
        data[str(i)]['DTC']['fpr'] = 0.0
    else:
        data[str(i)]['DTC']['fpr'] = round(fp/(fp+tn)*100, 2)

    if pd.isna(round(fn/(fn+tp)*100, 2)):
        data[str(i)]['DTC']['fnr'] = 0.0
    else:
        data[str(i)]['DTC']['fnr'] = round(fn/(fn+tp)*100, 2)

    if pd.isna(round(tn/(tn+fp)*100, 2)):
        data[str(i)]['DTC']['tnr'] = 0.0
    else:
        data[str(i)]['DTC']['tnr'] = round(tn/(tn+fp)*100, 2)

    if pd.isna(round(tn/(tn+fn)*100, 2)):
        data[str(i)]['DTC']['npv'] = 0.0
    else:
        data[str(i)]['DTC']['npv'] = round(tn/(tn+fn)*100, 2)

    if pd.isna(round(fp/(fp+tp)*100, 2)):
        data[str(i)]['DTC']['fdr'] = 0.0
    else:
        data[str(i)]['DTC']['fdr'] = round(fp/(fp+tp)*100, 2)

    if pd.isna(round(fn/(fn+tn)*100, 2)):
        data[str(i)]['DTC']['for'] = 0.0
    else:
        data[str(i)]['DTC']['for'] = round(fn/(fn+tn)*100, 2)

    if pd.isna(round(((2*tp)/(2*tp+fp+fn))*100, 2)):
        data[str(i)]['DTC']['f1'] = 0.0
    else:
        data[str(i)]['DTC']['f1'] = round(((2*tp)/(2*tp+fp+fn))*100, 2)


    #RFC
    RFC = RandomForestClassifier(
        bootstrap=False,
        max_depth=20,
        max_features='auto',
        n_estimators=500,
        random_state=random_state
    )
    data[str(i)]['RFC'] = {}
    _start_time = time.time()
    RFC.fit(X_train, y_train)
    train_time = round(time.time() - _start_time, 4)
    data[str(i)]['RFC']['Train_time'] = train_time
    _start_time = time.time()
    pred_result = RFC.predict(X_test)
    data[str(i)]['RFC']['Testing_time'] = round(time.time() - _start_time, 4)
    data[str(i)]['RFC']['Trained_accuracy'] = round(RFC.score(X_train, y_train) * 100, 2)
    data[str(i)]['RFC']['Testing_accuracy'] = round(RFC.score(X_test, y_test) * 100, 2)
    fpr_list, tpr_list, thresholds = roc_curve(y_test, pred_result)
    data[str(i)]['RFC']['fpr_list'] = fpr_list
    data[str(i)]['RFC']['tpr_list'] = tpr_list
    data[str(i)]['RFC']['thresholds'] = thresholds
    data[str(i)]['RFC']['AUC'] = auc(fpr_list, tpr_list)
    data[str(i)]['RFC']['Balance'] = balanced_accuracy_score(y_test, pred_result)
    cf_matrix = cm(y_test, pred_result)
    tn, fp, fn, tp = cm(y_test, pred_result).ravel()
    data[str(i)]['RFC']['TN'] = tn
    data[str(i)]['RFC']['FP'] = fp
    data[str(i)]['RFC']['FN'] = fn
    data[str(i)]['RFC']['TP'] = tp
    if pd.isna(round(tp / (tp + fp) * 100, 2)):
        data[str(i)]['RFC']['Precision'] = 0.0
    else:
        data[str(i)]['RFC']['Precision'] = round(tp / (tp + fp) * 100, 2)

    if pd.isna(round(tp / (tp + fn) * 100, 2)):
        data[str(i)]['RFC']['recall'] = 0.0
    else:
        data[str(i)]['RFC']['recall'] = round(tp / (tp + fn) * 100, 2)

    if pd.isna(round(fp / (fp + tn) * 100, 2)):
        data[str(i)]['RFC']['fpr'] = 0.0
    else:
        data[str(i)]['RFC']['fpr'] = round(fp / (fp + tn) * 100, 2)

    if pd.isna(round(fn / (fn + tp) * 100, 2)):
        data[str(i)]['RFC']['fnr'] = 0.0
    else:
        data[str(i)]['RFC']['fnr'] = round(fn / (fn + tp) * 100, 2)

    if pd.isna(round(tn / (tn + fp) * 100, 2)):
        data[str(i)]['RFC']['tnr'] = 0.0
    else:
        data[str(i)]['RFC']['tnr'] = round(tn / (tn + fp) * 100, 2)

    if pd.isna(round(tn / (tn + fn) * 100, 2)):
        data[str(i)]['RFC']['npv'] = 0.0
    else:
        data[str(i)]['RFC']['npv'] = round(tn / (tn + fn) * 100, 2)

    if pd.isna(round(fp / (fp + tp) * 100, 2)):
        data[str(i)]['RFC']['fdr'] = 0.0
    else:
        data[str(i)]['RFC']['fdr'] = round(fp / (fp + tp) * 100, 2)

    if pd.isna(round(fn / (fn + tn) * 100, 2)):
        data[str(i)]['RFC']['for'] = 0.0
    else:
        data[str(i)]['RFC']['for'] = round(fn / (fn + tn) * 100, 2)

    if pd.isna(round(((2 * tp) / (2 * tp + fp + fn)) * 100, 2)):
        data[str(i)]['RFC']['f1'] = 0.0
    else:
        data[str(i)]['RFC']['f1'] = round(((2 * tp) / (2 * tp + fp + fn)) * 100, 2)

    #GB
    GB = GradientBoostingClassifier(
        learning_rate=0.05,
        n_estimators=150,
        max_depth=7,
        random_state=random_state
    )
    data[str(i)]['GB'] = {}
    _start_time = time.time()
    GB.fit(X_train, y_train)
    train_time = round(time.time() - _start_time, 4)
    data[str(i)]['GB']['Train_time'] = train_time
    _start_time = time.time()
    pred_result = GB.predict(X_test)
    data[str(i)]['GB']['Testing_time'] = round(time.time() - _start_time, 4)
    data[str(i)]['GB']['Trained_accuracy'] = round(GB.score(X_train, y_train) * 100, 2)
    data[str(i)]['GB']['Testing_accuracy'] = round(GB.score(X_test, y_test) * 100, 2)
    fpr_list, tpr_list, thresholds = roc_curve(y_test, pred_result)
    data[str(i)]['GB']['fpr_list'] = fpr_list
    data[str(i)]['GB']['tpr_list'] = tpr_list
    data[str(i)]['GB']['thresholds'] = thresholds
    data[str(i)]['GB']['AUC'] = auc(fpr_list, tpr_list)
    data[str(i)]['GB']['Balance'] = balanced_accuracy_score(y_test, pred_result)
    cf_matrix = cm(y_test, pred_result)
    tn, fp, fn, tp = cm(y_test, pred_result).ravel()
    data[str(i)]['GB']['TN'] = tn
    data[str(i)]['GB']['FP'] = fp
    data[str(i)]['GB']['FN'] = fn
    data[str(i)]['GB']['TP'] = tp
    if pd.isna(round(tp / (tp + fp) * 100, 2)):
        data[str(i)]['GB']['Precision'] = 0.0
    else:
        data[str(i)]['GB']['Precision'] = round(tp / (tp + fp) * 100, 2)

    if pd.isna(round(tp / (tp + fn) * 100, 2)):
        data[str(i)]['GB']['recall'] = 0.0
    else:
        data[str(i)]['GB']['recall'] = round(tp / (tp + fn) * 100, 2)

    if pd.isna(round(fp / (fp + tn) * 100, 2)):
        data[str(i)]['GB']['fpr'] = 0.0
    else:
        data[str(i)]['GB']['fpr'] = round(fp / (fp + tn) * 100, 2)

    if pd.isna(round(fn / (fn + tp) * 100, 2)):
        data[str(i)]['GB']['fnr'] = 0.0
    else:
        data[str(i)]['GB']['fnr'] = round(fn / (fn + tp) * 100, 2)

    if pd.isna(round(tn / (tn + fp) * 100, 2)):
        data[str(i)]['GB']['tnr'] = 0.0
    else:
        data[str(i)]['GB']['tnr'] = round(tn / (tn + fp) * 100, 2)

    if pd.isna(round(tn / (tn + fn) * 100, 2)):
        data[str(i)]['GB']['npv'] = 0.0
    else:
        data[str(i)]['GB']['npv'] = round(tn / (tn + fn) * 100, 2)

    if pd.isna(round(fp / (fp + tp) * 100, 2)):
        data[str(i)]['GB']['fdr'] = 0.0
    else:
        data[str(i)]['GB']['fdr'] = round(fp / (fp + tp) * 100, 2)

    if pd.isna(round(fn / (fn + tn) * 100, 2)):
        data[str(i)]['GB']['for'] = 0.0
    else:
        data[str(i)]['GB']['for'] = round(fn / (fn + tn) * 100, 2)

    if pd.isna(round(((2 * tp) / (2 * tp + fp + fn)) * 100, 2)):
        data[str(i)]['GB']['f1'] = 0.0
    else:
        data[str(i)]['GB']['f1'] = round(((2 * tp) / (2 * tp + fp + fn)) * 100, 2)

    #XGBoost
    XGB = XGBClassifier(
        eta=0.2,
        max_depth=5,
        subsample=1,
        colsample_bytree=0.7,
        gamma=0,
        eval_metric='mlogloss',
        tree_method='gpu_hist'
    )

    data[str(i)]['XGB'] = {}
    _start_time = time.time()
    XGB.fit(X_train, y_train)
    train_time = round(time.time() - _start_time, 4)
    data[str(i)]['XGB']['Train_time'] = train_time
    _start_time = time.time()
    pred_result = XGB.predict(X_test)
    data[str(i)]['XGB']['Testing_time'] = round(time.time() - _start_time, 4)
    data[str(i)]['XGB']['Trained_accuracy'] = round(XGB.score(X_train, y_train) * 100, 2)
    data[str(i)]['XGB']['Testing_accuracy'] = round(XGB.score(X_test, y_test) * 100, 2)
    fpr_list, tpr_list, thresholds = roc_curve(y_test, pred_result)
    data[str(i)]['XGB']['fpr_list'] = fpr_list
    data[str(i)]['XGB']['tpr_list'] = tpr_list
    data[str(i)]['XGB']['thresholds'] = thresholds
    data[str(i)]['XGB']['AUC'] = auc(fpr_list, tpr_list)
    data[str(i)]['XGB']['Balance'] = balanced_accuracy_score(y_test, pred_result)
    cf_matrix = cm(y_test, pred_result)
    tn, fp, fn, tp = cm(y_test, pred_result).ravel()
    data[str(i)]['XGB']['TN'] = tn
    data[str(i)]['XGB']['FP'] = fp
    data[str(i)]['XGB']['FN'] = fn
    data[str(i)]['XGB']['TP'] = tp
    if pd.isna(round(tp / (tp + fp) * 100, 2)):
        data[str(i)]['XGB']['Precision'] = 0.0
    else:
        data[str(i)]['XGB']['Precision'] = round(tp / (tp + fp) * 100, 2)

    if pd.isna(round(tp / (tp + fn) * 100, 2)):
        data[str(i)]['XGB']['recall'] = 0.0
    else:
        data[str(i)]['XGB']['recall'] = round(tp / (tp + fn) * 100, 2)

    if pd.isna(round(fp / (fp + tn) * 100, 2)):
        data[str(i)]['XGB']['fpr'] = 0.0
    else:
        data[str(i)]['XGB']['fpr'] = round(fp / (fp + tn) * 100, 2)

    if pd.isna(round(fn / (fn + tp) * 100, 2)):
        data[str(i)]['XGB']['fnr'] = 0.0
    else:
        data[str(i)]['XGB']['fnr'] = round(fn / (fn + tp) * 100, 2)

    if pd.isna(round(tn / (tn + fp) * 100, 2)):
        data[str(i)]['XGB']['tnr'] = 0.0
    else:
        data[str(i)]['XGB']['tnr'] = round(tn / (tn + fp) * 100, 2)

    if pd.isna(round(tn / (tn + fn) * 100, 2)):
        data[str(i)]['XGB']['npv'] = 0.0
    else:
        data[str(i)]['XGB']['npv'] = round(tn / (tn + fn) * 100, 2)

    if pd.isna(round(fp / (fp + tp) * 100, 2)):
        data[str(i)]['XGB']['fdr'] = 0.0
    else:
        data[str(i)]['XGB']['fdr'] = round(fp / (fp + tp) * 100, 2)

    if pd.isna(round(fn / (fn + tn) * 100, 2)):
        data[str(i)]['XGB']['for'] = 0.0
    else:
        data[str(i)]['XGB']['for'] = round(fn / (fn + tn) * 100, 2)

    if pd.isna(round(((2 * tp) / (2 * tp + fp + fn)) * 100, 2)):
        data[str(i)]['XGB']['f1'] = 0.0
    else:
        data[str(i)]['XGB']['f1'] = round(((2 * tp) / (2 * tp + fp + fn)) * 100, 2)
with open('../data/data.json', 'w') as outfile:
    json.dump(data, outfile,cls=NpEncoder,indent=4)
