#!/usr/bin/env python
# coding: utf-8

import os, sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl
import xlrd
from sklearn.preprocessing import normalize, StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
# from nilearn.connectome import ConnectivityMeasure
from pandas import DataFrame
import scipy as sc
from scipy import io
from scipy.stats import pearsonr
from os.path import join, exists, dirname
from glob import glob
from brainspace import gradient
import nibabel as nib


# # Load Data

import nibabel as nib

store = 'storename'
path_data = join(store, 'datapath')
workfolder = 'ABIDE_1'
atlas_path = join(store,'atlaspath')
demo = pd.read_excel(join(path_data, 'Phenotypic.xlsx'), sheet_name='surf_n=211', skiprows=0)


sub_list = demo['FILE_ID']
file_name =[]

# Label setting
label = demo['DX_GROUP']
ASD_index=np.where(label == 1)[0]                
TD_index=np.where(label == 2)[0]
sorted_idx = np.concatenate((ASD_index,TD_index), axis = 0)

# Check file list

file_list = []

for i in range(len(sub_list)):
    file_list.append(glob(join(path_data,f'{workfolder}',f'{sub_list[i]}','surf_conn_mat.npy'))) # noaff_surf_par_mat_top30.npy # surf_par_conn_mat.npy
file_list = np.array(file_list).reshape(-1)
print(len(file_list))


conn_mat_ASD = []
conn_mat_TD = []

for i in ASD_index:
    conn_mat_ASD.append(np.load(file_list[i]))
    
for i in TD_index:
    conn_mat_TD.append(np.load(file_list[i]))
    
conn_mat_ASD = np.array(conn_mat_ASD)
conn_mat_TD = np.array(conn_mat_TD)
conn_mat = np.concatenate((conn_mat_ASD, conn_mat_TD))
print('conn_mat_ASD : ', conn_mat_ASD.shape, '     ', 'conn_mat_TD : ', conn_mat_TD.shape, '     ', 'conn_mat : ', conn_mat.shape)


# # Make affine matrix
affine_conn_mat = []

for i, x in enumerate(conn_mat):
    print(i, ' ', end = '' , flush = True)
    z_conn_mat = np.nan_to_num(np.arctanh(np.nan_to_num(x, nan=0.0)), nan=0.0)
    noaff_conn_mat = gradient.compute_affinity(z_conn_mat, sparsity=0.7)
    affine_conn_mat.append(noaff_conn_mat)
    
affine_conn_mat = np.array(affine_conn_mat)
np.save(savepath)


grpmean_conn_mat = np.nan_to_num(np.arctanh(np.nan_to_num(np.mean(conn_mat, axis = 0), nan=0.0)), nan = 0.0)

noaff_grpmean_conn_mat = gradient.compute_affinity(grpmean_conn_mat, sparsity=0.7) # sparsity로 thresholding ratio 조절
noaff_grpmean_conn_mat = np.nan_to_num(noaff_grpmean_conn_mat,nan=0.0)

grpmean_conn_mat = noaff_grpmean_conn_mat
np.save(savepath)

# # symmetry check

from brainspace.gradient import is_symmetric
for i in range(len(file_list)):
    print(is_symmetric(np.load(file_list[i])))


# # Make connectivity matrix to vector 

il1 = np.tril_indices(360)

conn_vec = []

for i in range(len(file_list)):

    vec = np.load(file_list[i])[il1]
    conn_vec.append(vec)
    
conn_vec = np.array(conn_vec).T


conn_vec_ASD = conn_vec[:,ASD_index]
conn_vec_TD = conn_vec[:,TD_index]
conn_vec_sorted = np.concatenate((conn_vec_ASD, conn_vec_TD), axis =1)
print('ASD : ', conn_vec_ASD.shape, '  ', 'TD : ', conn_vec_TD.shape, '  ', 'conn_vec : ', conn_vec_sorted.shape)
np.save(savepath)


# # Sparse Dictrionary Learning
from sklearn.decomposition import DictionaryLearning

SDL = DictionaryLearning(n_components = 100, alpha = 100)

Dict_ASD = SDL.fit_transform(conn_vec_ASD)
repres_ASD = SDL.components_

Dict_TD = SDL.fit_transform(conn_vec_TD)
repres_TD = SDL.components_


from sklearn.decomposition import DictionaryLearning

SDL = DictionaryLearning(n_components = 15, alpha = 13)

Dict_total = SDL.fit_transform(conn_vec_sorted)
repres_total = SDL.components_
print('Dictionary ASD : ', Dict_ASD.shape, '   ', 'Dictionary TD : ', Dict_TD.shape)
print('Dictionary Total : ', Dict_total.shape)


plt.figure(1)
plt.matshow(Dict_ASD[:50,:], cmap = 'jet')

plt.figure(2)
plt.matshow(Dict_TD[:50,:], cmap = 'jet')


plt.figure(1)
plt.matshow(Dict_total[:50,:])


plt.figure(1)
plt.matshow(repres_ASD, cmap = 'jet')

plt.figure(2)
plt.matshow(repres_TD, cmap = 'jet')


plt.figure(1)
plt.matshow(repres_total[:,0:5], cmap = 'jet')


Spec_vec_ASD = np.dot(Dict_ASD, repres_ASD)
Spec_vec_TD = np.dot(Dict_TD, repres_TD)
Spec_vec = np.concatenate((Spec_vec_ASD, Spec_vec_TD), axis =1)
print('DX matrix ASD : ',np.dot(Dict_ASD, repres_ASD).shape, '   ', 'DX matrix TD : ', np.dot(Dict_TD, repres_TD).shape, '   ', 'Spec_vec : ', Spec_vec.shape)


Spec_vec_total = np.dot(Dict_total, repres_total)
print('DX matrix Total : ',np.dot(Dict_total, repres_total).shape)

# # Make Refined 

Refiend_vec_ASD = conn_vec_ASD - Spec_vec_ASD
Refiend_vec_TD = conn_vec_TD - Spec_vec_TD
Refiend_vec_total = conn_vec_sorted - Spec_vec_total


# # Dictionary x sparse coding

DX_mat_ASD = []
DX_mat_TD = []

print('ASD start')

for i in range(ASD_index.shape[0]):
    
    print(i,' ', end = '', flush = True)
    tri = np.zeros((360,360))
    tri[np.tril_indices(360)] = np.dot(Dict_ASD, repres_ASD)[:,i]
    tri = tri + tri.T
    DX_mat_ASD.append(tri)

print('')
print('TD start')

for i in range(TD_index.shape[0]):
    
    print(i,' ', end = '', flush = True)
    tri = np.zeros((360,360))
    tri[np.tril_indices(360)] = np.dot(Dict_TD, repres_TD)[:,i]
    tri = tri + tri.T
    DX_mat_TD.append(tri)
    
DX_mat_ASD = np.array(DX_mat_ASD)
DX_mat_TD = np.array(DX_mat_TD)
print('DX_mat_ASD : ', DX_mat_ASD.shape, '     ', 'DX_mat_TD : ', DX_mat_TD.shape)


# # Group Specific Matrix (DX) t-test
import statsmodels as sm
from statsmodels.stats.multitest import multipletests

row_save = []
col_save = []

for row in range(360): # range(360)
    X = DX_mat_ASD[:,row,:]
    Y = DX_mat_TD[:,row,:]


    [s_1,p_1] = sc.stats.ttest_ind(X,Y, equal_var=False, axis=0)


    p_1_fdr = sm.stats.multitest.multipletests(p_1,alpha=0.05,method='bonferroni')
    print('Row : ', row, '  ', np.where(p_1_fdr[0]==True),'   ', p_1_fdr[1][np.where(p_1_fdr[0]==True)], '\n') #, np.where(p_1_fdr[0]==True)[0].shape)

plt.matshow(DX_mat_ASD[102])


# # Make Refined FC
Refined_mat_ASD = conn_mat_ASD - DX_mat_ASD
Refined_mat_TD = conn_mat_TD - DX_mat_TD
Refined_mat = np.concatenate((Refined_mat_ASD, Refined_mat_TD))
print('Refined_mat_ASD : ', Refined_mat_ASD.shape, '     ', 'Refined_mat_TD : ', Refined_mat_TD.shape,  '     ', 'Refined_mat : ', Refined_mat.shape)


index = 50

plt.figure(1)
plt.matshow(conn_mat_ASD[index], cmap = 'jet' )
plt.xlabel('ROI')
plt.ylabel('ROI')

plt.figure(2)
plt.matshow(DX_mat_ASD[index], cmap = 'jet')
plt.xlabel('ROI')
plt.ylabel('ROI')

plt.figure(3)
plt.matshow(Refined_mat_ASD[index], cmap = 'jet')
plt.xlabel('ROI')
plt.ylabel('ROI')


affine_mat = []

for i, x in enumerate(Refined_mat):
    print(i, ' ', end = '' , flush = True)
    conn_mat = x
    z_conn_mat = np.nan_to_num(conn_mat, nan=0.0)
    noaff_conn_mat = gradient.compute_affinity(z_conn_mat, sparsity=0.7)
    affine_mat.append(noaff_conn_mat)
    
affine_mat = np.array(affine_mat)
np.save(savepath)


grpmean_conn_mat = np.nan_to_num(np.arctanh(np.nan_to_num(np.mean(Refined_mat, axis = 0), nan=0.0)), nan = 0.0)

noaff_grpmean_conn_mat = gradient.compute_affinity(grpmean_conn_mat, sparsity=0.7) # sparsity로 thresholding ratio 조절
noaff_grpmean_conn_mat = np.nan_to_num(noaff_grpmean_conn_mat,nan=0.0)

grpmean_refine_mat = noaff_grpmean_conn_mat
np.save(savepath)


# # Gradient analysis

comp_num = 10

path_work = workpath

Refined = False

print('Refined : ', Refined)

if Refined:
    list_aff = np.load(join(path_work,  f'affine_refine_mat.npy'))  # affine_refine_mat
    grp_aff = np.load(join(path_work,  f'grpmean_refine_mat.npy'))  # grpmean_refine_mat
    
else:
    list_aff = np.load(join(path_work,  f'affine_conn_mat.npy'))  # affine_refine_mat
    grp_aff = np.load(join(path_work,  f'grpmean_conn_mat.npy'))  # grpmean_refine_mat

emb = gradient.embedding.PCAMaps(n_components = comp_num)

# make referece
emb.fit(grp_aff)
ref_lam = emb.lambdas_ 
ref_PC = emb.maps_ 

# gradient analysis
n = len(list_aff)

lam, grad, vec = [None] * n, [None] * n, [None] * n
for i, x1 in enumerate(list_aff):
    print(i,' ', end = '', flush = True)
    emb.fit(x1)
    lam[i], grad[i] = emb.lambdas_ , emb.maps_
    
pa = gradient.ProcrustesAlignment(n_iter=10)
pa.fit(grad, reference=ref_PC)
aligned = np.array(pa.aligned_)


# # Parameter check
# feature extraction : calculate euculidean distance, cosine similarity between reference experianced variance and alignmented each variance

label = demo['DX_GROUP']
label_sorted = np.concatenate((np.array(label[ASD_index]), np.array(label[TD_index])))

site_id = demo['SITE_ID']
site_id_sorted = np.concatenate((np.array(site_id[ASD_index]), np.array(site_id[TD_index])))

site_label = demo['SITE_Label']
site_label_sorted = np.concatenate((np.array(site_label[ASD_index]), np.array(site_label[TD_index])))

Age = demo['AGE_AT_SCAN']
Age_sorted = np.concatenate((np.array(Age[ASD_index]), np.array(Age[TD_index])))

FD = demo['func_mean_fd']
FD_sorted = np.concatenate((np.array(FD[ASD_index]), np.array(FD[TD_index])))

FIQ = demo['FIQ']
FIQ_sorted = np.concatenate((np.array(FIQ[ASD_index]), np.array(FIQ[TD_index])))

pc_num = 1       

# # Feature selection
# ASD vs TD t-test and fdr correction

import statsmodels as sm
from statsmodels.stats.multitest import multipletests

print('Refined : ', Refined)

print(grp_ASD_PCs.shape)
print(grp_TD_PCs.shape)

[s1,p1] = sc.stats.ttest_ind(Refined_vec_ASD.T, Refined_vec_TD.T, equal_var=False, axis=0) # Refined_mat_ASD # grp_ASD_PCs

p1_fdr = sm.stats.multitest.multipletests(p1,alpha=0.05,method='fdr_bh')

print(p1_fdr[0].sum())
print(np.where(p1_fdr[0]==True),'\n')
print(p1_fdr[1][np.where(p1_fdr[0]==True)])

feature = np.where(p1_fdr[0]==True)[0]


# # Classification
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from scipy import stats, interp
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, recall_score, f1_score, accuracy_score, auc
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import seaborn as sns

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = 17)


x_data = conn_vec_sorted.T  # conn_vec_sorted # Spec_vec # Refiend_vec_total
    
site_data = np.array(site_label_sorted)
y_data = np.array(label_sorted)

# make Total list
Total_tpr = []
Total_tnr = []
Total_ppv = []
Total_f1 = []
Total_acc = []
Total_corr = []
Total_corr_p = []
Total_chi = []

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize = [10, 7]) 

for i, (train_idx,test_idx) in enumerate(skf.split(x_data, site_data)): 
    X_train = np.array(x_data)[train_idx] 
    X_test = np.array(x_data)[test_idx]
    y_train = np.array(y_data)[train_idx] 
    y_test = np.array(y_data)[test_idx]

    # train, test number
    print('train number, train ASD : ', X_train.shape[0], len(np.where(y_train ==1)[0]))
    print('test number, test ASD : ', X_test.shape[0], len(np.where(y_test ==1)[0]))
    print(set(train_idx) & set(test_idx))
    print(test_idx)

    # Model Training
    rft=RandomForestClassifier()
    classifier = rft.fit(X_train,y_train)
    
    # prediction
    y_pred = rft.predict(X_test)
    score=rft.score(X_test, y_test.reshape(-1,1)) # accuracy
    
    # accuracy
    print('Accuracy is : %.3f'% accuracy_score(y_test,y_pred))
    Total_acc.append(accuracy_score(y_test,y_pred))
    
    # sensitivity(recall, TPR), specifity(TNR), f1_score
    f1 = f1_score(y_test, y_pred, average='binary')
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    tnr = tn/(tn+fp)
    tpr = tp/(tp+fn) # recall_score(y_test, y_pred, average='binary')
    ppv = tp/(tp+fp)

    
    print("Sensitivity(TPR) = %.3f" %(tpr))
    print("Specifity(TNR) = %.3f" %(tnr))
    print("Precision(PPV) = %.3f" %(ppv))
    print("f1 score = %.3f" %(f1))
    print('\n ----- \n')
    Total_tpr.append(tpr)
    Total_tnr.append(tnr)
    Total_ppv.append(ppv)
    Total_f1.append(f1)
    
    # confusion matrix 
    sns.set(style = 'white')
    
    disp = plot_confusion_matrix(classifier, X_test, y_test.reshape(-1,1), display_labels= ['ASD', "HC"], cmap=plt.cm.Purples)
    plt.title('Fold {} confusion matrix'.format(i+1))

    # plot roc curve
    viz = metrics.plot_roc_curve(rft,X_test,y_test.reshape(-1,1),name='ROC fold {}'.format(i+1),alpha=0.5, lw=1.2, ax=ax)
    
    interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

# Mean roc  
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b', label='Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=1)

# roc 커브 표준편차 추가
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Random forest classification ROC curve")
ax.legend(loc="lower right") 
plt.show()

print('\n')
print('Fold accuracy = ', np.round(Total_acc,3))
print('Mean accuracy = ',np.array(Total_acc).mean(), np.array(Total_acc).std())

print('Fold Sensitivity = ', np.round(Total_tpr,3))
print('Mean Sensitivity = ',np.array(Total_tpr).mean(), np.array(Total_tpr).std())

print('Fold Specifity = ', np.round(Total_tnr,3))
print('Mean Specifity = ',np.array(Total_tnr).mean(), np.array(Total_tnr).std())

print('Fold Precision = ', np.round(Total_ppv,3))
print('Mean Precision = ',np.array(Total_ppv).mean(), np.array(Total_ppv).std())

print('Fold f1 score = ', np.round(Total_f1,3))
print('Mean f1 score = ',np.array(Total_f1).mean(), np.array(Total_f1).std())



# Evaluation
mv = np.array([Total_tpr, Total_tnr, Total_ppv, Total_f1, Total_acc])
measure_value = pd.DataFrame(mv, index = ['Sensitivity(TPR)', 'Specifity(TNR)', 'Precision(ppv)', 'f1 score', 'Accuracy'],
                             columns = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])
                             #['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5','Fold 6', 'Fold 7', 'Fold 8', 'Fold 9', 'Fold 10'])
                             #['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5'])
                             
pd.options.display.float_format = '{:.3f}'.format 

measure_value.mean(axis=1)
measure_value.std(axis=1)

measure_value['Mean'] = measure_value.mean(axis=1)
measure_value['Std'] = measure_value.std(axis=1)
measure_value = measure_value.T
measure_value