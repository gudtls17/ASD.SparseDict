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

# # Clustering
# load group-wise specific vector

Spec_vec_ASD = np.load('Z:\\hschoi\\backup\\hschoi\\1.asd.grad\\data\\sparse.classification\\Spec_vec_ASD.npy')
Spec_vec_TD = np.load('Z:\\hschoi\\backup\\hschoi\\1.asd.grad\\data\\sparse.classification\\Spec_vec_TD.npy')
Spec_vec = np.load('Z:\\hschoi\\backup\\hschoi\\1.asd.grad\\data\\sparse.classification\\Spec_vec.npy')
print('Group wise matrix ASD : ',Spec_vec_ASD.shape, '   ', 'Group wise matrix TD : ', Spec_vec_TD.shape, '   ', 'Spec_vec : ', Spec_vec.shape)


from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

H_clus = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
data = H_clus.fit(Spec_vec_ASD.T) # Spec_vec.T # Spec_vec_ASD.T


def plot_dendrogram(M, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(M.children_.shape[0])
    n_samples = len(M.labels_)
    for i, merge in enumerate(M.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([M.children_, M.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


plot_dendrogram(data, truncate_mode='level', p=3)


H_clus = AgglomerativeClustering(n_clusters=3)

cluster_labels = H_clus.fit_predict(Spec_vec_ASD.T) # Spec_vec.T # Spec_vec_ASD.T
cluster_labels


# # Kmeans Clusteirng

from sklearn.cluster import KMeans

K_clus = KMeans(n_clusters=3, max_iter=1000, random_state=None) # precomputed # rbf
cluster_labels = K_clus.fit_predict(Spec_vec.T) # Spec_vec.T # Spec_vec_ASD.T
cluster_labels


# # Silhouette score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import davies_bouldin_score

for i in range(2,10):
    K_clus = KMeans(n_clusters=i, max_iter=1000, random_state=None) # precomputed # rbf
    cluster_labels = K_clus.fit_predict(Spec_vec_ASD.T) # Spec_vec.T # Spec_vec_ASD.T

    silhouette_avg = davies_bouldin_score(Spec_vec_ASD.T, cluster_labels)
    print("For n_clusters =", i,"The average silhouette_score is :", silhouette_avg)


# # Davies Bouledin score
# Davies bouldin score plot

plt.figure(1)
input1 = [0, 0, 2.930755866442926, 2.827201330955969, 2.6431571000713303, 2.5190458085945773, 2.5227719013501155, 2.5367862330303015, 2.4525232816748415, 2.5221604188276814, 2.402996845037538]
plt.plot(input1,'-o')
plt.xlabel("Cluster number")
plt.ylabel("Intra-cluster to inter-cluster")


# # Consensus Clsutering
import numpy as np
from itertools import combinations
import bisect
import matplotlib.pyplot as plt



class ConsensusCluster:
    """
      Implementation of Consensus clustering, following the paper
      https://link.springer.com/content/pdf/10.1023%2FA%3A1023949509487.pdf
      Args:
        * cluster -> clustering class
        * NOTE: the class is to be instantiated with parameter `n_clusters`,
          and possess a `fit_predict` method, which is invoked on data.
        * L -> smallest number of clusters to try
        * K -> biggest number of clusters to try
        * H -> number of resamplings for each cluster number
        * resample_proportion -> percentage to sample
        * Mk -> consensus matrices for each k (shape =(K,data.shape[0],data.shape[0]))
                (NOTE: every consensus matrix is retained, like specified in the paper)
        * Ak -> area under CDF for each number of clusters 
                (see paper: section 3.3.1. Consensus distribution.)
        * deltaK -> changes in areas under CDF
                (see paper: section 3.3.1. Consensus distribution.)
        * self.bestK -> number of clusters that was found to be best
      """

    def __init__(self, cluster, L, K, H, resample_proportion=0.5):
        assert 0 <= resample_proportion <= 1, "proportion has to be between 0 and 1"
        self.cluster_ = cluster
        self.resample_proportion_ = resample_proportion
        self.L_ = L
        self.K_ = K
        self.H_ = H
        self.Mk = None
        self.Ak = None
        self.deltaK = None
        self.bestK = None

    def _internal_resample(self, data, proportion):
        """
        Args:
          * data -> (examples,attributes) format
          * proportion -> percentage to sample
        """
        resampled_indices = np.random.choice(
            range(data.shape[0]), size=int(data.shape[0]*proportion), replace=False)
        return resampled_indices, data[resampled_indices, :]

    def fit(self, data, verbose=False):
        """
        Fits a consensus matrix for each number of clusters

        Args:
          * data -> (examples,attributes) format
          * verbose -> should print or not
        """
        Mk = np.zeros((self.K_-self.L_, data.shape[0], data.shape[0]))
        Is = np.zeros((data.shape[0],)*2)
        plot_Ak = []
        for k in range(self.L_, self.K_):  # for each number of clusters
            i_ = k-self.L_
            if verbose:
                print("At k = %d, aka. iteration = %d" % (k, i_))
            for h in range(self.H_):  # resample H times
                if verbose:
                    print("\tAt resampling h = %d, (k = %d)" % (h, k))
                resampled_indices, resample_data = self._internal_resample(
                    data, self.resample_proportion_)
                Mh = self.cluster_(n_clusters=k).fit_predict(resample_data)
                # find indexes of elements from same clusters with bisection
                # on sorted array => this is more efficient than brute force search
                id_clusts = np.argsort(Mh)
                sorted_ = Mh[id_clusts]
                for i in range(k):  # for each cluster
                    ia = bisect.bisect_left(sorted_, i)
                    ib = bisect.bisect_right(sorted_, i)
                    is_ = id_clusts[ia:ib]
                    ids_ = np.array(list(combinations(is_, 2))).T
                    # sometimes only one element is in a cluster (no combinations)
                    if ids_.size != 0:
                        Mk[i_, ids_[0], ids_[1]] += 1
                # increment counts
                ids_2 = np.array(list(combinations(resampled_indices, 2))).T
                Is[ids_2[0], ids_2[1]] += 1
            Mk[i_] /= Is+1e-8  # consensus matrix
            # Mk[i_] is upper triangular (with zeros on diagonal), we now make it symmetric
            Mk[i_] += Mk[i_].T
            Mk[i_, range(data.shape[0]), range(
                data.shape[0])] = 1  # always with self
            Is.fill(0)  # reset counter
        self.Mk = Mk
        # fits areas under the CDFs
        self.Ak = np.zeros(self.K_-self.L_)
        for i, m in enumerate(Mk):
            hist, bins = np.histogram(m.ravel(), density=True)
            self.Ak[i] = np.sum(h*(b-a)
                             for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)))
            
            for b, a, h in zip(bins[1:], bins[:-1], np.cumsum(hist)):
                plot_Ak.append(h*(b-a))
#             print(plot_Ak)
            
        # fits differences between areas under CDFs
        self.deltaK = np.array([(Ab-Aa)/Aa if i > 2 else Aa
                                for Ab, Aa, i in zip(self.Ak[1:], self.Ak[:-1], range(self.L_, self.K_-1))])
        self.bestK = np.argmax(self.deltaK) +             self.L_ if self.deltaK.size > 0 else self.L_
        print(self.Ak)
        print(self.deltaK)
       
        

    def predict(self):
        """
        Predicts on the consensus matrix, for best found cluster number
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            1-self.Mk[self.bestK-self.L_])

    def predict_data(self, data):
        """
        Predicts on the data, for best found cluster number
        Args:
          * data -> (examples,attributes) format 
        """
        assert self.Mk is not None, "First run fit"
        return self.cluster_(n_clusters=self.bestK).fit_predict(
            data)
    
from sklearn.cluster import SpectralClustering


cc = ConsensusCluster(KMeans, 3, 10, 1000, resample_proportion=0.9) # KMeans # AgglomerativeClustering # SpectralClustering
cc.fit(Spec_vec_ASD.T) # prox_mat_sorted_n211

cluster_labels = cc.predict_data(Spec_vec_ASD.T)
cluster_labels


Css_CDF_UnArea_change = np.array([0,0, 0,  0.30902006, 0.18648233, 0.10972098, 0.08561067, 0.06745405, 0.04399576])

plt.figure(1,(10,5))

plt.plot(Css_CDF_UnArea_change,'-o')
plt.xlim(1.5,8.5)
plt.xlabel('Number of clusters K')
plt.ylabel('Consensus matrix CDF difference')
plt.title('Consensus matrix area under CDF change')


# # Total
# Subject Number each subtype's ASD / TD

for i in range(cluster_labels.max()+1):
    print(np.where((cluster_labels==i)&(label_sorted==1))[0].shape, '/', np.where((cluster_labels==i)&(label_sorted==2))[0].shape,'\n')


# # Only ASD

for i in range(cluster_labels.max()+1):
    print(np.where(cluster_labels==i)[0].shape, '/', np.where(cluster_labels==i)[0].shape,'\n')


# # Check subtype's data number

ADI_social = np.nan_to_num(np.array(demo['ADI_R_SOCIAL_TOTAL_A']), nan = -2) # -9999 -> -1, nan -> -2    Full scale IQ
ADI_social = np.where(ADI_social == -9999, -1, ADI_social)
ADI_social_sorted = ADI_social[sorted_idx]

ADI_verbal = np.nan_to_num(np.array(demo['ADI_R_VERBAL_TOTAL_BV']), nan = -2) # -9999 -> -1, nan -> -2    Full scale IQ
ADI_verbal = np.where(ADI_verbal == -9999, -1, ADI_verbal)
ADI_verbal_sorted = ADI_verbal[sorted_idx]

ADI_behav = np.nan_to_num(np.array(demo['ADI_RRB_TOTAL_C']), nan = -2) # -9999 -> -1, nan -> -2    Full scale IQ
ADI_behav = np.where(ADI_behav == -9999, -1, ADI_behav)
ADI_behav_sorted = ADI_behav[sorted_idx]


# ADOS_Score 

label = np.array(demo['DX_GROUP'])

SRS = np.nan_to_num(np.array(demo['SRS_RAW_TOTAL']), nan = -2) # -9999 -> -1, nan -> -2    Full scale IQ
SRS = np.where(SRS == -9999, -1, SRS)
SRS_sorted = SRS[sorted_idx]

ADOS_Total = np.nan_to_num(np.array(demo['ADOS_TOTAL']), nan = -2) # -9999 -> -1, nan -> -2
ADOS_Total = np.where(ADOS_Total == -9999, -1, ADOS_Total)
ADOS_Total
ADOS_Total_sorted = ADOS_Total[sorted_idx]

ADOS_Total_idx = np.where(ADOS_Total >= 0)[0]
ADOS_Total_idx_sorted = np.where(ADOS_Total_sorted >= 0)[0]
ADOS_missing_idx = np.where(ADOS_Total < 0)[0]

ADOS_ASD_idx = np.where((ADOS_Total >= -3) & (label == 1))[0]

ADOS_comm = np.nan_to_num(np.array(demo['ADOS_COMM']), nan = -2)
ADOS_comm = np.where(ADOS_comm == -9999, -1, ADOS_comm)
ADOS_comm_sorted = ADOS_comm[sorted_idx]

ADOS_social =  np.nan_to_num(np.array(demo['ADOS_SOCIAL']), nan = -2)
ADOS_social = np.where(ADOS_social == -9999, -1, ADOS_social)
ADOS_social_sorted = ADOS_social[sorted_idx]

ADOS_behav =  np.nan_to_num(np.array(demo['ADOS_STEREO_BEHAV']), nan = -2)
ADOS_behav = np.where(ADOS_behav == -9999, -1, ADOS_behav)
ADOS_behav_sorted = ADOS_behav[sorted_idx]

ADOS_exist_score = ADOS_Total[ADOS_Total_idx]
ADOS_exist_comm = ADOS_comm[ADOS_Total_idx]
ADOS_exist_social = ADOS_social[ADOS_Total_idx]
ADOS_exist_behav = ADOS_behav[ADOS_Total_idx]
ADOS_exist_Label = label[ADOS_Total_idx]
ADOS_exist_age = np.array(Age[ADOS_Total_idx])

# Sorted

ADOS_exist_score_sorted = ADOS_Total_sorted[ADOS_Total_idx_sorted]
ADOS_exist_comm_sorted = ADOS_comm_sorted[ADOS_Total_idx_sorted]
ADOS_exist_social_sorted = ADOS_social_sorted[ADOS_Total_idx_sorted]
ADOS_exist_behav_sorted = ADOS_behav_sorted[ADOS_Total_idx_sorted]
ADOS_exist_Label_sorted = label_sorted[ADOS_Total_idx_sorted]
ADOS_exist_age_sorted = np.array(Age[sorted_idx][ADOS_Total_idx_sorted])

ADOS_Miss_score = ADOS_Total[ADOS_missing_idx]
ADOS_Miss_comm = ADOS_comm[ADOS_missing_idx]
ADOS_Miss_social = ADOS_social[ADOS_missing_idx]
ADOS_Miss_behav = ADOS_behav[ADOS_missing_idx]
ADOS_Miss_Label = label[ADOS_missing_idx]
ADOS_Miss_age = np.array(Age[ADOS_missing_idx])

import seaborn as sns


score = SRS_sorted # ADOS_Total_sorted # ADOS_comm_sorted # ADOS_social_sorted # ADOS_behav_sorted # SRS_sorted # ADI_behav_sorted # ADI_verbal_sorted # ADI_social_sorted
idx = np.where(score>=0)[0]
y = score[idx] # ADOS_Total_sorted # ADOS_comm_sorted # ADOS_social_sorted # ADOS_behav_sorted

df = DataFrame([y, cluster_labels[idx]+ 1]) # cluster_labels[ADOS_Total_idx_sorted] SRS_exist_idx_sorted
df = df.T
title_1 = 'SRS'
df.columns = [title_1,'Subtype']

plt.figure(1,(8,8))
sns.set(style = 'whitegrid', font_scale=1.5)
sns.boxplot(x = 'Subtype' , y = title_1, data = df)
sns.swarmplot(x = 'Subtype' , y = title_1, data = df, color = '.25')


# # t-test Total 

import statsmodels as sm
from statsmodels.stats.multitest import multipletests

# ADI_behav_sorted # ADI_verbal_sorted # ADI_social_sorted
# ADOS_Total_sorted # ADOS_comm_sorted # ADOS_social_sorted # ADOS_behav_sorted # SRS_sorted

score_total = ADOS_Total_sorted
score_comm = ADOS_comm_sorted
score_social = ADOS_social_sorted
score_behav = ADOS_behav_sorted
score_SRS = SRS_sorted
score_ADI_social = ADI_social_sorted
score_ADI_verbal = ADI_verbal_sorted
score_ADI_behav = ADI_behav_sorted

idx_total = np.where(score_total>=0)[0]
idx_comm = np.where(score_comm>=0)[0]
idx_social = np.where(score_social>=0)[0]
idx_behav = np.where(score_behav>=0)[0]
idx_SRS = np.where(score_SRS>=0)[0]
idx_ADI_social = np.where(score_ADI_social>=0)[0]
idx_ADI_verbal = np.where(score_ADI_verbal>=0)[0]
idx_ADI_behav = np.where(score_ADI_behav>=0)[0]

i = 0
j = 2

[s_1,p_1] = sc.stats.ttest_ind(score_total[np.intersect1d(np.where(cluster_labels==i)[0],idx_total)], 
                               score_total[np.intersect1d(np.where(cluster_labels==j)[0],idx_total)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_2,p_2] = sc.stats.ttest_ind(score_comm[np.intersect1d(np.where(cluster_labels==i)[0],idx_comm)], 
                               score_comm[np.intersect1d(np.where(cluster_labels==j)[0],idx_comm)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_3,p_3] = sc.stats.ttest_ind(score_social[np.intersect1d(np.where(cluster_labels==i)[0],idx_social)], 
                               score_social[np.intersect1d(np.where(cluster_labels==j)[0],idx_social)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_4,p_4] = sc.stats.ttest_ind(score_behav[np.intersect1d(np.where(cluster_labels==i)[0],idx_behav)], 
                               score_behav[np.intersect1d(np.where(cluster_labels==j)[0],idx_behav)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_5,p_5] = sc.stats.ttest_ind(score_SRS[np.intersect1d(np.where(cluster_labels==i)[0],idx_SRS)], 
                               score_SRS[np.intersect1d(np.where(cluster_labels==j)[0],idx_SRS)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_6,p_6] = sc.stats.ttest_ind(score_ADI_social[np.intersect1d(np.where(cluster_labels==i)[0],idx_ADI_social)], 
                               score_ADI_social[np.intersect1d(np.where(cluster_labels==j)[0],idx_ADI_social)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_7,p_7] = sc.stats.ttest_ind(score_ADI_verbal[np.intersect1d(np.where(cluster_labels==i)[0],idx_ADI_verbal)], 
                               score_ADI_verbal[np.intersect1d(np.where(cluster_labels==j)[0],idx_ADI_verbal)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_8,p_8] = sc.stats.ttest_ind(score_ADI_behav[np.intersect1d(np.where(cluster_labels==i)[0],idx_ADI_behav)], 
                               score_ADI_behav[np.intersect1d(np.where(cluster_labels==j)[0],idx_ADI_behav)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

# p_0_fdr = sm.stats.multitest.multipletests(p_0,alpha=0.05,method='fdr_bh')
p_1_fdr = sm.stats.multitest.multipletests(p_1,alpha=0.05,method='fdr_bh')
p_2_fdr = sm.stats.multitest.multipletests(p_2,alpha=0.05,method='fdr_bh')
p_3_fdr = sm.stats.multitest.multipletests(p_3,alpha=0.05,method='fdr_bh')
p_4_fdr = sm.stats.multitest.multipletests(p_4,alpha=0.05,method='fdr_bh')
p_5_fdr = sm.stats.multitest.multipletests(p_5,alpha=0.05,method='fdr_bh')
p_6_fdr = sm.stats.multitest.multipletests(p_6,alpha=0.05,method='fdr_bh')
p_7_fdr = sm.stats.multitest.multipletests(p_7,alpha=0.05,method='fdr_bh')
p_8_fdr = sm.stats.multitest.multipletests(p_8,alpha=0.05,method='fdr_bh')


# print('0 : ', np.where(p_0_fdr[0]==True),'\n', np.where(p_0_fdr[0]==True)[0].shape)
print('1 : ', np.where(p_1_fdr[0]==True),'\n', np.where(p_1_fdr[0]==True)[0].shape)
print('2 : ', np.where(p_2_fdr[0]==True),'\n', np.where(p_2_fdr[0]==True)[0].shape)
print('3 : ', np.where(p_3_fdr[0]==True),'\n', np.where(p_3_fdr[0]==True)[0].shape)
print('4 : ', np.where(p_4_fdr[0]==True),'\n', np.where(p_4_fdr[0]==True)[0].shape)
print('5 : ', np.where(p_5_fdr[0]==True),'\n', np.where(p_5_fdr[0]==True)[0].shape)
print('6 : ', np.where(p_6_fdr[0]==True),'\n', np.where(p_6_fdr[0]==True)[0].shape)
print('7 : ', np.where(p_7_fdr[0]==True),'\n', np.where(p_7_fdr[0]==True)[0].shape)
print('8 : ', np.where(p_8_fdr[0]==True),'\n', np.where(p_8_fdr[0]==True)[0].shape)

# print('0 : ', p_0_fdr)
print('1 : ', p_1_fdr)
print('2 : ', p_2_fdr)
print('3 : ', p_3_fdr)
print('4 : ', p_4_fdr)
print('5 : ', p_5_fdr)
print('6 : ', p_6_fdr)
print('7 : ', p_7_fdr)
print('8 : ', p_8_fdr)

# # Anova Total

import statsmodels as sm
from statsmodels.stats.multitest import multipletests

score_total = ADOS_Total_sorted
score_comm = ADOS_comm_sorted
score_social = ADOS_social_sorted
score_behav = ADOS_behav_sorted
score_SRS = SRS_sorted
score_ADI_social = ADI_social_sorted
score_ADI_verbal = ADI_verbal_sorted
score_ADI_behav = ADI_behav_sorted

idx_total = np.where(score_total>=0)[0]
idx_comm = np.where(score_comm>=0)[0]
idx_social = np.where(score_social>=0)[0]
idx_behav = np.where(score_behav>=0)[0]
idx_SRS = np.where(score_SRS>=0)[0]
idx_ADI_social = np.where(score_ADI_social>=0)[0]
idx_ADI_verbal = np.where(score_ADI_verbal>=0)[0]
idx_ADI_behav = np.where(score_ADI_behav>=0)[0]


[f_1,p_1] = sc.stats.f_oneway(score_total[np.intersect1d(np.where(cluster_labels==0)[0],idx_total)], 
                              score_total[np.intersect1d(np.where(cluster_labels==1)[0],idx_total)],
                              score_total[np.intersect1d(np.where(cluster_labels==2)[0],idx_total)])

[f_2,p_2] = sc.stats.f_oneway(score_comm[np.intersect1d(np.where(cluster_labels==0)[0],idx_comm)], 
                              score_comm[np.intersect1d(np.where(cluster_labels==1)[0],idx_comm)],
                              score_comm[np.intersect1d(np.where(cluster_labels==2)[0],idx_comm)])

[f_3,p_3] = sc.stats.f_oneway(score_social[np.intersect1d(np.where(cluster_labels==0)[0],idx_social)], 
                              score_social[np.intersect1d(np.where(cluster_labels==1)[0],idx_social)], 
                              score_social[np.intersect1d(np.where(cluster_labels==2)[0],idx_social)])

[f_4,p_4] = sc.stats.f_oneway(score_behav[np.intersect1d(np.where(cluster_labels==0)[0],idx_behav)], 
                              score_behav[np.intersect1d(np.where(cluster_labels==1)[0],idx_behav)], 
                              score_behav[np.intersect1d(np.where(cluster_labels==2)[0],idx_behav)])

[f_5,p_5] = sc.stats.f_oneway(score_SRS[np.intersect1d(np.where(cluster_labels==0)[0],idx_SRS)], 
                              score_SRS[np.intersect1d(np.where(cluster_labels==1)[0],idx_SRS)], 
                              score_SRS[np.intersect1d(np.where(cluster_labels==2)[0],idx_SRS)])

p_1_fdr = sm.stats.multitest.multipletests(p_1,alpha=0.05,method='fdr_bh')
print(np.where(p_1_fdr[0]==True),'\n', np.where(p_1_fdr[0]==True)[0].shape)

p_2_fdr = sm.stats.multitest.multipletests(p_2,alpha=0.05,method='fdr_bh')
print(np.where(p_2_fdr[0]==True),'\n', np.where(p_2_fdr[0]==True)[0].shape)

p_3_fdr = sm.stats.multitest.multipletests(p_3,alpha=0.05,method='fdr_bh')
print(np.where(p_3_fdr[0]==True),'\n', np.where(p_3_fdr[0]==True)[0].shape)

p_4_fdr = sm.stats.multitest.multipletests(p_4,alpha=0.05,method='fdr_bh')
print(np.where(p_4_fdr[0]==True),'\n', np.where(p_4_fdr[0]==True)[0].shape)

p_5_fdr = sm.stats.multitest.multipletests(p_5,alpha=0.05,method='fdr_bh')
print(np.where(p_5_fdr[0]==True),'\n', np.where(p_5_fdr[0]==True)[0].shape)

print('1 : ', p_1_fdr)
print('2 : ', p_2_fdr)
print('3 : ', p_3_fdr)
print('4 : ', p_4_fdr)
print('5 : ', p_5_fdr)

# ANOVA_pval = p_1_fdr[1]


# # Only ASD
import seaborn as sns

score = SRS_sorted # ADOS_Total_sorted # ADOS_comm_sorted # ADOS_social_sorted # ADOS_behav_sorted # SRS_sorted # ADI_behav_sorted # ADI_verbal_sorted # ADI_social_sorted
idx = np.where(score[:103]>=0)[0]
y = score[:103][idx] # ADOS_Total_sorted # ADOS_comm_sorted # ADOS_social_sorted # ADOS_behav_sorted

df = DataFrame([y, cluster_labels[idx]+ 1]) # cluster_labels[ADOS_Total_idx_sorted] SRS_exist_idx_sorted
df = df.T
title_1 = 'SRS'
df.columns = [title_1,'Subtype']

plt.figure(1,(8,8))
sns.set(style = 'whitegrid', font_scale=1.5)
sns.boxplot(x = 'Subtype' , y = title_1, data = df)
sns.swarmplot(x = 'Subtype' , y = title_1, data = df, color = '.25')



import statsmodels as sm
from statsmodels.stats.multitest import multipletests

# ADI_behav_sorted # ADI_verbal_sorted # ADI_social_sorted
# ADOS_Total_sorted # ADOS_comm_sorted # ADOS_social_sorted # ADOS_behav_sorted # SRS_sorted


score_total = ADOS_Total_sorted
score_comm = ADOS_comm_sorted
score_social = ADOS_social_sorted
score_behav = ADOS_behav_sorted
score_SRS = SRS_sorted
score_ADI_social = ADI_social_sorted
score_ADI_verbal = ADI_verbal_sorted
score_ADI_behav = ADI_behav_sorted

idx_total = np.where(score_total>=0)[0]
idx_comm = np.where(score_comm>=0)[0]
idx_social = np.where(score_social>=0)[0]
idx_behav = np.where(score_behav>=0)[0]
idx_SRS = np.where(score_SRS>=0)[0]
idx_ADI_social = np.where(score_ADI_social>=0)[0]
idx_ADI_verbal = np.where(score_ADI_verbal>=0)[0]
idx_ADI_behav = np.where(score_ADI_behav>=0)[0]

i = 0
j = 1

[s_1,p_1] = sc.stats.ttest_ind(score_total[np.intersect1d(np.where(cluster_labels==i)[0],idx_total)], 
                               score_total[np.intersect1d(np.where(cluster_labels==j)[0],idx_total)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_2,p_2] = sc.stats.ttest_ind(score_comm[np.intersect1d(np.where(cluster_labels==i)[0],idx_comm)], 
                               score_comm[np.intersect1d(np.where(cluster_labels==j)[0],idx_comm)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_3,p_3] = sc.stats.ttest_ind(score_social[np.intersect1d(np.where(cluster_labels==i)[0],idx_social)], 
                               score_social[np.intersect1d(np.where(cluster_labels==j)[0],idx_social)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_4,p_4] = sc.stats.ttest_ind(score_behav[np.intersect1d(np.where(cluster_labels==i)[0],idx_behav)], 
                               score_behav[np.intersect1d(np.where(cluster_labels==j)[0],idx_behav)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_5,p_5] = sc.stats.ttest_ind(score_SRS[np.intersect1d(np.where(cluster_labels==i)[0],idx_SRS)], 
                               score_SRS[np.intersect1d(np.where(cluster_labels==j)[0],idx_SRS)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_6,p_6] = sc.stats.ttest_ind(score_ADI_social[np.intersect1d(np.where(cluster_labels==i)[0],idx_ADI_social)], 
                               score_ADI_social[np.intersect1d(np.where(cluster_labels==j)[0],idx_ADI_social)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_7,p_7] = sc.stats.ttest_ind(score_ADI_verbal[np.intersect1d(np.where(cluster_labels==i)[0],idx_ADI_verbal)], 
                               score_ADI_verbal[np.intersect1d(np.where(cluster_labels==j)[0],idx_ADI_verbal)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]

[s_8,p_9] = sc.stats.ttest_ind(score_ADI_behav[np.intersect1d(np.where(cluster_labels==i)[0],idx_ADI_behav)], 
                               score_ADI_behav[np.intersect1d(np.where(cluster_labels==j)[0],idx_ADI_behav)], equal_var=False, axis=0) # [ADOS_Total_idx_sorted]


# p_0_fdr = sm.stats.multitest.multipletests(p_0,alpha=0.05,method='fdr_bh')
p_1_fdr = sm.stats.multitest.multipletests(p_1,alpha=0.05,method='fdr_bh')
p_2_fdr = sm.stats.multitest.multipletests(p_2,alpha=0.05,method='fdr_bh')
p_3_fdr = sm.stats.multitest.multipletests(p_3,alpha=0.05,method='fdr_bh')
p_4_fdr = sm.stats.multitest.multipletests(p_4,alpha=0.05,method='fdr_bh')
p_5_fdr = sm.stats.multitest.multipletests(p_5,alpha=0.05,method='fdr_bh')


# print('0 : ', np.where(p_0_fdr[0]==True),'\n', np.where(p_0_fdr[0]==True)[0].shape)
print('1 : ', np.where(p_1_fdr[0]==True),'\n', np.where(p_1_fdr[0]==True)[0].shape)
print('2 : ', np.where(p_2_fdr[0]==True),'\n', np.where(p_2_fdr[0]==True)[0].shape)
print('3 : ', np.where(p_3_fdr[0]==True),'\n', np.where(p_3_fdr[0]==True)[0].shape)
print('4 : ', np.where(p_4_fdr[0]==True),'\n', np.where(p_4_fdr[0]==True)[0].shape)
print('5 : ', np.where(p_5_fdr[0]==True),'\n', np.where(p_5_fdr[0]==True)[0].shape)
print('6 : ', np.where(p_6_fdr[0]==True),'\n', np.where(p_6_fdr[0]==True)[0].shape)
print('7 : ', np.where(p_7_fdr[0]==True),'\n', np.where(p_7_fdr[0]==True)[0].shape)
print('8 : ', np.where(p_8_fdr[0]==True),'\n', np.where(p_8_fdr[0]==True)[0].shape)


# print('0 : ', p_0_fdr)
print('1 : ', p_1_fdr)
print('2 : ', p_2_fdr)
print('3 : ', p_3_fdr)
print('4 : ', p_4_fdr)
print('5 : ', p_5_fdr)
print('6 : ', p_6_fdr)
print('7 : ', p_7_fdr)
print('8 : ', p_8_fdr)

# # Anova ASD
import statsmodels as sm
from statsmodels.stats.multitest import multipletests

score_total = ADOS_Total_sorted[:103]
score_comm = ADOS_comm_sorted[:103]
score_social = ADOS_social_sorted[:103]
score_behav = ADOS_behav_sorted[:103]
score_SRS = SRS_sorted[:103]

idx_total = np.where(score_total>=0)[0]
idx_comm = np.where(score_comm>=0)[0]
idx_social = np.where(score_social>=0)[0]
idx_behav = np.where(score_behav>=0)[0]
idx_SRS = np.where(score_SRS>=0)[0]


[f_1,p_1] = sc.stats.f_oneway(score_total[np.intersect1d(np.where(cluster_labels==0)[0],idx_total)], 
                              score_total[np.intersect1d(np.where(cluster_labels==1)[0],idx_total)],
                              score_total[np.intersect1d(np.where(cluster_labels==2)[0],idx_total)])

[f_2,p_2] = sc.stats.f_oneway(score_comm[np.intersect1d(np.where(cluster_labels==0)[0],idx_comm)], 
                              score_comm[np.intersect1d(np.where(cluster_labels==1)[0],idx_comm)],
                              score_comm[np.intersect1d(np.where(cluster_labels==2)[0],idx_comm)])

[f_3,p_3] = sc.stats.f_oneway(score_social[np.intersect1d(np.where(cluster_labels==0)[0],idx_social)], 
                              score_social[np.intersect1d(np.where(cluster_labels==1)[0],idx_social)], 
                              score_social[np.intersect1d(np.where(cluster_labels==2)[0],idx_social)])

[f_4,p_4] = sc.stats.f_oneway(score_behav[np.intersect1d(np.where(cluster_labels==0)[0],idx_behav)], 
                              score_behav[np.intersect1d(np.where(cluster_labels==1)[0],idx_behav)], 
                              score_behav[np.intersect1d(np.where(cluster_labels==2)[0],idx_behav)])

[f_5,p_5] = sc.stats.f_oneway(score_SRS[np.intersect1d(np.where(cluster_labels==0)[0],idx_SRS)], 
                              score_SRS[np.intersect1d(np.where(cluster_labels==1)[0],idx_SRS)], 
                              score_SRS[np.intersect1d(np.where(cluster_labels==2)[0],idx_SRS)])

p_1_fdr = sm.stats.multitest.multipletests(p_1,alpha=0.05,method='fdr_bh')
print(np.where(p_1_fdr[0]==True),'\n', np.where(p_1_fdr[0]==True)[0].shape)

p_2_fdr = sm.stats.multitest.multipletests(p_2,alpha=0.05,method='fdr_bh')
print(np.where(p_2_fdr[0]==True),'\n', np.where(p_2_fdr[0]==True)[0].shape)

p_3_fdr = sm.stats.multitest.multipletests(p_3,alpha=0.05,method='fdr_bh')
print(np.where(p_3_fdr[0]==True),'\n', np.where(p_3_fdr[0]==True)[0].shape)

p_4_fdr = sm.stats.multitest.multipletests(p_4,alpha=0.05,method='fdr_bh')
print(np.where(p_4_fdr[0]==True),'\n', np.where(p_4_fdr[0]==True)[0].shape)

p_5_fdr = sm.stats.multitest.multipletests(p_5,alpha=0.05,method='fdr_bh')
print(np.where(p_5_fdr[0]==True),'\n', np.where(p_5_fdr[0]==True)[0].shape)

print('1 : ', p_1_fdr)
print('2 : ', p_2_fdr)
print('3 : ', p_3_fdr)
print('4 : ', p_4_fdr)
print('5 : ', p_5_fdr)

# ANOVA_pval = p_1_fdr[1]


# # Visualization

# Prepare visualization

import vtk

from vtk import vtkPolyDataNormals

from brainspace.mesh.mesh_io import read_surface
from brainspace.mesh.mesh_operations import combine_surfaces
from brainspace.utils.parcellation import reduce_by_labels
from brainspace.vtk_interface import wrap_vtk, serial_connect

template_path = "Z:/hschoi/backup/hschoi/template/MMP"
template_L = "S900.L.midthickness_MSMAll.10k_fs_LR.surf.gii" # S900.L.midthickness_MSMAll.10k_fs_LR.surf.gii # L.very_inflated_MSMAll.10k_fs_LR.surf.gii
template_R = "S900.R.midthickness_MSMAll.10k_fs_LR.surf.gii" # S900.R.midthickness_MSMAll.10k_fs_LR.surf.gii # R.very_inflated_MSMAll.10k_fs_LR.surf.gii

surfs = [None] * 2

surfs[0] = read_surface(join(template_path,template_L)) 
nf = wrap_vtk(vtkPolyDataNormals, splitting=False, featureAngle=0.1)
surf_lh = serial_connect(surfs[0], nf)

surfs[1] = read_surface(join(template_path,template_R)) 
nf = wrap_vtk(vtkPolyDataNormals, splitting=False, featureAngle=0.1)
surf_rh = serial_connect(surfs[1], nf)

# Visualization

from brainspace.datasets import load_group_fc, load_parcellation, load_conte69
from brainspace.gradient import GradientMaps
from brainspace.plotting import plot_hemispheres
from brainspace.utils.parcellation import map_to_labels

atlas = np.load("Z:\\hschoi\\backup\\hschoi\\template\\MMP\\MMP.10k_fs_LR.npy")

pc_num = 2  
ref_PCs = ref_PC[:,pc_num]
X = ref_PCs 


labeling = atlas 

conn_matrix = X # X # ref_PCs 


mask = labeling != 0

grad = map_to_labels(conn_matrix, labeling, mask=mask, fill=np.nan)


plot_hemispheres(surf_lh, surf_rh, array_name=grad, size=(1300, 200),
                 color_bar=True, cmap='jet', zoom=1.85) #'jet' # 'viridis_r',   'Blues',  , 'seismic' # color_range = (-0.1,0.16)

