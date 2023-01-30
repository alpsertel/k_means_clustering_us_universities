# c. assignment-5 

#import necessary libraries
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# document location for the dataset "Universities.xls"
doc_loc =r"C:\Users\Lenovo\Desktop\ass5\Universities.xls"

# #of iterations of k-means alg
iter_max = 100

# k range to be tested
k_max = 30

# retrieve data for universities 
univ = pd.read_excel(doc_loc)

# clean data with empty entries
univ_2 = univ.dropna()

# upload collumn names for simplicity
univ_2.columns = univ_2.iloc[0]
univ_2 = univ_2[1:]

# select the categorical attributes
univ_2_categ = ["College Name", "State", "Public (1)/ Private (2)"]

# normalize non-categorical attributes
univ_3 = univ_2.copy()

for att in univ_3.columns:
    if att not in univ_2_categ:
        univ_3[att] = (univ_3[att]-univ_3[att].min())/(univ_3[att].max()-univ_3[att].min())

# drop categorical data
univ_4 = univ_3.drop(columns = univ_2_categ)

# calculate sse
univ_SSE = {}
for k in range(1, k_max):
    kmeans_fit = KMeans(n_clusters=k, max_iter=iter_max).fit(univ_4)
    # "inertia" is the equivelant of SSE, and kmeans_fit.inertia_ is used for the calculation of SSE in this assignment.
    univ_SSE[k] = kmeans_fit.inertia_ 
    
# plot SSE values for different k values
plt.figure()
plt.plot(list(univ_SSE.keys()), list(univ_SSE.values()))
plt.xlabel("# of clusters")
plt.ylabel("SSE")
plt.title("1.1")
plt.show()

# SSE reduction rate for each k
sse_red = {}
for l in range(1,len(univ_SSE)):
    sse_red[l] = (univ_SSE[l]-univ_SSE[l+1])/univ_SSE[l]
    
# k ia selected as the k where reduction raye is smaller than 0.10 (elbow method but in a systematic way)
k_opt = 4

# run k-means clustering
kmeans_fit_opt = KMeans(n_clusters=k_opt, max_iter=1000).fit(univ_4)

# get cluster labels
clusters = kmeans_fit_opt.labels_

univ_5 = univ_4.copy()

# add cluster labels to the dataset
univ_5['Cluster'] = clusters


# calculate summary statistics for each cluster
stat_analysis = np.zeros(((k_opt,4,len(univ_5.columns))))
for c in range(k_opt):
    cluster_data = univ_5[univ_5['Cluster'] == c]
    print("Cluster {}:".format(c))
    print("Mean:")
    print(cluster_data.mean())
    stat_analysis[c,0] = cluster_data.mean()
    print("Max.:")
    print(cluster_data.max())
    stat_analysis[c,1] = cluster_data.max()
    print("Min.:")
    print(cluster_data.min())
    stat_analysis[c,2] = cluster_data.min()
    print("Var.:")
    print(cluster_data.var())
    stat_analysis[c,3] = cluster_data.var()
    print("\n")
  
# record statistics for each cluster
cluster_0 = pd.DataFrame(stat_analysis[0])
cluster_1 = pd.DataFrame(stat_analysis[1])
cluster_2 = pd.DataFrame(stat_analysis[2])
cluster_3 = pd.DataFrame(stat_analysis[3])

cluster_0.columns = univ_5.columns
cluster_1.columns = univ_5.columns
cluster_2.columns = univ_5.columns
cluster_3.columns = univ_5.columns

# group clustered data for plots and analysis
univ_6 = univ_5.copy()
univ_6["College Name"] = univ_2["College Name"]
univ_6["State"] = univ_2["State"]
univ_6["Public (1)/ Private (2)"] = univ_2["Public (1)/ Private (2)"]

group_state = univ_6.groupby(['Cluster', 'State']).size().reset_index(name='counts')
group_public_priv = univ_6.groupby(['Cluster', 'Public (1)/ Private (2)']).size().reset_index(name='counts')

groups_combined = univ_6.groupby(['Cluster', 'State', 'Public (1)/ Private (2)']).size().reset_index(name='counts')

sns.set(style="ticks")
sns_plot = sns.catplot(x="State", y="counts", hue="Public (1)/ Private (2)", col="Cluster", data=groups_combined, kind="bar", height=4, aspect=4)

sns_plot = sns.catplot(x="State", y="counts", hue="Public (1)/ Private (2)", row="Cluster", data=groups_combined, kind="bar", height=4, aspect=4)

# plot the private/public counts
sns_plot2 = sns.catplot(x="Public (1)/ Private (2)", y="counts", col="Cluster", data=groups_combined, kind="bar", height=4, aspect=.7);

# plot the state counts
sns_plot3 = sns.catplot(x="State", y="counts", col="Cluster", data=groups_combined, kind="bar", height=4, aspect=4);
sns_plot3 = sns.catplot(x="State", y="counts", row="Cluster", data=groups_combined, kind="bar", height=4, aspect=4)

# euclidean distance function for distance calculations
def euclidean_distance(a, b):
    dist = 0
    for i in range(len(b)):
        if math.isnan(a[i]) or math.isnan(b[i]):
            continue
        dist += (a[i] - b[i])**2
    dist = (dist)**(1/2)
    return dist

# original data but cleaner format
univ_c = univ.copy()
univ_c.columns = univ_c.iloc[0]
univ_c = univ_c[2:]
for att in univ_c.columns:
    if att not in univ_2_categ:
        univ_c[att] = (univ_c[att]-univ_c[att].min())/(univ_c[att].max()-univ_c[att].min())

# get tufts data to update
tufts_data = univ_c.iloc[475]
tufts_data = tufts_data[3:].tolist()

# calculate distances to clusters
cluster_distances = []
for c in range(k_opt):
    cluster_data = univ_5[univ_5['Cluster'] == c]
    cluster_mean = cluster_data.mean()
    d = euclidean_distance(cluster_mean, tufts_data)
    cluster_distances.append(d)

# get the closest cluster
closest_cluster = cluster_distances.index(min(cluster_distances))
print("The closest Cluster to Tufts University is:")
print(closest_cluster)

closest_cluster_data = univ_5[univ_5['Cluster'] == closest_cluster]
closest_cluster_mean = closest_cluster_data.mean()

tufts_data_upd = tufts_data.copy()
tufts_data_upd[0] = closest_cluster_mean[0]
tufts_data_upd[1] = closest_cluster_mean[1]
tufts_data_upd[2] = closest_cluster_mean[2]
tufts_data_upd[9] = closest_cluster_mean[9]

univ_c_renorm = univ_c.copy()
tufts_data_nonnorm = tufts_data_upd.copy()

univ_c_renorm.iloc[475]['Math SAT'] = tufts_data_upd[0]
univ_c_renorm.iloc[475]['Verbal SAT'] = tufts_data_upd[1]
univ_c_renorm.iloc[475]['ACT'] = tufts_data_upd[2]
univ_c_renorm.iloc[475]['# PT undergrad'] = tufts_data_upd[9]

# un-normalization
for att in univ_3.columns:
    if att not in univ_2_categ:
        univ_c_renorm.iloc[475][att] = univ_c_renorm.iloc[475][att]*(univ_2[att].max()-univ_2[att].min())+univ_2[att].min()


# 3 closest data points from the closest cluster
distances_to_pt = []
for jj in range(len(closest_cluster_data)):
    dist_temp = euclidean_distance(closest_cluster_data.iloc[jj], tufts_data)
    distances_to_pt.append(dist_temp)
    
closest_cluster_data['distance_to_pt'] = distances_to_pt
closest_cluster_data = closest_cluster_data.sort_values(by=['distance_to_pt'])
closest_data_points = closest_cluster_data[:3]

# compute the mean of the 3 closest data points
closest_data_points_mean = closest_data_points.mean()

# insert the missing values
tufts_data_upd_3pt = tufts_data.copy()
tufts_data_upd_3pt[0] = closest_data_points_mean[0]
tufts_data_upd_3pt[1] = closest_data_points_mean[1]
tufts_data_upd_3pt[2] = closest_data_points_mean[2]
tufts_data_upd_3pt[9] = closest_data_points_mean[9]

univ_c_renorm.iloc[475]['Math SAT'] = tufts_data_upd_3pt[0]
univ_c_renorm.iloc[475]['Verbal SAT'] = tufts_data_upd_3pt[1]
univ_c_renorm.iloc[475]['ACT'] = tufts_data_upd_3pt[2]
univ_c_renorm.iloc[475]['# PT undergrad'] = tufts_data_upd_3pt[9]

# un-normalization
for att in univ_3.columns:
    if att not in univ_2_categ:
        univ_c_renorm.iloc[475][att] = univ_c_renorm.iloc[475][att]*(univ_2[att].max()-univ_2[att].min())+univ_2[att].min()
