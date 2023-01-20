"""Required libraries"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit
import math

def data_frame(data):
    """Data frame for clustering"""
    cname = list(data['Country Name'])
    year = list(data['2015'])
    dframe = {"Countries":cname,"2015":year}
    dframe = pd.DataFrame(dframe)
    print("Sample data for clustering : ")
    display(dframe.head())
    return dframe

def clustering(dframe):
    """Taking 3 clusters"""
    km = KMeans(n_clusters=3)
    y_pred = km.fit_predict(dframe[['2015']])
    centeroids = km.cluster_centers_ # finding the centroid for the clustering
    dframe.insert(len(dframe.columns), 'Cluster', list(y_pred))
    print("Sample data after clustering")
    display(dframe.head())
    return dframe,centeroids

def scatter_plot(x,y,z):
    """plotting the clusters"""
    plt.figure(figsize=(25,10))
    plt.scatter(x['Countries'],x['2015'],color='blue',label='GDP')
    plt.scatter(y['Countries'],y['2015'],color='green',label='GDP')
    plt.scatter(z['Countries'],z['2015'],color='yellow',label='GDP')
    plt.xlabel('Countries',size=20)
    plt.ylabel('Year 2015',size=20)
    plt.title("Country VS Year Clustering",size=25)
    plt.legend()
    plt.show()
    plt.close()

def func(x,a,b):
    return (a*(x**2))+b

def curve_fitting(data):
    india = data.loc[15, :].values.flatten().tolist()[1:]
    l = [int(i) for i in list(data.columns[1:])]
    popt,pocv = curve_fit(func,l,india)
    x = np.arange(0.0,5.0,0.01)
    plt.plot(x,func(x,*popt))
    plt.title("Curve fitting")
    plt.show()
    plt.close()

"""Reading the data"""
data = pd.read_csv("gdp_per_capita.csv")
print("Sample data taken : ")
display(data.head())

dframe = data_frame(data)

dframe,centeroids = clustering(dframe)

"""clustering graph"""
dframe1 = dframe[dframe.Cluster==0]
dframe2 = dframe[dframe.Cluster==1]
dframe3 = dframe[dframe.Cluster==2]

scatter_plot(dframe1,dframe2,dframe3)

"""Cluster membership vs Cluster centers"""
print("Cluster centers : ")
print(centeroids)
print()
plt.plot(centeroids,marker=".",mec="r",mfc="r")
plt.xlabel("Cluster membership")
plt.ylabel("Cluster centers")
plt.title("Cluster membership vs Cluster centers")
plt.show()
plt.close()

"""Curve fitting"""
curve_fitting(data)
