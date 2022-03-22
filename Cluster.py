import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import Birch
import ast
import math

def tim(res):   #convert time list in 'Time' column into better format
    ans=[]
    res = ast.literal_eval(res)
    for i in range(len(res)):
        x=res[i]
        i=0
        hour=""
        minute=""
        while(x[i]!=":"):
            hour+=x[i]
            i+=1
        hour=int(hour)
        i+=1
        while(i<len(x)):
            minute+=x[i]
            i+=1
        minute=int(minute)
        ans.append(hour*60+minute)   
    return ans

df1 = pd.read_csv(r'C:\Users\Ashish\The Last Credit\The Last Credit\staypoint.csv')

stored_data=[]   # storing the data in a well-format so that it could be used to retrieve information (note-it is different from the input for clustering)
for i in range(1000):
    for j in range(len(tim(df1['Time'][i]))):
        stored_data.append([df1['Vehicle No'][i],(df1['Latitude'][i],df1['Longitude'][i]),ast.literal_eval(df1['Time'][i])[j]])

# setting up distance coordinates
"""we know that distance between gps coordinates does not directly correspond to distance in kms
Therefore to scale that we should know how the distance is calculated.
In general the distance between two gps coordinates is defined as,
d = acos( sin φ1 ⋅ sin φ2 + cos φ1 ⋅ cos φ2 ⋅ cos Δλ ) ⋅ R

where,
φ1 = lat1 * Math.PI/180
φ2 = lat2 * Math.PI/180
Δλ = (lon2-lon1) * Math.PI/180
R = 6378.8*1000 metres

But in case of shorter distances, this formula reduces to,
d = R ⋅ √(x² + y²)

where,
x = (λ2-λ1) * Math.cos((φ1+φ2)/2)
x = (λ2-λ1) * 0.877 (approx.)
x = (λ2*0.877 - λ1* 0.877)
y = (φ2-φ1) 

x(i) = R*λ(i)*0.877 = R*lon(i)*math.pi/180
y(i) = R*lat(i)*math.pi/180
d(i)=√({x(i)}²+{y(i)}²)
"""

# scaling of time wrt distance
"""Current scale is: 1 minute on time-axis is equivalent to 1 km on distance axis.
The current equation of sphere is (delta_x)^2 + (delta_y)^2 + (delta_t)^2 = (radius)^2
As the current radius of uncertainity wrt distance is 30 metres and current time uncertainity from mean time is 30 minutes.
Taking radius as 30 implies that maximum time variation from mean point at any moment will be 30 minutes according to 
the scale we defined in the first line. We will get the maximum time variation at any instance when variation wrt to 
distance is zero at that moment. Taking the radius as 30 also implies that maximum uncertainity wrt distance at any moment
is 30 metres and this uncertainity will only achieve if the time variation from the mean time is zero at that instance.

According to this scale (stated in first line) 'current radius of uncertainity wrt distance' and 'current time uncertainity
from mean time' will get an equal weightage.
"""


"""Changing the scale will alter the weightage of both the uncertainities.
"""
# The Scaling 
current_radius_of_uncertainity = 30 # in metres
current_time_uncertainity = 30 # in minutes
scale = current_radius_of_uncertainity /float(current_time_uncertainity)   # this will scale the time axis wrt distance axes
radius = current_radius_of_uncertainity

data=[]    #creating  an array of the data to be clustered in the appropriate scaled format (note-it will be used as input for finding clusters)
for i in range(1000):
    for j in range(len(tim(df1['Time'][i]))):
        data.append([1000*6378.8*df1['Latitude'][i]*(math.pi/180),1000*6378.8*df1['Longitude'][i]*0.877*(math.pi/180),scale*(tim(df1['Time'][i])[j])])

# clustering process starts
brc = Birch(n_clusters=None,threshold=radius,compute_labels=True)   # setting up initial conditions for birch clustering
y=brc.fit(data)  # input data has  been fit for birch clustering
labels=y.labels_   # an array in which index is the staypoint no. and index value is its cluster no.

#storing the array(output) in a dictionary
k={}
for i,label in enumerate(labels):
    k[label]=k.get(label,[])
    k[label].append(i)

# writing a function to retrieve information from stored data which has been created in line no.30
def information(value):
    arr=[[],[],[]]
    for i in range(len(value)):
        I=value[i]
        arr[0].append(stored_data[I][0])
        arr[1].append(stored_data[I][1])
        arr[2].append(stored_data[I][2])
    return arr

# Changing the output format to same as previous one
centers=y.subcluster_centers_
arr=[]
for i in range(len(centers)):
    a=centers[i][0]/(6378.8*1000*(math.pi/180))
    b=centers[i][1]/(6378.8*1000*(math.pi/180)*0.877)  
    c=str(int((centers[i][2]/scale)//60)).zfill(2)+":"+str(int((centers[i][2]/scale)%60)).zfill(2)
    arr.append([a,b,c])

# saving the modified data into final array
final_arr=[]
for key,value in k.items():
    arr[key].append(information(value))
    final_arr.append(arr[key])

df2=pd.DataFrame(columns=['Mean Latitude','Mean Longitude','Mean Time','Total Staypoints','Availability','Vehicles','Chances','Vehicle No.s','Vehicle Coordinates','Vehicle Stay Times','Vehicle Coordinates Mean','Vehicle Stay Times Mean'])

# writing down into the output into a new dataframe
for i in range(len(final_arr)):
    df2.loc[i]=None
    df2['Mean Latitude'][i]=final_arr[i][0]
    df2['Mean Longitude'][i]=final_arr[i][1]
    df2['Mean Time'][i]=final_arr[i][2]
    df2['Total Staypoints'][i]=len(final_arr[i][3][0])
    
    # for Availability and vehicles
    store=dict()
    for element in final_arr[i][3][0]:
        store[element]=store.get(element,0)+1
        
    chances=[]
    vehicles=[]
    for key,value in store.items():
        vehicles.append(key)
        chances.append(value)
        
    df2['Chances'][i]=chances
    df2['Availability'][i]=len(store)
    df2['Vehicles'][i]=vehicles
    df2['Vehicle No.s'][i]=final_arr[i][3][0]
    df2['Vehicle Coordinates'][i]=final_arr[i][3][1]
    df2['Vehicle Stay Times'][i]=final_arr[i][3][2]
    
    # for vehicle coordinates mean
    sum_lat=0
    sum_long=0
    for j in range(len(final_arr[i][3][1])):
        sum_lat+=float(final_arr[i][3][1][j][0])
        sum_long+=float(final_arr[i][3][1][j][1])
    mean_lat=sum_lat/len(final_arr[i][3][1])
    mean_long=sum_long/len(final_arr[i][3][1])
    
    df2['Vehicle Coordinates Mean'][i]=(mean_lat,mean_long)
    # for vehicle stay times mean
    minutes=0
    res = final_arr[i][3][2]
    for l in range(len(res)):
        x=res[l]
        j=0
        hour=""
        minute=""
        while(x[j]!=":"):
            hour+=x[j]
            j+=1
        hour=int(hour)
        j+=1
        while(j<len(x)):
            minute+=x[j]
            j+=1
        minute=int(minute)
        minutes+=hour*60+minute
    minutes=minutes/len(res)
    hour=int(minutes//60)
    minute=int(minutes%60)
    
    df2['Vehicle Stay Times Mean'][i]=str(hour)+":"+str(minute)

df2=df2.sort_values(by=['Availability'], ascending=False)           #sorting in accordance with no. of unique taxis in a cluster

df2.to_csv(r'C:\Users\Ashish\The Last Credit\The Last Credit\clust.csv')