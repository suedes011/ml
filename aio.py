##--------------------------NAIVE BAYES---------------------------##

import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/Datasets/Buy_Computer.csv")
df = pd.DataFrame(df)
df.head()

df.drop('id', inplace = True, axis = 1)
df.rename(columns = {'Buy_Computer':'classes'}, inplace = True )
df.columns

countOfClasses = df["classes"].value_counts().to_dict()
countOfClasses

"""## Conditional Probability Table ( CPT )"""

cpt = {}
for i in df.columns:
  if i != "classes":
    cpt[i] = {}
    for j in df["classes"].unique():
      for k in df[i].unique():
         count = 0
         for p in range(len(df)):
           if df[i][p] == k and df["classes"][p] == j:
             count += 1
         cpt[i][(j,k)] = count/countOfClasses[j]
cpt

"""## Testing Data"""

x = {'age':'senior', 'income':'medium', 'student':'no', 'credit_rating':'excellent'}

denominator = 0
for i in df["classes"].unique():
  temp = 1
  for j in x.keys():
    temp *= cpt[j][(i,x[j])]
  temp *= countOfClasses[i]/len(df)
  denominator += temp

dum = 0
classification = ""
for i in df["classes"].unique():
  numerator = 1
  for j in x.keys():
    numerator *= cpt[j][(i,x[j])]
  numerator *= countOfClasses[i]/len(df)
  if numerator/denominator > dum:
    dum = numerator/denominator
    classification = i
print("Given data points are classified as :",classification)

##--------------------------------SIGMOID IRIS-----------------------------------##

import math
import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

iris = datasets.load_iris()
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df['classification'] = iris.target

df = df[:100]
df

x = []
x.append([1]*100)

w = []
for i in range(len(df.columns)):
    w.append(round(np.random.uniform(-1,1),1))

for i in df.columns:
  if i!="classification":
      x.append(df[i].tolist())

y = df["classification"].tolist()

prevW = []
epoch = 0
eta = 0.5
while(w != prevW):
    print("Epoch   :",epoch+1)
    ycap = []
    epoch+=1
    prevW = w.copy()
    for i in range(len(x[0])):
        wt = np.array(w)
        xmat = []   
        for j in x:
            xmat.append(j[i])
        wtx = np.dot(wt,xmat)
        wtx = 1 / (1+math.exp(-1*wtx))
        if wtx >= 0.5:
            ycap.append(1)
        else:
            ycap.append(0)
        if y[i] == ycap[i]:
             continue
        else:
          for j in range(len(x)):
            w[j] = w[j] + eta * (y[i] - wtx) * (wtx) * (1 - wtx) * xmat[j]
    print("Weights :",w)
    print("ycap    :",ycap)
    print("\n")

##--------------------------------PERCEPTRON-----------------------------------------##

import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

iris = datasets.load_iris()
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df['classification'] = iris.target

df = df[:100]
df

x = []
x.append([1]*100)

w = []
for i in range(len(df.columns)):
    w.append(round(np.random.uniform(-1,1),1))

for i in df.columns:
  if i!="classification":
      x.append(df[i].tolist())

y = df["classification"].tolist()

prevW = []
epoch = 0
while(w != prevW):
    print("Epoch   :",epoch+1)
    ycap = []
    epoch+=1
    prevW = w.copy()
    for i in range(len(x[0])):
        wt = np.array(w)
        xmat = []   
        for j in x:
            xmat.append(j[i])
        wtx = np.dot(wt,xmat)
        if wtx >= 0:
            ycap.append(1)
        else:
            ycap.append(0)
        if y[i] == ycap[i]:
             continue
        elif y[i] == 0 and ycap[i] == 1:
            for j in range(len(x)):
                w[j] = w[j] - xmat[j]
        elif y[i] == 1 and ycap[i] == 0:
            for j in range(len(x)):
                w[j] = w[j] + xmat[j] 
    print("Weights :",w)
    print("ycap    :",ycap)
    print("\n")
