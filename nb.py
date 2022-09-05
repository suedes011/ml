import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

df = pd.read_csv("/content/drive/MyDrive/Datasets/Buy_Computer.csv")
df = pd.DataFrame(df)
df.head()

df.drop('id', inplace = True, axis = 1)
actual = df['Buy_Computer']
df.drop('Buy_Computer', inplace = True, axis = 1)
df.columns

countOfClasses = df["Buy_Computer"].value_counts().to_dict()
countOfClasses

"""## Conditional Probability Table ( CPT )"""

cpt = {}
for i in df.columns:
  if i != "classes":
    cpt[i] = {}
    for j in actual.unique():
      for k in df[i].unique():
         count = 0
         for p in range(len(df)):
           if df[i][p] == k and actual[p] == j:
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

prediction = []
for i in range(len(df)):
  x = df.loc[[i]].to_dict()
  dum = 0
  classification = ""
  for j in actual.unique():
    numerator = 1
    for k in x.keys():
      numerator *= cpt[k][(j,x[k][i])]
    numerator *= countOfClasses[j]/len(df)
    if numerator/denominator > dum:
      dum = numerator/denominator
      classification = j
  prediction.append(classification)

"""# Model Evaluation"""

confusion_matrix = metrics.confusion_matrix(actual, prediction)
cMatrix = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
cMatrix.plot()
plt.show()

tp=0
tn=0
fp=0
fn=0
for i in range(len(df)):
  y=df.iloc[i,-1]
  ypred=prediction[i]
  if y==ypred:
    if y=='positive':
      tp+=1
    else:
      tn+=1
  else:
    if ypred=='positive':
      fp+=1
    else:
      fn+=1
