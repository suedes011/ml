import math as m
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('punkt')

from google.colab import drive
drive.mount('/content/drive')

data = pd.read_csv("/content/spam_ham_dataset.csv")
data.head()

print(data.columns)
print(data.shape)

df = data
df.info()

"""### Text Preprocessing

"""

for i in df.iterrows():
    print("Class Label: {}\nMail: \n{}\n\n".format(i[1][1], i[1][2]))
    if i[0] == 2: break

df['text']=df['text'].str.lower()

df['text']=df['text'].apply(lambda X: word_tokenize(X))

stop = stopwords.words('english')
def remove_stopwords(text):
    result = []
    for token in text:
        if token not in stop:
            result.append(token)
            
    return result

df['text'] = df['text'].apply(remove_stopwords)

from nltk.tokenize import RegexpTokenizer

def remove_punct(text):
    
    tokenizer = RegexpTokenizer(r"\w+")
    lst=tokenizer.tokenize(' '.join(text))
    return lst

df['text'] = df['text'].apply(remove_punct)

from nltk.stem import PorterStemmer

def stemming(text):
    porter = PorterStemmer()
    
    result=[]
    for word in text:
        result.append(porter.stem(word))
    return result

df['text']=df['text'].apply(stemming)

df.text[0]

df['text1'] = df['text'].apply(' '.join)
df['text'] = df['text1'].apply(lambda x:' '.join(x.split(' ')[1:]))
df['text']

df['text'] = df['text'].str.replace('\d+', '')
df['text'][0]

"""### Logistic Regression (Sigmoid neuron)"""

X = list(df.text)
y = list(df.label_num)

count_vect = CountVectorizer()

count_vect.fit(X)

count = len( count_vect.vocabulary_.keys())
print('NO.of Tokens: ',(count))

dtv = count_vect.transform(X)
dtv = dtv.toarray()

dtv.shape[0]

x = [[]]
for _ in range(dtv.shape[0]):
  x[0].append(1)
#x
# len(x)

w = []
for i in range(5172):
    w.append(round(np.random.uniform(-1,1),1))
#w

for i in dtv:
      x.append(i.tolist())

prevW = []
epoch = 0
eta = 0.5
c = 0 
while (epoch != 10) :
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
        wtx = 1 / (1+m.exp(-1*wtx))
        if wtx >= 0.5:
            ycap.append(1)
        else:
            ycap.append(0)
        if [i] == ycap[i]:
             continue
        else:
          for j in range(len(x)):
            w[j] = w[j] + eta * (y[i] - wtx) * (wtx) * (1 - wtx) * xmat[j]

    print("Weights :",w)
    print("ycap    :",ycap)
    print("\n")

len(ycap)

"""## Accuracy percentage """

(sum(1 for x,y in zip(ycap,y) if x == y) / len(ycap))*100
