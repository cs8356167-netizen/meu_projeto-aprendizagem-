# Previsão

import pandas as pd
import pickle as p1
from sklearn import linear_model
#import numpy as np

!pip3 install -U ucimlrepo
from ucimlrepo import fetch_ucirepo

# fetch dataset
data = fetch_ucirepo(id=186)
data_X=data.data.features[1:]
data_Y=data.data.targets[1:]

n=5351
print(data_X.iloc[n].values)
Modelogravado = p1.load(open('wineQpredictor', 'rb'))


#data_x=input("introduza valores do wine\n")
#data_x=list(map(float, input("Números: ").split()))

data_x=list(map(float, input("introduza valores do wine\n").split()))
#print(data_x)

col=list(data_X.columns)
#print(col)

data_x=pd.DataFrame([data_x],columns=col)

print(data_x.iloc[0])

y_pred=Modelogravado.predict(data_x)

print()

print("wine quality",n,"    ",int(y_pred[0,0]))
print()

print("real wine quality",n,"    ",data_Y.iloc[n].values[0])


print()