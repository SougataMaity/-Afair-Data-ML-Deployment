import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

data = sm.datasets.fair.load_pandas().data
data['affairs'] = data['affairs'].apply(lambda x: 1 if x>0 else 0)
y = data['affairs']
x = data.drop(['affairs'], axis=1)

lr_model = LogisticRegression()
lr_model.fit(x,y)



