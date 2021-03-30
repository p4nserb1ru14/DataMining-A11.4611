# import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import dataset
dataset = pd.read_csv(
        'dataset.csv',
        delimiter=';', 
        header='infer', 
        index_col=False
        )
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# nilai kosong

from sklearn.impute import SimpleImputer

# ganti NaN dengan mean kolom

imputer = SimpleImputer(
        missing_values=np.nan, 
        strategy='mean'
        )
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
# kodekan data kategori

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(),[0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# kodekan nama provinsi (kolom ke-0)

# kode hanya sebatas penanda
encoder_X = ColumnTransformer(
        [('province_encoder', OneHotEncoder(), [0])], 
        remainder='passthrough'
        )
X = encoder_X.fit_transform(X).astype(float)
# mengembalikan ke dalam tipe 'float64'

# dataset
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

