from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

print("Import Dataset : \n", x)
print("\n", y)

# Menghilangkan Missing Value (nan)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

print("\nMenghilangkan Missing Value (nan) : \n", x)

# Encoding data kategori (Atribut)
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

print("\nEncoding Data Kategori (Atribut) : \n", x)

# Encoding data kategori (Class / Label)
le = LabelEncoder()
y = le.fit_transform(y)

print("\nEncoding Data Kategori (Class / Label) : \n", y)

# Membagi dataset ke dalam training set dan test set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)
print("\nData train x : \n", x_train)
print("\nData test x : \n", x_test)
print("\nData train y : \n", y_train)
print("\nData test y : \n", y_test)

# Feature Scaling
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])

print("\nFeature Scaling train x : \n", x_train)
print("\nFeature Scaling test x : \n", x_test)
