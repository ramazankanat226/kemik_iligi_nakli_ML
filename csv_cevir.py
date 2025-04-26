import pandas as pd

# ARFF dosyasını okuma
from scipy.io import arff

data, meta = arff.loadarff('bone-marrow.arff')

# Pandas DataFrame'e dönüştürme
df = pd.DataFrame(data)

# CSV olarak kaydetme
df.to_csv('dosya.csv', index=False)