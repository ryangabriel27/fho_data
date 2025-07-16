import pandas as pd

df = pd.read_csv('./dataset/medical_insurance.csv')


print(df.shape) ## (1338, 8)
print(df.info()) 

## Não há valores nulos no dataset

print(df.describe())