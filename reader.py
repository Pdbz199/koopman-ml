import pandas as pd
import numpy as np

# get dataframe from CSV file
coin_data = pd.read_csv('../coindatav4.csv')
# print(coin_data.columns)
coin_data = coin_data.drop(columns=['Unnamed: 0', 'coin', 'returns'])

g = coin_data.groupby('datetime').cumcount()
X = np.array((coin_data.set_index(['datetime',g])
        .unstack(fill_value=0)
        .stack().groupby(level=0)
        .apply(lambda x: np.array(x.values.tolist()).reshape(len(x)))
        .tolist()))
print(X.shape)