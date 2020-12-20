import pandas as pd

# get dataframe from CSV file
coin_data = pd.read_csv('../ExportedCoinData.csv')

print(coin_data.loc[0]) # prints first entry of dataframe