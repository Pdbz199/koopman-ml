import pandas as pd

# get dataframe from CSV file
coin_data = pd.read_csv('../joemug.csv')

print(coin_data.groupby('datetime').size())