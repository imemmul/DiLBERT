import pandas as pd

df = pd.read_csv('turkish_reddit_data_full.csv')

df.to_excel('turkish_reddit_data_full.xlsx', index=False)



