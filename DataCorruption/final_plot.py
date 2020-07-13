import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

df = pd.read_csv('../Amit/DataCorruption/airbnb_clean_error_score.csv')
df.columns
names = df.top_k_columns.tolist()
value = df.Error_AUC_score.tolist()


df=pd.DataFrame()
# df['value'] = [4,3,5]
# df['name'] = [3,2,1]
df['value'] = names
df['name'] = value
df = df.sort_values('value', ascending = False).reset_index(drop=True)


sns.set(rc={'figure.figsize':(100,50)})
ax = sns.pointplot(x=df.index, y=df.value)
ax.set_xlabel("Error_AUC_score")
ax.set_ylabel("top_k_columns")



# Use name column to label x ticks
_ = ax.set_xticklabels(df.name.astype(str).values)
_=ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
#_=ax.set(xlim=(0.660000, 0.657177))

## TODO need to plot in range
