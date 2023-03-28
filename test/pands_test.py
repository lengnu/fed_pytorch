import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(
    filepath_or_buffer='F:\\Federated Learning\\Fed_Agg\\dataset\mnist\split\mnist_split_iid_20\\report.csv'
    , header=1, index_col='client')
col_names = df.axes[1][:10]
for col_name in col_names:
    df[col_name] = (df[col_name] * df['Amount']).astype(int)

fig = df.iloc[:10, :10].plot(stacked=True, kind='barh')
fig.legend(bbox_to_anchor=(1, 0.7))
fig.set_xlabel('sample num')


plt.show()
