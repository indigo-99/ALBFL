import utils
import pandas as pd

merge=pd.read_csv('data/merge.csv')
ignore_cols = set(utils.QID + utils.INDEXES + utils.FAULTY + utils.COMBINEFL + utils.CODE)
for col in set(merge.columns) - ignore_cols:
    merge[col] = merge.groupby('qid')[col].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
merge.fillna(0, inplace=True)
merge.to_csv('data/merge_norm.csv', index=False)