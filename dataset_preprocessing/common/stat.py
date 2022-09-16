import os
import sys
import pandas as pd


if __name__ == "__main__":
    metadata_csv_path = sys.argv[1]
    metadata_df = pd.read_csv(metadata_csv_path)
    print(metadata_df.groupby(['split', 'lang']).sum().transform(lambda x: x/3600))
    print("-"*50)
    print(metadata_df.groupby(['split', 'lang'])["speaker"].describe()[["count", "unique"]])
