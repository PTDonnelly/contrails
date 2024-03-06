import gzip
import shutil
import pandas as pd
import os

# Define the directory containing your CSV files
directory_path = 'G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\Eurocontrol'

for filename in os.listdir(directory_path):
    if filename.endswith(".gz"):
        print(filename)
        with gzip.open(filename, 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        df = pd.read_csv('example.csv')
        print(df.head())