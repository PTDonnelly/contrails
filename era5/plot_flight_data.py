# Let's first read the uploaded file to understand its structure and content better.
import pandas as pd

# Load the dataset
file_path = 'G:\\My Drive\\Research\\Postdoc_2_CNRS_LATMOS\\data\\FlightRadar24\\2019-01-01.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
df.head()

