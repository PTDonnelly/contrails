import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('nedt.csv', sep='\t')
df = pd.DataFrame(data)
print(df.head(10))

# # Plotting
# plt.figure(figsize=(10, 6))
# plt.plot(df['NEDT'].index, df['NEDT'])
# plt.title('IASI Noise Equivalent Differential Temperature', fontsize=15)
# plt.xlabel('Spectral Channel', fontsize=15)
# plt.ylabel('NEDT (K)', fontsize=15)
# plt.xlim([0, 8461])
# plt.ylim([0.01, 11])
# plt.yscale('log')
# plt.tick_params(axis='both', labelsize=15)
# plt.grid(True)
# plt.savefig('nedt.png')

data = pd.read_csv('covariance_matrix.csv', sep='\t')
df = pd.DataFrame(data)
print(df.head(10))
