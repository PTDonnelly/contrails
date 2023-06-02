import glob
import numpy as np

# Set the directory path and input file pattern
datadir = '/home/amaattanen/IASI/IASI_BeCOOL/2022/'
infiles = '/data/amaattanen/IASI/IASI_BeCOOL/METOPC/2021/nuit/*.filt'

# Search for matching files
filelist = glob.glob(infiles)
nfiles = len(filelist) - 1

# Define longitude and latitude arrays
longit = np.arange(176) + 55.0
latit = np.arange(16) - 16.0
latit[latit > 5.0] -= 16.0

nl1 = len(longit)
nl2 = len(latit)

# Initialize data arrays
data1 = np.zeros((nl1, nl2))
data = np.zeros((nl1, nl2))
datac = np.zeros((nl1, nl2))
datacloud1 = np.zeros((nl1, nl2))
pres = np.zeros_like(data)
temp = np.zeros_like(pres)
frac = np.zeros_like(pres)

# Initialize iasidata array for storing data points
iasidata = np.zeros((10000000, 8))

lasku = 0
indeksi = 0
nbr = 0
nbrcloud = 0

# Loop through each file
for i in range(nfiles):
    filename = filelist[i]
    
    # Read the columns from the file
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Convert longitudes to positive east longitudes
    longi = np.array([float(line.split()[1]) for line in lines])
    longi[longi < -130.0] += 360.0
    
    pind = len(lati) - 1
    
    # Loop through each data point
    for l in range(pind+1):
        # Check conditions for data point inclusion
        if (-10.0 <= lati[pind] <= 5.0) and (55.0 <= longi[pind] <= 230.0):
            nbr += 1
            
            # Check conditions for cloud inclusion
            if ctype1[l] != 0 and np.isfinite(ctemp1[l]):
                nbrcloud += 1
                
                # Store data point in iasidata array
                iasidata[nbr-1, 0] = lati[pind]
                iasidata[nbr-1, 1] = longi[pind]
                iasidata[nbr-1, 2] = orbit[pind]
                iasidata[nbr-1, 3] = datetime[pind]
                iasidata[nbr-1, 4] = cfrac1[pind]
                iasidata[nbr-1, 5] = ctemp1[pind]
                iasidata[nbr-1, 6] = cpres1[pind]
                iasidata[nbr-1, 7] = ctype1[pind]
    
    lasku += 1

# Save the data to a file
np.savetxt('iasidata_clouds_metopc_2021.txt', iasidata[:nbr,:])

print('found', lasku, 'orbits')
