# -*- coding: utf-8 -*-
"""
ncm_format.py python script

usage:
python ncm_format.py formatted_fname 
where:
- formatted_fname is 6803236 bytes input file

Main program is only an example and has only stdout ouputs.

higher level function is:
read_ncm_compute_nedt : read NCM structure from formatted binary file, keep only 1 diagonal vector (there are 5) and compute nedt, reconstruct covariance matrix from this NCM structure read from formatted binary file (only keep covariance output, not separated diagonal and extradiagonal parts), compute nedt for diagonal elements; nedt results are the same (for diagonal elements) 
  
usefull user functions are :
ncm_format_write : write NCM structure in formatted binary file
ncm_format_read : read NCM structure from previously written formatted binary file
reconstruct_cov_from_struct_ncm : reconstruct covariance matrix from this NCM structure

first developper function is : 
load_struct_ncm : load NCM structure from npz files (eig_decompos.py output, reconstruct_NCM_1{B,C}_PN1.npz for example)


NCM structure is 6803236 bytes one defined in IA-TN-0000-3274-CNE 03.00 (remains big endian)
with one change : IRnmRadNoiseCovarMat1{b,c}  are splited into IRnmRadNoiseCovarMatDiagonalVectors1b{b,c} (5 diagonal vectors) / IRnmRadNoiseCovarMatDiagonalVectorsEigenVectors1b{b,c} (2 eigenvectors)

sotware prerequisites :
- python 2.7.5 and following modules :
 - datetime
 - math
 - matplotlib 1.2.0
 - numpy 1.7.1
  
"""
from __future__ import print_function
from __future__ import division

from builtins import str
from builtins import range
from past.utils import old_div
import matplotlib.pyplot as plt
import numpy as np
from sys import argv
import struct
import math

# for planck : begin
pi = np.pi
Cst_h = 6.6260755e-34    # Joules*Secondes
Cst_c = 2.99792458e+8    # m/s
Cst_k = 1.380658e-23     # Joules/Kelvin
Cst_sca1  = 2*Cst_h*Cst_c**2
Cst_sca2  = old_div(Cst_h*Cst_c,Cst_k)
# for planck : end

# empty 6803236 bytes structure : begin
STRUCT_NCM_EMPTY={
  'IDefIssueIcd':-1,
  'IDefRevisionIcd' : -2,
  'IDefCovID' : -3,
  'IDefCovDate_day' : -4,
  'IDefCovDate_ms' : -5,
  'IDefCovarMatEigenVal1b' : np.zeros((2,100),dtype=np.double),
  'IDefCovarMatEigenVal1c' : np.zeros((2,100),dtype=np.double),
  'IDefRnmSize1_1b' : 5,
  'IDefRnmSize2_1b' : 2,
  'IDefRnmSize1_1c' : 5,
  'IDefRnmSize2_1c' : 2,
  'IRnmRadNoiseCovarMatDiagonalVectors1b' : np.zeros((8500,50),dtype="float64"),
  'IRnmRadNoiseCovarMatEigenVectors1b' : np.zeros((8500,50),dtype="float64"),
  'IRnmRadNoiseCovarMatDiagonalVectors1c' : np.zeros((8500,50),dtype="float64"),
  'IRnmRadNoiseCovarMatEigenVectors1c' : np.zeros((8500,50),dtype="float64"),
  }
# empty 6803236 bytes structure : end

def ncm_format_write(fname_output, struct_ncm, verbose=True):
  """
  write struct_ncm in fname_output file
  """
  idfct="[ncm_format_write]"
  if verbose: print(idfct, "begin")
  if verbose: print(idfct, "fname_output=", fname_output)
  fout = open(fname_output, 'wb')
  if verbose: print(idfct, "struct_ncm['IDefIssueIcd']", struct_ncm['IDefIssueIcd'])
  fout.write(struct.pack('>i', struct_ncm['IDefIssueIcd']))
  if verbose: print(idfct, "struct_ncm['IDefRevisionIcd']", struct_ncm['IDefRevisionIcd'])
  fout.write(struct.pack('>i', struct_ncm['IDefRevisionIcd']))
  if verbose: print(idfct, "struct_ncm['IDefCovID']", struct_ncm['IDefCovID'])
  fout.write(struct.pack('>i', struct_ncm['IDefCovID']))
  if verbose: print(idfct, "struct_ncm['IDefCovDate_day']", struct_ncm['IDefCovDate_day'])
  fout.write(struct.pack('>i', struct_ncm['IDefCovDate_day']))
  if verbose: print(idfct, "struct_ncm['IDefCovDate_ms']", struct_ncm['IDefCovDate_ms'])
  fout.write(struct.pack('>i', struct_ncm['IDefCovDate_ms']))
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(struct_ncm['IDefCovarMatEigenVal1b'])), "IDefCovarMatEigenVal1b", struct_ncm['IDefCovarMatEigenVal1b'].shape, (np.nanmax(struct_ncm['IDefCovarMatEigenVal1b'])))
    print(idfct, my_str)
  for i in range(struct_ncm['IDefCovarMatEigenVal1b'].shape[0]):
    for j in range(struct_ncm['IDefCovarMatEigenVal1b'].shape[1]):
      fout.write(struct.pack('>d', struct_ncm['IDefCovarMatEigenVal1b'][i,j]))
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(struct_ncm['IDefCovarMatEigenVal1c'])), "IDefCovarMatEigenVal1c", struct_ncm['IDefCovarMatEigenVal1c'].shape, (np.nanmax(struct_ncm['IDefCovarMatEigenVal1c'])))
    print(idfct, my_str)
  for i in range(struct_ncm['IDefCovarMatEigenVal1c'].shape[0]):
    for j in range(struct_ncm['IDefCovarMatEigenVal1c'].shape[1]):
      fout.write(struct.pack('>d', struct_ncm['IDefCovarMatEigenVal1c'][i,j]))
  if verbose: print(idfct, "struct_ncm['IDefRnmSize1_1b']", struct_ncm['IDefRnmSize1_1b'])
  fout.write(struct.pack('>i', struct_ncm['IDefRnmSize1_1b']))
  if verbose: print(idfct, "struct_ncm['IDefRnmSize2_1b']", struct_ncm['IDefRnmSize2_1b'])
  fout.write(struct.pack('>i', struct_ncm['IDefRnmSize2_1b']))
  if verbose: print(idfct, "struct_ncm['IDefRnmSize1_1c']", struct_ncm['IDefRnmSize1_1c'])
  fout.write(struct.pack('>i', struct_ncm['IDefRnmSize1_1c']))
  if verbose: print(idfct, "struct_ncm['IDefRnmSize2_1c']", struct_ncm['IDefRnmSize2_1c'])
  fout.write(struct.pack('>i', struct_ncm['IDefRnmSize2_1c']))
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1b'])), "IRnmRadNoiseCovarMatDiagonalVectors1b", struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1b'].shape, (np.nanmax(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1b'])))
    print(idfct, my_str)
  for i in range(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1b'].shape[0]):
    for j in range(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1b'].shape[1]):
      fout.write(struct.pack('>f', struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1b'][i,j]))
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1b'])), "IRnmRadNoiseCovarMatEigenVectors1b", struct_ncm['IRnmRadNoiseCovarMatEigenVectors1b'].shape, (np.nanmax(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1b'])))
    print(idfct, my_str)
  for i in range(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1b'].shape[0]):
    for j in range(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1b'].shape[1]):
      fout.write(struct.pack('>f', struct_ncm['IRnmRadNoiseCovarMatEigenVectors1b'][i,j]))
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1c'])), "IRnmRadNoiseCovarMatDiagonalVectors1c", struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1c'].shape, (np.nanmax(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1c'])))
    print(idfct, my_str)
  for i in range(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1c'].shape[0]):
    for j in range(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1c'].shape[1]):
      fout.write(struct.pack('>f', struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1c'][i,j]))
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1c'])), "IRnmRadNoiseCovarMatEigenVectors1c", struct_ncm['IRnmRadNoiseCovarMatEigenVectors1c'].shape, (np.nanmax(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1c'])))
    print(idfct, my_str)
  for i in range(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1c'].shape[0]):
    for j in range(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1c'].shape[1]):
      fout.write(struct.pack('>f', struct_ncm['IRnmRadNoiseCovarMatEigenVectors1c'][i,j]))
  fout.close()
  if verbose: print(idfct, "end")
  

def ncm_format_read(fname_input, verbose=True):
  """
  read struct_ncm frim fname_input file
  """
  idfct="[ncm_format_read]"
  if verbose: print(idfct, "begin")
  if verbose: print(idfct, "fname_input=", fname_input)
  struct_ncm = STRUCT_NCM_EMPTY
  fin = open(fname_input, 'rb')
  fileContent = fin.read()
  fin.close()
  offset=0
  mysize=4
  if verbose: print(idfct, "offset", offset)
  struct_ncm['IDefIssueIcd'] = struct.unpack('>i',fileContent[offset:offset+mysize])[0]
  if verbose: print(idfct, "struct_ncm['IDefIssueIcd']", struct_ncm['IDefIssueIcd'])
  offset=offset+mysize
  if verbose: print(idfct, "offset", offset)
  struct_ncm['IDefRevisionIcd'] = struct.unpack('>i',fileContent[offset:offset+mysize])[0]
  if verbose: print(idfct, "struct_ncm['IDefRevisionIcd']", struct_ncm['IDefRevisionIcd'])
  offset=offset+mysize
  if verbose: print(idfct, "offset", offset)
  struct_ncm['IDefCovID'] = struct.unpack('>i',fileContent[offset:offset+mysize])[0]
  if verbose: print(idfct, "struct_ncm['IDefCovID']", struct_ncm['IDefCovID'])
  offset=offset+mysize
  if verbose: print(idfct, "offset", offset)
  struct_ncm['IDefCovDate_day'] = struct.unpack('>i',fileContent[offset:offset+mysize])[0]
  if verbose: print(idfct, "struct_ncm['IDefCovDate_day']", struct_ncm['IDefCovDate_day'])
  offset=offset+mysize
  if verbose: print(idfct, "offset", offset)
  struct_ncm['IDefCovDate_ms'] = struct.unpack('>i',fileContent[offset:offset+mysize])[0]
  if verbose: print(idfct, "struct_ncm['IDefCovDate_ms']", struct_ncm['IDefCovDate_ms'])
  verbose2 = False
  for i in range(struct_ncm['IDefCovarMatEigenVal1b'].shape[0]):
    verbose2 = verbose
    for j in range(struct_ncm['IDefCovarMatEigenVal1b'].shape[1]):
      offset=offset+mysize
      mysize=8
      struct_ncm['IDefCovarMatEigenVal1b'][i,j] = struct.unpack('>d',fileContent[offset:offset+mysize])[0]
      if verbose2 : 
        print(idfct, "offset", offset)
        my_str= 'IDefCovarMatEigenVal1b[' + str(i) +  ',' + str(j) +  ']=' + str(  struct_ncm['IDefCovarMatEigenVal1b'][i,j])
        print(idfct, my_str)
      if j>5: verbose2 = False 
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(struct_ncm['IDefCovarMatEigenVal1b'])), "IDefCovarMatEigenVal1b", struct_ncm['IDefCovarMatEigenVal1b'].shape, (np.nanmax(struct_ncm['IDefCovarMatEigenVal1b'])))
    print(idfct, my_str)
  for i in range(struct_ncm['IDefCovarMatEigenVal1c'].shape[0]):
    verbose2 = verbose
    for j in range(struct_ncm['IDefCovarMatEigenVal1c'].shape[1]):
      offset=offset+mysize
      mysize=8
      struct_ncm['IDefCovarMatEigenVal1c'][i,j] = struct.unpack('>d',fileContent[offset:offset+mysize])[0]
      if verbose2 : 
        print(idfct, "offset", offset)
        my_str= 'IDefCovarMatEigenVal1c[' + str(i) +  ',' + str(j) +  ']=' + str(  struct_ncm['IDefCovarMatEigenVal1c'][i,j])
        print(idfct, my_str)
      if j>5: verbose2 = False 
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(struct_ncm['IDefCovarMatEigenVal1c'])), "IDefCovarMatEigenVal1c", struct_ncm['IDefCovarMatEigenVal1c'].shape, (np.nanmax(struct_ncm['IDefCovarMatEigenVal1c'])))
    print(idfct, my_str)
  offset=offset+mysize
  if verbose: print(idfct, "offset", offset)
  mysize=4
  struct_ncm['IDefRnmSize1_1b'] = struct.unpack('>i',fileContent[offset:offset+mysize])[0]
  if verbose: print(idfct, "struct_ncm['IDefRnmSize1_1b']", struct_ncm['IDefRnmSize1_1b'])
  offset=offset+mysize
  if verbose: print(idfct, "offset", offset)
  mysize=4
  struct_ncm['IDefRnmSize2_1b'] = struct.unpack('>i',fileContent[offset:offset+mysize])[0]
  if verbose: print(idfct, "struct_ncm['IDefRnmSize2_1b']", struct_ncm['IDefRnmSize2_1b'])
  offset=offset+mysize
  if verbose: print(idfct, "offset", offset)
  mysize=4
  struct_ncm['IDefRnmSize1_1c'] = struct.unpack('>i',fileContent[offset:offset+mysize])[0]
  if verbose: print(idfct, "struct_ncm['IDefRnmSize1_1c']", struct_ncm['IDefRnmSize1_1c'])
  offset=offset+mysize
  if verbose: print(idfct, "offset", offset)
  mysize=4
  struct_ncm['IDefRnmSize2_1c'] = struct.unpack('>i',fileContent[offset:offset+mysize])[0]
  if verbose: print(idfct, "struct_ncm['IDefRnmSize2_1c']", struct_ncm['IDefRnmSize2_1c'])
  verbose2 = False
  for i in range(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1b'].shape[0]):
    for j in range(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1b'].shape[1]):
      if j < 5 : verbose2 = verbose and i==0
      offset=offset+mysize
      mysize=4
      struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1b'][i,j] = struct.unpack('>f',fileContent[offset:offset+mysize])[0]
      if verbose2 : 
        print(idfct, "offset", offset)
        my_str= 'IRnmRadNoiseCovarMatDiagonalVectors1b[' + str(i) +  ',' + str(j) +  ']=' + str(  struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1b'][i,j])
        print(idfct, my_str)
      verbose2 = False
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1b'])), "IRnmRadNoiseCovarMatDiagonalVectors1b", struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1b'].shape, (np.nanmax(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1b'])))
    print(idfct, my_str) 
  verbose2 = False
  for i in range(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1b'].shape[0]):
    for j in range(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1b'].shape[1]):
      if j < 2 : verbose2 = verbose and i==0
      offset=offset+mysize
      mysize=4
      struct_ncm['IRnmRadNoiseCovarMatEigenVectors1b'][i,j] = struct.unpack('>f',fileContent[offset:offset+mysize])[0]
      if verbose2 : 
        print(idfct, "offset", offset)
        my_str= 'IRnmRadNoiseCovarMatEigenVectors1b[' + str(i) +  ',' + str(j) +  ']=' + str(  struct_ncm['IRnmRadNoiseCovarMatEigenVectors1b'][i,j])
        print(idfct, my_str)
      verbose2 = False
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1b'])), "IRnmRadNoiseCovarMatEigenVectors1b", struct_ncm['IRnmRadNoiseCovarMatEigenVectors1b'].shape, (np.nanmax(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1b'])))
    print(idfct, my_str)
  verbose2 = False
  for i in range(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1c'].shape[0]):
    for j in range(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1c'].shape[1]):
      if j < 5 : verbose2 = verbose and i==0
      offset=offset+mysize
      mysize=4
      struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1c'][i,j] = struct.unpack('>f',fileContent[offset:offset+mysize])[0]
      if verbose2 : 
        print(idfct, "offset", offset)
        my_str= 'IRnmRadNoiseCovarMatDiagonalVectors1c[' + str(i) +  ',' + str(j) +  ']=' + str(  struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1c'][i,j])
        print(idfct, my_str)
      verbose2 = False
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1c'])), "IRnmRadNoiseCovarMatDiagonalVectors1c", struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1c'].shape, (np.nanmax(struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors1c'])))
    print(idfct, my_str)
  verbose2 = False
  for i in range(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1c'].shape[0]):
    for j in range(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1c'].shape[1]):
      if j < 2 : verbose2 = verbose and i==0
      offset=offset+mysize
      mysize=4
      struct_ncm['IRnmRadNoiseCovarMatEigenVectors1c'][i,j] = struct.unpack('>f',fileContent[offset:offset+mysize])[0]
      if verbose2 : 
        print(idfct, "offset", offset)
        my_str= 'IRnmRadNoiseCovarMatEigenVectors1c[' + str(i) +  ',' + str(j) +  ']=' + str(  struct_ncm['IRnmRadNoiseCovarMatEigenVectors1c'][i,j])
        print(idfct, my_str)
      verbose2 = False
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1c'])), "IRnmRadNoiseCovarMatEigenVectors1c", struct_ncm['IRnmRadNoiseCovarMatEigenVectors1c'].shape, (np.nanmax(struct_ncm['IRnmRadNoiseCovarMatEigenVectors1c'])))
    print(idfct, my_str)
  return struct_ncm
  if verbose: print(idfct, "end")
  



def load_struct_ncm(npz_fname_1b, npz_fname_1c, verbose=True):
  """
  read  npz file (eig_decompos.py output, reconstruct_NCM_PN1.npz for example)
  return struct_ncm, cov_red_1b, cov_red_1c, w_cov_1b, w_cov_1c, v_cov_1b and v_cov_1c
  only cov_red_1b and  cov_red_1c are needed, others are for check

  note that struct_ncm elements are converted from (W/m2/st/cm-1)2 to (W/m2/st/m-1)2, and others outputs too
  """
  idfct="[load_struct_ncm]"
  if verbose: print(idfct, "begin")
  if verbose: print(idfct, "npz_fname_1b=", npz_fname_1b)
  if verbose: print(idfct, "npz_fname_1c=", npz_fname_1c)
  struct_ncm = STRUCT_NCM_EMPTY
  # erase index1 and index2 from npz_lut : unused and may be null #'index1':'arr_4', 'index2':'arr_5'
  npz_lut={'cov':'arr_0','cov_red':'arr_1','cov_norm':'arr_2','cov_red_norm':'arr_3','v_cov':'arr_6','w_cov':'arr_7'}
  levels=['1b','1c']
  npz_fnames= {'1b':npz_fname_1b,'1c':npz_fname_1c}
  cov_red= {'1b':None,'1c':None}
  w_cov= {'1b':None,'1c':None}
  v_cov= {'1b':None,'1c':None}
  # levels loop : begin
  for level in levels:
    if verbose: print(idfct, "level=", level)
    npz_fname=npz_fnames[level]
    if verbose: print(idfct, "npz_fname=", npz_fname)
    npzfile=np.load(npz_fname)
    #np.savez(fname_output,cov, cov_red, cov_norm, cov_red_norm, index1, index2, v_cov, w_cov) 
    # if verbose:
      #for k in sorted(npzfile.keys()):
        #print idfct, ":", k,  type(npzfile[k]), npzfile[k].shape      
  for k in sorted(npz_lut.keys()):
    #print idfct, ":", k, type(npzfile[npz_lut[k]]), npzfile[npz_lut[k]].shape
    try:
      my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(npzfile[npz_lut[k]])), k, npzfile[npz_lut[k]].shape, (np.nanmax(npzfile[npz_lut[k]])))
    except:
      my_str= "%s (%s)" % (k, npzfile[npz_lut[k]].shape)
    print(idfct, my_str)
    IDefCovarMatEigenVal1 = np.zeros((2,100),dtype=np.double)
    for icd in range(2):
      for iev in range(2):
        IDefCovarMatEigenVal1[icd,iev] = npzfile[npz_lut['v_cov']][8460-iev] * 0.0001 #from (W/m2/st/cm-1)2 to (W/m2/st/m-1)2
    struct_ncm['IDefCovarMatEigenVal%s' % level] = IDefCovarMatEigenVal1
    struct_ncm['IDefRnmSize1_%s' % level] = 5
    struct_ncm['IDefRnmSize2_%s' % level] = 2
    IRnmRadNoiseCovarMatDiagonalVectors1 = np.zeros((8500,50),dtype=np.float)
    for idiag in range(struct_ncm['IDefRnmSize1_%s' % level]): # 5 diagonals
        IRnmRadNoiseCovarMatDiagonalVectors1[0:8461-idiag,idiag] = np.diag(npzfile[npz_lut['cov']],idiag) * 0.0001 #from (W/m2/st/cm-1)2 to (W/m2/st/m-1)2
    struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors%s' % level] = IRnmRadNoiseCovarMatDiagonalVectors1
    IRnmRadNoiseCovarMatEigenVectors1 = np.zeros((8500,50),dtype=np.float)
    for iev in range(struct_ncm['IDefRnmSize2_%s' % level]): # 2 last eigenvectors
      IRnmRadNoiseCovarMatEigenVectors1[0:8461,iev] = npzfile[npz_lut['w_cov']][0:8461,8460-iev]  #no * 0.0001 #from (W/m2/st/cm-1)2 to (W/m2/st/m-1)2
    struct_ncm['IRnmRadNoiseCovarMatEigenVectors%s' % level] = IRnmRadNoiseCovarMatEigenVectors1
    cov_red[level]=npzfile[npz_lut['cov_red']]  * 0.0001 #from (W/m2/st/cm-1)2 to (W/m2/st/m-1)2
    w_cov[level]=npzfile[npz_lut['w_cov']]  #no * 0.0001 #from (W/m2/st/cm-1)2 to (W/m2/st/m-1)2
    v_cov[level]=npzfile[npz_lut['v_cov']]  * 0.0001 #from (W/m2/st/cm-1)2 to (W/m2/st/m-1)2
  # levels loop : end
  if verbose: print(idfct, "end")
  return struct_ncm, cov_red['1b'], cov_red['1c'], w_cov['1b'], w_cov['1c'], v_cov['1b'], v_cov['1c']

def reconstruct_cov_from_struct_ncm(struct_ncm, level, verbose=True):
  """
  reconstruct covariance matrix from struct_ncm
  return 3 matrices : cov (=covband+covextra), covband (diagonal), covextra (extradiagonal)
  """
  idfct="[reconstruct_cov_from_struct_ncm]"
  if verbose: print(idfct, "begin")
  if verbose: print(idfct, "level", level)
  assert(level in ['1b', '1c'])
  cov = np.zeros((8461,8461))
  covband = np.zeros((8461,8461))
  w_cov = np.zeros((8461,8461))
  v_cov = np.zeros(8461)
  IDefCovarMatEigenVal1 = struct_ncm['IDefCovarMatEigenVal%s' % level]
  IRnmRadNoiseCovarMatDiagonalVectors1 = struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors%s' % level]
  IRnmRadNoiseCovarMatEigenVectors1 = struct_ncm['IRnmRadNoiseCovarMatEigenVectors%s' % level]
  for iev in range(struct_ncm['IDefRnmSize2_%s' % level]): # 2 last eigenvectors
    v_cov[8460-iev]=IDefCovarMatEigenVal1[0,iev] #icd=0
    w_cov[0:8461,8460-iev]=IRnmRadNoiseCovarMatEigenVectors1[0:8461,iev]
  # covband
  nb_diag = struct_ncm['IDefRnmSize1_%s' % level]
  for idiag in range(nb_diag): # 5 diagonals bands
    x = IRnmRadNoiseCovarMatDiagonalVectors1[0:8461-idiag,idiag]
    covband = covband + np.diag(x,idiag)
    if idiag>0 : covband = covband + np.diag(x,-idiag)
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(covband)), "covband", covband.shape, (np.nanmax(covband)))
    print(idfct, my_str)
  # covextra
  covextra=np.dot(w_cov,np.dot(np.diag(v_cov),np.transpose(w_cov))) # matrix computation
  for idiag in range(nb_diag): # remove 5 diagonals reconstruected from eigenvectors
    x_up = np.diag(np.diag(covextra,idiag),idiag)
    x_down = np.diag(np.diag(covextra,-idiag),-idiag)
    covextra = covextra - x_up
    if idiag>0 :
	    covextra = covextra - x_down
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(covextra)), "covextra", covextra.shape, (np.nanmax(covextra)))
    print(idfct, my_str)
  cov = covextra + covband # add cov extra diagonal + diagonal
  if verbose : 
    my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(cov)), "cov", cov.shape, (np.nanmax(cov)))
    print(idfct, my_str)
  if verbose: print(idfct, "end")
  return cov, covband, covextra

def load_write_reconstruct_ncm(npz_fname_1b, npz_fname_1c, formatted_fname, IDefIssueIcd, IDefRevisionIcd, IDefCovID, IDefCovDate_day, IDefCovDate_ms, verbose=True):
  """
  read NCM structure from npz files (eig_decompos.py output, reconstruct_NCM_PN1_1b.npz and reconstruct_NCM_PN1_1c.npz for example)
  also keep cov_red output for checking, but not w_cov and v_cov
  
  copy input parameters into NCM structure (DefIssueIcd, IDefRevisionIcd, IDefCovDate_day, IDefCovDate_ms)

  reconstruct covariance matrix from this NCM structure directly read from npz file
  only keep covariance output, not separated diagonal and extradiagonal parts
  make difference with reduced covariance matrix read from npz file
  difference is expected to be low

  write NCM structure in formatted binary file
  read NCM structure from previously written formatted binary file

  reconstruct covariance matrix from this NCM structure read from formatted binary file
  only keep covariance output, not separated diagonal and extradiagonal parts
  make difference with reduced covariance matrix read from npz file
  difference is expected to be low
  """
  idfct="[load_write_reconstruct_ncm]"
  if verbose: print(idfct, "begin")
  if verbose: print(idfct, "npz_fname_1b", npz_fname_1b)
  if verbose: print(idfct, "npz_fname_1c", npz_fname_1c)
  if verbose: print(idfct, "formatted_fname", formatted_fname)
  if verbose: print(idfct, "IDefIssueIcd", IDefIssueIcd)
  if verbose: print(idfct, "IDefRevisionIcd", IDefRevisionIcd)
  if verbose: print(idfct, "IDefCovID", IDefCovID)
  if verbose: print(idfct, "IDefCovDate_day", IDefCovDate_day)
  if verbose: print(idfct, "IDefCovDate_ms", IDefCovDate_ms)
  # read NCM structure from npz file (eig_decompos.py output, reconstruct_NCM_PN1.npz for example)
  # also keep cov_red output for checking, but not w_cov and v_cov
  struct_ncm_1, cov_red_1b, cov_red_1c, _, _, _, _ = load_struct_ncm(npz_fname_1b, npz_fname_1c, verbose=verbose)
  # copy input parameters into NCM structure
  struct_ncm_1['IDefIssueIcd'] = IDefIssueIcd
  if verbose: print(idfct, "struct_ncm_1['IDefIssueIcd']", struct_ncm_1['IDefIssueIcd'])
  struct_ncm_1['IDefRevisionIcd'] = IDefRevisionIcd
  if verbose: print(idfct, "struct_ncm_1['IDefRevisionIcd']", struct_ncm_1['IDefRevisionIcd'])
  struct_ncm_1['IDefCovID'] = IDefCovID
  if verbose: print(idfct, "struct_ncm_1['IDefCovID']", struct_ncm_1['IDefCovID'])
  struct_ncm_1['IDefCovDate_day'] = IDefCovDate_day
  if verbose: print(idfct, "struct_ncm_1['IDefCovDate_day']", struct_ncm_1['IDefCovDate_day'])
  struct_ncm_1['IDefCovDate_ms'] = IDefCovDate_ms 
  if verbose: print(idfct, "struct_ncm_1['IDefCovDate_ms']", struct_ncm_1['IDefCovDate_ms'])
  # reconstruct covariance matrix from this NCM structure directly read from npz file
  # only keep covariance output, not separated diagonal and extradiagonal parts
  cov_1b1, _, _ = reconstruct_cov_from_struct_ncm(struct_ncm_1, '1b', verbose=verbose)
  cov_1c1, _, _ = reconstruct_cov_from_struct_ncm(struct_ncm_1, '1c', verbose=verbose)
  # make difference with reduced covariance matrix read from npz file
  # difference is expected to be low
  cov_1b_diff1 = cov_1b1 - cov_red_1b
  my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(cov_1b_diff1)), "cov_1b_diff1", cov_1b_diff1.shape, (np.nanmax(cov_1b_diff1)))
  print(idfct, my_str)
  cov_1c_diff1 = cov_1c1 - cov_red_1c
  my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(cov_1c_diff1)), "cov_1c_diff1", cov_1c_diff1.shape, (np.nanmax(cov_1c_diff1)))
  print(idfct, my_str)
  # write NCM structure in formatted binary file
  ncm_format_write(formatted_fname, struct_ncm_1, verbose=verbose)
  # read NCM structure from previously written formatted binary file
  struct_ncm_2 = ncm_format_read(formatted_fname, verbose=verbose)
  # reconstruct covariance matrix from this NCM structure read from formatted binary file
  # only keep covariance output, not separated diagonal and extradiagonal parts
  cov_1b2, _, _ = reconstruct_cov_from_struct_ncm(struct_ncm_2, '1b', verbose=verbose)
  cov_1c2, _, _ = reconstruct_cov_from_struct_ncm(struct_ncm_2, '1c', verbose=verbose)
  # make difference with reduced covariance matrix read from npz file
  # difference is expected to be low
  cov_1b_diff2 = cov_1b2 - cov_red_1b
  my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(cov_1b_diff2)), "cov_1b_diff2", cov_1b_diff2.shape, (np.nanmax(cov_1b_diff2)))
  print(idfct, my_str)
  cov_1c_diff2 = cov_1c2 - cov_red_1c
  my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(cov_1c_diff2)), "cov_1c_diff2", cov_1c_diff2.shape, (np.nanmax(cov_1c_diff2)))
  print(idfct, my_str)
  print(idfct, "end")


def plkdirect(t,wave):
  """
  calculate forward Planck function t in K wave in m-1
  resultat w en watt/m**2/steradian/m-1
  """
  sca = old_div(Cst_sca2*wave,t)
  if ( sca < 100.0 ):
      denom = math.e**(sca)-1.0
      w = old_div(Cst_sca1*wave**3,denom)
  else:
      w = 0.0
  return w

def plkinverse(wave,w):
  """
  Inverse Planch funcrion w in W/m**2/str/m-1 wave in m-1
  result t in K
  """
  if ( w > 0.0 ):
      denom = (1.0/w)*Cst_sca1*wave**3
      sca = math.log(denom+1.0)
      t = old_div(Cst_sca2*wave,sca)
  else:
      t = 0.0
  return t

def plkderive(t,wave):
  """
  calculate forward Planck function t in K wave in m-1
  and its derivative with respect to the temperature t
  results
  w in w/m**2/str/m-1
  dwsdt in w/m**2/str/m-1/K
  """
  sca = old_div(Cst_sca2*wave,t)
  if ( sca < 100.0 ):
      denom = math.e**(sca)-1.0
      w = old_div(Cst_sca1*wave**3,denom)
      dwsdt = old_div(old_div(w,denom)*(denom+1.0)*sca,t)
  else:
      w = 0.0
      dwsdt = 0.0
  return w,dwsdt


def read_ncm_compute_nedt(formatted_fname, level, verbose=True):
  """
  read NCM structure from formatted binary file
  keep only 1 diagonal vector (there are 5) and compute nedt
  reconstruct covariance matrix from this NCM structure read from formatted binary file
  only keep covariance output, not separated diagonal and extradiagonal parts
  compute nedt on diagonal elements
  nedt results are the same (for diagonal elements)
  """
  idfct="[read_ncm_compute_nedt]"
  if verbose: print(idfct, "begin")
  if verbose: print(idfct, "formatted_fname", formatted_fname)
  if verbose: print(idfct, "level", level)
  # read NCM structure from formatted binary file
  struct_ncm = ncm_format_read(formatted_fname, verbose=verbose)
  cov_nedt=np.zeros(8461)
  T0=280.0
  w=0.
  dwsdt=0.
  print(idfct, "diagonal elements of struct_ncm (level : %s)" % level)
  print(idfct, "iSample, nu0, cov_nedt[i]")
  for iSample in range(0,8461,200):
      nu0=64500.0+25.*iSample  #m-1
      w, dwsdt = plkderive(T0, nu0)  #m-1
      #keep only 1 diagonal vector (there are 5)
      idiag = 0
      x = struct_ncm['IRnmRadNoiseCovarMatDiagonalVectors%s' % level][0:8461-idiag,idiag] #m-1
      cov_nedt[iSample] = old_div(np.sqrt(x[iSample]),dwsdt) #m-1
      my_str= "%4i %8.2f %12.6e" % (iSample, nu0, cov_nedt[iSample])
      print(my_str)
  # reconstruct covariance matrix from this NCM structure
  # only keep covariance output, not separated diagonal and extradiagonal parts
  cov, _, _ = reconstruct_cov_from_struct_ncm(struct_ncm, level, verbose=verbose)
  my_str= "%e <= %s (%s) <= %e" % ((np.nanmin(cov)), "cov", cov.shape, (np.nanmax(cov)))
  cov_nedt=np.zeros(8461)
  T0=280.0
  w=0.
  dwsdt=0.
  print(idfct, "diagonal elements of reconstructed matrix")
  print(idfct, "iSample, nu0, cov_nedt[i]")
  for iSample in range(0,8461,200):
    nu0=64500.0+25.*iSample  #m-1
    w, dwsdt = plkderive(T0, nu0)
    # cov_nedt[iSample] = np.sqrt(cov[iSample,iSample])/dwsdt
    cov_nedt[iSample] = old_div(np.sqrt(cov[iSample,iSample]),dwsdt)
    my_str= "%4i %8.2f %12.6e" % (iSample, nu0, cov_nedt[iSample])
    print(my_str)
  
  
  # Saving the numpy array to a file
  np.save('covariance.npy', cov_nedt)

  # Writing each element of the numpy array to a text file
  with open('covariance.txt', 'w') as f:
      for element in cov_nedt:
          f.write("%s\n" % element)
  

  print(idfct, "end")
 
if __name__ == "__main__":
  idfct="[ncm_format.py]"
  print(idfct, "begin")
  #print argv
  #print len(argv)
  if (len(argv)!=2) :
    print(__doc__)
  if (len(argv)==2) :
    formatted_fname=argv[1]
    for level in ['1b', '1c']:
      read_ncm_compute_nedt(formatted_fname, level, verbose=True)

  print(idfct, "end")
