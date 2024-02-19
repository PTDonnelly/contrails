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

def plkdirect(t, wave):
    """
    Calculate forward Planck function t in K, wave in m-1
    Result w in watt/m**2/steradian/m-1
    """
    sca = old_div(Cst_sca2 * wave, t)
    if sca < 100.0:
        denom = math.exp(sca) - 1.0  # Using math.exp for clarity
        w = old_div(Cst_sca1 * wave ** 3, denom)
    else:
        w = 0.0
    return w

def plkinverse(wave, w):
    """
    Inverse Planck function w in W/m**2/str/m-1, wave in m-1
    Result t in K
    """
    if w > 0.0:
        denom = (1.0 / w) * Cst_sca1 * wave ** 3
        sca = math.log(denom + 1.0)
        t = old_div(Cst_sca2 * wave, sca)
    else:
        t = 0.0
    return t

def plkderive(t, wave):
    """
    Calculate forward Planck function t in K, wave in m-1
    and its derivative with respect to the temperature t
    Results: w in w/m**2/str/m-1, dwsdt in w/m**2/str/m-1/K
    """
    sca = old_div(Cst_sca2 * wave, t)
    if sca < 100.0:
        denom = math.exp(sca) - 1.0  # Using math.exp for clarity
        w = old_div(Cst_sca1 * wave ** 3, denom)
        dwsdt = old_div(old_div(w, denom) * (denom + 1.0) * sca, t)
    else:
        w = 0.0
        dwsdt = 0.0
    return w, dwsdt


def ncm_format_write(fname_output, struct_ncm, verbose=True):
    """
    Write struct_ncm in fname_output file.
    """
    idfct = "[ncm_format_write]"
    if verbose: print(f"{idfct} begin")
    if verbose: print(f"{idfct} fname_output={fname_output}")

    with open(fname_output, 'wb') as fout:
        # Writing integer values
        for key in ['IDefIssueIcd', 'IDefRevisionIcd', 'IDefCovID', 'IDefCovDate_day', 'IDefCovDate_ms']:
            if verbose: print(f"{idfct} struct_ncm['{key}'] {struct_ncm[key]}")
            fout.write(struct.pack('>i', struct_ncm[key]))

        # Writing matrix values
        for key in ['IDefCovarMatEigenVal1b', 'IDefCovarMatEigenVal1c', 'IRnmRadNoiseCovarMatDiagonalVectors1b', 'IRnmRadNoiseCovarMatDiagonalVectors1c', 'IRnmRadNoiseCovarMatEigenVectors1b', 'IRnmRadNoiseCovarMatEigenVectors1c']:
            if verbose:
                my_str = f"{np.nanmin(struct_ncm[key]):e} <= {key} ({struct_ncm[key].shape}) <= {np.nanmax(struct_ncm[key]):e}"
                print(f"{idfct} {my_str}")
            for i in range(struct_ncm[key].shape[0]):
                for j in range(struct_ncm[key].shape[1]):
                    fout.write(struct.pack('>f', struct_ncm[key][i, j]))

        # Writing size information
        for key in ['IDefRnmSize1_1b', 'IDefRnmSize2_1b', 'IDefRnmSize1_1c', 'IDefRnmSize2_1c']:
            if verbose: print(f"{idfct} struct_ncm['{key}'] {struct_ncm[key]}")
            fout.write(struct.pack('>i', struct_ncm[key]))

    if verbose: print(f"{idfct} end")

def ncm_format_read(fname_input, verbose=True):
    """
    Read struct_ncm from fname_input file.
    """
    idfct = "[ncm_format_read]"
    if verbose: print(f"{idfct} begin")
    if verbose: print(f"{idfct} fname_input={fname_input}")
    
    struct_ncm = STRUCT_NCM_EMPTY  # Make sure STRUCT_NCM_EMPTY is defined before this function
    with open(fname_input, 'rb') as fin:
        fileContent = fin.read()

    offset = 0
    mysize = 4

    # Unpacking integers
    for key in ['IDefIssueIcd', 'IDefRevisionIcd', 'IDefCovID', 'IDefCovDate_day', 'IDefCovDate_ms']:
        struct_ncm[key] = struct.unpack('>i', fileContent[offset:offset+mysize])[0]
        offset += mysize
        if verbose: print(f"{idfct} struct_ncm['{key}'] {struct_ncm[key]}")

    # Unpacking matrices
    for matrix_key in ['IDefCovarMatEigenVal1b', 'IDefCovarMatEigenVal1c', 'IRnmRadNoiseCovarMatDiagonalVectors1b', 'IRnmRadNoiseCovarMatDiagonalVectors1c', 'IRnmRadNoiseCovarMatEigenVectors1b', 'IRnmRadNoiseCovarMatEigenVectors1c']:
        mysize = 8 if 'EigenVal' in matrix_key else 4
        for i in range(struct_ncm[matrix_key].shape[0]):
            for j in range(struct_ncm[matrix_key].shape[1]):
                value = struct.unpack('>d' if mysize == 8 else '>f', fileContent[offset:offset+mysize])[0]
                struct_ncm[matrix_key][i, j] = value
                offset += mysize
                if verbose and j < 5:  # Adjust according to your verbose output requirements
                    print(f"{idfct} {matrix_key}[{i},{j}] = {value}")

    # Unpacking sizes
    for size_key in ['IDefRnmSize1_1b', 'IDefRnmSize2_1b', 'IDefRnmSize1_1c', 'IDefRnmSize2_1c']:
        struct_ncm[size_key] = struct.unpack('>i', fileContent[offset:offset+mysize])[0]
        offset += mysize
        if verbose: print(f"{idfct} struct_ncm['{size_key}'] {struct_ncm[size_key]}")

    if verbose: print(f"{idfct} end")
    return struct_ncm


def load_struct_ncm(npz_fname_1b, npz_fname_1c, verbose=True):
    """
    Read npz file (eig_decompos.py output, reconstruct_NCM_PN1.npz for example)
    Return struct_ncm, cov_red_1b, cov_red_1c, w_cov_1b, w_cov_1c, v_cov_1b, and v_cov_1c.
    Only cov_red_1b and cov_red_1c are needed; others are for check.
    
    Note: struct_ncm elements are converted from (W/m2/st/cm-1)^2 to (W/m2/st/m-1)^2, and other outputs too.
    """
    idfct = "[load_struct_ncm]"
    if verbose: print(f"{idfct} begin")
    if verbose: print(f"{idfct} npz_fname_1b={npz_fname_1b}")
    if verbose: print(f"{idfct} npz_fname_1c={npz_fname_1c}")

    struct_ncm = STRUCT_NCM_EMPTY  # Make sure STRUCT_NCM_EMPTY is defined and accessible
    npz_lut = {'cov':'arr_0', 'cov_red':'arr_1', 'cov_norm':'arr_2', 'cov_red_norm':'arr_3', 'v_cov':'arr_6', 'w_cov':'arr_7'}
    levels = ['1b', '1c']
    npz_fnames = {'1b': npz_fname_1b, '1c': npz_fname_1c}

    cov_red, w_cov, v_cov = {}, {}, {}

    for level in levels:
        if verbose: print(f"{idfct} level={level}")
        npzfile = np.load(npz_fnames[level])

        if verbose:
            for k in sorted(npz_lut.keys()):
                try:
                    my_str = f"{np.nanmin(npzfile[npz_lut[k]]):e} <= {k} ({npzfile[npz_lut[k]].shape}) <= {np.nanmax(npzfile[npz_lut[k]]):e}"
                except:
                    my_str = f"{k} ({npzfile[npz_lut[k]].shape})"
                print(f"{idfct} {my_str}")

        # Processing data
        IDefCovarMatEigenVal1 = npzfile[npz_lut['v_cov']][-2:] * 0.0001  # Last two values, conversion applied
        struct_ncm[f'IDefCovarMatEigenVal{level}'] = IDefCovarMatEigenVal1

        struct_ncm[f'IDefRnmSize1_{level}'] = 5
        struct_ncm[f'IDefRnmSize2_{level}'] = 2

        IRnmRadNoiseCovarMatDiagonalVectors1 = np.diag(npzfile[npz_lut['cov']])[:5] * 0.0001  # First 5 diagonals, conversion applied
        struct_ncm[f'IRnmRadNoiseCovarMatDiagonalVectors{level}'] = IRnmRadNoiseCovarMatDiagonalVectors1

        IRnmRadNoiseCovarMatEigenVectors1 = npzfile[npz_lut['w_cov']][:, -2:]  # Last two columns
        struct_ncm[f'IRnmRadNoiseCovarMatEigenVectors{level}'] = IRnmRadNoiseCovarMatEigenVectors1

        cov_red[level] = npzfile[npz_lut['cov_red']] * 0.0001
        w_cov[level] = npzfile[npz_lut['w_cov']]
        v_cov[level] = npzfile[npz_lut['v_cov']] * 0.0001

    if verbose: print(f"{idfct} end")
    return struct_ncm, cov_red['1b'], cov_red['1c'], w_cov['1b'], w_cov['1c'], v_cov['1b'], v_cov['1c']

def reconstruct_cov_from_struct_ncm(struct_ncm, level, verbose=True):
    """
    Reconstruct covariance matrix from struct_ncm.
    Return 3 matrices: cov (covband + covextra), covband (diagonal), covextra (extradiagonal).
    """
    idfct = "[reconstruct_cov_from_struct_ncm]"
    if verbose: print(f"{idfct} begin")
    if verbose: print(f"{idfct} level {level}")
    assert level in ['1b', '1c'], "Level must be '1b' or '1c'"

    cov = np.zeros((8461, 8461))
    covband = np.zeros((8461, 8461))
    w_cov = np.zeros((8461, 8461))
    v_cov = np.zeros(8461)

    IDefCovarMatEigenVal1 = struct_ncm[f'IDefCovarMatEigenVal{level}']
    IRnmRadNoiseCovarMatDiagonalVectors1 = struct_ncm[f'IRnmRadNoiseCovarMatDiagonalVectors{level}']
    IRnmRadNoiseCovarMatEigenVectors1 = struct_ncm[f'IRnmRadNoiseCovarMatEigenVectors{level}']

    for iev in range(struct_ncm[f'IDefRnmSize2_{level}']):  # Last 2 eigenvectors
        v_cov[8460 - iev] = IDefCovarMatEigenVal1[0, iev]  # Assuming first row is relevant
        w_cov[:, 8460 - iev] = IRnmRadNoiseCovarMatEigenVectors1[:, iev]

    nb_diag = struct_ncm[f'IDefRnmSize1_{level}']
    for idiag in range(nb_diag):  # Diagonal bands
        x = IRnmRadNoiseCovarMatDiagonalVectors1[:8461 - idiag, idiag]
        np.fill_diagonal(covband[idiag:], x)
        if idiag > 0:
            np.fill_diagonal(covband[:, idiag:], x)

    if verbose:
        print(f"{idfct} {np.nanmin(covband):e} <= covband ({covband.shape}) <= {np.nanmax(covband):e}")

    # covextra calculation
    covextra = w_cov @ np.diag(v_cov) @ w_cov.T
    for idiag in range(nb_diag):  # Remove diagonals reconstructed from eigenvectors
        np.fill_diagonal(covextra[idiag:], 0)
        if idiag > 0:
            np.fill_diagonal(covextra[:, idiag:], 0)

    if verbose:
        print(f"{idfct} {np.nanmin(covextra):e} <= covextra ({covextra.shape}) <= {np.nanmax(covextra):e}")

    cov = covextra + covband
    if verbose:
        print(f"{idfct} {np.nanmin(cov):e} <= cov ({cov.shape}) <= {np.nanmax(cov):e}")
        print(f"{idfct} end")

    return cov, covband, covextra



def read_ncm_compute_nedt(formatted_fname, level, verbose=True):
    """
    read NCM structure from formatted binary file
    keep only 1 diagonal vector (there are 5) and compute nedt
    reconstruct covariance matrix from this NCM structure read from formatted binary file
    only keep covariance output, not separated diagonal and extradiagonal parts
    compute nedt on diagonal elements
    nedt results are the same (for diagonal elements)
    """
    idfct = "[read_ncm_compute_nedt]"
    if verbose: print((idfct, "begin"))
    if verbose: print((idfct, "formatted_fname", formatted_fname))
    if verbose: print((idfct, "level", level))
    # read NCM structure from formatted binary file
    struct_ncm = ncm_format_read(formatted_fname, verbose=verbose)
    cov_nedt = np.zeros(8461)
    T0 = 280.0
    w = 0.
    dwsdt = 0.
    print(f"{idfct} diagonal elements of struct_ncm (level : {level})")
    print(f"{idfct} iSample, nu0, cov_nedt[i]")
    for iSample in range(0, 8461, 200):
        nu0 = 64500.0 + 25. * iSample  # m-1
        w, dwsdt = plkderive(T0, nu0)  # m-1
        # keep only 1 diagonal vector (there are 5)
        idiag = 0
        x = struct_ncm[f'IRnmRadNoiseCovarMatDiagonalVectors{level}'][0:8461-idiag,idiag] # m-1
        cov_nedt[iSample] = old_div(np.sqrt(x[iSample]), dwsdt) # m-1
        my_str = f"{iSample:4} {nu0:8.2f} {cov_nedt[iSample]:12.6e}"
        print(my_str)
    # reconstruct covariance matrix from this NCM structure
    # only keep covariance output, not separated diagonal and extradiagonal parts
    cov, _, _ = reconstruct_cov_from_struct_ncm(struct_ncm, level, verbose=verbose)
    my_str = f"{np.nanmin(cov):e} <= cov ({cov.shape}) <= {np.nanmax(cov):e}"
    cov_nedt = np.zeros(8461)
    T0 = 280.0
    w = 0.
    dwsdt = 0.
    print(f"{idfct} diagonal elements of reconstructed matrix")
    print(f"{idfct} iSample, nu0, cov_nedt[i]")
    for iSample in range(0, 8461, 200):
        nu0 = 64500.0 + 25. * iSample  # m-1
        w, dwsdt = plkderive(T0, nu0)
        cov_nedt[iSample] = old_div(np.sqrt(cov[iSample,iSample]), dwsdt)
        my_str = f"{iSample:4} {nu0:8.2f} {cov_nedt[iSample]:12.6e}"
        print(my_str)
    

    np.save('covariance.npy', cov_nedt)
    with open('covariance.txt', mode='w') as f:
        for element in cov_nedt:
            f.write(f"{element}\n")
    print(np.shape(cov_nedt))
    print(f"{idfct} end")

if __name__ == "__main__":
    idfct = "[ncm_format.py]"
    print(f"{idfct} begin")
    print(argv)
    print(len(argv))
    if (len(argv) != 2):
        print(__doc__)
    if (len(argv) == 2):
        formatted_fname = argv[1]
        for level in ['1c']:
            read_ncm_compute_nedt(formatted_fname, level, verbose=True)
    print(f"{idfct} end")


