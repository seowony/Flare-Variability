import os
import sys
import glob
import string
import getopt
import math

import numpy as np
import numpy.ma as ma
from astropy.io import fits, ascii
from astropy.table import Table
import matplotlib.pyplot as plt

import warnings
# Suppress the specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
# Get the screen session name from the environment variable
screen_session = os.getenv('STY', 'default_session')
log_filename = 'error_log_{}.txt'.format(screen_session)

# Configure logging to use the unique log file name
logging.basicConfig(filename=log_filename, level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')

def convert_filters(filt):
    """
    Replcae values in an array
    """
    filt[filt=='u'] = 1
    filt[filt=='v'] = 2
    filt[filt=='g'] = 3 
    filt[filt=='r'] = 4
    filt[filt=='i'] = 5
    filt[filt=='z'] = 6
    return filt

def get_qmean(pick_prior, input_lc):
    """
    get qmean + qstd

    2024-06-17 remove rows with NaN values from a structured array like lc

    """

    nan_mag_psf_mask = np.isnan(input_lc['mag_psf'])
    lc = input_lc[~nan_mag_psf_mask]

    filters = ['u', 'v', 'g', 'r', 'i', 'z']
    m_lc = lc['mag_psf']
    f_lc = lc['filter']
    qstd = []
    for i in filters:
        t_mask = (f_lc == str(i))
        if len(m_lc[t_mask]) > 0:
            qstd.append(np.std(m_lc[t_mask]))
        else:
            qstd.append(0.0)
    qmean = [pick_prior['u_psf'][0], pick_prior['v_psf'][0], pick_prior['g_psf'][0], pick_prior['r_psf'][0], pick_prior['i_psf'][0], pick_prior['z_psf'][0]]
    return qmean, qstd

def get_estrms(filt, wmean, verbose=False):
    """
    2024-08-26 Update the relation of log(e_psf_filter) vs. psf_filter

    """

    # After 2024-08-26: include quantile level of 0.05
    rms_m_dr4 = [0.19, 0.21, 0.24, 0.24, 0.21, 0.21]
    rms_c_dr4 = [-5.1, -5.55, -6.78, -6.61, -5.9, -5.65]

    fidx = filt - 1
    # mean relation between MAG and RMS - v5.0
    if verbose:
        print(fidx, wmean)
    if wmean > 0:
        est_rms = math.pow(10, rms_m_dr4[fidx]*wmean + rms_c_dr4[fidx])
    else:
        est_rms = 0.
    return est_rms #, est_err

def get_wmean(input_lc, fmjd, qmean, qstd, stage, verbose=False):
    """
    # see DR1 paper (Wolf+ 2018) and DR2 paper (Onken+ 2019)
    # to estimate a "PSF" magnitude and error, taken as the weighted mean of the inputs and the error on the weighted mean
    """
    nan_date_mask = np.isnan(input_lc['date'].astype(float))  # Assuming 'date' can be converted to float for NaN checking
    nan_mag_psf_mask = np.isnan(input_lc['mag_psf'])
    nan_e_mag_psf_mask = np.isnan(input_lc['e_mag_psf'])

    #Combine masks using logical OR to identify rows with any NaN values
    nan_any_mask = nan_date_mask | nan_mag_psf_mask | nan_e_mag_psf_mask  # Add more masks as needed

    # Use the inverse of the combined mask to filter out rows with NaN values
    lc = input_lc[~nan_any_mask]

    t_lc, m_lc, e_lc, f_lc = lc['date'], lc['mag_psf'], lc['e_mag_psf'], convert_filters(lc['filter'])

    nimaflag_lc = lc['nimaflags']
    wmag, wstd = [], []
    for k in range(1,7): # filter NUMBER sequence: uvgriz
        t_mask = (np.abs(fmjd - t_lc) > 30./1440.) & (f_lc == str(k)) # | ((fmjd - t_lc < 0.) & (f_lc == k))  # flags for non-outliers
        if np.sum(t_mask) > 0:
            if verbose:
                print(fmjd, k, m_lc[t_mask], t_lc[t_mask])
            wmag.append(np.average(m_lc[t_mask],weights=1./e_lc[t_mask]))
            if np.sum(t_mask) > 1:
                wstd.append(np.std(m_lc[t_mask], ddof=1))
            else:
                estd = get_estrms(k, wmag[k-1], verbose=False) # simple estimation
                wstd.append(estd)
        else:
            if stage == 1:
                wmag.append(0.0)
                wstd.append(0.0)
            elif stage == 2:
                wmag.append(0.0)
                wstd.append(0.0)
    wmag = np.round(wmag, 3)
    wstd = np.round(wstd, 3)

    # check only riz magnitudes for stability due to non-uniform samplings or bad epochs
    # DR1.1: (i_psf - z_psf) = 0.434 * (r_psf - i_psf) - 0.041
    # Use this file: ParentSample_Mdwarf_DR3.fits
    # DR3.0: (i_psf - z_psf) = 0.388 * (r_psf - i_psf) - 0.011
    # Use this script for getting coefficients: python special_functions_v8.py
    # DR4: (i_psf - z_psf) = 0.123 + 0.225*(r_psf - i_psf) + 0.056*(r_psf - i_psf)*(r_psf - i_psf)
    rmi, qrmi = wmag[3] - wmag[4], qmean[3] - qmean[4]
    imz, qimz = wmag[4] - wmag[5], qmean[4] - qmean[5]
    dist, qdist = abs(imz - (0.123 + 0.225*rmi + 0.056*rmi*rmi)), abs(qimz - (0.123 + 0.225*qrmi + 0.056*qrmi*qrmi))
    if qdist < dist:
        for k in range(4,7):
            wmag[k-1] = qmean[k-1]

    # check g magnitudes for stability due to non-uniform samplings or bad epochs
    gmr, qgmr = abs(wmag[2] - (0.996*wmag[3]+1.)), abs(qmean[2] - (0.996*qmean[3]+1.))
    if qgmr < gmr and wmag[2] > 0 and qmean[2] > 0:
        wmag[2] = qmean[2]

    # check uvg magnitude
    if wmag[0] < wmag[2]:
        wmag[0] = 0.
    if wmag[1] < wmag[2]:
        wmag[1] = 0.
    if wmag[2] < wmag[3]:
        wmag[2] = 0.

    # find 'zero' qmean values: use estimated qmag; remain some relations from DR1.1
    xm_idx = np.argwhere(wmag == 0)
    for kk in reversed(xm_idx.tolist()):
        k = kk[0]
        if k == 0:
            if wmag[1] > 0:    # u-filter from v-filter
                wmag[k] = (0.98*wmag[1] + 0.646) + np.random.normal(0, 0.138, 10000)[-1]
            elif wmag[2] > 0:  # u-filter from g-filter
                wmag[k] = (0.965*wmag[2] + 3.327) + np.random.normal(0, 0.1197, 10000)[-1]
            elif wmag[3] > 0:  # u-filter from r-filter
                wmag[k] = (0.967*wmag[3] + 4.1785) + np.random.normal(0, 0.124, 10000)[-1]
            else:
                wmag[k] = qmean[k]
                wstd[k] = qstd[k]
        elif k == 1:
            if wmag[0] > 0:  # v-filter from u
                wmag[k] = (1.01*wmag[0] - 0.4746) + np.random.normal(0, 0.117, 10000)[-1]
            elif wmag[2] > 0:  # v-filter from g
                wmag[k] = (0.99*wmag[2] + 2.647) + np.random.normal(0, 0.099, 10000)[-1]
            elif wmag[3] > 0:  # v-filter from r, +-1 range
                wmag[k] = (0.993*wmag[3] + 3.506) + np.random.normal(0, 0.11, 10000)[-1]
            else:
                wmag[k] = qmean[k]
                wstd[k] = qstd[k]
        elif k == 2:
            if wmag[3] > 0:  # g-filter from r + STDDEV from Gaussian fitting
                wmag[k] = (0.996*wmag[3] + 1.) + np.random.normal(0, 0.08, 10000)[-1]
            else:
                wmag[k] = qmean[k]
                wstd[k] = qstd[k]
        elif k >= 3:
            wmag[k] = qmean[k]
            wstd[k] = qstd[k]
        if k <= 2:
            estd = get_estrms(k, wmag[k])
            wstd[k] = estd

    if verbose:
        print("++++++QMEAN++++++",qmean, np.round(qstd,3))
        print("++++++WMEAN++++++",wmag, wstd)
    return list(wmag), list(wstd)

def get_group_mixture(lc, verbose=False, verbose2=False):
    """
    new version of get_group function
    """
    all_group = []
    all_gidx = []
    all_tblk = []
    all_gc = []
    all_mode = []

    group = []
    gidx = []
    tblk = []
    mode = []
    gc = 0

    mjd, filt, expt = lc['date'], convert_filters(lc['filter']), lc['exp_time']
    for i in range(len(lc)):  # i refers to index
         checkin_mjd, checkin_filt, checkin_expt = mjd[i], int(filt[i]), expt[i]
         # survey modes: shallow is 'fs'; main is 'ms'
         if checkin_expt == 100:
             checkin_mode = 'ms'
         else:
             checkin_mode = 'fs'

         if len(group) == 0:
             group.append(checkin_filt)
             tblk.append(checkin_mjd)
             gidx.append(i)
             mode.append(checkin_mode) 
             if verbose2:
                 print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode)
             continue

         # check two different survey modes
         delt = (checkin_mjd - tblk[0])*1440.
         if checkin_expt < 100:
             # typical [u-v-g-r-i-z] sequence
             if checkin_filt not in group and (checkin_filt - group[-1]) > 0 and delt < 5.0:
                 group.append(checkin_filt)
                 tblk.append(checkin_mjd)
                 gidx.append(i)
                 mode.append(checkin_mode)
                 if verbose2:
                     print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
                 continue
             else:
                 if verbose:
                     print(gc, group, gidx, tblk[0], mode[-1])
                 if len(group) > 1:
                     gc+=1
                     all_group.append(group)
                     all_gidx.append(gidx)
                     all_tblk.append(tblk[0])
                     all_gc.append(gc)
                     all_mode.append(mode[-1])
             group = []
             gidx = []
             group.append(checkin_filt)
             gidx.append(i)
             tblk[0] = checkin_mjd
             mode[-1] = checkin_mode
             if verbose2:
                 print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
         elif checkin_expt == 100:
             # main colour sequence: [u-v-g-r-u-v-i-z-u-v] ..... [u-v-g-r-u-v-i-z-u-v]
             where_i = [idx for idx, value in enumerate(group) if value == 5]
             where_z = [idx for idx, value in enumerate(group) if value == 6]
             # special condition: [iziz] sequence within 10 minutes: 87920199, 259222047
             if delt< 30. and group[0] != 5: 
                 # additional colour pairs: two [g-r] or three [i-z]
                 if delt < 2.1:
                     if 3 in group or 4 in group:
                         checkin_mode = 'ms_gr'
                     elif 5 in group or 6 in group:
                         checkin_mode = 'ms_iz'
#                     mode[0] = checkin_mode
                 mode.append(checkin_mode)
                 group.append(checkin_filt)
                 tblk.append(checkin_mjd)
                 gidx.append(i)
                 if verbose2:
                     print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
                 continue
             elif delt< 30. and group[0] == 5 and len(where_i) <= 1 and len(where_z) < 1:
                 if delt < 2.1:
                     if 3 in group or 4 in group:
                         checkin_mode = 'ms_gr'
                     elif 5 in group or 6 in group:
                         checkin_mode = 'ms_iz'
                 mode.append(checkin_mode)
                 group.append(checkin_filt)
                 tblk.append(checkin_mjd)
                 gidx.append(i)
                 if verbose2:
                     print("  ????", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
                 continue
             else:
                 if verbose:
                     print(gc, group, gidx, tblk[0], mode[-1])
                 if len(group) > 1:
                     gc+=1
                     all_group.append(group)
                     all_gidx.append(gidx)
                     all_tblk.append(tblk[0])
                     all_gc.append(gc)
                     all_mode.append(mode[-1])
             if verbose2:
                 print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
             group = []
             gidx = []
             group.append(checkin_filt)
             gidx.append(i)
             tblk[0] = checkin_mjd
             mode[-1] = checkin_mode
    # this is for last group
    if len(group) > 1:
        if verbose:
            print(gc, group, gidx, tblk[0], mode[-1])
        gc+=1
        all_group.append(group)
        all_gidx.append(gidx)
        all_tblk.append(tblk[0])
        all_gc.append(gc)
        all_mode.append(mode[-1])
    return all_group, all_gidx, all_tblk, all_gc, all_mode

def get_DR4group_mixture(lc, verbose=False, verbose2=False):
    """
    new version of get_DR4group function
    
    2024-05-28 use image_type for grouping LC clumps
   
    (runzogy) seowony@manta:/priv/manta2/skymap/seowony/flare_smdr4>stilts tpipe in=/priv/manta2/skymap/seowony/flare_smdr4/splitLC/lc_167.fits cmd='rowrange 1 10'
Table name: LCdump_dr4_with_GaiaDR3.csv
+-----------+---------------------+----------------+---------+-----------+--------+-------+-----------+-----+---------+---------+----------+----------+----------------
    | object_id | source_id           | date           | mag_psf | e_mag_psf | filter | flags | nimaflags | ccd | x_img   | y_img   | exp_time | img_qual | use_in_clipped | image_type |
| 88207322  | 5742418278875244032 | 57158.34614583 | 19.3408 | 0.037     | i      | 0     | 0         | 1   | 1814.12 | 2801.09 | 100.0    | 1        | 1              | ms         |
| 88207322  | 5742418278875244032 | 57158.34336806 | 19.1476 | 0.0775    | i      | 0     | 0         | 1   | 1399.34 | 1555.14 | 100.0    | 1        | 1              | ms         |
+-----------+---------------------+----------------+---------+-----------+--------+-------+-----------+-----+---------+---------+----------+----------+----------------+------------+

    """
    all_group = []
    all_gidx = []
    all_tblk = []
    all_gc = []
    all_mode = []

    group = []
    gidx = []
    tblk = []
    mode = []
    gc = 0

    mjd, filt, expt, imgtype = lc['date'], convert_filters(lc['filter']), lc['exp_time'] ,lc['image_type']
    for i in range(len(lc)):  # i refers to index
         checkin_mjd, checkin_filt, checkin_expt, checkin_imgt = mjd[i], int(filt[i]), expt[i], imgtype[i]
         # survey modes: shallow is 'fs'; main is 'ms'
         # dr4_images.fits cmd='keepcols "image_type"; sort -down image_type; uniq -count'
         #Table name: dr4_images.csv
         #+----------+------------+
         #| DupCount | image_type | exp_time
         #+----------+------------+
         #| 47012    | std        | 3 (g,r), 5 (i), 10 (z), 15 (v), 20 (u) sec
         #| 122754   | ms         | 100 (uvgriz), 300 (uv), 400 (uv)
         #| 194795   | fs         | 5 (g,r) 10 (i), 20 (z,v), 40 (u)
         #| 20576    | bad        | 30, 45, 100, 200, 300, 400
         #| 32086    | 3ps        | ...
         #+----------+------------+
         checkin_mode = checkin_imgt

         if len(group) == 0:
             group.append(checkin_filt)
             tblk.append(checkin_mjd)
             gidx.append(i)
             mode.append(checkin_mode) 
             if verbose2:
                 print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode)
             continue

         # check five different survey modes based on image_type
         delt = (checkin_mjd - tblk[0])*1440.   # minutes
         if checkin_mode == 'fs':
             # typical [u-v-g-r-i-z] sequence: < 4 minute
             if checkin_filt not in group and (checkin_filt - group[-1]) > 0 and delt < 5.0:
                 group.append(checkin_filt)
                 tblk.append(checkin_mjd)
                 gidx.append(i)
                 mode.append(checkin_mode)
                 if verbose2:
                     print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
                 continue
             else:
                 if verbose:
                     print(gc, group, gidx, tblk[0], mode[-1])
                 if len(group) > 1:
                     gc+=1
                     all_group.append(group)
                     all_gidx.append(gidx)
                     all_tblk.append(tblk[0])
                     all_gc.append(gc)
                     all_mode.append(mode[-1])
             group = []
             gidx = []
             group.append(checkin_filt)
             gidx.append(i)
             tblk[0] = checkin_mjd
             mode[-1] = checkin_mode
             if verbose2:
                 print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
         elif checkin_mode == 'std':
             # typical [u-v-g-r-i-z] sequence: < 7-10 minutes?
             if checkin_filt not in group and (checkin_filt - group[-1]) > 0 and delt < 8.0:
                 group.append(checkin_filt)
                 tblk.append(checkin_mjd)
                 gidx.append(i)
                 mode.append(checkin_mode)
                 if verbose2:
                     print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
                 continue
             else:
                 if verbose:
                     print(gc, group, gidx, tblk[0], mode[-1])
                 if len(group) > 1:
                     gc+=1
                     all_group.append(group)
                     all_gidx.append(gidx)
                     all_tblk.append(tblk[0])
                     all_gc.append(gc)
                     all_mode.append(mode[-1])
             group = []
             gidx = []
             group.append(checkin_filt)
             gidx.append(i)
             tblk[0] = checkin_mjd
             mode[-1] = checkin_mode
             if verbose2:
                 print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
         elif checkin_mode == 'ms' and checkin_expt == 100:
             # main colour sequence: [u-v-g-r-u-v-i-z-u-v] ..... [u-v-g-r-u-v-i-z-u-v]
             where_i = [idx for idx, value in enumerate(group) if value == 5]
             where_z = [idx for idx, value in enumerate(group) if value == 6]
             # special condition: [iziz] sequence within 10 minutes: 87920199, 259222047
             if delt< 30. and group[0] != 5: 
                 # additional colour pairs: two [g-r] or three [i-z]
                 if delt < 2.1:
                     if 3 in group or 4 in group:
                         checkin_mode = 'ms_gr'
                     elif 5 in group or 6 in group:
                         checkin_mode = 'ms_iz'
#                     mode[0] = checkin_mode
                 mode.append(checkin_mode)
                 group.append(checkin_filt)
                 tblk.append(checkin_mjd)
                 gidx.append(i)
                 if verbose2:
                     print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
                 continue
             elif delt< 30. and group[0] == 5 and len(where_i) <= 1 and len(where_z) < 1:
                 if delt < 2.1:
                     if 3 in group or 4 in group:
                         checkin_mode = 'ms_gr'
                     elif 5 in group or 6 in group:
                         checkin_mode = 'ms_iz'
                 mode.append(checkin_mode)
                 group.append(checkin_filt)
                 tblk.append(checkin_mjd)
                 gidx.append(i)
                 if verbose2:
                     print("  ????", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
                 continue
             else:
                 if verbose:
                     print(gc, group, gidx, tblk[0], mode[-1])
                 if len(group) > 1:
                     gc+=1
                     all_group.append(group)
                     all_gidx.append(gidx)
                     all_tblk.append(tblk[0])
                     all_gc.append(gc)
                     all_mode.append(mode[-1])
             if verbose2:
                 print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
             group = []
             gidx = []
             group.append(checkin_filt)
             gidx.append(i)
             tblk[0] = checkin_mjd
             mode[-1] = checkin_mode
         # uv filters of 300 sec 
         elif checkin_mode == 'ms' and checkin_expt == 300:
             # [u-v] sequence: < 11~14 minutes?
             checkin_mode = 'ms_uv300'
             if checkin_filt not in group and (checkin_filt - group[-1]) > 0 and delt < 5.4:
                 group.append(checkin_filt)
                 tblk.append(checkin_mjd)
                 gidx.append(i)
                 mode.append(checkin_mode)
                 if verbose2:
                     print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
                 continue
             else:
                 if verbose:
                     print(gc, group, gidx, tblk[0], mode[-1])
                 if len(group) > 1:
                     gc+=1
                     all_group.append(group)
                     all_gidx.append(gidx)
                     all_tblk.append(tblk[0])
                     all_gc.append(gc)
                     all_mode.append(mode[-1])
             group = []
             gidx = []
             group.append(checkin_filt)
             gidx.append(i)
             tblk[0] = checkin_mjd
             mode[-1] = checkin_mode
             if verbose2:
                 print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
         # uv filters of 400 sec 
         elif checkin_mode == 'ms' and checkin_expt == 400:
             # [u-v] sequence: < 11~14 minutes?
             checkin_mode = 'ms_uv400'
             if checkin_filt not in group and (checkin_filt - group[-1]) > 0 and delt < 7.1:
                 group.append(checkin_filt)
                 tblk.append(checkin_mjd)
                 gidx.append(i)
                 mode.append(checkin_mode)
                 if verbose2:
                     print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
                 continue
             else:
                 if verbose:
                     print(gc, group, gidx, tblk[0], mode[-1])
                 if len(group) > 1:
                     gc+=1
                     all_group.append(group)
                     all_gidx.append(gidx)
                     all_tblk.append(tblk[0])
                     all_gc.append(gc)
                     all_mode.append(mode[-1])
             group = []
             gidx = []
             group.append(checkin_filt)
             gidx.append(i)
             tblk[0] = checkin_mjd
             mode[-1] = checkin_mode
             if verbose2:
                 print("  ####", i, checkin_mjd, checkin_filt, checkin_expt, checkin_mode, delt)
         # ignore 3PS & BAD data
    # this is for last group
    if len(group) > 1:
        if verbose:
            print(gc, group, gidx, tblk[0], mode[-1])
        gc+=1
        all_group.append(group)
        all_gidx.append(gidx)
        all_tblk.append(tblk[0])
        all_gc.append(gc)
        all_mode.append(mode[-1])
    return all_group, all_gidx, all_tblk, all_gc, all_mode


def get_phi_updateFS(group, gidx, qmean, lc, filt_pair):
    """
    get phi for FS, MS_gr, MS_iz modes
    2024-06-03 Added MS_uv300, MS_uv400, and STD modes?
    2024-09-04 Added Nan value condition

    """
    dm_pf = 0.
    dm_sf = 0.
    pp_pair = 0
    pp_flag = 0		# add new flag for each pair
    t_pair = phi_pair = 0.
    pf = filt_pair	# primary filter
    sf = filt_pair + 1	# secondary filter
    pf_mean = qmean[filt_pair-1]
    sf_mean = qmean[filt_pair]

    if pf in group and sf in group and pf_mean > 0. and sf_mean > 0.:
        # mjd
        t_pair = (lc['date'][gidx[group.index(sf)]] - lc['date'][gidx[group.index(pf)]])*1440 # unit: minutes
        # t_pair < 2.1 is a good condition for ms_gr/ms_iz pairs
        if (group.index(sf) - group.index(pf)) == 1 and t_pair < 2.1: 
            # mag and mag_err
            pf_mag = lc['mag_psf'][gidx[group.index(pf)]]
            pf_err = lc['e_mag_psf'][gidx[group.index(pf)]]
            sf_mag = lc['mag_psf'][gidx[group.index(sf)]]
            sf_err = lc['e_mag_psf'][gidx[group.index(sf)]]
            # 2024-09-04: manage NaN values in mag_psf
            if np.isnan(sf_mag) or np.isnan(pf_mag):
                phi_pair = 0.
                pp_pair = -1
                if np.isnan(pf_mag):
                    dm_pf = 0.
                if np.isnan(sf_mag):
                    dm_sf = 0.
            else:
                phi_pair = (pf_mag - pf_mean) / pf_err * (sf_mag - sf_mean) / sf_err
                if (pf_mag - pf_mean) < 0 and (sf_mag - sf_mean) < 0:	# flare candidate
                    pp_pair = 1
                elif (pf_mag - pf_mean) < 0 and (sf_mag - sf_mean) > 0: # null, primary filter < 0
                    pp_pair = 2
                elif (pf_mag - pf_mean) > 0 and (sf_mag - sf_mean) > 0: # eclipse candidate
                    pp_pair = 3
                elif (pf_mag - pf_mean) > 0 and (sf_mag - sf_mean) < 0: # null, secondary filter < 0
                    pp_pair = 4
                dm_pf = round(pf_mean - pf_mag, 3)	# pf_dm
                dm_sf = round(sf_mean - sf_mag, 3)	# sf_dm
	    # 20180511 - add flag (nimaflags) for each pair
            pp_flag = np.sum([lc['nimaflags'][gidx[group.index(pf)]], lc['nimaflags'][gidx[group.index(sf)]]])
        # ms_uv300
        elif (group.index(sf) - group.index(pf)) == 1 and filt_pair == 1 and t_pair < 5.5:
            # mag and mag_err
            pf_mag = lc['mag_psf'][gidx[group.index(pf)]]
            pf_err = lc['e_mag_psf'][gidx[group.index(pf)]]
            sf_mag = lc['mag_psf'][gidx[group.index(sf)]]
            sf_err = lc['e_mag_psf'][gidx[group.index(sf)]]
            if np.isnan(sf_mag) or np.isnan(pf_mag):
                phi_pair = 0.
                pp_pair = -1
                if np.isnan(pf_mag):
                    dm_pf = 0.
                if np.isnan(sf_mag):
                    dm_sf = 0.
            else:
                phi_pair = (pf_mag - pf_mean) / pf_err * (sf_mag - sf_mean) / sf_err
                if (pf_mag - pf_mean) < 0 and (sf_mag - sf_mean) < 0:	# flare candidate
                    pp_pair = 1
                elif (pf_mag - pf_mean) < 0 and (sf_mag - sf_mean) > 0: # null, primary filter < 0
                    pp_pair = 2
                elif (pf_mag - pf_mean) > 0 and (sf_mag - sf_mean) > 0: # eclipse candidate
                    pp_pair = 3
                elif (pf_mag - pf_mean) > 0 and (sf_mag - sf_mean) < 0: # null, secondary filter < 0
                    pp_pair = 4
                dm_pf = round(pf_mean - pf_mag, 3)	# pf_dm
                dm_sf = round(sf_mean - sf_mag, 3)	# sf_dm
            # 20180511 - add flag (nimaflags) for each pair
            pp_flag = np.sum([lc['nimaflags'][gidx[group.index(pf)]], lc['nimaflags'][gidx[group.index(sf)]]])
        # ms_uv400
        elif (group.index(sf) - group.index(pf)) == 1 and filt_pair == 1 and t_pair < 7.1:
            # mag and mag_err
            pf_mag = lc['mag_psf'][gidx[group.index(pf)]]
            pf_err = lc['e_mag_psf'][gidx[group.index(pf)]]
            sf_mag = lc['mag_psf'][gidx[group.index(sf)]]
            sf_err = lc['e_mag_psf'][gidx[group.index(sf)]]
            if np.isnan(sf_mag) or np.isnan(pf_mag):
                phi_pair = 0.
                pp_pair = -1
                if np.isnan(pf_mag):
                    dm_pf = 0.
                if np.isnan(sf_mag):
                    dm_sf = 0.
            else:
                phi_pair = (pf_mag - pf_mean) / pf_err * (sf_mag - sf_mean) / sf_err
                if (pf_mag - pf_mean) < 0 and (sf_mag - sf_mean) < 0:	# flare candidate
                    pp_pair = 1
                elif (pf_mag - pf_mean) < 0 and (sf_mag - sf_mean) > 0: # null, primary filter < 0
                    pp_pair = 2
                elif (pf_mag - pf_mean) > 0 and (sf_mag - sf_mean) > 0: # eclipse candidate
                    pp_pair = 3
                elif (pf_mag - pf_mean) > 0 and (sf_mag - sf_mean) < 0: # null, secondary filter < 0
                    pp_pair = 4
                dm_pf = round(pf_mean - pf_mag, 3)	# pf_dm
                dm_sf = round(sf_mean - sf_mag, 3)	# sf_dm
            # 20180511 - add flag (nimaflags) for each pair
            pp_flag = np.sum([lc['nimaflags'][gidx[group.index(pf)]], lc['nimaflags'][gidx[group.index(sf)]]])

    return pp_pair, np.round(phi_pair,2), t_pair, dm_pf, dm_sf, pp_flag

def get_phi_updateMS(group, gidx, qmean, lc, pf_idx, verbose=False, verbose2=False):
    """
    get phi for MS
    pf is a primary index of filter_pair in ms_ceq = [1, 2, 3, 4, 1, 2, 5, 6, 1, 2]
    REMEMBER pf_idx is starting from 1

    group = [1, 2, 3, 4, 1, 2, 5, 6, 1, 2], [3, 4, 5, 6], or several diffrent combinations
    """
    ms_ceq = [1, 2, 3, 4, 1, 2, 5, 6, 1, 2]
    sf_idx = pf_idx + 1     # index for secondary filter

    dm_pf, dm_sf, pp_pair, pp_flag = 0., 0., 0, 0
    t_pair = phi_pair = 0.
    # primary and secondary filters 
    pf, sf = ms_ceq[pf_idx-1], ms_ceq[sf_idx-1]
    pf_mean, sf_mean = qmean[pf-1], qmean[sf-1]
    pf_here, sf_here = -1, -1

    # REMINDER: enumerate() provides us all indices of a given filter
    where_pf, where_sf = [idx for idx, value in enumerate(group) if value == pf], [idx for idx, value in enumerate(group) if value == sf]

    # 20200218 - SWC
    # 67637471 [v u v v] only cases
    where_vdx = [idx for idx, value in enumerate(group) if value == 2]
    where_rdx = [idx for idx, value in enumerate(group) if value == 4]
    where_idx = [idx for idx, value in enumerate(group) if value == 5]
    where_zdx = [idx for idx, value in enumerate(group) if value == 6]

    # 14-Feb-2020: SWC
    # ValueError: 3,4,5,6 are not in list
    #where_gdx, where_rdx, where_idx, where_zdx = group.index(3), group.index(4), group.index(5), group.index(6)
    if where_pf and where_sf and pf_mean > 0. and sf_mean > 0.:
        # [uv_1, vg, gr, ru, uv_2, vi, iz, zu, uv_3]
        if pf != 1: # vgriz filters
            # REMINDER: index() returns the lowest index in list
            # griz filters have unique indices, so it is fine with [vg, gr, iz] pairs
            # PLEASE check index of uv filters for [ru, vi, zu] pairs
            # pf_idx = [2, 3, 4, 6, 7, 8]
            if pf_idx == 4:   # ru (4)
                if where_rdx and where_idx:
                    where_udx = [idx for idx, value in enumerate(where_sf) if (value > where_rdx[0]) and (value < where_idx[0])]
                    if where_udx:
                        pf_here, sf_here = group.index(pf), where_sf[where_udx[0]]
                elif where_rdx and not where_idx:
                    where_udx = [idx for idx, value in enumerate(where_sf) if (value - where_rdx[0]) == 1]
                    if where_udx:
                        pf_here, sf_here = group.index(pf), where_sf[where_udx[0]]
            elif pf_idx == 8: # zu (8)
                where_udx = [idx for idx, value in enumerate(where_sf) if (value - where_zdx[0]) == 1]
                if where_udx:
                    pf_here, sf_here = group.index(pf), where_sf[where_udx[0]]
            elif pf_idx == 6: # vi (6)
                if where_rdx and where_idx:
                    where_vdx = [idx for idx, value in enumerate(where_pf) if (value > where_rdx[0]) and (value < where_idx[0])]
                    if where_vdx:
                        pf_here, sf_here = where_pf[where_vdx[0]], group.index(sf)
                elif not where_rdx and where_idx:
                    where_vdx = [idx for idx, value in enumerate(where_pf) if (value - where_idx[0]) == 1]
                    if where_vdx:
                        pf_here, sf_here = where_pf[where_vdx[0]], group.index(sf)
            else: # vg (2), gr (3), iz (7)
                if (group.index(sf) - group.index(pf)) == 1:
                    pf_here, sf_here = group.index(pf), group.index(sf)
        else: # uv filter: what about [uv_1, uv_2, uv_3] pairs?
            # At least, riz should be in group list. NOT TRUE anymore, but it works in this IF block 
            #   #DR3: 67637471 > python fvarindex_v7.py -f /priv/manta2/skymap/seowony/flare_smdr4/splitLC/lc_76.fits
            #   6 [2, 1, 2, 2] [20, 21, 22, 23] 57362.6607407 ms
            # pf_idx = [1, 5, 9]
            if pf_idx == 1:   # uv_1
                if where_rdx:
                    where_udx = [idx for idx, value in enumerate(where_pf) if value < where_rdx[0]]
                    where_vdx = [idx for idx, value in enumerate(where_sf) if value < where_rdx[0]]
                    if where_udx and where_vdx:
                        pf_here, sf_here = where_pf[where_udx[0]], where_sf[where_vdx[0]]
                else:
                    where_udx, where_vdx = where_pf[0], where_sf[0]
                    if (where_vdx - where_udx) == 1:
                        pf_here, sf_here = where_udx, where_vdx
            elif pf_idx == 5: # uv_2
                if where_rdx and where_idx:
                    where_udx = [idx for idx, value in enumerate(where_pf) if (value > where_rdx[0]) and (value < where_idx[0])]
                    where_vdx = [idx for idx, value in enumerate(where_sf) if (value > where_rdx[0]) and (value < where_idx[0])]
                    if where_udx and where_vdx:
                        pf_here, sf_here = where_pf[where_udx[0]], where_sf[where_vdx[0]]
                elif where_rdx and not where_idx:
                    check_udx = [idx for idx, value in enumerate(where_pf) if (value - where_rdx[0]) == 1]
                    check_vdx = [idx for idx, value in enumerate(where_sf) if (value - where_rdx[0]) == 2]
                    if check_udx and check_vdx:
                        pf_here, sf_here = where_pf[check_udx[0]], where_sf[check_vdx[0]] 
                elif where_idx and not where_rdx:
                    check_udx = [idx for idx, value in enumerate(where_pf) if (where_idx[0] - value) == 2]
                    check_vdx = [idx for idx, value in enumerate(where_sf) if (where_idx[0] - value) == 1]
                    if check_udx and check_vdx:
                        pf_here, sf_here = where_pf[check_udx[0]], where_sf[check_vdx[0]]
                else:
                    # [v - u - v] condition; at least two v filters
                    if len(where_sf) > 1:
                        for uu_idx in where_pf:
                            check_middle = list(np.array(uu_idx) - np.array(where_sf))
                            # find (1, -1) values
                            check_pre2 = [idx for idx, value in enumerate(check_middle) if value == 2]
                            check_pre = [idx for idx, value in enumerate(check_middle) if value == 1]
                            check_post = [idx for idx, value in enumerate(check_middle) if value == -1]
                            if check_pre and check_post and not check_pre2:
                                pf_here, sf_here = uu_idx, uu_idx+1
            elif pf_idx == 9: # uv_3
                if where_idx:
                    where_udx = [idx for idx, value in enumerate(where_pf) if value > where_idx[0]]
                    where_vdx = [idx for idx, value in enumerate(where_sf) if value > where_idx[0]]
                    if where_udx and where_vdx:
                        pf_here, sf_here = where_pf[where_udx[0]], where_sf[where_vdx[0]]
                else:
                    where_udx, where_vdx = where_pf[-1], where_sf[-1]
                    if (where_vdx - where_vdx) == 1:
                        pf_here, sf_here = where_udx, where_vdx
        if verbose:
            print(group, gidx, pf_idx, "[", pf, sf, "] pairs", "--> ", pf_here, sf_here)
 
        # phi-statistics
        if pf_here >= 0 and sf_here >= 0:
            t_pair = (lc['date'][gidx[sf_here]] - lc['date'][gidx[pf_here]])*1440  # delta mjd (unit: minutes)
            if (sf_here - pf_here) == 1 and t_pair < 2.1:
                # mag and mag_err
                pf_mag = lc['mag_psf'][gidx[pf_here]]
                pf_err = lc['e_mag_psf'][gidx[pf_here]]
                sf_mag = lc['mag_psf'][gidx[sf_here]]
                sf_err = lc['e_mag_psf'][gidx[sf_here]]
                if np.isnan(sf_mag) or np.isnan(pf_mag):
                    phi_pair = 0.
                    pp_pair = -1
                    if np.isnan(pf_mag):
                        dm_pf = 0.
                    if np.isnan(sf_mag):
                        dm_sf = 0.
                else:
                    phi_pair = (pf_mag - pf_mean) / pf_err * (sf_mag - sf_mean) / sf_err
                    if (pf_mag - pf_mean) < 0 and (sf_mag - sf_mean) < 0:	# flare candidate
                        pp_pair = 1
                    elif (pf_mag - pf_mean) < 0 and (sf_mag - sf_mean) > 0: # null, primary filter < 0
                        pp_pair = 2
                    elif (pf_mag - pf_mean) > 0 and (sf_mag - sf_mean) > 0: # eclipse candidate
                        pp_pair = 3
                    elif (pf_mag - pf_mean) > 0 and (sf_mag - sf_mean) < 0: # null, secondary filter < 0
                       pp_pair = 4
                    dm_pf = round(pf_mean - pf_mag, 3)	# pf_dm
                    dm_sf = round(sf_mean - sf_mag, 3)	# sf_dm
                # 20180511 - add flag (nimaflags) for each pair
                pp_flag = np.sum([lc['nimaflags'][gidx[pf_here]], lc['nimaflags'][gidx[sf_here]]])
    else:
        if verbose2:
            print(group, gidx, pf_idx, "[", pf, sf, "] pairs", "--> ", pf_here, sf_here)
    return pp_pair, np.round(phi_pair,2), t_pair, dm_pf, dm_sf, pp_flag

def get_phi_initial(group, gidx, qmean, lc, filt_pair, gmode):
    """
    get initial PHI_{FilterFilter} values
    add a gmode option: 12-Feb-2020 SWC

    input: [3, 4, 5, 6], [21, 22, 23, 24], qmean, lc, filt_pair (1 to 6), ms
    """
    t_pair = 0
    # 2024-08-29 use fake phi value for sw_test
    phi_pair = 1.1
    pf = filt_pair                # primary filter
    sf = filt_pair + 1	          # secondary filter
    pf_mean = qmean[filt_pair-1]
    sf_mean = qmean[filt_pair]
    if pf in group and sf in group and pf_mean > 0. and sf_mean > 0.:
        # mag, mag_err, and delmjd
        # REMINDER: index() returns the lowest index in list
        pf_mag = lc['mag_psf'][gidx[group.index(pf)]]
        pf_err = lc['e_mag_psf'][gidx[group.index(pf)]]
        sf_mag = lc['mag_psf'][gidx[group.index(sf)]]
        sf_err = lc['e_mag_psf'][gidx[group.index(sf)]]
        t_pair = (lc['date'][gidx[group.index(sf)]] - lc['date'][gidx[group.index(pf)]])*1440 # unit: minutes

        # 2024-09-04: Nan value
        if np.isnan(sf_mag) or np.isnan(pf_mag):
            phi_pair = 0.
        else:
            # 'fs', 'ms_gr', and 'ms_iz' cases
            # 2024-06-03: added 'ms_uv300, ms_uv400' and 'std' cases 
            if gmode != 'ms': 
                if pf_err > 0 and sf_err > 0:
                    phi_pair = (pf_mag - pf_mean) / pf_err * (sf_mag - sf_mean) / sf_err
                else:
                    phi_pair = 0.
            else: # 'ms' olnly case [1, 2, 3, 4, 1, 2, 5, 6, 1, 2] three uv pairs + gr, ri, iz pairs
                # this is not proper estimation for uv_1, uv_2, and uv_3 pairs, but it's usefule to find an outlier epoch?
                if t_pair < 2.1 and (group.index(sf) - group.index(pf)) == 1:
                    if pf_err > 0 and sf_err > 0:
                        phi_pair = (pf_mag - pf_mean) / pf_err * (sf_mag - sf_mean) / sf_err
                    else:
                        phi_pair = 0.
    return np.round(phi_pair,3) #, t_pair
	
def get_sw_score(sw_arr, verbose=False):
    """
    do a simple statistical test for small number sample.
    for unbiased estimator: dof = 1, ?? dof = 0
    """

    # Define a dictionary to map filt_pair values to filter names
    filter_names = {
        1: 'uv',
        2: 'vg',
        3: 'gr',
        4: 'ri',
        5: 'iz'
    }

    all_token = []
    all_factor = []

    final_token = 0	# default
    stack = []
    ncase = int(np.max(np.unique(sw_arr[:,0])))
    
    # filter pairs, i.e., uv-vg-gr-ri-iz        
    for filt_pair in range(1,6):
        control = sw_arr[(sw_arr[:,0] == 0) & (sw_arr[:,filt_pair+1] > 0)][:,filt_pair+1]
        if len(control) > 1:
            wc = np.sqrt(np.sum(np.power(control, 2)) / (len(control) - 1))
        elif len(control) == 1:
            wc = np.sqrt(np.sum(np.power(control, 2)) / len(control))   # meaningless
        else:
            wc = 0.
        stack.append(wc)
        if verbose:
            print("=============")
            print(f"++ Case 0, {filter_names[filt_pair]}, {np.round(wc,1)}, {control}")  # ",(",len(control),")"
        for case in range(ncase):
            subset = sw_arr[(sw_arr[:,0] == case + 1) & (sw_arr[:,1] != case + 1) & (sw_arr[:,filt_pair+1] > 0)][:,filt_pair+1]

            if len(subset) > 1:
                w = np.sqrt(np.sum(np.power(subset, 2)) / (len(subset) - 1))
            elif len(subset) == 1:
                w = np.sqrt(np.sum(np.power(subset, 2)) / len(subset))  # meaningless
            else:
                w = 0.                
            stack.append(w)
            if verbose:
                print(f"++ Case {case+1}, {filter_names[filt_pair]}, {np.round(w,1)}, {subset}")
        n_stack = np.array(stack)

        # find minimum dispersion of phi distribution per each filter (per each observation block)
        xm = np.ma.masked_invalid(ma.masked_where(n_stack==0, n_stack))
        if xm.count() > 0:
            token = np.argmin(xm)
            factor = np.max(xm.compressed()) - np.min(xm.compressed())
        else:
            token = 0
            factor = 0.
        all_token.append(token)
        all_factor.append(factor)
        if verbose:
            print(xm, token, factor)
        stack = []
    if verbose:
        print("=============")
        print(all_token, all_factor)

    # get a token, _v5 is too simple to find a correct answer
    #_v5.1 adopts a reducing factor, but I'm not fully satisfied it.
    final_token = 0
    factor_max = 0
    for case in range(ncase + 1):
        token_mask = (np.asarray(all_token) == case)
        factor_sum = np.sum(np.asarray(all_factor)[token_mask])
        if factor_max < factor_sum:
            factor_max = factor_sum
            final_token = case
            if verbose:
                print(final_token, factor_max)
    if abs(factor_max) < 1.:
        final_token = 0
    if verbose:
        print(all_token, final_token)
    return final_token

def lc_plot(rawlc, qmean, qstd, file_idx):
    """
    lc_plot
    """
#    pngfile = str(file_idx) + '.png'
    
    plt.figure(figsize=(12,6))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("MJD", fontsize=20)
    plt.ylabel("DR4 MAG", fontsize=20)
    plt.title("%s" % file_idx, fontsize=20)
    # for plt.plot
    lc = rawlc[(rawlc['nimaflags']== 0) & (rawlc['flags']<4)]
    t_lc = lc['date'] #:,0]
    m_lc = lc['mag_psf'] #:,1]
    e_lc = lc['e_mag_psf']
    f_lc = lc['filter'] #[int(filt_str.decode('utf-8')) for filt_str in lc['filter']]   #:,3]
    # for scatter
    lc = rawlc #[(rawlc['nimaflags']== 0)] # & (rawlc['flags']<4)]
    t_rawlc = lc['date'] #:,0]
    m_rawlc = lc['mag_psf'] #:,1]
    e_rawlc = lc['e_mag_psf']
    f_rawlc = lc['filter'] #  [int(filt_str.decode('utf-8')) for filt_str in lc['filter']]
    color=['r', 'b', 'g', 'm', 'c', 'y']
#   shape=['s', '^', 'o', 'D', '+', 'x', '1', 'h', 'p']
    # for error plot
    err = rawlc[(rawlc['nimaflags']>0) | (rawlc['flags']>=4)]
    t_err = err['date']
    m_err = err['mag_psf']
    e_err = err['e_mag_psf']

#    filters = ['u', 'v', 'g', 'r', 'i', 'z']
    j = 0
    for i in range(1,7):
        t_mask = (f_rawlc == str(i)) #[(f_raw == i) for f_raw in f_rawlc]
        if len(m_rawlc[t_mask]) > 0:
            plt.scatter(t_rawlc[t_mask],m_rawlc[t_mask], color=color[j%len(color)],s=120)
#            t_masktwo = [(f_ == i) for f_ in f_lc]
            t_masktwo = (f_lc == str(i)) #[(f_ == i) for f_ in f_lc]
            plt.plot(t_lc[t_masktwo],m_lc[t_masktwo], color=color[j%len(color)])
            plt.errorbar(t_lc[t_masktwo],m_lc[t_masktwo], yerr=e_lc[t_masktwo], color=color[j%len(color)])
            plt.axhline(qmean[i-1], color=color[j%len(color)], linestyle='--')
#            plt.fill_between(t_lc[t_mask], qmean[i-1] + abs(qstd[i-1]), qmean[i-1] - abs(qstd[i-1]), facecolor=color[i%len(color)],alpha=0.2)
        j += 1
#    mode_mask = lc[lc['exp_time'] == 100]['date']
#    for mode in mode_mask:
#        plt.axvline(mode, color='k', linestyle='--')
    plt.gca().invert_yaxis()
    # error plot
    # https://matplotlib.org/3.1.0/gallery/color/named_colors.html
    plt.scatter(t_err, m_err, color='darkgray', s=70) #, marker="x")
#    plt.scatter(t_err, m_err, color='k', s=120, marker="x")

#    plt.axvline(tout, color='k', linestyle='-') #, zorder=2)
    plt.show()
#    plt.savefig(pngfile)
    return

#def fvarindex(lc_bigfile, quiet_log='', mosaic_log='', verbose=False):
def fvarindex(lc_bigfile, verbose=False, verbose2=False):
    #Append current directory for python libraries
    sys.path.append('./')

    #Check initial values.
    if lc_bigfile == '':
        print('#ERROR : Check the lchuck file')
        sys.exit()
    elif lc_bigfile != '':
#        with fits.open(lc_bigfile) as bigchuck:
#            lcdump = bigchuck[1].data
        lcdump = Table(fits.getdata(lc_bigfile, 1))

    # read object_id & prior mag info.
    quiet_log = './quiet_uvgriz_dr4_with_GaiaDR3.fits'
    if quiet_log != '':
        with fits.open(quiet_log) as qlist:
            prior_dr4 = qlist[1].data
    # read mosaic file
    mosaic_log = './mosaic_x0_y0.log' 
    if mosaic_log != '':   
        # Use Astropy to read a CSV file
        mosaic_x0y0 = ascii.read(mosaic_log, format='csv')

    # Output file saved in the same folder
    # 'fs' (five filter pairs) [u-v-g-r-i-z] +  'ms_gr' [g-r] and 'ms_iz' [i-z] (one filter pair)
    # 'ms_uv300', 'ms_uv400' (one filter pair), and 'std' (five filter pairs) [u-v-g-r-i-z] 

    # 2024-06-25 SWC Opens both files at the start using a with statement to ensure they are properly closed after the loop finishes
    outname_fs = str(lc_bigfile).strip('.fits') + '.fs'
    # only for 'ms': [u-v-g-r-u-v-i-z-u-v]
    outname_ms = str(lc_bigfile).strip('.fits') + '.ms'

    # Open both files at the beginning of the loop
    with open(outname_fs, 'w') as fsout, open(outname_ms, 'w') as msout:
        # Write headers
        fsout.write('#file_idx mode gc mjd phi_uv phi_vg phi_gr phi_ri phi_iz t_uv t_vg t_gr t_ri t_iz pp_uv pp_vg pp_gr pp_ri pp_iz umag urms vmag vrms gmag grms rmag rrms imag irms zmag zrms dm_u dm_v dm_g dm_r dm_i dm_z flag_uv flag_vg flag_gr flag_ri flag_iz mx_img my_img' + '\n')
        # fix p_zu -> pp_zu
        msout.write('#file_idx mode gc mjd phi_uv_1 phi_vg phi_gr phi_ru phi_uv_2 phi_vi phi_iz phi_zu phi_uv_3 t_uv_1 t_vg t_gr t_ru t_uv_2 t_vi t_iz t_zu t_uv_3 pp_uv_1 pp_vg pp_gr pp_ru pp_uv_2 pp_vi pp_iz pp_zu pp_uv_3 umag urms vmag vrms gmag grms rmag rrms imag irms zmag zrms dm_u_1 dm_v_1 dm_g dm_r dm_u_2 dm_v_2 dm_i dm_z dm_u_3 dm_v_3 flag_uv_1 flag_vg flag_gr flag_ru flag_uv_2 flag_vi flag_iz flag_zu flag_uv_3 mx_img my_img' + '\n')

        # remove duplicated entries
        # header: [object_id,date,mag_psf,e_mag_psf,filter,flags,nimaflags,ccd,x_img,y_img,exp_time]
        object_id = np.unique(lcdump['object_id']) #, return_index=True) 

        iplot = False
        for i in range(len(object_id)):
        #for i in range(1):
            # flare example: 13941 
            # full colour sequence of ms mode: 1802
            # special condition: [iziz] sequence within 10 minutes: 87920199, 259222051
            # weire uv paris: 67637471
            file_idx = object_id[i] # file_idx is object_id in DR3

            # header: [object_id, u_psf, v_psf, g_psf, r_psf, i_psf, z_psf, idseq] where idseq is retracted
            pick_prior = prior_dr4[prior_dr4['object_id'] == file_idx]
            if len(pick_prior) == 0:
                print('#PASS: ' + str(file_idx) + ' > python fvarindex_v8.py -f ' + lc_bigfile)
                continue

            print('#DR4: ' + str(file_idx) + ' > python fvarindex_v8.py -f ' + lc_bigfile)
            # extract a lightcurve sorted by mjd; apply additional filter of nimaflags == 0?
            rawlc = np.sort(np.array(lcdump[lcdump['object_id'] == file_idx]), order='date')
            # 2024-08-29 Remove data points with flags >= 4
            lc = rawlc[(rawlc['flags']<4)]

            # get initial mean magnitudes from master.dr4
            qmean, qstd = get_qmean(pick_prior, lc)
            if verbose:
                print(file_idx, qmean[0], qmean[1], qmean[2], qmean[3], qmean[4], qmean[5])

            # get obervation blocks under mixture of two survey modes, for example,
            # main_lc = lc[lc['exp_time'] ==100]
            # shallow_lc = lc[lc['exp_time'] <100]
            # 2024-06-03 use get_DR4group_mixture instead of get_group_mixture function
            all_group, all_gidx, all_tblk, all_gc, all_mode = get_DR4group_mixture(lc, verbose=True, verbose2=False)

            # original light curve and prior qmag
            if iplot:
                lc_plot(lc, qmean, qstd, file_idx)

            # check single-epoch observation to pass (# of gc = 1)
            if len(all_group) <= 1:
                continue

            # block iteration: "j" = observation block number, including two survey modes
            sw_test = []
            for j in range(len(all_gc)):
                # get weighted mean after removing a specific epoch
                tblk = all_tblk[j]
                wmean, wstd = get_wmean(lc, tblk, qmean, qstd, 1, verbose=False)
                if verbose:
                    print(all_group[j], all_gidx[j], all_gc[j], all_mode[j])  # group, gidx
                    # e.g., 13941 - [3, 4, 5, 6] [0, 1, 2, 3] 1 fs

                # the leave-one-out cross-validation algorithm to estimate the quiescent magnitudes of M dwarfs
                # simply check only five filter pairs for each obs. block
                # Phi_[uv, vg, gr, ri, iz] is always relevant for 'fs' mode
                # Phi_[gr or iz] is also relevant for 'ms_gr' and 'ms_iz' modes
                # this scheme is a bit tricky for 'ms' mode. 
                for gc in all_gc:
                    group = all_group[gc-1]
                    gidx = all_gidx[gc-1]
                    gmode = all_mode[gc-1]
                    # for control group
                    if j == 0:
                        t_, phi_ = np.zeros(5), np.zeros(5)
                        for filt_pair in range(1,6):
                            # SWC 2024-09-26 bug fixed
                            #phi_pair = get_phi_initial(group, gidx, qmean, lc, filt_pair, gmode)
                            phi_pair = get_phi_initial(group, gidx, wmean, lc, filt_pair, gmode)
                            phi_[filt_pair-1] = phi_pair
                        sw = [j, gc, phi_[0], phi_[1], phi_[2], phi_[3], phi_[4]] 
                        if verbose2:
                            print("---FILTER---", sw)
                        sw_test.append(sw)

                    # for treatment groups
                    t_, phi_ = np.zeros(5), np.zeros(5)
                    for filt_pair in range(1,6):	
                        phi_pair = get_phi_initial(group, gidx, wmean, lc, filt_pair, gmode)
                        phi_[filt_pair-1] = phi_pair
                    sw = [j+1, gc, phi_[0], phi_[1], phi_[2], phi_[3], phi_[4]] 
                    if verbose2:
                        print("---FILTER---", sw)
                    sw_test.append(sw)
            # 52344021: bad case in DR1.1? Is v5.2 function stable enough?
            qtoken = get_sw_score(np.array(sw_test), verbose=False)

            # final phi-statistics
            if qtoken == 0:
                qmean_values, wmean_values = qmean, wmean
                qstd_values, wstd_values = qstd, wstd

                # 2024-06-18: Replace NaN values in QMEAN with corresponding values from WMEAN
                replaced_mean = [wmean_values[i] if np.isnan(qmean) else qmean for i, qmean in enumerate(qmean_values)]
                replaced_std = [wstd_values[i] if np.isnan(qstd) else qstd for i, qstd in enumerate(qstd_values)]

                wmean = replaced_mean #qmean
                wstd = replaced_std #qstd
            else:
                tout = all_tblk[qtoken-1]
                wmean, wstd = get_wmean(lc, tout, qmean, qstd, 2, verbose=False)
            if iplot:
                lc_plot(lc, wmean, wstd, file_idx)

            # UPDATED 20200212 - SWC
            # shallow survey = 'fs'; uvgriz <= 4 min
            # main survey = 'ms'; uvgruvizuv or two gr pairs or up to three iz pairs, and uvgruvizuv
            # DEFINED: four observation blocks: fs, ms, ms_gr, ms_iz
            # UPDATED 20240603 - SWC
            # added standard survey = 'std'; uvgriz <= 8 min
            # DEFINED: seven observation blocks: fs, ms, ms_gr, ms_iz, ms_uv300, ms_uv400, std
            for gc in all_gc:
                group = all_group[gc-1]
                gidx = all_gidx[gc-1]
                tblk = all_tblk[gc-1]
                gmode = all_mode[gc-1]
                # 20180511 - add new features
                m_ccd = int(lc['ccd'][gidx][0])
                mx_img = np.median(lc['x_img'][gidx]) + float(list(mosaic_x0y0[m_ccd-1])[1])
                my_img = np.median(lc['y_img'][gidx]) + float(list(mosaic_x0y0[m_ccd-1])[2])
                if gmode != 'ms':
                    pp_, t_, phi_, dm_, flag_ = np.zeros(5), np.zeros(5), np.zeros(5), np.zeros(6), np.zeros(5)
                    for filt_pair in range(1,6):
                        pp_pair, phi_pair, t_pair, dm_pf, dm_sf, pp_flag = get_phi_updateFS(group, gidx, wmean, lc, filt_pair)
                        pp_[filt_pair-1] = pp_pair
                        t_[filt_pair-1] = t_pair
                        phi_[filt_pair-1] = phi_pair
                        flag_[filt_pair-1] = pp_flag
                        #if dm_pf != 0:
                        dm_[filt_pair-1] = dm_pf
                        #if dm_sf != 0:
                        dm_[filt_pair] = dm_sf

                    if np.sum(t_) > 0.:
                        try:
                            fsout.write('%10s %5s %2d %14.8f %7.1f %7.1f %7.1f %7.1f %7.1f %5.2f %5.2f %5.2f %5.2f %5.2f %1d %1d %1d %1d %1d %7.4f %6.3f %7.4f %6.3f %7.4f %6.3f %7.4f %6.3f %7.4f %6.3f %7.4f %6.3f %4.2f %4.2f %4.2f %4.2f %4.2f %4.2f %3d %3d %3d %3d %3d %4d %4d\n' % (file_idx, gmode, gc, tblk, phi_[0], phi_[1], phi_[2], phi_[3], phi_[4], t_[0], t_[1], t_[2], t_[3], t_[4], pp_[0], pp_[1], pp_[2], pp_[3], pp_[4], wmean[0], wstd[0], wmean[1], wstd[1], wmean[2], wstd[2], wmean[3], wstd[3], wmean[4], wstd[4], wmean[5], wstd[5], dm_[0], dm_[1], dm_[2], dm_[3], dm_[4], dm_[5], flag_[0], flag_[1], flag_[2], flag_[3], flag_[4], mx_img, my_img))
                            fsout.flush()    # Flush output periodically
                        except Exception as e:
                            error_message = "Error writing data for object_id index {}: {}".format(i, e)
                            print(error_message)
                            logging.error(error_message)  # Log the error details to a file
                else:
                    pp_, t_, phi_, dm_, flag_ = np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(10), np.zeros(9)
                    ms_ceq = [1, 2, 3, 4, 1, 2, 5, 6, 1, 2]
                    for filt_pair in range(1, len(ms_ceq)): # filt_pair is index of ms_ceq
                        pp_pair, phi_pair, t_pair, dm_pf, dm_sf, pp_flag = get_phi_updateMS(group, gidx, wmean, lc, filt_pair, verbose=False, verbose2=False)
                        pp_[filt_pair-1] = pp_pair
                        t_[filt_pair-1] = t_pair
                        phi_[filt_pair-1] = phi_pair
                        flag_[filt_pair-1] = pp_flag

                        #if dm_pf != 0:
                        dm_[filt_pair-1] = dm_pf
                        #if dm_sf != 0:
                        dm_[filt_pair] = dm_sf

                    if np.sum(t_) > 0.:
                        try:
                            msout.write('%10s %5s %2d %14.8f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %7.1f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %5.2f %1d %1d %1d %1d %1d %1d %1d %1d %1d %7.4f %6.3f %7.4f %6.3f %7.4f %6.3f %7.4f %6.3f %7.4f %6.3f %7.4f %6.3f %4.2f %4.2f %4.2f %4.2f %4.2f %4.2f %4.2f %4.2f %4.2f %4.2f %3d %3d %3d %3d %3d %3d %3d %3d %3d %4d %4d\n' % (file_idx, gmode, gc, tblk, phi_[0], phi_[1], phi_[2], phi_[3], phi_[4], phi_[5], phi_[6], phi_[7], phi_[8], t_[0], t_[1], t_[2], t_[3], t_[4], t_[5], t_[6], t_[7], t_[8], pp_[0], pp_[1], pp_[2], pp_[3], pp_[4], pp_[5], pp_[6], pp_[7], pp_[8], wmean[0], wstd[0], wmean[1], wstd[1], wmean[2], wstd[2], wmean[3], wstd[3], wmean[4], wstd[4], wmean[5], wstd[5], dm_[0], dm_[1], dm_[2], dm_[3], dm_[4], dm_[5], dm_[6], dm_[7], dm_[8], dm_[9], flag_[0], flag_[1], flag_[2], flag_[3], flag_[4], flag_[5], flag_[6], flag_[7], flag_[8], mx_img, my_img))
                            msout.flush()
                        except Exception as e:
                            error_message = "Error writing data for object_id index {}: {}".format(i, e)
                            print(error_message)
                            logging.error(error_message)  # Log the error details to a file

if __name__=='__main__':
	
    lchuck = '' # small (?) chuck of light-curves
	
    if len(sys.argv) == 1:
        print(help)
        sys.exit()
    #read command line option.
    try:
        optlist, args = getopt.getopt(sys.argv[1:], 'h:f:')
    except getopt.GetoptError as err:
        print(help)
        sys.exit()
	
    for o, a in optlist:
        if o in ('-h'):
            print(help)
            sys.exit()
        elif o in ('-f'):
            lchuck = a
        else:
            continue

    fvarindex(lchuck, verbose=False, verbose2=False)
