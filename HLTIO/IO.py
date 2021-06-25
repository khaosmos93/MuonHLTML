import sys
import glob
import numpy as np
import uproot
from HLTIO import preprocess
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from scipy import sparse
from pathlib import Path
import math
import pandas as pd
import tqdm
import gc

def dR(eta1, phi1, eta2, phi2):
    dr = math.sqrt((eta1-eta2)*(eta1-eta2) + (phi1-phi2)*(phi1-phi2))
    return dr

def setEtaPhi(x, y, z):
    perp = math.sqrt(x*x + y*y)
    eta = np.arcsinh(z/perp)
    phi = np.arccos(x/perp)
    return eta, phi

def dphi(phi1, phi2):
    tmpdphi = math.fabs(phi1-phi2)
    if tmpdphi >= math.pi:
        tmpdphi = 2*math.pi - tmpdphi
    return tmpdphi

def readSeedTree(path,treePath, minpt = 0, maxpt = 1e9, eta_bound = 0.9, isGNN = False):
    hasTree = False
    if len(glob.glob(path)) == 1:
        df = (uproot.open(path)[treePath]).arrays(library='pd')
                        # array_cache = '1000 MB',
                        # num_workers = 16)[treePath]
    else:
        df = uproot.concatenate(f'{path}:{treePath}',
                                num_workers = 16,
                                library='pd')

    df = df[ df['gen_pt'] < maxpt ]
    df = df[ df['gen_pt'] > minpt ]
    df.loc[:,'hasL2'] = df.apply(preprocess.hasL2, axis=1)
    df = preprocess.addDistHitL1Tk(df, addAbsDist=False)
    df = preprocess.setClassLabel(df)

    df_B = df[((df['tsos_eta'] < eta_bound) & (df['tsos_eta'] > -eta_bound))].copy()
    df_E = df[((df['tsos_eta'] > eta_bound) | (df['tsos_eta'] < -eta_bound))].copy()

    del df
    gc.collect()

    return preprocess.filterClass(df_B, isGNN), preprocess.filterClass(df_E, isGNN)

def sampleByLabel(df, df_add = None, n = 500000):
    out = pd.DataFrame()
    df_tmp = None
    df_add_tmp = None
    for il in range(4):
        df_tmp = df[df['y_label']==il]
        if df_tmp.shape[0] < n and df_add is not None:
            df_add_tmp = df_add[df_add['y_label']==il]
            df_tmp = df_tmp.append(df_add_tmp, ignore_index=True)
        if df_tmp.shape[0] > n:
            df_tmp = df_tmp.sample(n=min(n,df_tmp.shape[0]), axis=0, random_state=123456)
        out = out.append(df_tmp, ignore_index=True)

    del df, df_add, df_tmp, df_add_tmp
    gc.collect()

    return out

def dropDummyColumn(df):
    df_nunique = df.nunique()
    cols_to_drop = df_nunique[df_nunique == 1].index
    cols_to_drop = (df[cols_to_drop] == -99999.).all(axis=0)
    cols_to_drop = cols_to_drop[cols_to_drop==True].index
    if cols_to_drop.shape[0] > 0:
        df.drop(cols_to_drop,
                axis=1,
                inplace=True)
    return df

def dumpsvm(x, y, filename):
    dump_svmlight_file(x, y, filename, zero_based=True)

    return

def loadsvm(filepath):
    x, y = load_svmlight_file(filepath)
    x = x.toarray()

    return x, y
