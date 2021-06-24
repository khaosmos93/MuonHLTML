import sys
import glob
import numpy as np
# import ROOT
import uproot
from HLTIO import preprocess
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import load_svmlight_file
from scipy import sparse
from pathlib import Path
import math
import pandas as pd

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

"""
def Read(path,varlist):
    # Multi-thread
    ROOT.ROOT.EnableImplicitMT()

    f = ROOT.TFile.Open(path)
    t = f.Get("tree")

    mtx = t.AsMatrix(varlist)

    return mtx

def treeToDf(tree):
    npArr, cols = tree.AsMatrix(return_labels=True)
    df = pd.DataFrame(data=npArr, columns=cols)

    return df
"""

def readSeedTree(path,treePath,minpt,maxpt,isB):
    # ROOT.ROOT.EnableImplicitMT()
    # f = ROOT.TFile.Open(path)
    # tree = f.Get()
    # df = treeToDf(tree)

    tree = uproot.open(path)[treePath]
    df = tree.arrays(library='pd')

    # df = df[ df['truePU'] > 180. ]
    # df = df[ df['dR_minDRL1SeedP_AtVtx']   >= 0. ]
    # df = df[ df['dR_minDPhiL1SeedX_AtVtx'] >= 0. ]
    # df = df[ df['dR_minDRL2SeedP']         >= 0. ]
    # df = df[ df['dR_minDPhiL2SeedX']       >= 0. ]
    # df = df[ df['dR_L1TkMuSeedP']          >= 0. ]
    # df = df[ df['dR_minDRL1SeedP_AtVtx']   < 9999. ]
    # df = df[ df['dR_minDPhiL1SeedX_AtVtx'] < 9999. ]
    # df = df[ df['dR_minDRL2SeedP']         < 9999. ]
    # df = df[ df['dR_minDPhiL2SeedX']       < 9999. ]
    # df = df[ df['dR_L1TkMuSeedP']          < 9999. ]

    df = df[ df['gen_pt'] < maxpt ]
    df = df[ df['gen_pt'] > minpt ]
    if isB:
        df = df[ ( (df['tsos_eta'] < 0.9) & (df['tsos_eta'] > -0.9) ) ]
    else:
        df = df[ ( (df['tsos_eta'] > 0.9) | (df['tsos_eta'] < -0.9) ) ]

    return preprocess.getNclass(df)

def readMinSeeds(dir,treePath,minpt,maxpt,isB):
    filelist = glob.glob(dir)
    full = pd.DataFrame()
    y = np.array([]).reshape(0,)
    n = np.array([0,0,0,0])
    cut = 500000

    nfile = 0
    for path in filelist:
        print('Processing %dth file %s ...' % (nfile, path) )
        if np.all( n >= cut ):
            continue

        notBuilt, combi, simMatched, muMatched = readSeedTree(path,treePath,minpt,maxpt,isB)
        subset = pd.DataFrame()
        n_ = np.array([0,0,0,0])
        y_ = np.array([]).reshape(0,)
        if n[0] < cut:
            subset = subset.append(notBuilt,ignore_index=True)
            y_ = np.hstack( ( y_, np.full(notBuilt.shape[0],0) ) )
            n_[0] = notBuilt.shape[0]
        if n[1] < cut:
            subset = subset.append(combi,ignore_index=True)
            y_ = np.hstack( ( y_, np.full(combi.shape[0],1) ) )
            n_[1] = combi.shape[0]
        if n[2] < cut:
            subset = subset.append(simMatched,ignore_index=True)
            y_ = np.hstack( ( y_, np.full(simMatched.shape[0],2) ) )
            n_[2] = simMatched.shape[0]
        if n[3] < cut:
            subset = subset.append(muMatched,ignore_index=True)
            y_ = np.hstack( ( y_, np.full(muMatched.shape[0],3) ) )
            n_[3] = muMatched.shape[0]

        full = full.append(subset, ignore_index=True)
        n += n_
        y = np.hstack( (y,y_) )

        full = preprocess.filterClass(full)
        full['hasL2'] = full.apply(preprocess.hasL2, axis=1)
        full = preprocess.addDistHitL1Tk(full, addAbsDist=False)

        nfile = nfile+1

    print(treePath + ' | %d/%d files | (notBuilt, combi, simMatched, muMatched) = (%d, %d, %d, %d) seeds added' % \
        (nfile, len(filelist), n[0], n[1], n[2], n[3]))

    return full, y

def dumpsvm(x, y, filename):
    dump_svmlight_file(x, y, filename, zero_based=True)

    return

def loadsvm(filepath):
    x, y = load_svmlight_file(filepath)
    x = x.toarray()

    return x, y

"""
def maketest(mu,sigma,name):
    testfile = ROOT.TFile("./data/test"+name+".root","RECREATE")
    tree = ROOT.TTree("tree","test")
    v1 = np.empty((1), dtype="float32")
    v2 = np.empty((1), dtype="float32")
    v3 = np.empty((1), dtype="float32")
    v4 = np.empty((1), dtype="float32")
    v5 = np.empty((1), dtype="float32")
    tree.Branch("v1",v1,"v1/F")
    tree.Branch("v2",v2,"v2/F")
    tree.Branch("v3",v3,"v3/F")
    tree.Branch("v4",v4,"v4/F")
    tree.Branch("v5",v5,"v5/F")

    for i in range(10000):
        v1[0] = np.random.normal(mu,sigma,1)
        v2[0] = np.random.normal(mu,sigma,1)
        v3[0] = np.random.normal(mu,sigma,1)
        v4[0] = np.random.normal(mu,sigma,1)
        v5[0] = np.random.normal(mu,sigma,1)
        tree.Fill()
    testfile.Write()
    testfile.Close()

    return
"""
