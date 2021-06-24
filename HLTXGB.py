import sys
import multiprocessing
import numpy as np
import pandas as pd
from HLTIO import IO
from HLTIO import preprocess
from HLTvis import vis
from HLTvis import postprocess
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import hyperopt
import pickle
import math
import os
import time
import glob
import tqdm
import gc
import logging

def getBestParam(seedname,tag):
    # note that optimized parameters always depend on the training set i.e. needs to be re-optimized when using different set
    if seedname == 'NThltIterL3OI':
        if tag == 'Barrel':
            return {'eta': 0.06575256610822619, 'gamma': 3.092874778027949, 'lambda': 1.149946617809189, 'max_depth': 10, 'min_child_weight': 1302.7598075776639}
        if tag == 'Endcap':
            return {'eta': 0.0649164398943043, 'gamma': 3.792188468267796, 'lambda': 0.9036363051887085, 'max_depth': 9, 'min_child_weight': 69.87920184424019}
    if seedname == 'NThltIter2FromL1':
        if tag == 'Barrel':
            return {'eta': 0.11370670513701887, 'gamma': 0.8175150663273574, 'lambda': 0.5410160034001444, 'max_depth': 10, 'min_child_weight': 97.10666707815184}
        if tag == 'Endcap': # modified by hand
            return {'eta': 0.33525154433566323, 'gamma': 0.7307823685738455, 'lambda': 0.31169463543440357, 'max_depth': 10, 'min_child_weight': 148.29348974514608}
            # return {'eta': 0.05, 'gamma': 5.0, 'lambda': 2.0, 'max_depth': 8, 'min_child_weight': 1000.0}

    raise NameError('Please check seedname or tag!')

    return

def objective(params,dTrain):
    param = {
        'max_depth': int(params['max_depth']),
        'eta': params['eta'],
        'gamma': params['gamma'],
        'lambda': params['lambda'],
        'min_child_weight':params['min_child_weight'],
        'objective':'multi:softprob',
        'num_class': 4,
        'subsample':0.5,
        'eval_metric':'mlogloss',
        'tree_method':'gpu_hist',
        'nthread':4
    }

    xgb_cv = xgb.cv(dtrain=dTrain,nfold=5,num_boost_round=200,metrics='mlogloss',early_stopping_rounds=20,params=param)

    return xgb_cv['test-mlogloss-mean'].min()

def doTrain(version, seed, seedname, tag, doLoad, stdTransPar=None):
    plotdir = 'plot_'+version
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    colname = list(seed[0].columns)
    print(colname)
    print(seedname+"|"+tag + r' C0: %d, C1: %d, C2: %d, C3: %d' % \
        ( (seed[1]==0).sum(), (seed[1]==1).sum(), (seed[1]==2).sum(), (seed[1]==3).sum() ) )

    if doLoad :
        print("doLoad means you are attempting to load a model instead of train. Did you mean doXGB?")
        return

    x_train, x_mean, x_std = preprocess.stdTransform(seed[0])
    with open("scalefiles/%s_%s_%s_scale.txt" % (version, tag, seedname), "w") as f_scale:
        f_scale.write( "%s_%s_%s_ScaleMean = %s\n" % (version, tag, seedname, str(x_mean.tolist())) )
        f_scale.write( "%s_%s_%s_ScaleStd  = %s\n" % (version, tag, seedname, str(x_std.tolist())) )
        f_scale.close()

    y_wgtsTrain, wgts = preprocess.computeClassWgt(seed[1])
    dtrain = xgb.DMatrix(seed[0], weight=y_wgtsTrain, label=seed[1], feature_names=colname)

    weightSum = np.sum(y_wgtsTrain)

    param_space = {
        'max_depth': hyperopt.hp.quniform('max_depth',5,10,1),
        'eta': hyperopt.hp.loguniform('eta',-3,1), # from exp(-3) to exp(1)
        'gamma': hyperopt.hp.uniform('gamma',0,10),
        'lambda': hyperopt.hp.uniform('lambda',0,3),
        'min_child_weight': hyperopt.hp.loguniform('min_child_weight',math.log(weightSum/10000),math.log(weightSum/10))
    }

    trials = hyperopt.Trials()

    objective_ = lambda x: objective(x, dtrain)

    best = hyperopt.fmin(fn=objective_, space=param_space, max_evals=256, algo=hyperopt.tpe.suggest, trials=trials)

    with open('model/'+version+'_'+tag+'_'+seedname+'_trial.pkl','wb') as output:
        pickle.dump(trials, output, pickle.HIGHEST_PROTOCOL)

    print('Best parameters for '+version+'_'+tag+'_'+seedname+' are')
    print(best)

    return

def doXGB(version, seed, seedname, tag, doLoad, stdTransPar=None):
    plotdir = 'plot_'+version
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    colname = list(seed[0].columns)
    print(colname)
    print(seedname+"|"+tag + r' C0: %d, C1: %d, C2: %d, C3: %d' % \
        ( (seed[1]==0).sum(), (seed[1]==1).sum(), (seed[1]==2).sum(), (seed[1]==3).sum() ) )

    x_train, x_test, y_train, y_test = preprocess.split(seed[0], seed[1],0.2)

    if doLoad and stdTransPar==None:
        print("doLoad is True but stdTransPar==None --> return")
        return

    if stdTransPar==None:
        x_train, x_test, x_mean, x_std = preprocess.stdTransform(x_train, x_test)
        with open("scalefiles/%s_%s_%s_scale.txt" % (version, tag, seedname), "w") as f_scale:
            f_scale.write( str(x_mean.tolist()) )
            f_scale.write( str(x_std.tolist()) )
            f_scale.close()
    else:
        x_train, x_test = preprocess.stdTransformFixed(x_train, x_test, stdTransPar)

    y_wgtsTrain, y_wgtsTest, wgts = preprocess.computeClassWgt(y_train, y_test)

    dtrain = xgb.DMatrix(x_train, weight=y_wgtsTrain, label=y_train, feature_names=colname)
    dtest  = xgb.DMatrix(x_test,  weight=y_wgtsTest,  label=y_test,  feature_names=colname)

    evallist = [(dtest, 'eval'), (dtrain, 'train')]

    param = getBestParam(seedname,tag)

    param['objective'] = 'multi:softprob'
    param['num_class'] = 4
    param['subsample'] = 0.5
    param['eval_metric'] = 'mlogloss'

    # HERE CPU
    param['tree_method'] = 'gpu_hist'
    param['nthread'] = 4

    num_round = 200

    bst = xgb.Booster(param)

    if doLoad:
        bst.load_model('model/'+version+'_'+tag+'_'+seedname+'.model')
    else:
        bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=20, verbose_eval=50)
        bst.save_model('model/'+version+'_'+tag+'_'+seedname+'.model')

    dTrainPredict    = bst.predict(dtrain)
    dTestPredict     = bst.predict(dtest)

    dTrainPredictRaw = bst.predict(dtrain, output_margin=True)
    dTestPredictRaw  = bst.predict(dtest,  output_margin=True)

    labelTrain       = postprocess.softmaxLabel(dTrainPredict)
    labelTest        = postprocess.softmaxLabel(dTestPredict)

    # -- ROC -- #
    for cat in range(4):
        if ( np.asarray(y_train==cat,dtype=np.int).sum() < 2 ) or ( np.asarray(y_test==cat,dtype=np.int).sum() < 2 ): continue

        fpr_Train, tpr_Train, thr_Train, AUC_Train, fpr_Test, tpr_Test, thr_Test, AUC_Test = postprocess.calROC(
            dTrainPredict[:,cat],
            dTestPredict[:,cat],
            np.asarray(y_train==cat,dtype=np.int),
            np.asarray(y_test==cat, dtype=np.int)
        )
        vis.drawROC( fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, version+'_'+tag+'_'+seedname+r'_logROC_cat%d' % cat, plotdir)
        vis.drawROC2(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, version+'_'+tag+'_'+seedname+r'_linROC_cat%d' % cat, plotdir)
        vis.drawThr(  thr_Train, tpr_Train, thr_Test, tpr_Test,  version+'_'+tag+'_'+seedname+r'_logThr_cat%d' % cat, plotdir)
        vis.drawThr2( thr_Train, tpr_Train, thr_Test, tpr_Test,  version+'_'+tag+'_'+seedname+r'_linThr_cat%d' % cat, plotdir)

        fpr_Train, tpr_Train, thr_Train, AUC_Train, fpr_Test, tpr_Test, thr_Test, AUC_Test = postprocess.calROC(
            postprocess.sigmoid( dTrainPredictRaw[:,cat] ),
            postprocess.sigmoid( dTestPredictRaw[:,cat] ),
            np.asarray(y_train==cat,dtype=np.int),
            np.asarray(y_test==cat, dtype=np.int)
        )
        vis.drawROC( fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, version+'_'+tag+'_'+seedname+r'_logROCSigm_cat%d' % cat, plotdir)
        vis.drawROC2(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, version+'_'+tag+'_'+seedname+r'_linROCSigm_cat%d' % cat, plotdir)
        vis.drawThr(  thr_Train, tpr_Train, thr_Test, tpr_Test,  version+'_'+tag+'_'+seedname+r'_logThrSigm_cat%d' % cat, plotdir)
        vis.drawThr2( thr_Train, tpr_Train, thr_Test, tpr_Test,  version+'_'+tag+'_'+seedname+r'_linThrSigm_cat%d' % cat, plotdir)
    # -- ROC -- #

    # -- Confusion matrix -- #
    confMat, confMatAbs = postprocess.confMat(y_test,labelTest)
    vis.drawConfMat(confMat,   version+'_'+tag+'_'+seedname+'_testConfMatNorm', plotdir)
    vis.drawConfMat(confMatAbs,version+'_'+tag+'_'+seedname+'_testConfMat', plotdir, doNorm = False)

    confMatTrain, confMatTrainAbs = postprocess.confMat(y_train,labelTrain)
    vis.drawConfMat(confMatTrain,   version+'_'+tag+'_'+seedname+'_trainConfMatNorm', plotdir)
    vis.drawConfMat(confMatTrainAbs,version+'_'+tag+'_'+seedname+'_trainConfMat', plotdir, doNorm = False)
    # -- #

    # -- Score -- #
    TrainScoreCat3 = dTrainPredict[:,3]
    TestScoreCat3  = dTestPredict[:,3]

    TrainScoreCat3Sig_Xgb = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]==3 ] )
    TrainScoreCat3Bkg_Xgb = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]!=3 ] )
    vis.drawScore(TrainScoreCat3Sig_Xgb, TrainScoreCat3Bkg_Xgb, version+'_'+tag+'_'+seedname+r'_trainScore_cat3', plotdir)

    TestScoreCat3Sig_Xgb = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]==3 ] )
    TestScoreCat3Bkg_Xgb = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]!=3 ] )
    vis.drawScore(TestScoreCat3Sig_Xgb, TestScoreCat3Bkg_Xgb, version+'_'+tag+'_'+seedname+r'_testScore_cat3', plotdir)

    TrainScoreCat3 = postprocess.sigmoid( dTrainPredictRaw[:,3] )
    TestScoreCat3  = postprocess.sigmoid( dTestPredictRaw[:,3] )

    TrainScoreCat3Sig_Sigm = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]==3 ] )
    TrainScoreCat3Bkg_Sigm = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]!=3 ] )
    vis.drawScore(TrainScoreCat3Sig_Sigm, TrainScoreCat3Bkg_Sigm, version+'_'+tag+'_'+seedname+r'_trainScoreSigm_cat3', plotdir)

    TestScoreCat3Sig_Sigm = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]==3 ] )
    TestScoreCat3Bkg_Sigm = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]!=3 ] )
    vis.drawScore(TestScoreCat3Sig_Sigm, TestScoreCat3Bkg_Sigm, version+'_'+tag+'_'+seedname+r'_testScoreSigm_cat3', plotdir)

    TrainScoreCat3 = dTrainPredictRaw[:,3]
    TestScoreCat3  = dTestPredictRaw[:,3]

    TrainScoreCat3Sig_Raw = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]==3 ] )
    TrainScoreCat3Bkg_Raw = np.array( [ score for i, score in enumerate(TrainScoreCat3) if y_train[i]!=3 ] )
    vis.drawScoreRaw(TrainScoreCat3Sig_Raw, TrainScoreCat3Bkg_Raw, version+'_'+tag+'_'+seedname+r'_trainScoreRaw_cat3', plotdir)

    TestScoreCat3Sig_Raw = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]==3 ] )
    TestScoreCat3Bkg_Raw = np.array( [ score for i, score in enumerate(TestScoreCat3) if y_test[i]!=3 ] )
    vis.drawScoreRaw(TestScoreCat3Sig_Raw, TestScoreCat3Bkg_Raw, version+'_'+tag+'_'+seedname+r'_testScoreRaw_cat3', plotdir)
    # -- #

    # -- Importance -- #
    if not doLoad:
        gain = bst.get_score( importance_type='gain')
        cover = bst.get_score(importance_type='cover')
        vis.drawImportance(gain,cover,colname,version+'_'+tag+'_'+seedname+'_importance', plotdir)
    # -- #

    with open('model/'+version+'_'+tag+'_'+seedname+'_plotObj.pkl','wb') as output:
        pickle.dump([confMat, confMatAbs, TrainScoreCat3Sig_Xgb, TrainScoreCat3Bkg_Xgb, TestScoreCat3Sig_Xgb, TestScoreCat3Bkg_Xgb, TrainScoreCat3Sig_Raw, TrainScoreCat3Bkg_Raw, TestScoreCat3Sig_Raw, TestScoreCat3Bkg_Raw], output, pickle.HIGHEST_PROTOCOL)

    return

def run_quick(seedname, doLoad = False):

    # ntuple_path = 'data/ntuple_1-17.root'
    ntuple_path = '/home/common/TT_seedNtuple_GNN_v200622/ntuple_94.root'

    df_B, df_E = IO.readSeedTree(ntuple_path, 'seedNtupler/'+seedname)

    seed_label_B = (
        df_B.drop(['y_label'], axis=1),
        df_B.loc[:,'y_label'].values
    )

    seed_label_E = (
        df_E.drop(['y_label'], axis=1),
        df_E.loc[:,'y_label'].values
    )

    tag = 'Barrel'
    print("\n\nStart: %s|%s" % (seedname, tag))
    stdTrans = None
    if doLoad:
        scalefile = importlib.import_module("scalefiles."+tag+"_"+seedname+"_scale")
        scaleMean = getattr(scalefile, version+"_"+tag+"_"+seedname+"_ScaleMean")
        scaleStd  = getattr(scalefile, version+"_"+tag+"_"+seedname+"_ScaleStd")
        stdTrans = [ scaleMean, scaleStd ]
    doXGB('vTEST',seed_label_B,seedname,tag,doLoad,stdTrans)

    tag = 'Endcap'
    print("\n\nStart: %s|%s" % (seedname, tag))
    stdTrans = None
    if doLoad:
        scalefile = importlib.import_module("scalefiles."+tag+"_"+seedname+"_scale")
        scaleMean = getattr(scalefile, version+"_"+tag+"_"+seedname+"_ScaleMean")
        scaleStd  = getattr(scalefile, version+"_"+tag+"_"+seedname+"_ScaleStd")
        stdTrans = [ scaleMean, scaleStd ]
    doXGB('vTEST',seed_label_E,seedname,tag,doLoad,stdTrans)

    return

def load(seedname, ntuple_path):
    time_init = time.time()
    df_B, df_E = IO.readSeedTree(ntuple_path, 'seedNtupler/'+seedname)
    out = {
        'seedname': seedname,
        'df_B': df_B,
        'df_E': df_E,
        'time': (time.time()-time_init)
    }
    return out

def run(version, seedname, seed, tag, doLoad = False):
    time_init = time.time()

    stdTrans = None
    if doLoad:
        scalefile = open("scalefiles/"+version+"_"+tag+"_"+seedname+"_scale.txt",'r')
        scaleMean = json.loads(scalefile.readline())
        scaleStd  = json.loads(scalefile.readline())
        stdTrans = [ scaleMean, scaleStd ]

    print("\n\nStart: %s|%s" % (seedname, tag))

    doXGB(version, seed, seedname, tag, doLoad, stdTrans)
    # doTrain(version, seed, seedname, tag, doLoad, stdTrans)

    return seedname, tag, (time.time() - time_init)

if __name__ == '__main__':
    from warnings import simplefilter
    simplefilter(action='ignore', category=FutureWarning)

    # run_quick('NThltIter2FromL1')
    # sys.exit()

    ################################################

    gpu_id = sys.argv[2]
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu_id

    timer = {}

    VER = 'vTEST'
    NSAMPLE = 500000

    ntuple_path = '/home/common/TT_seedNtuple_GNN_v200622/ntuple_*.root'
    all_files = glob.glob(ntuple_path)

    seedlist = ['NThltIterL3OI',
                'NThltIter0','NThltIter2','NThltIter3',
                'NThltIter0FromL1','NThltIter2FromL1','NThltIter3FromL1']
    seedlist = list(sys.argv[1].split(','))

    print('-'*70)
    print(f'Version: {VER}')
    print(f'Input files: {len(all_files)} files from')
    print(f'             {ntuple_path}')
    print(f'N seeds per class: {NSAMPLE}')
    print('Seed types:')
    for seed_name in seedlist:
        print(f'\t{seed_name}')
    print('-'*70)

    jobs_load = [[seed_name, file_path] for seed_name in seedlist for file_path in all_files]
    jobs_load = np.array(jobs_load).T.tolist()
    assert len(jobs_load[0]) == len(jobs_load[1])
    njobs_load = len(jobs_load[0])

    import dask
    from dask.distributed import Client
    from distributed.diagnostics.progressbar import progress
    logger = logging.getLogger("distributed.utils_perf")
    logger.setLevel(logging.ERROR)
    dask.config.set({"temporary-directory": "/home/msoh/dask-temp/"})
    client = Client(processes=True,
                    n_workers=24,
                    threads_per_worker=1,
                    memory_limit='6GB',
                    silence_logs='error')
    print('*'*30)
    print('Dask Client:')
    print(client)
    print('Dashboard: {}'.format(client.dashboard_link))
    print('*'*30)

    print(f'\nLoading ntuples: # jobs = {njobs_load}')
    futures_load = client.map(load, *jobs_load, priority=100)
    progress(futures_load)
    print('>>> done!')
    gc.collect()

    print(f'\nMerging dataframes:')
    time_merge = time.time()
    results_load = {}
    for out in tqdm.tqdm(futures_load):
        res = out.result()
        seedname = res['seedname']

        if seedname not in results_load.keys():
            results_load[seedname] = {'df_B': res['df_B'], 'df_E': res['df_E']}
        else:
            results_load[seedname]['df_B'] = IO.sampleByLabel(results_load[seedname]['df_B'],
                                                              df_add = res['df_B'],
                                                              n = NSAMPLE)
            results_load[seedname]['df_E'] = IO.sampleByLabel(results_load[seedname]['df_E'],
                                                              df_add = res['df_E'],
                                                              n = NSAMPLE)

        if f'[1] Load {seedname} per file' not in timer.keys():
            timer[f'[1] Load {seedname} per file'] = res['time']/float(len(all_files))
        else:
            timer[f'[1] Load {seedname} per file'] += res['time']/float(len(all_files))

    gc.collect()

    workers = [w for w in client.scheduler_info()['workers'].keys()]
    client.retire_workers(workers=workers)
    client.close()
    timer[f'[2] Merge'] = time.time() - time_merge
    print('>>> done!')

    print('\nRunning xgboost:')
    run_list = []
    for seedname, res in results_load.items():

        seed_label_B = (
            IO.dropDummyColumn(res['df_B']).drop(['y_label'], axis=1),
            res['df_B'].loc[:,'y_label'].values
        )

        seed_label_E = (
            IO.dropDummyColumn(res['df_E']).drop(['y_label'], axis=1),
            res['df_E'].loc[:,'y_label'].values
        )

        run_list.append((VER, seedname, seed_label_B, 'Barrel'))
        run_list.append((VER, seedname, seed_label_E, 'Endcap'))

    pool = multiprocessing.Pool(processes=min(16,len(run_list)))
    results_run = pool.starmap(run,run_list)
    pool.close()
    pool.join()
    gc.collect()

    for seedname, tag, time_run in results_run:
        timer[f'[3] Run {seedname} {tag}'] = time_run
    print('>>> done!')

    # -- Timing summary -- #
    print('')
    print('-'*70)
    print(f'Timing summary: {VER}')
    time_total = 0
    for _key, _time in timer.items():
        time_total += _time
        unit = 'sec' if _time < 60. else 'min'
        time = round((_time/60. if _time > 60. else _time), 2)
        print(f'\t{_key}: {time} {unit}')
    unit_total = 'sec' if time_total < 60. else 'min'
    time_total = round((time_total/60. if time_total > 60. else time_total), 2)
    print(f'Total: {time_total} {unit_total}')
    print('-'*70)
    # -- #

    print('Finished')
