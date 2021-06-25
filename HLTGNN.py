import sys, os
import gc
import tqdm
import multiprocessing
import numpy as np
import pandas as pd
from HLTIO import IO
from HLTIO import preprocess
from HLTvis import vis
from HLTvis import postprocess

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn.conv import MessagePassing

class trackletGNN(MessagePassing):
    def __init__(self, in_channels, out_channels, propagate_dimensions, edge_dimensions, **kwargs):
        super(trackletGNN, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.propagate_dimensions = propagate_dimensions
        self.edge_dimensions = edge_dimensions

        self.lin_x = nn.Linear(in_channels, propagate_dimensions)
        self.lin_out = nn.Linear( propagate_dimensions, out_channels )
        self.lin_edge = nn.Linear( edge_dimensions, propagate_dimensions )

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_x.reset_parameters()
        self.lin_out.reset_parameters()
        self.lin_edge.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch=None):
        """"""
        to_prop = self.lin_x(x)
        out = self.propagate(edge_index, x=to_prop, edge_attr=edge_attr)

        return self.lin_out(out)

    def message(self, x_j, edge_attr):
        return x_j * self.lin_edge(edge_attr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                         self.in_channels, self.out_channels)

def expDistance(threeVec1, threeVec2):
    diff = threeVec1 - threeVec2
    distSq = torch.tensor( -np.sum(diff**2) )
    return torch.exp(distSq)

def buildGraph(row,y):
    # scheme e.g. for triplet l1(012)-hit(345) = l1-h1 l2-h2 l3-h3
    x = torch.tensor( [[0], [0], [0], [1], [1], [1]], dtype=torch.float )
    edge_index = torch.tensor([[0, 1, 2, 0, 0, 1, 3, 3, 4],
                               [3, 4, 5, 1, 2, 2, 4, 5, 5]], dtype=torch.long )

    threeVec = []
    threeVec.append( np.array([ row['l1x1'], row['l1y1'], row['l1z1'] ]) )
    threeVec.append( np.array([ row['l1x2'], row['l1y2'], row['l1z2'] ]) )
    threeVec.append( np.array([ row['l1x3'], row['l1y3'], row['l1z3'] ]) )

    threeVec.append( np.array([ row['hitx1'], row['hity1'], row['hitz1'] ]) )
    threeVec.append( np.array([ row['hitx2'], row['hity2'], row['hitz2'] ]) )
    threeVec.append( np.array([ row['hitx3'], row['hity3'], row['hitz3'] ]) )

    edge_attr = torch.tensor([[expDistance(threeVec[0],threeVec[3]),1],
                              [expDistance(threeVec[1],threeVec[4]),1],
                              [expDistance(threeVec[2],threeVec[5]),1],
                              [expDistance(threeVec[0],threeVec[1]),0],
                              [expDistance(threeVec[0],threeVec[2]),0],
                              [expDistance(threeVec[1],threeVec[2]),0],
                              [expDistance(threeVec[3],threeVec[4]),0],
                              [expDistance(threeVec[3],threeVec[5]),0],
                              [expDistance(threeVec[4],threeVec[5]),0]], dtype=torch.float )

    y = torch.full( ( 1, ), y, dtype=torch.long )

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data

def trackletDataset(inputData,y):
    data_list = []

    for idx, row in inputData.iterrows():
    # print('\n>>> Run trackletDataset')
    # for idx, row in tqdm.tqdm(inputData.iterrows()):
        data_list.append( buildGraph(row,y[idx]) )

    return data_list

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = trackletGNN(1, 9, 9, 2)
        self.conv2 = trackletGNN(9, 9, 9, 2)
        self.conv3 = trackletGNN(9, 4, 9, 2)
        self.flatten = nn.Flatten(start_dim=0)
        self.linear1 = nn.Linear(24,8)
        self.linear2 = nn.Linear(8,4)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = x.view(-1,6*4)
        x = self.linear1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear2(x)
        x = x.view(-1,4)

        return F.log_softmax(x, dim=1)

def train(train_loader,model,device,optimizer,wgts):
    model.train()
    loss_all = 0.

    for data in train_loader:
    # print('\n>>> Run train')
    # for data in tqdm.tqdm(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        label = data.y.to(device)
        loss = nn.NLLLoss(weight=torch.tensor(wgts,dtype=torch.float).to(device))
        lossOut = loss(output, data.y)
        lossOut.backward()
        loss_all += data.num_graphs * lossOut.item()
        optimizer.step()

    # print('loss_all = %d' % (loss_all))

    return loss_all / len(train_loader.dataset)

def evaluate(loader,model,device):
    model.eval()

    predictions = np.array([]).reshape(0,4)
    labels = np.array([]).reshape(0,)

    with torch.no_grad():
        for data in loader:

            data = data.to(device)
            pred = np.exp(model(data).detach().cpu().numpy())

            label = data.y.cpu().numpy()
            predictions = np.vstack( ( predictions, pred ) )
            labels = np.hstack( ( labels, label ) )

    return predictions, labels

def GNN(data_list, y, seedname, runname):
    plotdir = 'plot_GNN'
    if not os.path.isdir(plotdir):
        os.makedirs(plotdir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_train, x_test, y_train, y_test = preprocess.split(data_list, y)

    train_loader = DataLoader(x_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(x_test, batch_size=32, shuffle=True)
    model = Net().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    _, __, wgts = preprocess.computeClassWgt(y_train, y_test)

    # for epoch in range(100):
    print('\n>>> Epoch')
    for epoch in tqdm.tqdm(range(10)):
        # print('Epoch = %d' % (epoch))
        train(train_loader,model,device,optimizer,wgts)

    pred_train, y_train = evaluate(train_loader,model,device)
    pred_test, y_test = evaluate(test_loader,model,device)

    lab_train = postprocess.softmaxLabel(pred_train)
    lab_test = postprocess.softmaxLabel(pred_test)

    for cat in range(4):
        if ( np.asarray(y_train==cat,dtype=np.int).sum() < 2 ) or ( np.asarray(y_test==cat,dtype=np.int).sum() < 2 ): continue

        fpr_Train, tpr_Train, thr_Train, AUC_Train, fpr_Test, tpr_Test, thr_Test, AUC_Test = postprocess.calROC(
            pred_train[:,cat],
            pred_test[:,cat],
            np.asarray(y_train==cat,dtype=np.int),
            np.asarray(y_test==cat, dtype=np.int)
        )
        vis.drawROC( fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, runname+'_'+seedname+r'_ROC1_cat%d' % cat, plotdir)
        vis.drawROC2(fpr_Train, tpr_Train, AUC_Train, fpr_Test, tpr_Test, AUC_Test, runname+'_'+seedname+r'_ROC2_cat%d' % cat, plotdir)
        vis.drawThr(  thr_Train, tpr_Train, thr_Test, tpr_Test,  runname+'_'+seedname+r'_Thr1_cat%d' % cat, plotdir)
        vis.drawThr2( thr_Train, tpr_Train, thr_Test, tpr_Test,  runname+'_'+seedname+r'_Thr2_cat%d' % cat, plotdir)

    confMat, confMatAbs = postprocess.confMat(y_test,lab_test)
    vis.drawConfMat(confMat,   runname+'_'+seedname+'_testConfMatNorm', plotdir)
    vis.drawConfMat(confMatAbs,runname+'_'+seedname+'_testConfMat', plotdir, doNorm = False)

    confMatTrain, confMatTrainAbs = postprocess.confMat(y_train,lab_train)
    vis.drawConfMat(confMatTrain,   runname+'_'+seedname+'_trainConfMatNorm', plotdir)
    vis.drawConfMat(confMatTrainAbs,runname+'_'+seedname+'_trainConfMat', plotdir, doNorm = False)

    return

def run(seedname, runname):
    doLoad = False
    isB = ('Barrel' in runname)

    # ntuple_path = 'data/ntuple_1-17.root'
    ntuple_path = '/home/common/TT_seedNtuple_GNN_v200622/ntuple_*.root'

    print("\n\nStart: %s|%s" % (seedname, runname))
    data_y, df_E = IO.readSeedTree(ntuple_path, 'seedNtupler/'+seedname, isGNN = True)
    # data_y = data_y.append(df_E, ignore_index=True)
    data_y = data_y[['y_label', 'nHits',
                     'l1x1','l1y1','l1z1',
                     'hitx1','hity1','hitz1',
                     'l1x2','l1y2','l1z2',
                     'hitx2','hity2','hitz2',
                     'l1x3','l1y3','l1z3',
                     'hitx3','hity3','hitz3']]
                     # 'l1x4','l1y4','l1z4',
                     # 'hitx4','hity4','hitz4']]

    data_y = IO.sampleByLabel(data_y, n = 500000)

    # HERE
    print(data_y)

    data = data_y.drop(['y_label'], axis=1)
    y = data_y.loc[:,'y_label'].values

    del df_E, data_y
    gc.collect()

    # HERE
    # print(data)
    # print(y)

    select = data['nHits']>0
    select_np = select.to_numpy()
    data = data[select]
    y = y[select_np]

    # HERE
    # print(select)
    # print(select_np)
    # print(data)
    # print(y)
    # sys.exit()

    colname = list(data.columns)
    print(colname)
    print(seedname+"|"+runname + r' C0: %d, C1: %d, C2: %d, C3: %d' % \
        ( (y==0).sum(), (y==1).sum(), (y==2).sum(), (y==3).sum() ) )

    data_list = trackletDataset(data, y)

    # HERE
    # print(data_list)

    GNN(data_list,y,seedname,runname)

if __name__ == '__main__':
    run('NThltIter2FromL1','test')





