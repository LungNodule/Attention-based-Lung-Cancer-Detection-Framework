import numpy as np
from Evaluation import evaluation, net_evaluation
from Global_Vars import Global_Vars
from Model_PROPOSED import Model_PROPOSED
from deeplabv3 import A_TransDeepLabV3Plus


def objfun_cls(Soln):
    Images = Global_Vars.Images
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(Images.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = Images[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = Images[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, predict = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(predict, Test_Target)
            Fitn[i] = 1 / ((Eval[4]) + Eval[8])
        return Fitn
    else:
        learnper = round(Images.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = Images[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = Images[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, predict = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(predict, Test_Target)
        Fitn = 1 / ((Eval[4]) + Eval[8])
        return Fitn


def objfun_Seg(Soln):
    Images = Global_Vars.Images
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Eval, pred = A_TransDeepLabV3Plus(Images, sol)
            predict = pred
            Eval = net_evaluation(predict, Tar)
            Fitn[i] = 1 / (Eval[4] + Eval[6])
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        Eval, pred = A_TransDeepLabV3Plus(Images, sol)
        predict = pred
        Eval = net_evaluation(predict, Tar)
        Fitn = 1 / (Eval[4] + Eval[6])
        return Fitn
