import os
import numpy as np
import cv2 as cv
from imWeightedThresholdedheq import imWTHeq
import pydicom as dicom
import random as rn
from numpy import matlib
from EBOA import EBOA
from EEO import EEO
from Global_Vars import Global_Vars
from HHOA import HHOA
from LSO import LSO
from Model_BILSTM import Model_BILSTM
from Model_CNN import Model_CNN
from Model_DBN import Model_DBN
from Model_PROPOSED import Model_PROPOSED
from Objective_Function import objfun_cls, objfun_Seg
from PROPOSED import PROPOSED
from Plot_Results import *
from deeplabv3 import A_TransDeepLabV3Plus


## Read Dataset 1 -UnitoChest
an = 0
if an == 1:
    Images = []
    GroundTruth = []
    Path = './Dataset/UnitoChest'
    sub_dir = os.listdir(Path)
    for i in range(len(sub_dir)):
        if sub_dir[i] == 'Images':
            sub_dir_ = os.listdir(Path + '/' + sub_dir[i])
            for j in range(len(sub_dir_)):  # For all Patients
                ## Read Images
                sub_dir__ = os.listdir(Path + '/' + sub_dir[i] + '/' + sub_dir_[j] + '/exam_1/')
                image_name = []
                for k in range(len(sub_dir__)):  # For all Images
                    print(k)
                    image_path = Path + '/' + sub_dir[i] + '/' + sub_dir_[j] + '/exam_1/' + sub_dir__[k]
                    ds = dicom.dcmread(image_path)
                    image = (ds.pixel_array / 13).astype('uint8')
                    image = cv.resize(image, [128, 128])
                    image_name.append(sub_dir__[k].split('.')[0] + '_mask.png')
                    Images.append(image)

                ## Read Masks
                image_name = np.asarray(image_name)
                mask_sub_dir__ = np.asarray(os.listdir(Path + '/' + sub_dir[i + 1] + '/' + sub_dir_[j] + '/exam_1/'))
                for k in range(len(sub_dir__)):  # For all Images
                    if image_name[k] in mask_sub_dir__:
                        image_path = Path + '/' + sub_dir[i + 1] + '/' + sub_dir_[j] + '/exam_1/' + image_name[k]
                        image = cv.imread(image_path)
                        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                        image = cv.resize(image, [128, 128])
                    else:
                        image = np.zeros((128, 128)).astype('uint8')
                    GroundTruth.append(image)

    np.save('Images.npy', Images)
    np.save('GroundTruth.npy', GroundTruth)

### pre-processing ###
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Prep_Images = []
    for i in range(Images.shape[0]):
        print(i + 1, Images.shape[0])
        image = Images[i]
        image_v_heq, _ = imWTHeq(image, r=0.2, v=0.2)  # •	Weighted mean histogram equalization•
        figure_size = 2  # the dimension of the x and yaxis of the kernal.
        Filter_image = cv.blur(image_v_heq, (figure_size, figure_size))  # mean filtering
        Prep_Images.append(Filter_image)
    np.save('Prep_Images.npy', Prep_Images)

# Optimization for segmentation
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Tar = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Images = Images
    Global_Vars.Target = Tar
    Npop = 10
    Chlen = 3  # Here we optimized Hidden Neuron Count, No of epoches, Steps per epoch in Transunet3+
    xmin = matlib.repmat([5, 5, 300], Npop, 1)
    xmax = matlib.repmat([255, 50, 1000], Npop, 1)
    initsol = np.zeros(xmax.shape)
    for p1 in range(Npop):
        for p2 in range(Chlen):
            initsol[p1, p2] = np.random.uniform(xmin[p1, p2], xmax[p1, p2])
    fname = objfun_Seg
    Max_iter = 50

    print("LSO...")
    [bestfit1, fitness1, bestsol1, time1] = LSO(initsol, fname, xmin, xmax, Max_iter)

    print("HHOA...")
    [bestfit2, fitness2, bestsol2, time2] = HHOA(initsol, fname, xmin, xmax, Max_iter)

    print("EEO...")
    [bestfit3, fitness3, bestsol3, time3] = EEO(initsol, fname, xmin, xmax, Max_iter)

    print("EBOA...")
    [bestfit4, fitness4, bestsol4, time4] = EBOA(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]

    np.save('BestSol_seg.npy', sol)

# Segmentation for TransDeepLabV3Plus
an = 0
if an == 1:
    Images = np.load('Images.npy', allow_pickle=True)
    Bestsol = np.load('BestSol_seg.npy', allow_pickle=True)
    Seg = A_TransDeepLabV3Plus(Images, Bestsol)
    np.save('segmentation.npy', Seg)

# OPTIMIZATION FOR CLASSIFICATION
an = 0
if an == 1:
    Images = np.load('segmentation.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Images = Images
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3
    xmin = np.asarray([5, 5, 50]) * np.ones((Npop, 1))
    xmax = np.asarray([255, 50, 250]) * np.ones((Npop, 1))
    initsol = np.zeros(xmin.shape)
    for i in range(xmin.shape[0]):
        for j in range(xmin.shape[1]):
            initsol[i, j] = np.random.uniform(xmin[i, j], xmax[i, j])
    fname = objfun_cls
    max_iter = 50

    print('LSO....')
    [bestfit1, fitness1, bestsol1, Time1] = LSO(initsol, fname, xmin, xmax, max_iter)

    print('HHOA....')
    [bestfit2, fitness2, bestsol2, Time2] = HHOA(initsol, fname, xmin, xmax, max_iter)

    print('EEO....')
    [bestfit3, fitness3, bestsol3, Time3] = EEO(initsol, fname, xmin, xmax, max_iter)

    print('EBOA....')
    [bestfit4, fitness4, bestsol4, Time4] = EBOA(initsol, fname, xmin, xmax, max_iter)

    print('PROPOSED....')
    [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

    sol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]

    np.save('Best_cls_sol.npy', sol)

# KFOLD - Classification
an = 0
if an == 1:
    BestSol = np.load('Best_cls_sol.npy', allow_pickle=True)
    Image = np.load('segmentation.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    K = 5
    Per = 1 / 5
    Perc = round(Image.shape[0] * Per)
    eval = []
    for i in range(K):
        Eval = np.zeros((10, 14))
        Image = Image
        Test_Data = Image[i * Perc: ((i + 1) * Perc), :]
        Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
        test_index = np.arange(i * Perc, ((i + 1) * Perc))
        total_index = np.arange(Image.shape[0])
        train_index = np.setdiff1d(total_index, test_index)
        Train_Data = Image[train_index, :]
        Train_Target = Target[train_index, :]
        for j in range(BestSol.shape[0]):
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Eval[j, :], pred1 = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target, sol[j])
        Eval[5, :], pred2 = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[6, :], pred3 = Model_BILSTM(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[7, :], pred4 = Model_DBN(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[8, :], pred5 = Model_PROPOSED(Train_Data, Train_Target, Test_Data, Test_Target)
        Eval[9, :], pred6 = Eval[4, :]
        eval.append(Eval)
    np.save('Eval_Fold.npy', np.asarray(eval))

plot_results_kfold()
plot_confusion()
Plot_Results_Roc()
plot_results_conv()
Image_Results()
plot_results_dice()

