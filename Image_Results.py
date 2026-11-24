import numpy as np
import cv2 as cv
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

no_of_dataset = 1


def Image_Results():
    for n in range(1):
        Orig = np.load('Images.npy', allow_pickle=True)
        pre = np.load('Prep_Images.npy', allow_pickle=True)
        Gt = np.load('GroundTruth.npy', allow_pickle=True)
        unet = np.load('unet_segmentation.npy', allow_pickle=True)
        unet3 = np.load('unet3+_segmentation.npy', allow_pickle=True)
        deep = np.load('Deeplabv3_segmentation.npy', allow_pickle=True)
        trans = np.load('Transdeeplabv3_segmentation.npy', allow_pickle=True)
        segment = np.load('segmentation.npy', allow_pickle=True)
        ind = [3639, 3651, 3652, 4917, 4919]
        for j in range(len(ind)):
            original = Orig[ind[j]]
            prepr = pre[ind[j]]
            GroundTruth = Gt[ind[j]].astype('uint8')
            une = unet[ind[j]]
            une3 = unet3[ind[j]]
            deepv = deep[ind[j]]
            Transdeeplabv = trans[ind[j]]
            seg = segment[ind[j]]
            fig, ax = plt.subplots(4, 2)
            plt.suptitle("Image %d" % (j + 1), fontsize=20)
            plt.subplot(2, 4, 1)
            plt.title('Original')
            plt.imshow(original)
            plt.axis('off')
            plt.subplot(2, 4, 2)
            plt.title('Pre Process')
            plt.imshow(prepr)
            plt.axis('off')
            plt.subplot(2, 4, 3)
            plt.title('GroundTruth')
            plt.imshow(GroundTruth)
            plt.axis('off')

            plt.subplot(2, 4, 4)
            plt.title('Unet')
            plt.imshow(une)
            plt.axis('off')

            plt.subplot(2, 4, 5)
            plt.title('Unet3+')
            plt.imshow(une3)
            plt.axis('off')

            plt.subplot(2, 4, 6)
            plt.title('Deeplabv3')
            plt.imshow(deepv)
            plt.axis('off')

            plt.subplot(2, 4, 7)
            plt.title('TransDeeplabv3')
            plt.imshow(Transdeeplabv)
            plt.axis('off')

            plt.subplot(2, 4, 8)
            plt.title('segmentation')
            plt.imshow(seg)
            plt.axis('off')
            path1 = "./Results/Image_Res/Dataset1_image_%s.png" % (j + 1)
            plt.savefig(path1)
            plt.show()
            cv.imwrite('./Results/orig-' + str(j + 1) + '.png', original)
            cv.imwrite('./Results/pre-proc-' + str(j + 1) + '.png', prepr)
            cv.imwrite('./Results/ground-' + str(j + 1) + '.png', GroundTruth)

            cv.imwrite('./Results/unet-' + str(j + 1) + '.png', une)
            cv.imwrite('./Results/une3-' + str(j + 1) + '.png', une3)
            cv.imwrite('./Results/deepv-' + str(j + 1) + '.png', deepv)
            cv.imwrite('./Results/Transdeeplabv-' + str(j + 1) + '.png', Transdeeplabv)
            cv.imwrite('./Results/segment-' + str(j + 1) + '.png', seg)


if __name__ == '__main__':
    Image_Results()  # used
