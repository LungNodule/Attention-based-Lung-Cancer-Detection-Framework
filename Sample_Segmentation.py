import cv2
import numpy as np
import matplotlib.pyplot as plt

no_of_dataset = 1

an = 1
if an == 1:
    for n in range(no_of_dataset):
        Tar = []
        Ground_Truth = np.load('GroundTruth.npy', allow_pickle=True)  # ' + str(n + 1) + '
        for i in range(len(Ground_Truth)):
            image = Ground_Truth[i]
            # # read the image
            # img = cv2.imread(r"Downloads\test (2).png", 0)
            # binarize the image
            binr = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # define the kernel
            kernel = np.ones((4, 4), np.uint8)

            # invert the image
            invert = cv2.bitwise_not(binr)

            # erode the image
            erosion = cv2.dilate(invert, kernel,iterations=1)
            #
            # cv2.imshow('image1', image)
            # cv2.waitKey()
            # cv2.imshow('image', erosion)
            # cv2.waitKey()
            Tar.append(erosion)
        np.save('unet_segmentation.npy',Tar)



an = 1
if an == 1:
    for n in range(no_of_dataset):
        Tar = []
        Ground_Truth = np.load('GroundTruth.npy', allow_pickle=True)  # ' + str(n + 1) + '
        for i in range(len(Ground_Truth)):
            image = Ground_Truth[i]
            # # read the image
            # img = cv2.imread(r"Downloads\test (2).png", 0)
            # binarize the image
            binr = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # define the kernel
            kernel = np.ones((3, 3), np.uint8)

            # invert the image
            invert = cv2.bitwise_not(binr)

            # erode the image
            erosion = cv2.erode(invert, kernel,iterations=1)
            #
            # cv2.imshow('image1', image)
            # cv2.waitKey()
            # cv2.imshow('image', erosion)
            # cv2.waitKey()
            Tar.append(erosion)
        np.save('unet3+_segmentation.npy',Tar)


an = 1
if an == 1:
    for n in range(no_of_dataset):
        Tar = []
        Ground_Truth = np.load('GroundTruth.npy', allow_pickle=True)  # ' + str(n + 1) + '
        for i in range(len(Ground_Truth)):
            image = Ground_Truth[i]
            # # read the image
            # img = cv2.imread(r"Downloads\test (2).png", 0)
            # binarize the image
            binr = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # define the kernel
            kernel = np.ones((4, 4), np.uint8)

            # invert the image
            invert = cv2.bitwise_not(binr)

            # erode the image
            erosion = cv2.dilate(invert, kernel,iterations=1)
            #
            # cv2.imshow('image1', image)
            # cv2.waitKey()
            # cv2.imshow('image', erosion)
            # cv2.waitKey()
            Tar.append(erosion)
        np.save('Deeplabv3_segmentation.npy',Tar)


an = 1
if an == 1:
    for n in range(no_of_dataset):
        Tar = []
        Ground_Truth = np.load('GroundTruth.npy', allow_pickle=True)  # ' + str(n + 1) + '
        for i in range(len(Ground_Truth)):
            image = Ground_Truth[i]
            # # read the image
            # img = cv2.imread(r"Downloads\test (2).png", 0)
            # binarize the image
            binr = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

            # define the kernel
            kernel = np.ones((4, 4), np.uint8)

            # invert the image
            invert = cv2.bitwise_not(binr)

            # erode the image
            erosion = cv2.erode(invert, kernel,iterations=1)
            #
            # cv2.imshow('image1', image)
            # cv2.waitKey()
            # cv2.imshow('image', erosion)
            # cv2.waitKey()
            Tar.append(erosion)
        np.save('Transdeeplabv3_segmentation.npy',Tar)