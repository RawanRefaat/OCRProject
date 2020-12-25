import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import cv2
import pandas
import csv
import numpy
from PIL import Image
from sklearn.impute import SimpleImputer
from csv import reader
from skimage import data
from skimage.color import rgb2gray

trainImagesFile= "E:\Ai Project\data\TrainData\csvTrainImages 60k x 784.csv"
trainLabelsFile = "E:\Ai Project\data\TrainData\csvTrainLabel 60k x 1.csv"
testImagesFile = "E:\Ai Project\data\TestData\csvTestImages 10k x 784.csv"
testLabelsFile = "E:\Ai Project\data\TestData\csvTestLabel 10k x 1.csv"
imagePath = "E:\maxresdefault.jpg"


def readImages():
    labelsCount = 10
    digitsList = []
    digit0,digit1,digit2,digit3,digit4,digit5,digit6,digit7,digit8,digit9 = [],[],[],[],[],[],[],[],[],[],
    with open(trainImagesFile, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for i in range(1):
           pixelsRow = next(csv_reader)
           digit0.append(pixelsRow)
           pixelsRow = next(csv_reader)
           digit1.append(pixelsRow)
           pixelsRow = next(csv_reader)
           digit2.append(pixelsRow)
           pixelsRow = next(csv_reader)
           digit3.append(pixelsRow)
           pixelsRow = next(csv_reader)
           digit4.append(pixelsRow)
           pixelsRow = next(csv_reader)
           digit5.append(pixelsRow)
           pixelsRow = next(csv_reader)
           digit6.append(pixelsRow)
           pixelsRow = next(csv_reader)
           digit7.append(pixelsRow)
           pixelsRow = next(csv_reader)
           digit8.append(pixelsRow)
           pixelsRow = next(csv_reader)
           digit9.append(pixelsRow)
    csv_file.close()
    digitsList.append(digit0)
    digitsList.append(digit1)
    digitsList.append(digit2)
    digitsList.append(digit3)
    digitsList.append(digit4)
    digitsList.append(digit5)
    digitsList.append(digit6)
    digitsList.append(digit8)
    digitsList.append(digit8)
    digitsList.append(digit9)


def processMissingValues(trainData):
    meanImputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    meanImputer = meanImputer.fit(trainData)
    trainData = meanImputer.transform(trainData)
    return trainData
  
def preprocessImage(imagePath):
    image = cv2.imread(imagePath)
    mask = np.zeros(image.shape, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,51,9)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,2))
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
       peri = cv2.arcLength(c, True)
       approx = cv2.approxPolyDP(c, 0.04 * peri, True)
       x,y,w,h = cv2.boundingRect(c)
       area = w * h
       ar = w / float(h)
       if area > 1200 and area < 50000 and ar < 6:
          cv2.drawContours(mask, [c], -1, (255,255,255), -1)

    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    result = cv2.bitwise_and(image, image, mask=mask)
    result[mask==0] = (255,255,255) 
    #cv2.imshow('mask', mask)
    cv2.imshow('thresh', thresh)
    cv2.waitKey()
    fig, axes = plt.subplots(1, 2, figsize=(8,4))
    ax = axes.ravel()
    ax[0].imshow(image)
    ax[0].set_title("Before")
    ax[1].imshow(thresh)
    ax[1].set_title("Final")
    fig.tight_layout()
    plt.show()
   
def main():
    trainImages = pandas.read_csv(trainImagesFile, sep=',', header=None)
    trainLabels = pandas.read_csv(trainLabelsFile, sep=',', header=None)
    trainFinalData = pandas.concat([trainLabels, trainImages], axis=1)
    preprocessImage(imagePath)

if __name__ == "__main__":
    main()
       
    



