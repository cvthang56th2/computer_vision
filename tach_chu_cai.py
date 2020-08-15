import numpy as np
import cv2
import argparse
from skimage.filters import threshold_local

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized

def sort_contours(cnts, method = "left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-botom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def extract_letter (image):
    image = resize(image,width=320)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #Rectangular kernel with size 5x5
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

    #apply blackhat and otsu thresholding
    blackhat = cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
    _,thresh = cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
    thresh = cv2.dilate(thresh,None)        #dilate thresholded image for better segmentation

    #find external contours
    cnts,t = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cnts = cnts[1]
    cnts, boxes = sort_contours(cnts)
    
    avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])      #contourArea for digit approximation

    digits = []
    boxes = []


    for c in cnts:
        if cv2.contourArea(c)<avgCntArea/10:
            continue
        mask = np.zeros(gray.shape,dtype="uint8")   #empty mask for each iteration

        (x,y,w,h) = cv2.boundingRect(c)
        hull = cv2.convexHull(c)
        cv2.drawContours(mask,[hull],-1,255,-1)     #draw hull on mask
        mask = cv2.bitwise_and(thresh,thresh,mask=mask) #segment digit from thresh

        digit = mask[y-8:y+h+8,x-8:x+w+8]       #just for better approximation
        digit = cv2.resize(digit,(28,28))
        boxes.append((x,y,w,h))
        digits.append(digit)

    # digits = np.array(digits)
    # model = load_model(args["model"])
    # #digits = digits.reshape(-1,784)    #for Multi-Layer-Perceptron
    # digits = digits.reshape(digits.shape[0],28,28,1)    #for Convolution Neural Networks
    # labels = model.predict_classes(digits)

    cv2.imshow("Original",image)
    cv2.imshow("Thresh",thresh)

    # #draw bounding boxes and print digits on them
    # for (x,y,w,h),label in sorted(zip(boxes,labels)):
    #     cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
    #     cv2.putText(image,str(label),(x+2,y-5),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
    #     cv2.imshow("Recognized",image)
    #     cv2.waitKey(0)

    #draw bounding boxes and print digits on them
    for r in boxes:
        (x,y,w,h) = r
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
        cv2.imshow("Recognized",image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    
filename = 'test_1.jpg'
image = cv2.imread(filename)
extract_letter(image)
