import os
import cv2
import numpy as np

from sklearn.neighbors import KNeighborsClassifier

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


def sort_contours(cnts, method="left-to-right"):
    """Sort contours

    :param cnts: contours to sort
    :param method: 'left-to-right' or 'right-to-left' or 'top-to-bottom' or 'bottom-to-top'
    :return:
    """
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

class CharRecognizer:
    def __init__(self, model_file=None, char_size=(28, 28)):
        self.train_mat = None
        self.char_size = char_size
        # if model_file is not None:
        #     self.model = load_model(model_file)

    def train_model_nearest_centroid(self, train_file, test_file, image_file):
        trainData, trainLabels = self.load_dataset(train_file)
        print("Size of feature vector", trainData.shape)
        print("Size of label", trainLabels.shape)

        # create a model
        clf = KNeighborsClassifier()
        clf.fit(trainData, trainLabels)

        # read,resize and convert to grayscale
        image = cv2.imread(image_file)
        image = resize(image, width=320)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Rectangular kernel with size 5x5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # apply blackhat and otsu thresholding
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        thresh = cv2.dilate(thresh, None)  # dilate thresholded image for better segmentation

        # find external contours
        cnts, temp = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[1]
        cnts, boxes = sort_contours(cnts)
        avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])  # contourArea for digit approximation

        for c in cnts:
            if cv2.contourArea(c) < avgCntArea / 10:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")  # empty mask for each iteration

            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            hull = cv2.convexHull(c)
            cv2.drawContours(mask, [hull], -1, 255, -1)  # draw hull on mask
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)  # segment digit from thresh

            digit = mask[y - 8:y + h + 8, x - 8:x + w + 8]  # just for better approximation
            digit = cv2.resize(digit, (28, 28))
            digit = digit.reshape((1, 28*28))
            label = clf.predict(digit)
            print(label, chr(label[0]))

            cv2.putText(image, '[' + chr(label[0]) + ']', (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Recognized", image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    def load_dataset(self, path, delimiter=","):
        """
        Loads dataset from a given path and delimiter

        :param path: Path to dataset that needed to be loaded
        :param delimiter: delimiter (default=",")
        :return: dataset,labels
        """
        arr = np.loadtxt(path, delimiter=delimiter, dtype="int32")
        labels = arr[:, 0]
        data = arr[:, 1:]
        # data = data.reshape(-1,28,28)
        return data, labels

model = CharRecognizer()
model.train_model_nearest_centroid("character_new_chars.train", "", "test/c.png")