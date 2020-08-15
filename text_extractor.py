__author__ = 'saideeptalari'
import numpy as np
import cv2
import os
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.models import load_model
from skimage.filters import threshold_local

from sklearn.neighbors import NearestCentroid, KNeighborsClassifier
from sklearn import svm

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

    def train_model_svm(self, train_file, test_file, image_file):
        trainData, trainLabels = self.load_dataset(train_file)
        print("Size of feature vector", trainData.shape)
        print("Size of label", trainLabels.shape)

        # create a model
        clf = svm.SVC()
        clf.fit(trainData, trainLabels)

        print("Training set score: %f" % clf.score(trainData, trainLabels))

        print(trainData[0, :].shape)
        print("Actual", trainLabels[0])
        print("Predict", clf.predict([trainData[0, :]]))

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
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[1]
        cnts, boxes = sort_contours(cnts)
        avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])  # contourArea for digit approximation

        for c in cnts:
            if cv2.contourArea(c) < avgCntArea / 10:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")  # empty mask for each iteration

            (x, y, w, h) = cv2.boundingRect(c)
            hull = cv2.convexHull(c)
            cv2.drawContours(mask, [hull], -1, 255, -1)  # draw hull on mask
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)  # segment digit from thresh

            digit = mask[y - 8:y + h + 8, x - 8:x + w + 8]  # just for better approximation
            digit = cv2.resize(digit, (28, 28))
            digit = digit.reshape((1, 28*28))
            label = clf.predict(digit)

            cv2.putText(image, str(label), (x + 2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Recognized", image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    def train_model_nearest_centroid(self, train_file, test_file, image_file):
        trainData, trainLabels = self.load_dataset(train_file)
        print("Size of feature vector", trainData.shape)
        print("Size of label", trainLabels.shape)

        # create a model
        # clf = NearestCentroid()
        clf = KNeighborsClassifier()
        clf.fit(trainData, trainLabels)

        # print(trainData[0, :].shape)
        # print("Actual", trainLabels[0])
        # print("Predict", clf.predict([trainData[0, :]]))

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


    def train_model_keras(self, train_file, test_file, out_file):
        trainData,trainLabels = self.load_dataset(train_file)
        print("Size of feature vector", trainData.shape)
        print("Size of label", trainLabels.shape)
        # trainLabels = self.encode(trainLabels)

        # load testing set and encode
        testData, testLabels = self.load_dataset(test_file)
        # testLabels = self.encode(testLabels)

        # convert to float
        trainData = trainData.astype("float32")
        testData = testData.astype("float32")

        # normalize
        trainData /= 255
        testData /= 255

        # create model
        # model = Sequential()
        # model.add(Dense(input_dim=784, output_dim=256, activation='relu', init="normal"))
        # model.add(Dense(output_dim=256, activation='relu', init="normal"))
        # model.add(Dense(output_dim=10, activation="softmax"))

        # compile and fit
        model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
        model.fit(trainData, trainLabels, batch_size=100, nb_epoch=25, verbose=2,
                  validation_data=(testData, testLabels))

        print(model.summary())
        score = model.evaluate(testData, testLabels)
        print('Test cost:', score[0])
        print('Test accuracy:', score[1])

        # save model to disk
        model.save(out_file)

    def recognize_keras(self, image_file):
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
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[1]
        avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])  # contourArea for digit approximation

        digits = []
        boxes = []

        for c in cnts:
            if cv2.contourArea(c) < avgCntArea / 10:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")  # empty mask for each iteration

            (x, y, w, h) = cv2.boundingRect(c)
            hull = cv2.convexHull(c)
            cv2.drawContours(mask, [hull], -1, 255, -1)  # draw hull on mask
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)  # segment digit from thresh

            digit = mask[y - 8:y + h + 8, x - 8:x + w + 8]  # just for better approximation
            digit = cv2.resize(digit, (28, 28))
            boxes.append((x, y, w, h))
            digits.append(digit)

        digits = np.array(digits)
        digits = digits.reshape(-1,784)    #for Multi-Layer-Perceptron
        # digits = digits.reshape(digits.shape[0],28,28,1)    #for Convolution Neural Networks
        labels = self.model.predict_classes(digits)

        #draw bounding boxes and print digits on them
        for (x,y,w,h),label in sorted(zip(boxes,labels)):
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),1)
            cv2.putText(image,str(label),(x+2,y-5),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),2)
            cv2.imshow("Recognized",image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()

    def create_train_and_test_data_file(self, data_folder_path, out_train_file, out_test_file, num_test=10):
        """

        :param data_folder_path:
        :param out_train_file:
        :param out_test_file:
        :param num_test:
        :return:
        """
        SEPARATOR = ","
        f_train = open(out_train_file, "w")
        f_test = open(out_test_file, "w")

        for dirname, dirnames, filenames in os.walk(data_folder_path):
            for subdirname in dirnames:
                # label is name of folder, ie: 'a', 'b',...
                label = subdirname
                # Path to folder of a character in face data
                subject_path = os.path.join(dirname, subdirname)
                # Get list of image file names of a character
                image_files = os.listdir(subject_path)
                # get a list of image in folder of current character for testing
                test_num = []
                while len(test_num) < num_test:
                    rdn = np.random.randint(len(image_files), size=1)[0]
                    if rdn not in test_num:
                        test_num.append(rdn)

                cur_img = 0
                for filename in image_files:
                    # ignore system files like .DS_Store
                    if filename.startswith("."):
                        continue

                    abs_path = "%s/%s" % (subject_path, filename)
                    label_path = "%s%s%s" % (abs_path, SEPARATOR, label)

                    if cur_img in test_num:
                        f_test.write(label_path + '\n')
                    else:
                        f_train.write(label_path + '\n')

                    cur_img += 1

        f_train.close()
        f_test.close()

    def extract_char_and_save(self, data_file, out_file):
        self.train_mat = np.zeros((1, (self.char_size[0]*self.char_size[1])+1), dtype='uint8')
        # read all line of file, each line is an image filename
        file = open(data_file, 'r')
        train_files = file.readlines()
        file.close()

        # let's go through each directory and read images within it
        for line in train_files:
            if line.strip() == "":
                continue
            # extract label number of subject from im_file
            label = int(line[line.rfind(";") + 1])
            # labels.append(label)
            # extract filename from im_file
            image_name = line[:line.rfind(";")]

            # read image
            image = cv2.imread(image_name)
            self.extract_char_from_image(image, label)

        self.save_dataset(out_file)

    def encode(self, y):
        """
        One-hot encodes the labels
        :param y: labels to be encoded
        :return: one-hot encoded labels
        """
        Y = np.zeros((y.shape[0], len(np.unique(y))))
        for i in range(y.shape[0]):
            Y[i, y[i]] = 1
        return Y

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

    def save_dataset(self, output_filename, delimiter=","):
        self.train_mat = np.delete(self.train_mat, 0, 0)
        f_handle = open(output_filename, 'w')
        np.savetxt(f_handle, self.train_mat, fmt='%.6f', delimiter=delimiter)
        f_handle.close()

    def align_image(self, image):
        pass

    def extract_char_from_image(self, image, label):
        # resize and convert to grayscale
        image = resize(image, width=320)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Rectangular kernel with size 5x5
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # apply blackhat and otsu thresholding
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # dilate thresholded image for better segmentation
        thresh = cv2.dilate(thresh, None, iterations=2)
        thresh = cv2.erode(thresh, None, iterations=1)

        # find external contours
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[1]
        avgCntArea = np.mean([cv2.contourArea(k) for k in cnts])  # contourArea for digit approximation
        # sort contours from left to right
        cnts, boxes = sort_contours(cnts)
        digits = []
        boxes = []

        for c in cnts:
            if cv2.contourArea(c) < avgCntArea / 10:
                continue
            mask = np.zeros(gray.shape, dtype="uint8")  # empty mask for each iteration

            (x, y, w, h) = cv2.boundingRect(c)
            hull = cv2.convexHull(c)
            # draw hull on mask with inside is filled
            cv2.drawContours(mask, [hull], -1, 255, -1)
            cv2.imshow("Mask", mask)
            # cv2.waitKey(0)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)  # segment digit from thresh

            char_i = mask[y - 8:y + h + 8, x - 8:x + w + 8]  # just for better approximation
            char_i = cv2.resize(char_i, self.char_size)
            char_i = char_i.reshape((1, self.char_size[0] * self.char_size[1]))
            self.train_mat = np.vstack((self.train_mat, np.hstack((label, char_i[0, :]))))
            cv2.imshow("Crop", char_i)
            cv2.waitKey(0)
            boxes.append((x, y, w, h))
            digits.append(char_i)

        cv2.imshow("Original", image)
        cv2.imshow("Thresh", thresh)

        # draw bounding boxes and print digits on them
        for r in boxes:
            (x, y, w, h) = r
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.imshow("Recognized", image)
            cv2.waitKey(0)

        cv2.destroyAllWindows()


model = CharRecognizer()
# model.train_model_svm("mnist_train.csv", "mnist_test.csv", "numbers.jpg")
# model.train_model_nearest_centroid("mnist_train.csv", "mnist_test.csv", "numbers.jpg")
model.train_model_nearest_centroid("character_new_chars.train", "", "test/b.png")
# model.train_model("mnist_train.csv", "mnist_test.csv", "output/char.model")
# # model.create_train_and_test_data_file('character_data', 'train.txt', 'test.txt', 2)
# # model.extract_data_and_save('train.txt', 'character.train')
# model.train_model_keras("character_new_chars.train", "character_new_chars.test", "output/character_new_chars.model")

# model= CharRecognizer(model_file="output/mnist2.model")
# model.recognize_keras('numbers.jpg')