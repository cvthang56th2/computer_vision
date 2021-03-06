# coding: utf-8
# Face Recognition with OpenCV
import cv2
import os
import math
import numpy as np
from PIL import Image
import dlib

# In[3]: Detect, Align and Crop Face From Image
def Distance(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.sqrt(dx * dx + dy * dy)

def ScaleRotateTranslate(image, angle, center=None, new_center=None, scale=None, resample=Image.BICUBIC):
    if (scale is None) and (center is None):
        return image.rotate(angle=angle, resample=resample)
    nx, ny = x, y = center
    sx = sy = 1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = (scale, scale)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    a = cosine / sx
    b = sine / sx
    c = x - nx * a - ny * b
    d = -sine / sy
    e = cosine / sy
    f = y - nx * d - ny * e

    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=resample)

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

def rect_to_bb(rect):
    """ take a bounding predicted by dlib and convert it
    to the format (x, y, w, h) as we would normally do with OpenCV

    :param rect: rectangle predicted by dlib
    :return: rectangle (x, y, w, h)
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return x, y, w, h

class FaceRecongizerCV:
    def __init__(self, model_name=None, recognizer_type='LBP', detect_type='dlib'):
        self.detector = None
        self.det_type = detect_type
        self.reg_type = recognizer_type

        if self.det_type == 'dlib':
            self.detector = dlib.get_frontal_face_detector()
        else:
            self.detector = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt2.xml')

        if self.reg_type == 'LBP':
            self.recognizer = cv2.face_LBPHFaceRecognizer.create()
        elif self.reg_type == 'EIGEN':
            self.recognizer = cv2.face_EigenFaceRecognizer.create()
        else: # 'FISHER'
            self.recognizer = cv2.face_FisherFaceRecognizer.create()
            # self.recognizer = cv2.face.FisherFaceRecognizer_create()

        if model_name is not None:
            self.recognizer.read(model_name)

        self.train_faces = []  # list to hold all subject faces
        self.train_labels = []  # list to hold labels for all subjects

    def train_recognizer(self, data_folder_path, out_train_file, out_test_file, out_model_file, num_test=10):
        print("Preparing data...")
        self.create_data_file(data_folder_path, out_train_file, out_test_file, num_test)
        self.prepare_training_data(out_train_file)

        # print total faces and labels
        print("Total train faces: ", len(self.train_faces))
        print("Total train labels: ", len(self.train_labels))

        # train our face recognizer of our training faces
        self.recognizer.train(self.train_faces, np.array(self.train_labels))
        self.recognizer.save(out_model_file)

    def test_recognizer(self, test_file):
        true_predicted = 0
        num_test = 0
        print("==========================")
        print("Predicting images in testing set")

        # Read all lines in testing file
        f = open(test_file, 'r')
        test_files = f.readlines()
        f.close()

        # Predict all testing file
        for im_file in test_files:
            # get true label and image filename
            true_label = int(im_file[im_file.rfind(";") + 1])
            image_path = im_file[:im_file.rfind(";")]
            print(image_path)
            image = cv2.imread(image_path)

            # predict
            predicted_img, predicted_label = self.predict(image)
            if predicted_label == -1:
                print("Cannot predict file: " + image_path)
                continue
            if predicted_label == true_label:
                true_predicted += 1
            num_test += 1

            cv2.imshow(str(true_label), cv2.resize(predicted_img, (400, 500)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        print("     Total testing face: " + str(num_test))
        print("     Number of predicted face: " + str(true_predicted))
        print("     Percent of recognition: " + str(100.0 * true_predicted / num_test))
        print("Prediction complete")

    def create_data_file(self, data_folder_path, out_train_file, out_test_file, num_test=10):
        SEPARATOR = ";"
        f_train = open(out_train_file, "w")
        f_test = open(out_test_file, "w")

        for dirname, dirnames, filenames in os.walk(data_folder_path):
            for subdirname in dirnames:
                # our subject directories start with letter 's' so
                # ignore any non-relevant directories if any
                if not subdirname.startswith("s"):
                    continue

                # extract label number of subject from dir_name
                # format of dir name = slabel,
                # so removing letter 's' from dir_name will give us label
                label = int(subdirname.replace("s", ""))

                # Path to folder of a person in face data
                subject_path = os.path.join(dirname, subdirname)
                # Get list of image file names of a person
                image_files = os.listdir(subject_path)
                # get a list of image in folder of current person for testing
                test_num = []
                while len(test_num) < num_test:
                    rdn = np.random.randint(len(image_files), size=1)[0]
                    if rdn not in test_num:
                        test_num.append(rdn)
                print(len(image_files))
                print(test_num)

                cur_img = 0
                for filename in image_files:
                    # ignore system files like .DS_Store
                    if filename.startswith("."):
                        continue

                    abs_path = "%s/%s" % (subject_path, filename)
                    label_path = "%s%s%d" % (abs_path, SEPARATOR, label)

                    if cur_img in test_num:
                        f_test.write(label_path + '\n')
                    else:
                        f_train.write(label_path + '\n')

                    cur_img += 1

        f_train.close()
        f_test.close()

    def prepare_training_data(self, training_filename):
        # read all line of file, each line is an image filename
        file = open(training_filename, 'r')
        train_files = file.readlines()
        file.close()

        self.train_faces = []  # list to hold all subject faces
        self.train_labels = []  # list to hold labels for all subjects
        labels = []

        # let's go through each directory and read images within it
        for line in train_files:
            if line.strip() == "":
                continue
            # extract label number of subject from im_file
            label = int(line[line.rfind(";") + 1])
            labels.append(label)
            # extract filename from im_file
            image_name = line[:line.rfind(";")]

            # read image
            image = cv2.imread(image_name)

            # display an image window to show the image
            cv2.imshow("Training on image...", resize(image, 400))
            cv2.waitKey(100)

            # detect and aligned face
            face, rect = self.detect_and_align_face_dlib(image)

            # for the purpose of this tutorial
            # we will ignore faces that are not detected
            if face is not None:
                self.train_faces.append(face)  # add face to training faces
                self.train_labels.append(label)  # add label tro training face

        cv2.destroyAllWindows()
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return labels

    def detect_and_align_face_dlib(self, img):
        # convert the test image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # let's detect multiscale (some images may be closer to camera than others) images
        # result is a list of faces
        faces = self.detector(gray, 0)

        # if no faces are detected then return original img
        if len(faces) == 0:
            return None, None

        # under the assumption that there will be only one face,
        # extract the face area
        (x, y, w, h) = rect_to_bb(faces[0])

        # get the face part of the image
        roi = gray[y:y + w, x:x + h]

        return roi, (x, y, w, h)
    def detect_and_align_face_dlib_new(self, img):
        # convert the test image to gray image as opencv face detector expects gray images
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # let's detect multiscale (some images may be closer to camera than others) images
        # result is a list of faces
        faces = self.detector(gray, 0)

        # if no faces are detected then return original img
        if len(faces) == 0:
            return None, None

        # under the assumption that there will be only one face,
        # extract the face area
        listXYWH = []
        listRoi = []
        for face in faces:
            (x, y, w, h) = rect_to_bb(face)
            listXYWH.append((x, y, w, h))
            # get the face part of the image
            roi = gray[y:y + w, x:x + h]
            listRoi.append(roi)

        return listRoi, listXYWH

    def draw_rectangle(self, img, rect):
        """ function to draw rectangle on image according to given (x, y) coordinates and given width and heigh

        :param img:
        :param rect:
        :return:
        """
        (x, y, w, h) = rect
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    def draw_text(self, img, text, x, y):
        """ function to draw text on give image starting from passed (x, y) coordinates.

        :param img:
        :param text:
        :param x:
        :param y:
        :return:
        """
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    # In[9]: Prediction
    def predict(self, test_img, subjects=None):
        # make a copy of the image as we don't want to chang original image
        img = test_img.copy()
        # detect face from the image
        faces, rects = self.detect_and_align_face_dlib_new(img)

        if faces is None:
            return None, -1

        # predict the image using our face recognizer
        for i in range(0, len(faces)):
            rect = rects[i]
            face = faces[i]
            if face is None:
                face = []
            if len(face) == 0 or rect[0] < 0 or rect[1] < 0 or rect[2] < 0 or rect[3] < 0:
                return None, -1
            label, confidence = self.recognizer.predict(face)
            print("score: " + str(confidence))
            # get name of respective label returned by face recognizer
            # print(label)
            if label and confidence and confidence <= 100:
                if subjects is not None:
                    label_text = subjects[label]
                else:
                    label_text = str(label)
            else:
                label_text = 'unknown'

            # draw a rectangle around face detected
            self.draw_rectangle(img, rect)
            # draw name of predicted person
            self.draw_text(img, label_text, rect[0], rect[1] - 5)

        return img, 'label'


# =================================TRAING and TESING================================
if __name__ == '__main__':
    # Path to folder of Face Database
    DATA_PATH = "training-data"
    # training file
    TRAIN_FILE = "training_list.txt"
    # testing file
    TEST_FILE = "testing_list.txt"
    # output model filename
    OUTPUT_MODEL = "OpenCVFaceRecognizer.yml"

    reg = FaceRecongizerCV(recognizer_type='LBP', detect_type='dlib')
    reg.train_recognizer(DATA_PATH, TRAIN_FILE, TEST_FILE, OUTPUT_MODEL,num_test=2)
    reg.test_recognizer(TEST_FILE)

