#!/usr/bin/python
# The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
#
# This example program shows how you can use dlib to make a HOG based object
# detector for things like faces, pedestrians, and any other semi-rigid
# object.  In particular, we go though the steps to train the kind of sliding
# window object detector first published by Dalal and Triggs in 2005 in the
# paper Histograms of Oriented Gradients for Human Detection.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html.
import os
import glob

import dlib
import cv2
from skimage import io
import argparse
import numpy as np


class HOGDetector(object):
    def __init__(self, options=None, detector_filename=None):
        """Create or load an object detector

        :param options: options for training a detector
        :param loadPath: name of a trained detector
        """
        # create detector options
        self.options = options
        if self.options is None:
            self.options = dlib.simple_object_detector_training_options()

        # load the trained detector (for testing)
        if detector_filename is not None:
            self._detector = dlib.simple_object_detector(detector_filename)

    def _prepare_annotations(self, annotations):
        """Create a list of dlib rectanges from list of pixel locations of the edges
        of the boxes.

        :param annotations: list of pixel locations, ie, left=329, top=78, right=437, bottom=186
        :return:
        """
        annots = []
        for (x, y, xb, yb) in annotations:
            annots.append([dlib.rectangle(left=int(x), top=int(y), right=int(xb), bottom=int(yb))])
        return annots

    def _prepare_images(self, image_names):
        """Read a list of images given list of image names

        :param image_names: List of image names
        :return:
        """
        images = []
        for imPath in image_names:
            image = cv2.imread(imPath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        return images

    def train2(self, image_names, annotations, visualize=False, detector_output_filename=None):
        """Train the object detector.
        Note: If you use the tool gather_annotations.py to annotate objects, you should use this function
        instead of train to train the object detector.

        :param image_names: a numpy array of type unicode containing paths to images
        :param annotations: List of pixel locations of the edges of the boxes. ie, left=329, top=78, right=437, bottom=186
        :param visualize:
        :param detector_output_filename:
        :return:
        """
        annotations = self._prepare_annotations(annotations)
        images = self._prepare_images(image_names)
        self._detector = dlib.train_simple_object_detector(images, annotations, self.options)

        print("Visualze the HOG filter we have learned")
        if visualize:
            win = dlib.image_window()
            win.set_image(self._detector)
            dlib.hit_enter_to_continue()

        # save detector to disk
        if detector_output_filename is not None:
            self._detector.save(detector_output_filename)

        # If you have already loaded your training
        # images and bounding boxes for the objects then you can call it as shown below.
        print("\nTraining accuracy: {}".format(
            dlib.test_simple_object_detector(images, annotations, self._detector)))

    def train(self, training_xml_path, testing_xml_path, detector_output_filename, visualize=False):
        """This function does the actual training.  It will save the final detector to
        detector.svm.  The input is an XML file that lists the images in the training dataset
        and also contains the positions of the face boxes.  To create your
        own XML files you can use the imglab tool which can be found in the
        tools/imglab folder.  It is a simple graphical tool for labeling objects in
        images with boxes.  To see how to use it read the tools/imglab/README.txt
        file.  But for this example, we just use the training.xml file included with
        dlib.

        :param training_xml_path: an XML file that lists the images in the training dataset
        and also contains the positions of the face boxes.
        :param testing_xml_path: an XML file that lists the images in the testing dataset
        :param detector_output_filename: Output detector filename
        :param visualize:
        :return:
        """
        print("Train the detector...")
        dlib.train_simple_object_detector(training_xml_path, detector_output_filename, self.options)

        # Now that we have a face detector we can test it.  The first statement tests
        # it on the training data.  It will print(the precision, recall, and then)
        # average precision.
        print("")  # Print blank line to create gap from previous output
        print("Training accuracy: {}".format(
            dlib.test_simple_object_detector(training_xml_path, detector_output_filename)))
        # However, to get an idea if it really worked without overfitting we need to
        # run it on images it wasn't trained on.  The next line does this.  Happily, we
        # see that the object detector works perfectly on the testing images.
        print("Testing accuracy: {}".format(
            dlib.test_simple_object_detector(testing_xml_path, detector_output_filename)))

        print("Visualze the HOG filter we have learned")
        if visualize:
            self._detector = dlib.simple_object_detector(detector_output_filename)
            win = dlib.image_window()
            win.set_image(self._detector)
            dlib.hit_enter_to_continue()

    def detect_from_folder(self, image_folder, ext="*.jpg"):
        """Run the detector over the images in the faces folder and display the results

        :param image_folder: Folder containing image to detect
        :param ext: extension of the image
        :return:
        """
        print("Showing detections on the images in the faces folder...")
        win = dlib.image_window()
        for f in glob.glob(os.path.join(image_folder, ext)):
            print("Processing file: {}".format(f))
            img = io.imread(f)
            dets = self._detector(img)
            print("Number of faces detected: {}".format(len(dets)))
            for k, d in enumerate(dets):
                print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                    k, d.left(), d.top(), d.right(), d.bottom()))

            win.clear_overlay()
            win.set_image(img)
            win.add_overlay(dets)
            dlib.hit_enter_to_continue()

    def predict(self, image, upsampling=0):
        """Detect objects from image

        :param image: Image to search for objects
        :return: List of detected objects in form of boxes, ie: (left, top, right, bottom)
        """
        boxes = self._detector(image, upsampling)
        if boxes is None:
            return None
        preds = []
        for box in boxes:
            (x, y, xb, yb) = [box.left(), box.top(), box.right(), box.bottom()]
            preds.append((x, y, xb, yb))
        return preds

    def detect_and_display(self, image, annotate=None):
        """Detect objects from image, draw the bounding boxes and display
        it on window

        :param image: Image to search for objects
        :param annotate:
        :return:
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preds = self.predict(image)
        for (x, y, xb, yb) in preds:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # draw and annotate on image
            cv2.rectangle(image, (x, y), (xb, yb), (0, 0, 255), 2)
            if annotate is not None and type(annotate) == str:
                cv2.putText(image, annotate, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 255, 0), 2)

        cv2.imshow("Detected", image)
        cv2.waitKey(0)


# =================================DETECT FROM FOLDER================================
# detector = HOGDetector(detector_filename="hand_model.svm")
# detector.detect_from_folder("hand_imgs", "*.jpg")

# =================================TRAIN================================
if __name__ == '__main__':
    train_path = "test_set/kien/kien-train.xml"
    test_path = "test_set/kien/kien-test.xml"
    out_path = "kien_model.svm"
    options = dlib.simple_object_detector_training_options()
    # Since faces are left/right symmetric we can tell the trainer to train a
    # symmetric detector.  This helps it get the most value out of the training
    # data.
    options.add_left_right_image_flips = True
    # The trainer is a kind of support vector machine and therefore has the usual
    # SVM C parameter.  In general, a bigger C encourages it to fit the training
    # data better but might lead to overfitting.  You must find the best C value
    # empirically by checking how well the trained detector works on a test set of
    # images you haven't trained on.  Don't just leave the value set at 5.  Try a
    # few different C values and see what works best for your data.
    options.C = 5
    options.detection_window_size = 6400
    # Tell the code how many CPU cores your computer has for the fastest training.
    options.num_threads = 4
    options.be_verbose = True

    detector = HOGDetector(options=options)
    print("[INFO] creating & saving object detector")

    # detector.train(args["train_xml"], args["test_xml"], args["detector"], visualize=True)
    detector.train(train_path, test_path, out_path, visualize=True)

# =================================TRAIN WITH COMMAND PROMPT================================
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--train_xml", required=True, help="path to training xml file...")
# ap.add_argument("-e", "--test_xml", required=True, help="path to test xml file...")
# ap.add_argument("-d", "--detector", default=None, help="path to save the trained detector...")
# args = vars(ap.parse_args())

# options = dlib.simple_object_detector_training_options()
# # Since faces are left/right symmetric we can tell the trainer to train a
# # symmetric detector.  This helps it get the most value out of the training
# # data.
# options.add_left_right_image_flips = True
# # The trainer is a kind of support vector machine and therefore has the usual
# # SVM C parameter.  In general, a bigger C encourages it to fit the training
# # data better but might lead to overfitting.  You must find the best C value
# # empirically by checking how well the trained detector works on a test set of
# # images you haven't trained on.  Don't just leave the value set at 5.  Try a
# # few different C values and see what works best for your data.
# options.C = 5
# # Tell the code how many CPU cores your computer has for the fastest training.
# options.num_threads = 4
# options.be_verbose = True

# detector = HOGDetector(options=options)
# print("[INFO] creating & saving object detector")

# detector.train(args["train_xml"], args["test_xml"], args["detector"], visualize=True)

# # =================================TRAIN WITH COMMAND PROMPT 2================================
# # ======This method is used when you use the tool gather_annotations.py to label objects in image=====
# ap = argparse.ArgumentParser()
# ap.add_argument("-a","--annotations",required=True,help="path to saved annotations...")
# ap.add_argument("-i","--images",required=True,help="path to saved image paths...")
# ap.add_argument("-d","--detector",default=None,help="path to save the trained detector...")
# args = vars(ap.parse_args())

# print("[INFO] loading annotations and images")
# annots = np.load(args["annotations"])
# imagePaths = np.load(args["images"])

# detector = ObjectDetector()
# print("[INFO] creating & saving object detector")

# detector.train2(imagePaths,annots, visualize=True, detector_output_filename=args["detector"])

# # ================================TEST================================
# ap = argparse.ArgumentParser()
# ap.add_argument("-d","--detector",required=True,help="path to trained detector to load...")
# ap.add_argument("-i","--image",required=True,help="path to an image for object detection...")
# ap.add_argument("-a","--annotate",default=None,help="text to annotate...")
# args = vars(ap.parse_args())

# detector = ObjectDetector(loadPath=args["detector"])

# imagePath = args["image"]
# image = cv2.imread(imagePath)
# detector.detect(image,annotate=args["annotate"])