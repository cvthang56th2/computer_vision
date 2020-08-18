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

# =================================DETECT FROM FOLDER================================

# =================================TRAIN================================
if __name__ == '__main__':
    train_path = "test_set/corgi/corgi-train100.xml"
    test_path = "test_set/corgi/corgi-test100.xml"
    out_path = "corgi_model100.svm"
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
