__author__ = 'thang'
import codecs
import numpy as np
import cv2
import os

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

    def create_train_and_test_data_file(self, data_folder_path, out_train_file, out_test_file, num_test=10):
        """

        :param data_folder_path:
        :param out_train_file:
        :param out_test_file:
        :param num_test:
        :return:
        """
        SEPARATOR = ";"
        f_train = codecs.open(out_train_file, "w", "utf-8")
        f_test = codecs.open(out_test_file, "w", "utf-8")

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
        file = codecs.open(data_file, 'r', "utf-8")
        train_files = file.readlines()
        file.close()

        # let's go through each directory and read images within it
        for line in train_files:
            if line.strip() == "":
                continue
            # extract label number of subject from im_file
            label = line[line.rfind(";") + 1]
            if label == 'â':
                label = 97
            elif label == 'ă':
                label = 97
            elif label == 'ê':
                label = 101
            elif label == 'ô':
                label = 111
            elif label == 'đ':
                label = 100
            elif label == 'ơ':
                label = 111
            elif label == 'ư':
                label = 117
            else:
                label = ord(label)
            print(line[line.rfind(";") + 1], label)
            # labels.append(label)
            # extract filename from im_file
            image_name = line[:line.rfind(";")]
            print('image_name', image_name)

            # read image
            image = cv2.imread(image_name)
            if image is None:
              image = cv2.imdecode(np.fromfile(image_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            self.extract_char_from_image(image, label)

        self.save_dataset(out_file)

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
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
            # cv2.imshow("Mask", mask)
            # cv2.waitKey(0)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)  # segment digit from thresh

            char_i = mask[y - 8:y + h + 8, x - 8:x + w + 8]  # just for better approximation
            if char_i.size == 0:
                print('Image Failed')
                return
            char_i = cv2.resize(char_i, self.char_size)
            char_i = char_i.reshape((1, self.char_size[0] * self.char_size[1]))
            self.train_mat = np.vstack((self.train_mat, np.hstack((label, char_i[0, :]))))
            # cv2.imshow("Crop", char_i)
            # cv2.waitKey(0)
            boxes.append((x, y, w, h))
            digits.append(char_i)

        # cv2.imshow("Original", image)
        # cv2.imshow("Thresh", thresh)

        # draw bounding boxes and print digits on them
        # for r in boxes:
        #     (x, y, w, h) = r
        #     cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 1)
        #     cv2.imshow("Recognized", image)
        #     # cv2.waitKey(0)

        # cv2.destroyAllWindows()

    def save_dataset(self, output_filename, delimiter=","):
      self.train_mat = np.delete(self.train_mat, 0, 0)
      f_handle = open(output_filename, 'w')
      np.savetxt(f_handle, self.train_mat, fmt='%.6f', delimiter=delimiter)
      f_handle.close()


model = CharRecognizer()
model.create_train_and_test_data_file('new_chars', 'train_new_chars.txt', 'test_new_chars.txt', 2)
model.extract_char_and_save('train_new_chars.txt', 'character_new_chars.train')
model.extract_char_and_save('test_new_chars.txt', 'character_new_chars.test')