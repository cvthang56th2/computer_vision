#!/usr/bin/env python
import numpy as np
import cv2
import dlib
from tkinter import *
from threading import Thread, Event
import datetime
from PIL import Image
from PIL import ImageTk
from FaceRecOpenCV_File import FaceRecongizerCV

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


class FPS:
    def __init__(self):
        # store the start time, end time, and total number of frames
        # that were examined between the start and end intervals
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        # start the timer
        self._start = datetime.datetime.now()
        return self

    def stop(self):
        # stop the timer
        self._end = datetime.datetime.now()

    def update(self):
        # increment the total number of frames examined during the
        # start and end intervals
        self._numFrames += 1
        self._end = datetime.datetime.now()

    def elapsed(self):
        # return the total number of seconds between the start and
        # end interval
        return (self._end - self._start).total_seconds()

    def fps(self):
        # compute the (approximate) frames per second
        return self._numFrames / self.elapsed()


class WebcamVideoStream:
    def __init__(self, src=0):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def __del__(self):
        self.stream.release()

    def start(self):
        # start the thread to read frames from the video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class FaceController(Frame):

    def __init__(self, parent=None):                            # if not a frame
        Frame.__init__(self, parent)                       # provide container
        self.pack(expand=YES, fill=BOTH)
        self.config(relief=RIDGE, border=2)                # reconfig to change

        self.video_src = 0
        self.cam = None
        self.frame = None
        self.save_cnt = 0
        self.save_frame = None
        self.detector = dlib.get_frontal_face_detector()
        self.recognizer = FaceRecongizerCV("OpenCVFaceRecognizer.yml", recognizer_type='LBP', detect_type='dlib')

        self.stopEvent = None
        self.thread = None
        self.panel = None
        self.canvas = None

        # creae Gui
        self.create_gui()

    def create_gui(self):
        """Create all the necessary GUI widgets

        :return:
        """
        Label(self, text='OBJECT DETECTION', font=("Times", 10, "bold"), relief=GROOVE, bg='light blue',
              fg='brown').pack(side=TOP, fill=BOTH)

        # View Raw video; Histogram backprojection; Histogram bins.
        frame1 = Frame(master=self)
        frame1.pack(side=LEFT)
        Button(master=frame1, text='Start', command=self.on_start).pack(fill=X)
        Button(master=frame1, text='Stop', command=self.on_stop).pack(fill=X)
        Button(master=frame1, text='Take Image', command=self.on_get_image).pack(fill=X)
        Button(master=frame1, text='Save Image', command=self.on_save_image).pack(fill=X)

        frame2 = Frame(master=self)
        frame2.pack(side=LEFT)
        self.panel = Label(frame2, width=640, height=480, bg='lightgray', relief=SUNKEN)
        self.panel.pack(side=LEFT, fill=BOTH, expand=YES)
        # self.canvas = Canvas(frame2, relief=RAISED, bg="lightblue", width=640, height=480)
        # self.canvas.pack()

    def on_start(self):
        self.cam = WebcamVideoStream(self.video_src).start()
        self.frame = self.cam.read()
        cv2.namedWindow('Object Detection')
        # start the thread to read frames from the video stream
        self.stopEvent = Event()
        self.thread = Thread(target=self.run, args=())
        self.thread.start()

    def on_stop(self):
        self.stopEvent.set()
        self.cam.stop()
        cv2.destroyAllWindows()

    def on_get_image(self):
        # OpenCV represents images in BGR order; however PIL
        # represents images in RGB order, so we need to swap
        # the channels, then convert to PIL and ImageTk format

        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # self.canvas.create_image(0, 0, image=image, anchor=NW)

        self.panel.configure(image=image)
        self.panel.image = image

    def on_save_image(self):
        self.save_frame = self.frame.copy()
        self.save_cnt += 1
        # OpenCV represents images in BGR order; however PIL
        # represents images in RGB order, so we need to swap
        # the channels, then convert to PIL and ImageTk format
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale image
        rects = self.detector(gray, 0)

        for (i, rect) in enumerate(rects):
            (x, y, w, h) = rect_to_bb(rect)
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        image = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = ImageTk.PhotoImage(image)

        # self.canvas.create_image(0, 0, image=image, anchor=NW)

        self.panel.configure(image=image)
        self.panel.image = image
        if self.save_frame is not None:
            cv2.imwrite(str(self.save_cnt) + ".jpg", self.save_frame)

    def run(self):
        """This function is for the processing thread

        :return:
        """
        fps = FPS().start()
        # keep looping over frames until we are instructed to stop
        while not self.stopEvent.is_set():
            # grab the current frame
            self.frame = self.cam.read()

            # resize the frame, blur it, and convert it to the HSV color space
            vis = resize(self.frame, width=400)

            # recognition
            image, label = self.recognizer.predict(vis, ('', 'thang', 'kien'))
            if image is not None:
                vis = image.copy()

            # show the frame to our screen and increment the frame counter
            cv2.imshow("Object Detection", vis)



if __name__ == '__main__':
    appWindow = Tk() # creates the application window (you can use any name)
    appWindow.wm_title("COLOR TRACKER") # displays title at the top left
    appWindow.config(bg="#037481")
    appWindow.geometry("700x600")

    csc = FaceController(appWindow)

    appWindow.mainloop()
