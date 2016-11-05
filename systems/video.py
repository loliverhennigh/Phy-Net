
import numpy as np
import tensorflow as tf
import cv2

class Video:
  # You can give this a video to compress

  def __init__(self):
    # name of video file 
    self.video_file = "../systems/goldfish.webm"
    self.cap = cv2.VideoCapture(self.video_file)

    # number of frames per data point
    self.FRAMES_NUM = 4

    # resize shape
    self.SHAPE = (28,28)

    # the frame
    self.frames = np.zeros((self.SHAPE[0], self.SHAPE[1], self.FRAMES_NUM))

  def get_28x28x4(self): 
    for i in xrange(self.FRAMES_NUM):
      ret, frame = self.cap.read()
      if not ret:
        self.cap.release()
        self.cap = cv2.VideoCapture(self.video_file)
        ret, frame = self.cap.read()
      frame = cv2.resize(frame, self.SHAPE, interpolation = cv2.INTER_CUBIC)
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      self.frames[:, :, i] = frame
      self.frames = self.frames/255.0

  def generate_28x28x4(self, batch_size, num_steps):
    x = np.zeros([batch_size, num_steps, self.SHAPE[0], self.SHAPE[1], self.FRAMES_NUM])
    for i in xrange(batch_size):
      for j in xrange(num_steps):
        self.get_28x28x4()
        x[i, j, :, :, :] = self.frames
    return x


