
import numpy as np
import matplotlib.pyplot as plt
from lxml import etree
import glob
from tqdm import *
import sys
import os.path
import gc
import skfmm
import time
import psutil as ps

from Queue import Queue
import threading

class Sailfish_data:
  def __init__(self, base_dir, size, dim, max_queue=50, nr_threads=1):

    # base dir where all the xml files are
    self.base_dir = base_dir
    self.size = size
    self.dim = dim

    # lists to store the datasets
    self.geometries    = []
    self.steady_flows = []

    # make queue
    self.max_queue = max_queue
    self.queue = Queue() # to stop halting when putting on the queue
    self.queue_batches = []

    # Start threads
    for i in xrange(nr_threads):
      get_thread = threading.Thread(target=self.data_worker)
      get_thread.daemon = True
      get_thread.start()

  def create_dataset(self, num_sim=4, num_steps=60000):

    print("clearing old data...")
    self.geometries = []
    self.steady_flows = []
    self.queue_batches = []
    with self.queue.mutex:
      self.queue.queue.clear()
    with open(os.devnull, 'w') as devnull:
      print('rm -r ' + self.base_dir + "size_" + str(self.size) + "/dim_" + str(self.dim) + "/*")
      p = ps.subprocess.Popen(('rm -r ' + self.base_dir + "size_" + str(self.size) + "/dim_" + str(self.dim) + "/*").split(' '), stdout=devnull, stderr=devnull)
      p.communicate()

    time.sleep(1.0)

    print("generating simulations...")
    for i in tqdm(xrange(num_sim)):
      save_dir = self.base_dir + "size_" + str(self.size) + "/dim_" + str(self.dim) + "/" + "sim_" + str(i)
      cmd = ("../systems/turbulent_flow_" + str(self.dim) + "d.py " 
           + "--checkpoint_file=" + save_dir + "/flow " 
           + "--sim_size=" + str(self.size) + " "
           + "--checkpoint_every=120 "
           + "--max_iters=" + str(num_steps))
      print(cmd)

      with open(os.devnull, 'w') as devnull:
        p = ps.subprocess.Popen(('mkdir -p ' + save_dir).split(' '), stdout=devnull, stderr=devnull)
        p.communicate()
        p = ps.subprocess.Popen(cmd.split(' '), stdout=devnull, stderr=devnull)
        print(p)
        p.communicate()

    print("parsing new data")
    self.parse_data()

  def data_worker(self):
    while True:
      geometry_file, steady_flow_files = self.queue.get()

      # load geometry file
      geometry_array = np.load(geometry_file)
      geometry_array = np.expand_dims(geometry_array, axis=0)
      geometry_array = geometry_array[:,1:-1,1:-1]


      # load flow file
      steady_flow_array = []
      for flow_file in steady_flow_files:
        steady_flow = np.load(flow_file)
        steady_flow = steady_flow.f.dist0a[:,1:-1,1:self.size+1]
        steady_flow = np.swapaxes(steady_flow, 0, 1)
        steady_flow = np.swapaxes(steady_flow, 1, 2)
        steady_flow = np.expand_dims(steady_flow, axis=0)
        steady_flow_array.append(steady_flow)
      steady_flow_array = np.concatenate(steady_flow_array, axis=0)
      steady_flow_array = steady_flow_array.astype(np.float32)
  
      # add to que
      self.queue_batches.append((geometry_array, steady_flow_array))
      self.queue.task_done()
  
  def parse_data(self): 
    # get list of all simulation runs
    sim_dir = glob.glob(self.base_dir + "size_" + str(self.size) + "/dim_" + str(self.dim) + "/*/")

    # clear lists
    self.geometries = []
    self.steady_flows = []

    print("parsing dataset")
    for d in tqdm(sim_dir):
      # get needed filenames
      geometry_file    = d + "flow_geometry.npy"
      steady_flow_file = glob.glob(d + "*.0.cpoint.npz")
      steady_flow_file.sort()

      # check file for geometry
      if not os.path.isfile(geometry_file):
        continue

      if len(steady_flow_file) == 0:
        continue

      # store name
      self.geometries.append(geometry_file)
      self.steady_flows.append(steady_flow_file)

    self.num_sim = len(self.geometries)

  def minibatch(self, batch_size=32, seq_length=5):

    for i in xrange(self.max_queue - len(self.queue_batches) - self.queue.qsize()):
      sim_index = np.random.randint(0, self.num_sim)
      if len(self.steady_flows[sim_index]) - seq_length < 1:
        exit()
        continue
      sim_start_index = np.random.randint(0, len(self.steady_flows[sim_index])-seq_length)
      self.queue.put((self.geometries[sim_index], self.steady_flows[sim_index][sim_start_index:sim_start_index+seq_length]))
   
    #print("num in queue")
    #print(len(self.queue_batches))
    while len(self.queue_batches) < batch_size:
      time.sleep(0.01)

    batch_boundary = []
    batch_data = []
    for i in xrange(batch_size): 
      batch_boundary.append(self.queue_batches[0][0].astype(np.float32))
      batch_data.append(self.queue_batches[0][1])
      self.queue_batches.pop(0)
    batch_boundary = np.stack(batch_boundary, axis=0)
    batch_data = np.stack(batch_data, axis=0)
    return batch_boundary, batch_data

"""
#dataset = Sailfish_data("../../data/", size=32, dim=3)
dataset = Sailfish_data("/data/sailfish_flows/", size=512, dim=2)
#dataset.create_dataset()
dataset.parse_data()
batch_boundary, batch_data = dataset.minibatch(batch_size=8)
for i in xrange(100):
  batch_boundary, batch_data = dataset.minibatch(batch_size=8)
  time.sleep(.8)
  print("did batch")
  plt.imshow(batch_data[0,0,:,:,0])
  plt.show()
  plt.imshow(batch_data[0,0,:,:,1])
  plt.show()
  plt.imshow(batch_data[0,0,:,:,-1])
  plt.show()
  plt.imshow(batch_data[0,0,:,:,2])
  plt.show()
  plt.imshow(batch_data[0,0,:,:,-2])
  plt.show()
  plt.imshow(np.sum(batch_data[0,0], axis=2))
  print(np.sum(batch_data[0,0]))
  plt.show()
  plt.imshow(batch_boundary[0,0,:,:,0])
  plt.show()
"""


