"""
Train agent to perform image classification using reinforcement learning.

Author: Jeffrey M. Ede
Email: j.m.ede@warwick.ac.uk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from collections.abc import Iterable
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from time import sleep
import threading
import queue
import copy

import pickle

import tensorflow as tf
import sonnet as snt

import numpy as np
import cv2
import math

from PIL import Image

from scipy.signal import convolve
from scipy.misc import imread

## Allow hyperparameters to be passed by command line. Default hyperparameters are provided.
FLAGS = tf.flags.FLAGS

#Experiment number
tf.flags.DEFINE_integer("exper_num", 1000+1, "Number for log and notes files.")

NOTES = """Cluttered translated MNIST. Multiple scales. 
Dorsal gradients do not backpropagate through ventral network. 
Action losses no longer divided by (worker_steps-1).\n"""

#Training options
tf.flags.DEFINE_integer("val_period", 10, """Period between validations. Other iterations are for training.""")

tf.flags.DEFINE_string("model_dir",
                       "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/recurrent_conv-1",
                       "Working directory.")
tf.flags.DEFINE_string("trainwriter_dir",
                       FLAGS.model_dir+"/trainwriter",
                       "Directory for TensorBoard event files")
tf.flags.DEFINE_string("ckpt_dir", FLAGS.model_dir+f"/model-{FLAGS.exper_num}/", "Checkpoint directory.")
tf.flags.DEFINE_string("log_file", FLAGS.model_dir+f"/log-{FLAGS.exper_num}.txt", "Error log file.")
tf.flags.DEFINE_string("notes_file", FLAGS.model_dir+f"/notes-{FLAGS.exper_num}.txt", "Notes file.")
tf.flags.DEFINE_integer("ckpt_interval", 3600, "Checkpointing time interval in secs.")

tf.flags.DEFINE_string("conv_embedder_ckpt",
                       FLAGS.model_dir + "/cifar100-3/model.ckpt-100000",
                       "Checkpoint to transfer convolutional embedder learning from at start of training.")

tf.flags.DEFINE_integer("report_interval", 1, """Iterations between reports (samples, valid loss).""")

tf.flags.DEFINE_bool(
    "transfer_learning", False, """Whether to transfer learning for the convolutional embedder.""")

tf.flags.DEFINE_bool("tfdbg", False, """Use tfdbg for debugging.""")

tf.flags.DEFINE_integer("max_iters", 500_000, """Maximum training iterations.""")
tf.flags.DEFINE_integer("save_period", 2_500, """Save model after every this many iterations.""")

tf.flags.DEFINE_float("gamma", 0.99, "Discount factor for future rewards.")

tf.flags.DEFINE_bool("time", False, """Whether to input remainining time as an action.""")

# Task parameters
tf.flags.DEFINE_float("required_top_1", 0.85, "Minibatch size per realization.")
tf.flags.DEFINE_float("accuracy_beta", 0.99, "Decay rate for averaging classification accuracy.")
tf.flags.DEFINE_float("avg_reward_beta", 0.997, """Decay rate for average reward per step.""")
tf.flags.DEFINE_integer("batch_size", 128, """Minibatch size per realization.""")
tf.flags.DEFINE_integer("worker_steps", 6, """Number of image observations per asynchronous update.""")

tf.flags.DEFINE_integer("num_workers", 1, """Number of worker threads.""")

tf.flags.DEFINE_float("entropy_scale", 0.02, """Scale factor for entropy reward.""")

# Model parameters
tf.flags.DEFINE_integer("base_size", 12, "Size of spatial convolutional embedder instesput.")
tf.flags.DEFINE_integer("min_crop_size", FLAGS.base_size, """Minimum crop size allowed.""")
tf.flags.DEFINE_integer("img_size", 2048, "Number of colour channels in images.")
tf.flags.DEFINE_integer("channels", 1, "Number of colour channels in images.")
tf.flags.DEFINE_float(
    "soft_update", 
    0.003, #0.001 is the value used in https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
    """Weighting for soft update of target network.""")


#tf.app.flags.DEFINE_float('action_stddev', 0.05, "Std dev of Gaussian perturbation to add to unscaled actions.")
tf.app.flags.DEFINE_float("darker_than_black_val", 0., "Value to pad images with for crops outside their support.")

tf.flags.DEFINE_integer("input_attention_span", 1, "Number of previous obersations to pay attention to.")

tf.flags.DEFINE_integer("num_scales", 1, "Number of scales to collect input at.")
tf.flags.DEFINE_float("scale_factor", 2, "Multiples of base scale to use if there is more than one scale.")

# Dataset
tf.flags.DEFINE_string("filenames_pickle",
                       FLAGS.model_dir+"/dnc_filenames_pickle.P",
                       "File to save pickled list of names to for quick load.")
tf.flags.DEFINE_string("data_dir", "//Desktop-sa1evjv/f/stills_hq", """Micrograph directory.""")
tf.flags.DEFINE_integer("prefetch_buffer_size", 
                        2*FLAGS.batch_size, 
                        "Maximum number of batches to prepare in advance.")
tf.flags.DEFINE_integer("max_image_side",
                        1024,
                        "Cap maximum image side length in initial experiments to improve efficiency")

# Optimizer parameters
tf.app.flags.DEFINE_float('actor_lr', 1.e-3, """Base actor learning rate.""")
tf.app.flags.DEFINE_float('critic_lr', 1.e-3, """Base actor learning rate.""")
tf.app.flags.DEFINE_float('rmsprop_eps', 1.e-10, """Epsilon for actor optimizer if RMSProp.""")
tf.app.flags.DEFINE_float('max_grad_norm', 20, """Max multiple of gradient norm to clip to.""")

#Position, size and gamma of crop and the gamma correction of the image to crop from
CropParams = collections.namedtuple('CropParams', ('h', 'w', 'size'))#, 'gamma', 'sharpness'))

NUM_ACTIONS = 3

#Take notes
with open(FLAGS.notes_file, 'a') as f:
    f.write(NOTES)

# Utility

def knuth_fisher_yates(arrays):
    """Implements Knuth-Fisher-Yates shuffle algorithm along zeroeth axis to shuffle
    multiple arrays in the same order."""
    for old_index in range(arrays[0].shape[0]):
        new_index = np.random.randint(old_index+1)

        for i in range(len(arrays)):
            arrays[i][old_index,...], arrays[i][new_index,...] = arrays[i][new_index,...], arrays[i][old_index,...]

    return arrays

def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm

loss_capper_counter = 0
def loss_capper(x):
    return x
    #"""Track running means to calculate variance."""
    #global loss_capper_counter
    #loss_capper_counter += 1
    #lcc = loss_capper_counter

    #def cap(x):
    #    sigma = tf.sqrt(mu2 - mu**2+1.e-8)
    #    cases = [ (tf.greater(x, mu+3*sigma), lambda: x/tf.stop_gradient(x/(mu+3*sigma))),
    #              (tf.less(x, mu-3*sigma), lambda: x/tf.stop_gradient(x/(mu-3*sigma))) ]
    #    capped_x = tf.case( cases, default=lambda: x )
    #    return capped_x

    #mu = tf.get_variable(f"mu-{lcc}", initializer=tf.constant(0, dtype=tf.float32))
    #mu2 = tf.get_variable(f"mu2-{lcc}", initializer=tf.constant(100, dtype=tf.float32))

    #x = cap(x)

    #with tf.control_dependencies([mu.assign(0.99*mu+0.001*x), mu2.assign(0.99*mu2+0.001*x**2)]):
    #    return tf.cond(x <= 1, lambda: x, lambda: tf.sqrt(x + 1.e-8))

def tf_print(*args):
    with tf.control_dependencies([tf.print(t) for t in [*args]]):
        return tf.no_op()

def sample_from_normal_dist(mean, stddev, name=None):
	return tf.random.normal(
		# shape=mean.get_shape(),
		shape=tf.shape(mean),
    	mean=mean,
    	stddev=stddev,
    	dtype=tf.float32,
    	seed=None,
    	name=name,
		)

def variables_in_scopes(scopes):
    #Convert single argument to list
    if not isinstance(scopes, Iterable):
        scopes = [scopes]

    #Collect trainable variables from scopes
    variables = []
    for scope in list(scopes):
        variables += tf.trainable_variables(scope)
    return variables

def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img - min)/(max - min)

    return img.astype(np.float32)

def disp(img):
    #if len(img.shape) == 3:
    #    img = np.sum(img, axis=2)
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return

def int_shape(x):
    """Shape of tensor as a list"""
    return list(map(int, x.get_shape()))

def maxpool2D(x, size=2, stride=2, pad='SAME'):

    x = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=size,
        strides=stride,
        padding=pad)

    return x

def OU_perturb(x, shape=(FLAGS.batch_size,NUM_ACTIONS), theta=0.03, sigma=0.5):
    """Ornstein-Uhlembeck perturbation. Using Gaussian Wiener process."""
    noise_perturb = -theta*x + sigma*tf.random_normal(shape=shape)
    return x + noise_perturb


def sample_sensible_crop(img_h, img_w):
    """Sample a crop that is within the limits of what is considered sensible
    for an image.
    """

    #Get limits on acceptible crops for this image
    limits = sensible_crop_limits(img_h, img_w)

    #Side length of crop
    size = limits.min_size + tf.random.uniform()*(limits.max_size - limits.min_size)

    #Position of crop top-left pixel
    h = limits.h_start + tf.random.uniform()*(limits.h_end - limits.h_start) - size/2
    w = limits.w_start + tf.random.uniform()*(limits.w_end - limits.w_start) - size/2

    #Convert to integers
    size = tf.to_int32(size)
    h = tf.to_int32(h)
    w = tf.to_int32(w)

    return CropParams(h=h, w=w, size=size)

def update_dqn_params(from_scope, to_scope):
    #Assign variables in one variable scope to another variable scope

    from_variables = tf.trainable_variables(from_scope)
    to_variables = tf.trainable_variables(to_scope)

    assign_op = []
    for from_var, to_var in zip(from_variables, to_variables):
        assign_op = tf.assign(to_var, from_var)
        assign_ops.append( assign_op )

# Visual domain access

class VisualDomain(object):
    """Efficient image cropping at multiple scales. This is achieved by
    creating an image pyramid that can be reused.
    """

    def __init__(self):
        self.images = None

    def sensible_start(self):

        start = tf.constant()

        return start

        actuations = []
        for image in images:

            actuation = tf.py_func(
                self._sensible_start,
                [image],
                [tf.float32])
            actuation = tf.reshape(actuation, [NUM_ACTIONS])
            actuations.append(actuation)

        actuations = tf.stack(actuations)

        return actuations

    def _sensible_start(self, image):
        """Sensible starting position to examine new image. It is the smallest 
        view that the entire image fits in.
        """

        image = image[0] #Batch dimension not needed

        #First view contains entire image...
        h, w, d = image.shape
        size = max(image.shape)

        if h > w:
            h0 = 0
            w0 = (w - h) // 2
        elif w > h:
            w0 = 0
            h0 = (h - w) // 2
        else:
            h0 = 0
            w0 = 0

        actuation = np.asarray([np.random.random(), np.random.random()], dtype=np.float32)
        actuation

        return actuation

    def inspect(self, actuations):
        """Extract crop."""

        visions = []
        vision_shape = [FLAGS.batch_size, FLAGS.base_size, FLAGS.base_size, FLAGS.channels]
        if FLAGS.num_scales > 1:
            vision_shape += [FLAGS.num_scales]

        vision = tf.py_func(
            self._inspect,
            [actuations], 
            [tf.float32])
        vision = tf.reshape(vision, vision_shape)

        return vision

    def _inspect(self, actuations):
        """This function describes how the pyramid is operated on to retrieve a crop.
        
        Args:
            new_state: New state relative to the old state.
            old_state: Absolute; not relative, old state.
            reset: Whether to run the image queue to the next image.

        Returns:
            A crop from the image pyramid and the state.
        """

        actuations = 0.5*(actuations + 1.) #Scale from [-1,1] to [0,1]

        image_b, image_h, image_w, image_c = self.images.shape

        crops = []
        for i in range(image_b):
            actuation = actuations[i]
            image = self.images[i]

            max_side_len = max(image_h, image_w)

            xs = []
            for i in range(FLAGS.num_scales):

                size = FLAGS.base_size#min(0.05, 0.05 + 0.95*actuation[2])*max_side_len

                #Get crop coordinates
                h = actuation[0]*(max_side_len - min(size, max_side_len))
                w = actuation[1]*(max_side_len - min(size, max_side_len))

                unscaled_size = size
                size *= FLAGS.scale_factor**i# * actuation[2]
        
                h -= (size - unscaled_size)/2
                w -= (size - unscaled_size)/2

                h0 = h
                w0 =  w
                size0 = size

                max_crop_size = 2*max(image_h, image_w)
                if size > max_crop_size:
                    size_diff = size - max_crop_size
            
                    size = max_crop_size
                    h += size_diff/2
                    w += size_diff/2

                if size < FLAGS.min_crop_size:
                    size_diff = FLAGS.min_crop_size - size
            
                    size = FLAGS.min_crop_size
                    h -= size_diff/2
                    w -= size_diff/2

                h = np.clip(h, -size/2, image_h - size/2)
                w = np.clip(w, -size/2, image_w - size/2)

                #Get crop size
                extent = size
                extent = int(extent)
        
                #Cast cropping positions to indices
                h_start = int(np.floor(h))
                w_start = int(np.floor(w))
        
                h_end = int(h + extent)
                w_end = int(w + extent)

                #Pad the image so the crop is in it
                pad_h_start = max(-h_start, 0)
                pad_w_start = max(-w_start, 0)
                pad_h_end = max(h_end - image_h, 0)
                pad_w_end = max(w_end - image_w, 0)

                #Limit crop to support
                h0_start = h_start + pad_h_start
                h0_end = h_end - pad_h_end
                w0_start = w_start + pad_w_start
                w0_end = w_end - pad_w_end

                #Create darker than black canvas
                if pad_h_start or pad_h_end or pad_w_start or pad_w_end:
                    x = np.full(
                        shape=(extent, extent, image_c),
                        fill_value=FLAGS.darker_than_black_val, 
                        dtype=np.float32)
            
                    if h0_end > h0_start and w0_end > w0_start: 
                        if not pad_h_end: 
                            pad_h_end = None 
                        else: 
                            pad_h_end = -pad_h_end 
                        if not pad_w_end: 
                            pad_w_end = None 
                        else: 
                            pad_w_end = -pad_w_end 

                        #print(pad_h_start, pad_h_end, pad_w_start, pad_w_end, image_h, image_w)
                        #Place crop on canvas
                        x[pad_h_start:(pad_h_start+h0_end-h0_start), 
                          pad_w_start:(pad_w_start+w0_end-w0_start), 
                          0:image_c] = \
                              image[h0_start:h0_end, w0_start:w0_end, 0:image_c]
                else:
                    x = image[h0_start:h0_end, w0_start:w0_end, 0:image_c]

                #Resize for embedder
                if extent != FLAGS.base_size:
                    x = self._resize(image=x, extent=extent)

                #disp(x)
                xs.append(x)

            crop = np.stack(xs, axis=-1)
            crops.append(crop)

        crops = np.stack(crops, axis=0)

        return crops

    def _resize(self, image, extent):

        image = image.astype(np.float32)

        resamp_factor = image.shape[0] / extent
        stddev = np.sqrt(resamp_factor)
        ksize = int(2*np.ceil(3*stddev) + 1)
        x = cv2.GaussianBlur(image, (ksize,ksize), stddev)

        x = cv2.resize(x, (FLAGS.base_size, FLAGS.base_size), interpolation=cv2.INTER_LINEAR)

        if FLAGS.channels == 1:
            x = np.expand_dims(x, axis=-1)

        return x

class Dataset(snt.AbstractModule):

    def __init__(self, train=True, val=True, test=True, name="dataset"):

        super(Dataset, self).__init__(name=name)

        self._images_train = self.get_iterator("train") if train else None
        self._images_val = self.get_iterator("val") if val else None
        self._images_test = self.get_iterator("test") if test else None

    def _build(self):
        ds = [self._images_train, self._images_val, self._images_test]
        return ds

    def get_iterator(self, subset, shuffle_buffer_size=5000, num_parallel_calls=6):

        with tf.device('/cpu:0'):

            dataset = tf.data.Dataset.list_files(FLAGS.data_dir+"/"+subset+"/"+"*.tif")
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
            dataset = dataset.repeat()
            dataset = dataset.map(
                lambda file: tf.py_func(self.record_parser, [file], [tf.float32]),
                num_parallel_calls=num_parallel_calls)
            dataset = dataset.map(self.reshaper, num_parallel_calls=num_parallel_calls)
            dataset = dataset.batch(batch_size=FLAGS.batch_size)
            dataset = dataset.prefetch(buffer_size=FLAGS.prefetch_buffer_size)

            iter = dataset.make_one_shot_iterator()
            return iter.get_next()

    @staticmethod
    def reshaper(img):
        img = tf.reshape(img, [FLAGS.img_size,FLAGS.img_size, FLAGS.channels])
        return img

    @staticmethod
    def load_image(addr, resize_size=(FLAGS.img_size,FLAGS.img_size, FLAGS.channels), img_type=np.float32):
        """Read an image and make sure it is of the correct type. Optionally resize it"""
    
        try:
            img = imread(addr, mode='F')
        except:
            img = 0.5*np.ones(resize_size) #Blank image
            print("Image read failed")

        if resize_size != img.shape:
            img = cv2.resize(img, resize_size, interpolation=cv2.INTER_AREA)

        return img.astype(img_type)

    @staticmethod
    def scale0to1(img):
        """Rescale image between 0 and 1"""

        min = np.min(img)
        max = np.max(img)

        if min == max:
            img.fill(0.5)
        else:
            img = (img-min) / (max-min)

        return img.astype(np.float32)

    @staticmethod
    def flip_rotate(img):
        """Applies a random flip || rotation to the image, possibly leaving it unchanged"""

        choice = int(8*np.random.rand())
    
        if choice == 0:
            return img
        if choice == 1:
            return np.rot90(img, 1)
        if choice == 2:
            return np.rot90(img, 2)
        if choice == 3:
            return np.rot90(img, 3)
        if choice == 4:
            return np.flip(img, 0)
        if choice == 5:
            return np.flip(img, 1)
        if choice == 6:
            return np.flip(np.rot90(img, 1), 0)
        if choice == 7:
            return np.flip(np.rot90(img, 1), 1)

    def preprocess(self, img):

        img[np.isnan(img)] = 0.5
        img[np.isinf(img)] = 0.5

        return self.scale0to1(self.flip_rotate(img))

    def record_parser(self, record):

        img = self.load_image(record)
        img = self.preprocess(img)

        return img

# Training

class AgentCore(object):

    def __init__(self, name):

        with tf.variable_scope(name):

            with tf.variable_scope("conv_embedder"):

                #Actor and critic networks will share the same convolutional
                #feature extractor. However, the fully connected layers after the convolutions
                #will be trained separately for each network.
                if FLAGS.transfer_learning:
                    self.conv_embedder = snt.Sequential(
                        [snt.Conv2D(output_channels=32, kernel_shape=3, stride=1),
                         tf.nn.relu,
                         snt.Conv2D(output_channels=64, kernel_shape=3, stride=1),
                         maxpool2D,
                         tf.nn.relu,
                         snt.Conv2D(output_channels=128, kernel_shape=3, stride=1),
                         maxpool2D,
                         tf.nn.relu,
                         snt.Conv2D(output_channels=128, kernel_shape=3, stride=1),
                         maxpool2D,
                         tf.nn.relu,
                         tf.layers.flatten])
                else:
                    #self.conv_embedder = snt.Sequential(
                    #    [snt.Conv2D(output_channels=32, kernel_shape=3, stride=1),
                    #     tf.nn.relu,
                    #     snt.Conv2D(output_channels=64, kernel_shape=3, stride=2),
                    #     tf.nn.relu,
                    #     snt.Conv2D(output_channels=128, kernel_shape=3, stride=2),
                    #     tf.nn.relu,
                    #     tf.layers.flatten])
                    self.conv_embedder = snt.Sequential(
                         [tf.layers.flatten,
                         snt.Linear(output_size=256),
                         tf.nn.relu,
                         snt.Linear(output_size=128),
                         tf.nn.relu])

            with tf.variable_scope("transfer"):
                self.action_embedder = snt.Linear(output_size=128)
                self.vision_embedder = snt.Linear(output_size=128)

            with tf.variable_scope("context"):
                self.context_hidden = snt.Linear(output_size=256)
                self.context_cell = snt.Linear(output_size=256)

                self.attention_input = snt.Linear(output_size=256)
                self.attention_hidden = snt.Linear(output_size=256)

            with tf.variable_scope("action_chooser"):
                self.action_chooser_mean = snt.Linear(output_size=NUM_ACTIONS, use_bias=False)
                self.action_chooser_stddev = snt.Linear(output_size=NUM_ACTIONS, use_bias=False)

                #self.stop_or_continue = snt.Linear(output_size=2, use_bias=False)

            with tf.variable_scope("value"):
                self.value = snt.Linear(output_size=1, use_bias=False)

            with tf.variable_scope("critic"):
                self.critic = snt.Sequential(
                    [snt.Linear(output_size=1024),
                    tf.nn.leaky_relu,
                    snt.Linear(output_size=512),
                    tf.nn.leaky_relu,
                    snt.Linear(output_size=256),
                    tf.nn.leaky_relu,
                    snt.Linear(output_size=1, use_bias=False)])

                self.critic_action_embedder = snt.Linear(output_size=256)

            with tf.variable_scope("belief"):

                #Belief recurrent network
                self.dorsal_rnn = snt.LSTM(hidden_size=256)
                self.ventral_rnn = snt.LSTM(hidden_size=256)

            with tf.variable_scope("classifier"):
                #Classify the image by using the output of the belief RNN
                self.classifier = snt.Sequential(
                    [snt.Linear(output_size=256),
                        tf.nn.relu,
                        snt.Linear(output_size=256),
                        tf.nn.relu,
                        snt.Linear(output_size=10, use_bias=False)])

class Agent(VisualDomain):

    def __init__(self, sess):

        self._sess = sess

        self.core_name = "agent_core"
        self.target_name = "target_core"

        #Create agent and target agent for soft updates
        self._agent_core = AgentCore(name=self.core_name)
        self._target_core = AgentCore(name=self.target_name)

        self.core_variable_scopes = [self.core_name]
        self.target_variable_scopes = [self.target_name]

        self.actor_variable_scopes = [
            "belief", "classifier", "baseliner", "action_chooser", "context", "transfer", "value"]
        if not FLAGS.transfer_learning:
            self.actor_variable_scopes += ["conv_embedder"]

        self.critic_variable_scopes = ["critic"]

        self.actor_variable_scopes = [self.core_name + "/" + s for s in self.actor_variable_scopes]
        self.critic_variable_scopes = [self.core_name + "/" + s for s in self.critic_variable_scopes]

        #Trainable variables
        self.core_variables = variables_in_scopes(self.core_variable_scopes)
        self.target_variables = variables_in_scopes(self.target_variable_scopes)

        self._noise = tf.zeros((FLAGS.batch_size, NUM_ACTIONS))

        actor_loss, critic_loss, accuracy, supervised_loss, actions = self.examine(
            training=True)

        # Training
        self._global_step = tf.get_variable(
            name="global_step",
            shape=[],
            dtype=tf.int64,
            initializer=tf.zeros_initializer(),
            trainable=False,
            collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

        self._learning_rate = tf.placeholder(tf.float32)

        self._actor_train_op = self._optimize(
            variable_scopes=self.actor_variable_scopes,
            loss=actor_loss,
            learning_rate=self._learning_rate * FLAGS.actor_lr, 
            epsilon=FLAGS.rmsprop_eps, 
            max_grad_norm=FLAGS.max_grad_norm,
            global_step=self._global_step)

        self._critic_train_op = self._optimize(
            variable_scopes=self.critic_variable_scopes,
            loss=critic_loss,
            learning_rate=self._learning_rate * FLAGS.critic_lr, 
            epsilon=FLAGS.rmsprop_eps, 
            max_grad_norm=FLAGS.max_grad_norm,
            global_step=self._global_step)

        self._core_train_op = [self._actor_train_op, self._critic_train_op]

        self._train_op = self._soft_update_op()

        self._performance = {'actor_loss': actor_loss,
                             'critic_loss': critic_loss,
                             'accuracy': accuracy,
                             'action': actions[:,0,:],
                             'supervised_loss': supervised_loss}

    def examine(self, training=True):

        action = self.sensible_start()

        ventral_state = None
        dorsal_state = None

        #Use fixed number of steps
        actions = [action]
        inputs_buffer = []
        qs = []
        vs = []
        estimate_qs = []
        estimate_vs = []
        log_policies = []
        for i in range(FLAGS.worker_steps):

            action, value, log_policy, inputs_buffer, ventral_output, dorsal_output, ventral_state, dorsal_state = self._agent(
                action, inputs_buffer, self._agent_core, i, ventral_state, dorsal_state)

            if i < FLAGS.worker_steps - 1:

                q = self._critic(dorsal_output, action, self._agent_core)

                estimate_v = q - log_policy

                if i < FLAGS.worker_steps - 2:
                    _, target_value, _, _, _, target_dorsal_output, _, _ = self._agent(
                        action, inputs_buffer, self._target_core, i, ventral_state, dorsal_state)
 
                    estimate_q = FLAGS.gamma*target_value
                    estimate_qs.append(estimate_q)

                actions.append(action)
                qs.append(q)
                vs.append(value)
                estimate_vs.append(estimate_v)
                log_policies.append(log_policy)

            #TODO: remove labels
            label = tf.ones([FLAGS.batch_size, 10])/10

            if i == FLAGS.worker_steps - 1:
                logits = self._agent_core.classifier(ventral_output)
                classification_loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)

                pred = tf.argmax(logits, axis=1)
                label_val = tf.argmax(label, axis=1)
                discrete_reward = tf.cast(tf.equal(pred, label_val), tf.float32)

                #estimate_q = -tf.expand_dims(classification_loss, -1)
                estimate_q = tf.expand_dims(discrete_reward, axis=-1)

                estimate_qs.append(estimate_q)

        if not training:
            return tf.nn.softmax(logits)
        else:
            #Compute losses
            actions = tf.stack(actions)

            critic_losses = [(q - tf.stop_gradient(estimate_q))**2 for q, estimate_q in zip(qs, estimate_qs)]
            value_losses = [(v - tf.stop_gradient(estimate_v))**2 for v, estimate_v in zip(vs, estimate_vs)]
            actor_losses = [-(q - log_policy) for q, log_policy in zip(qs, log_policies)]
            
            with tf.control_dependencies([tf.print(x) for x in [critic_losses[-1], qs[-1]]]):
                critic_losses = tf.stack(critic_losses)
                value_losses = tf.stack(value_losses)
                actor_losses = tf.stack(actor_losses)

            accuracy = tf.reduce_mean(discrete_reward)
            
            mean_classification_loss = tf.reduce_mean(classification_loss)
            actor_loss = tf.reduce_mean(actor_losses) + tf.reduce_mean(value_losses) + mean_classification_loss
            critic_loss = tf.reduce_mean(critic_losses)

            return actor_loss, critic_loss, accuracy, mean_classification_loss, actions

    def input_embedder(self, conv_embedding, action, agent_core):

        action = tf.stop_gradient(action)

        action_embedding = agent_core.action_embedder(action)
        vision_embedding = agent_core.vision_embedder(conv_embedding)
        embedding = tf.concat([action_embedding, vision_embedding], axis=1)

        return embedding

    def _attention(self, inputs, hidden, input_embedder, hidden_embedder):
        
        embedded_hidden = tf.tile(tf.expand_dims(hidden_embedder(hidden), axis=0), [len(inputs), 1, 1])

        embedded_inputs = [input_embedder(input) for input in inputs]
        embedded_inputs = tf.stack(embedded_inputs)

        summed = tf.tanh(embedded_hidden + embedded_inputs)

        probs = tf.nn.softmax(summed, axis=0)
        
        stacked_inputs = tf.stack(inputs)

        embedding = tf.reduce_sum(probs*stacked_inputs, axis=0)

        return embedding

    def _belief(self, input, prev_state, rnn):
        """Model environmental belief as it is partially observed."""

        output, state = tf.nn.static_rnn(
            cell=rnn,
            inputs=[input],
            initial_state=prev_state,
            dtype=tf.float32)

        return output, state

    def _agent(self, action, inputs_buffer, image, agent_core, i, ventral_state, dorsal_state):

        vision = self.inspect(action)

        conv_embedding = agent_core.conv_embedder(vision)

        input_embedding = self.input_embedder(conv_embedding, action, agent_core)

        inputs_buffer.append(input_embedding)

        if FLAGS.input_attention_span > 1:

            inputs_buffer_overflow = len(inputs_buffer) - FLAGS.input_attention_span
            if inputs_buffer_overflow > 0:
                inputs_buffer = inputs_buffer[inputs_buffer_overflow:]

            if i:
                input_embedding = self._attention(
                    inputs_buffer, dorsal_output, agent_core.attention_input, agent_core.attention_hidden)

        if not i:
            #context_embedding = tf.stop_gradient(input_embedding)
            #context_hidden = agent_core.context_hidden(context_embedding)
            #context_cell = agent_core.context_cell(context_embedding)
            #dorsal_state = snt.LSTMState(hidden=context_hidden, cell=context_cell)

            dorsal_state = agent_core.ventral_rnn.initial_state(FLAGS.batch_size, dtype=tf.float32)
            ventral_state = agent_core.ventral_rnn.initial_state(FLAGS.batch_size, dtype=tf.float32)

        ventral_output, ventral_state = self._belief(input_embedding, ventral_state, agent_core.ventral_rnn)
        ventral_output = ventral_output[0]

        if i != FLAGS.worker_steps - 1: #Explore

            dorsal_output, dorsal_state = self._belief(tf.stop_gradient(ventral_output), dorsal_state, agent_core.dorsal_rnn)
            dorsal_output = dorsal_output[0]

            #Predict optimal action
            action_mean = agent_core.action_chooser_mean(dorsal_output)
            action_stddev = agent_core.action_chooser_stddev(dorsal_output)

            dist = tf.contrib.distributions.MultivariateNormalDiag(loc=action_mean, scale_diag=action_stddev)

            #Following soft actor critic implementation in https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/model.py
            x = dist.sample()
            action = tf.tanh(x)

            log_policy = dist.log_prob(x)
            log_policy -= tf.reduce_sum(tf.log(1 - action**2 + 1.e-6), axis=-1) #Squash correction
            log_policy *= FLAGS.entropy_scale
            log_policy = tf.expand_dims(log_policy, axis=-1)

            if FLAGS.time:
                time = ((FLAGS.worker_steps-i-1)/FLAGS.worker_steps)*tf.ones([FLAGS.batch_size, 1])
                action = tf.concat([action, time], axis=1)

            #Estimate state value
            value = agent_core.value(dorsal_output)

            self._noise = OU_perturb(self._noise)
        else:
            dorsal_output = None
            dorsal_state = None
            action = None
            value = None
            log_policy = None

        return action, value, log_policy, inputs_buffer, ventral_output, dorsal_output, ventral_state, dorsal_state

    def _critic(self, hidden, action, agent_core):

        embedded_action = agent_core.critic_action_embedder(action)
        state = tf.concat([hidden, embedded_action], axis=1)

        q = agent_core.critic(state)

        return q

    def core2target(self, run=True):
        """Overwrite target network variables with the active agent's."""

        #Assign agent variable to target variables
        assign_ops = [tf.assign(t, a) for t, a in zip(self.target_variables, self.core_variables)]
        if run:
            self._sess.run(assign_ops)
        else:
            return assign_ops

    def _optimize(self, variable_scopes, loss, learning_rate, epsilon, max_grad_norm, 
                  global_step):
        """Optimation operations to be performed in training step."""

        #Get trainable variables from variable scopes
        variables = variables_in_scopes(variable_scopes)
                
        #l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in target_variables])
        #loss += l2_loss

        #Construct optimizer that clips gradients by a multiple of their global norm
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9)
        #optimizer =  tf.train.RMSPropOptimizer(learning_rate, epsilon=epsilon)
        if max_grad_norm:
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(
                optimizer, clip_norm=max_grad_norm)

        #Compute gradients for target values
        #gradients = optimizer.compute_gradients(loss, var_list=variables)

        train_op = optimizer.minimize(loss, var_list=variables)

        return train_op

    def _soft_update_op(self):

            with tf.control_dependencies([*self._core_train_op]):
                train_op = [tf.assign(t, (1 - FLAGS.soft_update)*t + FLAGS.soft_update*v) for 
                                 t, v in zip(self.target_variables, self.core_variables)]
                train_op = tf.group(*train_op)
                return train_op

    def train(self, images, learning_rate, train=True):
        """Feed examples to the network for training. 
        
        Args:
            images: Images to examine.
            prev_state: State and decisions by made by a worker in a previous step.

        Returns:
            Loss statistics, final_state
        """

        self.images = images
        
        if train:
            train_op = [self._train_op]
            feed_dict.update( { self._learning_rate: learning_rate } )
        else:
            train_op =[]

        #Train and get final state to return
        results = self._sess.run(
            [v for _, v in self._performance.items()] +
            train_op, 
            feed_dict=feed_dict)        

        #Bind losses in named tuple
        losses = results[:len(self._performance)]
        losses = { k: losses[i] for i, k in enumerate(self._performance) }
        
        return losses

async def log_losses(log_file, output):
    """Write losses to file, blocking further writes until current write finishes."""
    await log_file.write(output)

def trainer(agent, sess, saver, image_train_iter, image_val_iter, counter_queue, log_file):

    accuracy = 1/10
    accuracy_beta = np.sqrt(0.999**FLAGS.batch_size)

    counter = -1
    while accuracy < FLAGS.required_top_1 or counter < FLAGS.max_iters:

        if not counter % FLAGS.val_period:
            validating = True
            image_iter = image_val_iter
        else:
            validating = False
            image_iter = image_train_iter

        #Update shared counter
        counter = counter_queue.get()
        counter_queue.put(counter+1)

        images = sess.run(image_iter)

        step = int(counter/2_500)
        learning_rate = np.float32(0.97**step)

        #Asynchronus gradient update
        performance = agent.train(
            images=images,
            learning_rate=learning_rate,
            train=not validating)
        
        accuracy  = accuracy_beta*accuracy + (1 - accuracy_beta)*performance['accuracy']

        #String losses
        output = f"Exper: {FLAGS.exper_num}, Iter: {counter}"
        for k in performance:
            if k != 'action':
                output += f", {k}: {performance[k]}"

        if not counter % FLAGS.report_interval:
            log_file.write(output)

        output += f", action: {performance['action']}"
        output += f", moving_accuracy: {accuracy}"
        output += f", validating: {validating}"
        print(output)

        if not counter%FLAGS.save_period:
            output += "\n"
            saver.save(sess, FLAGS.ckpt_dir, global_step=counter)

def train():
    """Asynchronous dvantage actor critic_training."""

    images, images_val, _ = Dataset(test=False)()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.force_gpu_compatible = True

    with tf.Session(config=config) as sess, open(FLAGS.log_file, "a") as log_file:

        if FLAGS.tfdbg:
            #Wrap session in debugger interface
            from tensorflow.python import debug as tf_debug #Imported here to avoid tfdbg interface appearing
            sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type='readline')

        print("Graphing agent...")

        global examples_queue
        examples_queue = queue.Queue(maxsize=10)

        #Create the agent
        agent = Agent(sess=sess)

        #Only keep 2 save checkpoints
        saver = tf.train.Saver( max_to_keep=2 )
        sess.run( tf.global_variables_initializer() )
        train_writer = tf.summary.FileWriter( FLAGS.trainwriter_dir, sess.graph )

        #Queue to count training steps
        counter_queue = queue.Queue()

        print("Initializing variables...")

        last_counter = 0
        counter_queue.put(last_counter)
        if last_counter:
            #Restore save model
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir)) 

        else:
            if FLAGS.transfer_learning:
                #Collect variables to initialize
                scopes = agent.actor_variable_scopes + agent.critic_variable_scopes

                vars = []
                for s in scopes:
                    vars += tf.trainable_variables(s)
                sess.run(tf.variables_initializer(vars))

                print("Transferring learning")

                #Transfer learning
                collection = tf.get_collection(
                    tf.GraphKeys.GLOBAL_VARIABLES, scope="agent_core/conv_embedder")
           
                #Use transfer learning to initialize the convolutional embedder
                conv_embedder_saver = tf.train.Saver(var_list=collection)
                conv_embedder_saver.restore(sess, FLAGS.conv_embedder_ckpt)
            else:
                #Initialize all variables
                sess.run(tf.global_variables_initializer())

        #Assign agent variables to target variables to they start off with the same values
        agent.core2target()

        print("Starting worker...")

        target=trainer(
            agent,
            sess,
            saver,
            images,
            images_val,
            counter_queue,  
            log_file)

def main(unused_argv):
    tf.logging.set_verbosity(3)  # Print INFO log messages
    train()

if __name__ == "__main__":
    tf.app.run()

#def ornstein_uhlembeck_noise(shape, mean=0., sigma=, dt=0.15):
#    x = tf.get_variable("noise", initializer=tf.constant(0, dtype=tf.float32))
#    y = x - dt*(x - mean) + tf.sqrt(2*dt)*tf.random_normal(shape)
#    with tf.control_dependencies([x.assign(y)]):
#        return tf.identity(x)
