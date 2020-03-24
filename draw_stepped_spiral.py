import numpy as np

from PIL import Image
from PIL import ImageDraw

import itertools

import tensorflow as tf

import cv2

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 128, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 64, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 0, "Maximum absolute value of controller and dnc outputs.")
tf.flags.DEFINE_bool("use_batch_norm", True, "Use batch normalization in generator.")

tf.flags.DEFINE_string("model", "LSTM", "LSTM or DNC.")

tf.flags.DEFINE_integer("projection_size", 0, "Size of projection layer. Zero for no projection.")

tf.flags.DEFINE_bool("is_input_embedder", False, "Embed inputs before they are input.")

# Optimizer parameters.
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("replay_size", 25000, "Maximum examples in ring buffer.")
tf.flags.DEFINE_integer("avg_replays", 4, "Mean frequency each experience is used.")
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10, "Epsilon used for RMSProp optimizer.")
tf.flags.DEFINE_float("L2_norm", 1.e-5, "Decay rate for L2 regularization. 0 for no regularization.")

# Task parameters
tf.flags.DEFINE_integer("img_side", 96, "Number of image pixels for square image")
tf.flags.DEFINE_integer("num_steps", 20, "Number of image pixels for square image")
tf.flags.DEFINE_integer("step_size", 20, "Distance STEM probe moves at each step (in px).")
tf.flags.DEFINE_integer("num_actions", 2, "Number of parameters to describe actions.")
tf.flags.DEFINE_integer("shuffle_size", 2000, "Size of moving buffer to sample data from.")
tf.flags.DEFINE_integer("prefetch_size", 10, "Number of batches to prepare in advance.")

# Training options.
tf.flags.DEFINE_float("actor_lr", 0.0007, "Actor learning rate.")
tf.flags.DEFINE_float("critic_lr", 0.001, "Critic learning rate.")
tf.flags.DEFINE_float("generator_lr", 0.003, "Generator learning rate.")

tf.flags.DEFINE_float("gamma", 0.97, "Reward/loss decay.")

tf.flags.DEFINE_bool("is_advantage_actor_critic", False, "Use advantage rather than Q errors for actor.")

tf.flags.DEFINE_bool("is_cyclic_generator_learning_rate", True, "True for sawtooth oscillations.")
tf.flags.DEFINE_bool("is_decaying_generator_learning_rate", True, "True for decay envelope for sawtooth oscillations.")

tf.flags.DEFINE_integer("supervision_iters", 100_000, "Starting value for supeversion.")
tf.flags.DEFINE_float("supervision_start", 1., "Starting value for supeversion.")
tf.flags.DEFINE_float("supervision_end", 0., "Starting value for supeversion.")

if FLAGS.supervision_iters:
    #Flag will not be used
    tf.flags.DEFINE_float("supervision", 0.5, "Weighting for known discounted future reward.")
else:
    #Flag will be used
    tf.flags.DEFINE_float("supervision", 0.0, "Weighting for known discounted future reward.")

tf.flags.DEFINE_bool("is_target_actor", True and FLAGS.supervision != 1, "True to use target actor.")
tf.flags.DEFINE_bool("is_target_critic", True and FLAGS.supervision != 1, "True to use target critic.")
tf.flags.DEFINE_bool("is_target_generator", False, "True to use target generator.")

tf.flags.DEFINE_integer("update_frequency", 0, "Frequency of hard target network updates. Zero for soft updates.")
tf.flags.DEFINE_float("target_decay", 0.9997, "Decay rate for target network soft updates.")
tf.flags.DEFINE_bool("is_generator_batch_norm_tracked", False, "True to track generator batch normalization.")

tf.flags.DEFINE_bool("is_positive_qs", True, "Whether to clip qs to be positive.")

tf.flags.DEFINE_bool("is_infilled", False, "True to use infilling rather than generator.")

tf.flags.DEFINE_bool("is_prev_position_input", True, "True to input previous positions.")

tf.flags.DEFINE_bool("is_ornstein_uhlenbeck", True, "True for O-U exploration noise.")
tf.flags.DEFINE_bool("is_noise_decay", True, "Decay noise if true.")
tf.flags.DEFINE_float("ou_theta", 0.1, "Drift back to mean.")
tf.flags.DEFINE_float("ou_sigma", 0.2, "Size of random process.")

tf.flags.DEFINE_bool("is_rel_to_truth", False, "True to normalize losses using expected losses.")

tf.flags.DEFINE_bool("is_clipped_reward", True, "True to clip rewards.")
tf.flags.DEFINE_bool("is_clipped_critic", False, "True to clip critic predictions for actor training.")

tf.flags.DEFINE_float("over_edge_penalty", 0.05, "Penalty for action going over edge of image.")

tf.flags.DEFINE_bool("is_prioritized_replay", False, "True to prioritize the replay of difficult experiences.")
tf.flags.DEFINE_bool("is_biased_prioritized_replay", False, "Priority sampling without bias correction.")

tf.flags.DEFINE_bool("is_relative_to_spirals", False, "True to compare generator losses against losses for spirals.")

tf.flags.DEFINE_bool("is_self_competition", False, "Oh it is on. True to compete against past versions of itself.")

tf.flags.DEFINE_float("norm_generator_losses_decay", 0.997, "Divide generator losses by their running mean. Zero for no normalization.")

tf.flags.DEFINE_bool("is_minmax_reward", False, "True to use highest losses for actor loss.")

tf.flags.DEFINE_integer("start_iter", 0, "Starting iteration")
tf.flags.DEFINE_integer("train_iters", 500_000, "Training iterations")
tf.flags.DEFINE_integer("val_examples", 20_000, "Number of validation examples")

tf.flags.DEFINE_float("style_loss", 1., "Weighting of style loss. Zero for no style loss.")

tf.flags.DEFINE_string("data_file",
                       "//Desktop-sa1evjv/h/96x96_stem_crops.npy",
                       "Datafile containing 19769 96x96 downsampled STEM crops.")

tf.flags.DEFINE_integer("report_freq", 10, "How often to print losses to the console.")


def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return


def construct_partial_scans(actions, observations):
    """
    actions: [batch_size, num_steps, 2]
    observations: [batch_size, num_steps, 10]
    """

    #Last action unused and the first action is always the same
    actions = np.concatenate((np.ones([FLAGS.batch_size, 1, 2]), actions[:,:-1,:]), axis=1)

    starts = 0.5*FLAGS.img_side + FLAGS.step_size*(np.cumsum(actions, axis=1) - actions)

    #starts = np.zeros(actions.shape)
    #starts[:,0,:] = actions[:,0,:]
    #for i in range(1, FLAGS.num_steps):
    #    starts[:,i,:] = actions[:,i,:] + starts[:,i-1,:]
    #starts -= actions
    #starts *= FLAGS.step_size
    #starts += 0.5*FLAGS.img_side

    positions = np.stack([starts + i*actions for i in range(FLAGS.step_size)], axis=-2)
    x = np.minimum(np.maximum(positions, 0), FLAGS.img_side-1)

    indices = []
    for j in range(FLAGS.batch_size):
        for k in range(FLAGS.num_steps):
            for i in range(FLAGS.step_size):
                indices.append( [j, int(x[j,k,i,0]), int(x[j,k,i,1])] )
    indices = np.array(indices)
    indices = tuple([indices[:,i] for i in range(3)])

    partial_scans = np.zeros([FLAGS.batch_size, FLAGS.img_side, FLAGS.img_side])
    masks = np.zeros([FLAGS.batch_size, FLAGS.img_side, FLAGS.img_side])
    
    partial_scans[indices] = observations.reshape([-1])
    masks[indices] = 1

    partial_scans /= np.maximum(masks, 1)
    masks = np.minimum(masks, 1)

    partial_scans = np.stack([partial_scans, masks], axis=-1)

    return partial_scans


def make_observations(actions, starts, full_scans):

    starts *= FLAGS.img_side
    x = np.minimum(np.maximum(np.stack([starts + i*actions for i in range(FLAGS.step_size)]), 0), FLAGS.img_side-1)

    indices = []
    for j in range(FLAGS.batch_size):
        for i in range(FLAGS.step_size):
            indices.append( [j, int(x[i][j][0]), int(x[i][j][1]), 0] )
    indices = tuple([np.array(indices)[:,i] for i in range(4)])

    observations = full_scans[indices].reshape([-1, FLAGS.step_size])

    return observations


def stepped_spiral_actions(theta_incr=np.pi/180):
    
    coverage = FLAGS.num_steps*FLAGS.step_size/FLAGS.img_side**2

    start_theta = np.pi/4
    start_r = np.sqrt(2)*FLAGS.step_size
    start_position = np.ones([2])/2

    alpha = 3.4
    theta0 = -start_r/alpha

    actions = []
    positions = [start_position]
    for _ in range(0, FLAGS.num_steps):
        
        for i in itertools.count(start=1):
            theta = start_theta + i*theta_incr
            r = alpha*(theta - theta0)

            if np.sqrt( (r*np.cos(theta) - start_r*np.cos(start_theta))**2 + 
                        (r*np.sin(theta) - start_r*np.sin(start_theta))**2 ) >= np.sqrt(2)*FLAGS.step_size:
                
                vect = np.array([r*np.cos(theta) - start_r*np.cos(start_theta), 
                                 r*np.sin(theta) - start_r*np.sin(start_theta)])
                vect /= np.sum(np.sqrt(vect**2))
                vect *= np.sqrt(2)
                start_position += FLAGS.step_size*vect/FLAGS.img_side



                actions.append( vect )
                positions.append( start_position )

                start_theta = theta
                start_r = r

                break

    actions.append( np.ones([2]) ) #Discarded

    actions = np.stack(actions)
    actions = np.stack([actions]*FLAGS.batch_size)

    positions = np.stack(positions)
    positions = np.stack([positions]*FLAGS.batch_size)

    return actions, positions


def get_obs_seq(actions, positions, imgs):

    actions = np.concatenate([np.ones([FLAGS.batch_size, 1, 2]), actions[:,1:]], axis=1)
    observations = [make_observations(actions[:,i,:], positions[:,i,:], imgs) for i in range(FLAGS.num_steps)]

    observations = np.stack(observations, axis=1)

    return observations


imgs = np.random.random([32, 96, 96, 1])

actions, positions = stepped_spiral_actions()
observations = get_obs_seq(actions, positions, imgs)
#actions = np.concatenate([actions[:,1:], actions[:, :1]], axis=1) #Last action not used, first set automatically...


partial_scans = construct_partial_scans(actions, observations)

print("Coverage:", np.sum(partial_scans[0,:,:,:1] != 0)/96**2)

disp(partial_scans[0,:,:,:1])