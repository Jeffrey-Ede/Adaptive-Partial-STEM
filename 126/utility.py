import tensorflow as tf

import itertools

import numpy as np

FLAGS = tf.flags.FLAGS


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
    actions = np.stack([actions]*FLAGS.batch_size).astype(np.float32)

    positions = np.stack(positions)
    positions = np.stack([positions]*FLAGS.batch_size).astype(np.float32)

    return actions, positions


def make_observations(actions, starts, full_scans):

    x = np.minimum(np.maximum(np.stack([starts + i*actions for i in range(FLAGS.step_size)]), 0), FLAGS.img_side-1)

    indices = []
    for j in range(FLAGS.batch_size):
        for i in range(FLAGS.step_size):
            indices.append( [j, int(x[i][j][0]), int(x[i][j][1]), 0] )
    indices = tuple([np.array(indices)[:,i] for i in range(4)])

    observations = full_scans[indices].reshape([-1, FLAGS.step_size])

    return observations


def spiral_generator(scans):

    actions0, positions = stepped_spiral_actions()
    actor_actions = tf.convert_to_tensor(actions0[:,:-1], dtype=tf.float32)

    positions *= FLAGS.img_side

    def py_spiral_generator(imgs):

        actions = np.concatenate([np.ones([FLAGS.batch_size, 1, 2]), actions0[:,1:]], axis=1)
        observations = [make_observations(actions[:,i,:], positions[:,i,:], imgs) for i in range(FLAGS.num_steps)]
        
        observations = np.stack(observations, axis=1)

        return observations

    observations = tf.py_func(py_spiral_generator, [scans], tf.float32)
    observations = tf.reshape(observations, [FLAGS.batch_size, FLAGS.num_steps, FLAGS.step_size])

    return observations, actor_actions


def auto_name(name):
    """Append number to variable name to make it unique.
    
    Inputs:
        name: Start of variable name.

    Returns:
        Full variable name with number afterwards to make it unique.
    """

    scope = tf.contrib.framework.get_name_scope()
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    names = [v.name for v in vars]
    
    #Increment variable number until unused name is found
    for i in itertools.count():
        short_name = name + "_" + str(i)
        sep = "/" if scope != "" else ""
        full_name = scope + sep + short_name
        if not full_name in [n[:len(full_name)] for n in names]:
            return short_name

def alrc(
    loss, 
    num_stddev=3, 
    decay=0.999, 
    mu1_start=2, 
    mu2_start=3**2, 
    in_place_updates=False
    ):
    """Adaptive learning rate clipping (ALRC) of outlier losses.
    
    Inputs:
        loss: Loss function to limit outlier losses of.
        num_stddev: Number of standard deviation above loss mean to limit it
        to.
        decay: Decay rate for exponential moving averages used to track the first
        two raw moments of the loss.
        mu1_start: Initial estimate for the first raw moment of the loss.
        mu2_start: Initial estimate for the second raw moment of the loss.
        in_place_updates: If False, add control dependencies for moment tracking
        to tf.GraphKeys.UPDATE_OPS. This allows the control dependencies to be
        executed in parallel with other dependencies later.
    Return:
        Loss function with control dependencies for ALRC.
    """

    #Varables to track first two raw moments of the loss
    mu = tf.get_variable(
        auto_name("mu1"), 
        initializer=tf.constant(mu1_start, dtype=tf.float32))
    mu2 = tf.get_variable(
        auto_name("mu2"), 
        initializer=tf.constant(mu2_start, dtype=tf.float32))

    #Use capped loss for moment updates to limit the effect of outlier losses on the threshold
    sigma = tf.sqrt(mu2 - mu**2+1.e-8)
    loss = tf.where(loss < mu+num_stddev*sigma, 
                   loss, 
                   loss/tf.stop_gradient(loss/(mu+num_stddev*sigma)))

    #Update moment moving averages
    mean_loss = tf.reduce_mean(loss)
    mean_loss2 = tf.reduce_mean(loss**2)
    update_ops = [mu.assign(decay*mu+(1-decay)*mean_loss), 
                  mu2.assign(decay*mu2+(1-decay)*mean_loss2)]
    if in_place_updates:
        with tf.control_dependencies(update_ops):
            loss = tf.identity(loss)
    else:
        #Control dependencies that can be executed in parallel with other update
        #ops. Often, these dependencies are added to train ops e.g. alongside
        #batch normalization update ops.
        for update_op in update_ops:
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
        
    return loss


if __name__ == "__main__":
    
    pass