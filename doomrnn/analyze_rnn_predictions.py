# Code that helps avoid overusing memory

import tensorflow as tf
tf_config = tf.ConfigProto(device_count = {'GPU':0}) #To use only CPU
#tf_config.gpu_options.allow_growth = True

#Importing the VAE and RNN.
import os
import sys
#Adding WorldModels path to pythonpath
nb_dir = os.path.split(os.getcwd())[0]
print(nb_dir)
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

#Importing the VAE
from doomrnn import ConvVAE as VAE
from doomrnn import Model as RNN
from collections import namedtuple

from gym.utils import seeding
np_random, seed = seeding.np_random(None)
import numpy as np

LATENT_SPACE_DIMENSIONALITY = 64
RNN_SIZE = 512
model_restart_factor = 10.
VAE_PATH = "old_tf_models"


def default_prediction_hps(num_mixtures):

  HyperParams = namedtuple('HyperParams', ['max_seq_len', 'seq_width',
		'rnn_size',
		'batch_size',
		'grad_clip',
		'num_mixture',
		'restart_factor',
		'learning_rate',
		'decay_rate',
		'min_learning_rate',
		'use_layer_norm',
		'use_recurrent_dropout',
		'recurrent_dropout_prob',
		'use_input_dropout',
		'input_dropout_prob',
		'use_output_dropout',
		'output_dropout_prob',
		'is_training',
		])

  return HyperParams(max_seq_len=2, # KOEChange. Was 500. Ha also uses 2 when sampling.
                     seq_width=LATENT_SPACE_DIMENSIONALITY,    # KOEChange. Was 32
                     rnn_size=RNN_SIZE,    # number of rnn cells
                     batch_size=1,   # minibatch sizes
                     grad_clip=1.0,
                     num_mixture=num_mixtures,   # number of mixtures in MDN
                     restart_factor=model_restart_factor, # factor of importance for restart=1 rare case for loss.
                     learning_rate=0.001,
                     decay_rate=0.99999,
                     min_learning_rate=0.00001,
                     use_layer_norm=0, # set this to 1 to get more stable results (less chance of NaN), but slower
                     use_recurrent_dropout=0,
                     recurrent_dropout_prob=0.90,
                     use_input_dropout=0,
                     input_dropout_prob=0.90,
                     use_output_dropout=0,
                     output_dropout_prob=0.90,
                     is_training=0)

def get_pi_idx(x, pdf):
  # samples from a categorial distribution
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  random_value = np.random.randint(N)
  #print('error with sampling ensemble, returning random', random_value)
  return random_value



class RNNAnalyzer:

    def __init__(self, rnn_load_path, num_mixtures, temperature):
        #RNN parameters - modelled after hps_sample in doomrnn.py
        self.vae=VAE(z_size=LATENT_SPACE_DIMENSIONALITY,
                      batch_size=1,
                      is_training=False,
                      reuse=False,
                      gpu_mode=False)

        self.vae.load_json(os.path.join(VAE_PATH, 'vae.json'))
        hps = default_prediction_hps(num_mixtures)
        self.rnn = RNN(hps, gpu_mode=False)

        self.rnn.load_json(os.path.join(rnn_load_path, 'rnn.json'))
        self.frame_count = 0
        self.temperature = temperature
        self.zero_state = self.rnn.sess.run(self.rnn.zero_state)
        self.outwidth = self.rnn.hps.seq_width
        self.restart = 1
        self.rnn_state = self.zero_state

    def _reset(self, initial_z):
        #Resets RNN, with an initial z.
        self.rnn_state = self.zero_state
        self.z = initial_z
        self.restart = 1
        self.frame_count = 0

    def decode_with_vae(self, latent_vector_sequence):
        reconstructions = self.vae.decode(np.array(latent_vector_sequence))
        return reconstructions


    def predict_one_step(self, action, previous_z=[]):
        #Predicts one step ahead from the previous state.
        #If previous z is given, we predict with that as input. Otherwise, we dream from the previous output we generated.
        print("Test")
        self.frame_count += 1
        prev_z = np.zeros((1, 1, self.outwidth))
        if len(previous_z)>0:
            prev_z[0][0] = previous_z
        else:
            prev_z[0][0] = self.z

        prev_action = np.zeros((1, 1))
        prev_action[0] = action

        prev_restart = np.ones((1, 1))
        prev_restart[0] = self.restart

        s_model = self.rnn

        feed = {s_model.input_z: prev_z,
                s_model.input_action: prev_action,
                s_model.input_restart: prev_restart,
                s_model.initial_state: self.rnn_state
               }

        [logmix, mean, logstd, logrestart, next_state] = s_model.sess.run([s_model.out_logmix,
                                                                           s_model.out_mean,
                                                                           s_model.out_logstd,
                                                                           s_model.out_restart_logits,
                                                                           s_model.final_state],
                                                                          feed)

        OUTWIDTH = self.outwidth
        # adjust temperatures
        logmix2 = np.copy(logmix)/self.temperature
        logmix2 -= logmix2.max()
        logmix2 = np.exp(logmix2)
        logmix2 /= logmix2.sum(axis=1).reshape(OUTWIDTH, 1)

        mixture_idx = np.zeros(OUTWIDTH)
        chosen_mean = np.zeros(OUTWIDTH)
        chosen_logstd = np.zeros(OUTWIDTH)
        for j in range(OUTWIDTH):
          idx = get_pi_idx(np_random.rand(), logmix2[j])
          mixture_idx[j] = idx
          chosen_mean[j] = mean[j][idx]
          chosen_logstd[j] = logstd[j][idx]

        rand_gaussian = np_random.randn(OUTWIDTH)*np.sqrt(self.temperature)
        next_z = chosen_mean+np.exp(chosen_logstd)*rand_gaussian
        self.restart = 0
        next_restart = 0 #Never telling it that we got a restart.
        #if (logrestart[0] > 0):
        #next_restart = 1

        self.z = next_z
        self.restart = next_restart
        self.rnn_state = next_state


        return next_z, logmix2



