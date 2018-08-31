'''
Uses pretrained VAE to process dataset to get mu and logvar for each frame, and stores
all the dataset files into one dataset called series/series.npz
'''
import glob

import numpy as np
import os
import json
import tensorflow as tf
np.set_printoptions(precision=4, edgeitems=6, linewidth=100, suppress=True)

import random
from doomrnn import reset_graph, ConvVAE

DATA_DIR = "record"
SERIES_DIR = "series"
model_path_name = "tf_vae"

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # disable GPU

if not os.path.exists(SERIES_DIR):
  os.makedirs(SERIES_DIR)

def load_raw_data_list(filelist):
  data_list = []
  action_list = []
  counter = 0
  for i in range(len(filelist)):
    filename = filelist[i]
    raw_data = np.load(os.path.join(DATA_DIR, filename))
    data_list.append(raw_data['obs'])
    action_list.append(raw_data['action'])
    if ((i+1) % 1000 == 0):
      print("loading file", (i+1))
  return data_list, action_list

def encode(img):
  simple_obs = np.copy(img).astype(np.float)/255.0
  simple_obs = simple_obs.reshape(1, 64, 64, 3)
  mu, logvar = vae.encode_mu_logvar(simple_obs)
  z = (mu + np.exp(logvar/2.0) * np.random.randn(*logvar.shape))[0]
  return mu[0], logvar[0], z

def decode(z):
  # decode the latent vector
  img = vae.decode(z.reshape(1, 64)) * 255.
  img = np.round(img).astype(np.uint8)
  img = img.reshape(64, 64, 3)
  return img


# Hyperparameters for ConvVAE
z_size=32
batch_size=1
learning_rate=0.0001
kl_tolerance=0.5

#Kais data loading code ---------------------
obs_data = []
action_data = []
loadfolder = "../../WorldModels/data_small_episodes/"
obs_filename_base = 'obs_data_doomrnn_'
actions_filename_base = 'action_data_doomrnn_'
obs_file_pattern = os.path.join(loadfolder, obs_filename_base + '*')
action_file_pattern = os.path.join(loadfolder, actions_filename_base + '*')
for file_number in range(1, len(glob.glob(obs_file_pattern)) + 1):
  obs_file = os.path.join(loadfolder, obs_filename_base) + str(file_number) + ".npy"
  action_file = os.path.join(loadfolder, actions_filename_base) + str(file_number) + ".npy"
  print("Loading obs file ", obs_file)
  for episode in np.load(obs_file):
    obs_data.append(episode)
  print("loading action file ", action_file)
  for episode in np.load(action_file):
    action_data.append(episode)
print("-----LOADING FILES DONE -------")

obs_data = np.array(obs_data)
action_data = np.array(action_data)

print("Obs data has shape ", obs_data.shape)
print("action data has shape ", action_data.shape)

# Need to store each ep separately. we cant predict btw episodes
# TODO Note: There are equally many actions and observations. I guess the final action can just be discarded?
z_sequences = []  # One for each ep
action_sequences = []  # One for each ep
for episode_number in range(len(obs_data)):
  observations = np.array(obs_data[episode_number])
  # Generating all latent codes for this episode
  latent_values = vae.generate_latent_variables(observations)
  z_sequences.append(latent_values)
  action_sequences.append(np.array(action_data[episode_number]))

  print("Added latent sequences of length ", len(latent_values), " and action sequence of length ",
        len(action_sequences[-1]))
  print("Array sizes: ", len(z_sequences), ", ", len(action_sequences))
z_sequences = np.array(z_sequences)  # Will this work? Has sub-arrays of differing lengths.

#End Kais data loading code


dataset, action_dataset = z_sequences, action_sequences

reset_graph()

vae = ConvVAE(z_size=z_size,
              batch_size=batch_size,
              learning_rate=learning_rate,
              kl_tolerance=kl_tolerance,
              is_training=False,
              reuse=False,
              gpu_mode=False)

vae.load_json(os.path.join(model_path_name, 'vae.json'))

mu_dataset = []
logvar_dataset = []
for i in range(len(dataset)):
  data = dataset[i]
  datalen = len(data)
  mu_data = []
  logvar_data = []
  for j in range(datalen):
    img = data[j]
    mu, logvar, z = encode(img)
    mu_data.append(mu)
    logvar_data.append(logvar)
  mu_data = np.array(mu_data, dtype=np.float16)
  logvar_data = np.array(logvar_data, dtype=np.float16)
  mu_dataset.append(mu_data)
  logvar_dataset.append(logvar_data)
  if (i+1) % 100 == 0:
    print(i+1)

dataset = np.array(dataset)
action_dataset = np.array(action_dataset)
mu_dataset = np.array(mu_dataset)
logvar_dataset = np.array(logvar_dataset)

np.savez_compressed(os.path.join(SERIES_DIR, "series.npz"), action=action_dataset, mu=mu_dataset, logvar=logvar_dataset)
