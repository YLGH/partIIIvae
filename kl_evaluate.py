#!/usr/bin/env python

from get_models import get_model as get_model

from get_models_ import get_model as get_model_

# from get_models import get_model

from keras import backend as K
import numpy as np

from Flickr30k.get_data import Flickr30kLoader
from pascal50S.get_data_refactor import Pascal10SLoader

import random

from CaptionCaptionEvaluator import *
from CaptionImageEvaluator import *

from sklearn.manifold import TSNE
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


for i in range(1, 22):
    if i == 11:
      continue

    vae = get_model(i)
    print(vae.model_name)
    vae.define_model()
    vae.load_epoch(120)

    z_h_mean, z_h_log_var = vae.sentence_to_params(random.sample(vae.all_captions_train, 5000))

    kl_loss = - 0.5 * \
        np.mean(1 + z_h_log_var - np.square(z_h_mean) -
               np.exp(z_h_log_var), axis=1)

    print('train', kl_loss.mean())


    z_h_mean, z_h_log_var = vae.sentence_to_params(random.sample(vae.all_captions_val, 5000))

    kl_loss = - 0.5 * \
        np.mean(1 + z_h_log_var - np.square(z_h_mean) -
               np.exp(z_h_log_var), axis=1)

    print('val', kl_loss.mean())
