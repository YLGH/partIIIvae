#!/usr/bin/env python

from vae_models import *

from keras import backend as K
import numpy as np

from Flickr30k.get_data import Flickr30kLoader
from pascal50S.get_data_refactor import Pascal10SLoader


pascal_loader = Pascal10SLoader(
    'pascal50S/')
flickr_loader = Flickr30kLoader(
    'Flickr30k/')

vae = BiLSTM_Gaussian_dilate_regular(encoder_dim=512, latent_dim=256, data_loader=pascal_loader,
                                     model_name='bilstm_gauss_dilate_regular_pascal', use_glove_embedding=True)

vae.define_model()
vae.validate_model()
