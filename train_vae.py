#!/usr/bin/env python

from keras import backend as K
import numpy as np

from Flickr30k.get_data import Flickr30kLoader
from pascal50S.get_data_refactor import Pascal10SLoader

import sys

from get_models import get_model


index = int(sys.argv[1])

vae = get_model(index, True)
vae.define_model()
vae.train_model_vae(batch_size=64, end_epoch=120,
                    kl_anneal_rate=vae.kl_beta / 80)
