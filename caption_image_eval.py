#!/usr/bin/env python
from VAE_base import VAE_base
from vae_models import *
from keras import backend as K
import numpy as np

from get_models import get_model

from Flickr30k.get_data import Flickr30kLoader
from pascal50S.get_data_refactor import Pascal10SLoader

from CaptionImageEvaluator import *
import sys

print('sysargv', sys.argv)

index = int(sys.argv[1])

vae = get_model(index)
#vae.define_model()
#vae.load_epoch(120)
#
#caption_image = CaptionImageEvaluator(
#    vae,
#    save_aux='VAE-EPOCH-120_hidden_1536')
#

caption_image = CaptionImageEvaluator(vae, save_aux='BOW-hidden-1536') 
caption_image.load_model(
    hidden_size=1536, is_bow=True)

caption_image.load_test_set('test_1')
i = 10
while i < 150:
  caption_image.train(load_epoch=i)
  caption_image.evaluate_test()
  i+=10
