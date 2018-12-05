#!/usr/bin/env python
from CaptionImageEvaluator import *
import sys

from get_models import get_model

index = int(sys.argv[1])

vae = get_model(index)

#vae.define_model()
#vae.load_epoch(120)

#caption_image = CaptionImageEvaluator(
#    vae, save_aux='VAE-EPOCH-120_hidden_1536')

caption_image = CaptionImageEvaluator(vae, save_aux='BOW-hidden-1536')
caption_image.load_model(
    hidden_size=1536, is_bow=True)
caption_image.train(batch_size=256, steps_per_epoch=50,
                    start=0, end=500, save_every=10, validation=False)
