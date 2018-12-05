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


for i in range(1, 11):

    vae = get_model(i)
    print(vae.model_name)
    vae.define_model()
    vae.load_epoch(120)

    BLEU_scores = []
    for x in random.sample(vae.all_captions_train, 200):
        mu = vae.sentence_to_params([x])[0][0]
        one = vae.tokenizer.texts_to_sequences([x])[0]
        generated = vae.generate_from_mu(mu)

        gen_strip = ""
        for word in generated.split(" "):
            if word != '</e>':
                gen_strip += word
                gen_strip += " "
            else:
                break

        two = vae.tokenizer.texts_to_sequences([gen_strip])[0]

        BLEU_scores.append(nltk.translate.bleu_score.sentence_bleu(
            [one], two, smoothing_function=SmoothingFunction().method3))

    print('FLICKR TRAIN', np.asarray(BLEU_scores).mean())

    BLEU_scores = []
    for x in random.sample(vae.all_captions_val, 200):
        mu = vae.sentence_to_params([x])[0][0]
        one = vae.tokenizer.texts_to_sequences([x])[0]
        generated = vae.generate_from_mu(mu)

        gen_strip = ""
        for word in generated.split(" "):
            if word != '</e>':
                gen_strip += word
                gen_strip += " "
            else:
                break

        two = vae.tokenizer.texts_to_sequences([gen_strip])[0]

        BLEU_scores.append(nltk.translate.bleu_score.sentence_bleu(
            [one], two, smoothing_function=SmoothingFunction().method3))

    print('FLICKR VAL', np.asarray(BLEU_scores).mean())


for i in range(12, 22):

    vae = get_model(i)
    print(vae.model_name)
    vae.define_model()
    vae.load_epoch(120)

    BLEU_scores = []
    for x in random.sample(vae.all_captions_train, 200):
        mu = vae.sentence_to_params([x])[0][0]
        one = vae.tokenizer.texts_to_sequences([x])[0]
        generated = vae.generate_from_mu(mu)

        gen_strip = ""
        for word in generated.split(" "):
            if word != '</e>':
                gen_strip += word
                gen_strip += " "
            else:
                break

        two = vae.tokenizer.texts_to_sequences([gen_strip])[0]

        BLEU_scores.append(nltk.translate.bleu_score.sentence_bleu(
            [one], two, smoothing_function=SmoothingFunction().method3))

    print('PASCAL TRAIN', np.asarray(BLEU_scores).mean())

    BLEU_scores = []
    for x in random.sample(vae.all_captions_val, 200):
        mu = vae.sentence_to_params([x])[0][0]
        one = vae.tokenizer.texts_to_sequences([x])[0]
        generated = vae.generate_from_mu(mu)

        gen_strip = ""
        for word in generated.split(" "):
            if word != '</e>':
                gen_strip += word
                gen_strip += " "
            else:
                break

        two = vae.tokenizer.texts_to_sequences([gen_strip])[0]

        BLEU_scores.append(nltk.translate.bleu_score.sentence_bleu(
            [one], two, smoothing_function=SmoothingFunction().method3))

    print('PASCAL VAL', np.asarray(BLEU_scores).mean())
