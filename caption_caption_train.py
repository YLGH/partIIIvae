#!/usr/bin/env python


from VAE_base import VAE_base
from encoders import encoder_BiLSTM
from latent import gaussian as latent_gaussian
from decoder import decoder_dilated_conv_small, decoder_dilated_conv_medium

from keras import backend as K
import numpy as np

from Flickr30k.get_data import Flickr30kLoader

from CaptionCaptionEvaluator import *


flickr_loader = Flickr30kLoader(
    'Flickr30k/')


class BiLSTM_Gaussian_conv_med(VAE_base):
    def define_encoder(self):
        encoder_BiLSTM(self, 512)

    def define_latent(self):
        latent_gaussian(self, 128)

    def define_decoder(self):
        decoder_dilated_conv_medium(
            self)

    def auxiliary(self):
        self.get_params = K.function([self.sequence_input], [
                                     self.z_h_mean, self.z_h_log_var])
        self.generate = K.function([self.z_h, self.decoder_input], [
                                   self.decoder_distribution_outputs])


vae = BiLSTM_Gaussian_conv_med(
    model_name='bilstm_gasus_dilate_med', data_loader=flickr_loader, word_embedding_dim=300)
vae.define_model()
vae.train_model_vae(load_epoch=190)

vae.save_embeddings('epoch_190')
vae.load_embeddings('epoch_190')

caption_caption = CaptionCaptionEvaluator(
    vae)
caption_caption.load_model()
caption_caption.train(batch_size=512, steps_per_epoch=30,
                      save_aux='VAE-EPOCH-190')
