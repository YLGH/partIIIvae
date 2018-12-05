from VAE_base import VAE_base
from encoders import *
from latent import *
from decoder import *

from keras import backend as K


class LSTM_Gaussian_LSTM(VAE_base):
    def __init__(self, encoder_dim, latent_dim, decoder_dim, *args, **kwargs):
        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        self.decoder_dim = decoder_dim

        super(LSTM_Gaussian_LSTM, self).__init__(
            *args, **kwargs)

    def define_encoder(self):
        encoder_LSTM(
            self, self.encoder_dim)

    def define_latent(self):
        latent_gaussian(
            self, self.latent_dim)

    def define_decoder(self):
        decoder_LSTM(
            self, self.decoder_dim)


class BiLSTM_Gaussian_dilate_small(VAE_base):
    def __init__(self, encoder_dim, latent_dim, *args, **kwargs):
        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        super(BiLSTM_Gaussian_dilate_small,
              self).__init__(*args, **kwargs)

    def define_encoder(self):
        encoder_BiLSTM(
            self, self.encoder_dim)

    def define_latent(self):
        latent_gaussian(
            self, self.latent_dim)

    def define_decoder(self):
        decoder_dilated_conv_small(self)


class BiLSTM_Gaussian_dilate_medium(VAE_base):
    def __init__(self, encoder_dim, latent_dim, *args, **kwargs):
        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        super(BiLSTM_Gaussian_dilate_medium,
              self).__init__(*args, **kwargs)

    def define_encoder(self):
        encoder_BiLSTM(
            self, self.encoder_dim)

    def define_latent(self):
        latent_gaussian(
            self, self.latent_dim)

    def define_decoder(self):
        decoder_dilated_conv_medium(
            self)


class BiLSTM_Gaussian_dilate_large(VAE_base):
    def __init__(self, encoder_dim, latent_dim, *args, **kwargs):
        self.encoder_dim = encoder_dim
        self.latent_dim = latent_dim
        super(BiLSTM_Gaussian_dilate_large,
              self).__init__(*args, **kwargs)

    def define_encoder(self):
        encoder_BiLSTM(
            self, self.encoder_dim)

    def define_latent(self):
        latent_gaussian(
            self, self.latent_dim)

    def define_decoder(self):
        decoder_dilated_conv_large(self)


class LSTM_Gumbel_LSTM(VAE_base):
    def __init__(self, encoder_dim, latent_M, latent_N,
                 decoder_dim, *args, **kwargs):
        self.encoder_dim = encoder_dim

        self.latent_M = latent_M
        self.latent_N = latent_N

        self.decoder_dim = decoder_dim

        super(LSTM_Gumbel_LSTM, self).__init__(
            *args, **kwargs)

    def define_encoder(self):
        encoder_LSTM(
            self, self.encoder_dim)

    def define_latent(self):
        latent_gumbel(
            self, self.latent_M, self.latent_N)

    def define_decoder(self):
        decoder_LSTM(
            self, self.decoder_dim)


class BiLSTM_Gumbel_dilate_small(VAE_base):
    def __init__(self, encoder_dim, latent_M, latent_N, *args, **kwargs):
        self.encoder_dim = encoder_dim

        self.latent_M = latent_M
        self.latent_N = latent_N
        super(BiLSTM_Gumbel_dilate_small,
              self).__init__(*args, **kwargs)

    def define_encoder(self):
        encoder_BiLSTM(
            self, self.encoder_dim)

    def define_latent(self):
        latent_gumbel(
            self, self.latent_M, self.latent_N)

    def define_decoder(self):
        decoder_dilated_conv_small(self)


class BiLSTM_Gumbel_dilate_medium(VAE_base):
    def __init__(self, encoder_dim, latent_M, latent_N, *args, **kwargs):
        self.encoder_dim = encoder_dim

        self.latent_M = latent_M
        self.latent_N = latent_N
        super(BiLSTM_Gumbel_dilate_medium,
              self).__init__(*args, **kwargs)

    def define_encoder(self):
        encoder_BiLSTM(
            self, self.encoder_dim)

    def define_latent(self):
        latent_gumbel(
            self, self.latent_M, self.latent_N)

    def define_decoder(self):
        decoder_dilated_conv_medium(
            self)


class BiLSTM_Gumbel_dilate_large(VAE_base):
    def __init__(self, encoder_dim, latent_M, latent_N, *args, **kwargs):
        self.encoder_dim = encoder_dim

        self.latent_M = latent_M
        self.latent_N = latent_N

        super(BiLSTM_Gumbel_dilate_large,
              self).__init__(*args, **kwargs)

    def define_encoder(self):
        encoder_BiLSTM(
            self, self.encoder_dim)

    def define_latent(self):
        latent_gumbel(
            self, self.latent_M, self.latent_N)

    def define_decoder(self):
        decoder_dilated_conv_large(self)

class BOW_encoder(VAE_base):
    def __init__(self, *args, **kwargs):
        self.encoder_name = 'bow'
        self.latent_name = ''
        self.decoder_name = ''
        super(BOW_encoder,
              self).__init__(*args, **kwargs)

    def define_encoder(self):
      pass

    def define_latent(self):
      pass

    def define_decoder(self):
      pass

