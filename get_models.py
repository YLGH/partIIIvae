from VAE_base import VAE_base
from vae_models import *

from Flickr30k.get_data import Flickr30kLoader
from pascal50S.get_data_refactor import Pascal10SLoader

# flickr_loader = Flickr30kLoader('Flickr30k/')
pascal_loader = Pascal10SLoader('pascal50S/')


def get_model(index, use_glove_embedding=False):
    if index == 0:
        vae = LSTM_Gaussian_LSTM(encoder_dim=512, latent_dim=256, decoder_dim=512, data_loader=flickr_loader,
                                 model_name='lstm_gauss_lstm_beta_0_flickr', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=0.0)
    elif index == 1:
        vae = LSTM_Gaussian_LSTM(encoder_dim=512, latent_dim=256, decoder_dim=512, data_loader=flickr_loader,
                                 model_name='lstm_gauss_lstm_beta_1_flickr', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 2:
        vae = BiLSTM_Gaussian_dilate_small(encoder_dim=512, latent_dim=256, data_loader=flickr_loader,
                                           model_name='bilstm_gauss_dilate_small_beta_1_flickr', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 3:
        vae = BiLSTM_Gaussian_dilate_medium(encoder_dim=512, latent_dim=256, data_loader=flickr_loader,
                                            model_name='bilstm_gauss_dilate_medium_beta_1_flickr', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 4:
        vae = BiLSTM_Gaussian_dilate_large(encoder_dim=512, latent_dim=256, data_loader=flickr_loader,
                                           model_name='bilstm_gauss_dilate_large_beta_1_flickr', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 5:
        vae = BiLSTM_Gaussian_dilate_small(encoder_dim=512, latent_dim=256, data_loader=flickr_loader,
                                           model_name='bilstm_gauss_dilate_small_beta_8_flickr', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=8.0)
    elif index == 6:
        vae = BiLSTM_Gaussian_dilate_medium(encoder_dim=512, latent_dim=256, data_loader=flickr_loader,
                                            model_name='bilstm_gauss_dilate_medium_beta_8_flickr', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=8.0)
    elif index == 7:
        vae = BiLSTM_Gaussian_dilate_large(encoder_dim=512, latent_dim=256, data_loader=flickr_loader,
                                           model_name='bilstm_gauss_dilate_large_beta_8_flickr', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=8.0)
    elif index == 8:
        vae = BiLSTM_Gaussian_dilate_small(encoder_dim=512, latent_dim=256, data_loader=flickr_loader,
                                           model_name='bilstm_gauss_dilate_small_beta_64_flickr', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=64.0)
    elif index == 9:
        vae = BiLSTM_Gaussian_dilate_medium(encoder_dim=512, latent_dim=256, data_loader=flickr_loader,
                                            model_name='bilstm_gauss_dilate_medium_beta_64_flickr', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=64.0)
    elif index == 10:
        vae = BiLSTM_Gaussian_dilate_large(encoder_dim=512, latent_dim=256, data_loader=flickr_loader,
                                           model_name='bilstm_gauss_dilate_large_beta_64_flickr', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=64.0)
    if index == 11:
        vae = LSTM_Gaussian_LSTM(encoder_dim=512, latent_dim=256, decoder_dim=512, data_loader=pascal_loader,
                                 model_name='lstm_gauss_lstm_beta_0_pascal', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=0.0)
    elif index == 12:
        vae = LSTM_Gaussian_LSTM(encoder_dim=512, latent_dim=256, decoder_dim=512, data_loader=pascal_loader,
                                 model_name='lstm_gauss_lstm_beta_1_pascal', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 13:
        vae = BiLSTM_Gaussian_dilate_small(encoder_dim=512, latent_dim=256, data_loader=pascal_loader,
                                           model_name='bilstm_gauss_dilate_small_beta_1_pascal', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 14:
        vae = BiLSTM_Gaussian_dilate_medium(encoder_dim=512, latent_dim=256, data_loader=pascal_loader,
                                            model_name='bilstm_gauss_dilate_medium_beta_1_pascal', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 15:
        vae = BiLSTM_Gaussian_dilate_large(encoder_dim=512, latent_dim=256, data_loader=pascal_loader,
                                           model_name='bilstm_gauss_dilate_large_beta_1_pascal', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 16:
        vae = BiLSTM_Gaussian_dilate_small(encoder_dim=512, latent_dim=256, data_loader=pascal_loader,
                                           model_name='bilstm_gauss_dilate_small_beta_8_pascal', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=8.0)
    elif index == 17:
        vae = BiLSTM_Gaussian_dilate_medium(encoder_dim=512, latent_dim=256, data_loader=pascal_loader,
                                            model_name='bilstm_gauss_dilate_medium_beta_8_pascal', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=8.0)
    elif index == 18:
        vae = BiLSTM_Gaussian_dilate_large(encoder_dim=512, latent_dim=256, data_loader=pascal_loader,
                                           model_name='bilstm_gauss_dilate_large_beta_8_pascal', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=8.0)
    elif index == 19:
        vae = BiLSTM_Gaussian_dilate_small(encoder_dim=512, latent_dim=256, data_loader=pascal_loader,
                                           model_name='bilstm_gauss_dilate_small_beta_64_pascal', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=64.0)
    elif index == 20:
        vae = BiLSTM_Gaussian_dilate_medium(encoder_dim=512, latent_dim=256, data_loader=pascal_loader,
                                            model_name='bilstm_gauss_dilate_medium_beta_64_pascal', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=64.0)
    elif index == 21:
        vae = BiLSTM_Gaussian_dilate_large(encoder_dim=512, latent_dim=256, data_loader=pascal_loader,
                                           model_name='bilstm_gauss_dilate_large_beta_64_pascal', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=64.0)
    elif index == 22:
        vae = LSTM_Gumbel_LSTM(encoder_dim=512, latent_M=16, latent_N=16, decoder_dim=512,
                               data_loader=flickr_loader,
                               model_name='lstm_gumbel_lstm_beta_1_flickr',
                               word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 23:
        vae = BiLSTM_Gumbel_dilate_small(encoder_dim=512, latent_M=16, latent_N=16,
                                         data_loader=flickr_loader,
                                         model_name='bilstm_gumbel_dilate_small_beta_1_flickr',
                                         word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 24:
        vae = BiLSTM_Gumbel_dilate_medium(encoder_dim=512, latent_M=16, latent_N=16,
                                          data_loader=flickr_loader,
                                          model_name='bilstm_gumbel_dilate_medium_beta_1_flickr',
                                          word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 25:
        vae = BiLSTM_Gumbel_dilate_large(encoder_dim=512, latent_M=16, latent_N=16,
                                         data_loader=flickr_loader,
                                         model_name='bilstm_gumbel_dilate_large_beta_1_flickr',
                                         word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 126:
        vae = LSTM_Gumbel_LSTM(encoder_dim=512, latent_M=16, latent_N=16, decoder_dim=512,
                               data_loader=pascal_loader,
                               model_name='lstm_gumbel_lstm_beta_1_pascal_',
                               word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 127:
        vae = BiLSTM_Gumbel_dilate_small(encoder_dim=512, latent_M=16, latent_N=16,
                                         data_loader=pascal_loader,
                                         model_name='bilstm_gumbel_dilate_small_beta_1_pascal_',
                                         word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 128:
        vae = BiLSTM_Gumbel_dilate_medium(encoder_dim=512, latent_M=16, latent_N=16,
                                          data_loader=pascal_loader,
                                          model_name='bilstm_gumbel_dilate_medium_beta_1_pascal_',
                                          word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 129:
        vae = BiLSTM_Gumbel_dilate_large(encoder_dim=512, latent_M=16, latent_N=16,
                                         data_loader=pascal_loader,
                                         model_name='bilstm_gumbel_dilate_large_beta_1_pascal_',
                                         word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 30:
        vae = BiLSTM_Gaussian_dilate_small(encoder_dim=512, latent_dim=256, data_loader=flickr_loader,
                                           model_name='bilstm_gauss_dilate_small_beta_1_flickr_sum_categorical', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)
    elif index == 31:
        vae = BiLSTM_Gaussian_dilate_medium(encoder_dim=512, latent_dim=256, data_loader=flickr_loader,
                                            model_name='bilstm_gauss_dilate_medium_beta_1_flickr_sum_categorical', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)

    elif index == 32:
        vae = BiLSTM_Gaussian_dilate_medium(encoder_dim=512, latent_dim=256, data_loader=flickr_loader,
                                            model_name='bilstm_gauss_dilate_medium_beta_1_flickr_sum_categorical', word_embedding_dim=300, use_glove_embedding=use_glove_embedding, kl_beta=1.0)

    elif index == 33:
        vae = BOW_encoder(data_loader=flickr_loader, model_name='bow_encoder_flickr', use_glove_embedding=True)

    elif index == 34:
        vae = BOW_encoder(data_loader=pascal_loader, model_name='bow_encoder_pascal', use_glove_embedding=True)


    return vae
  

