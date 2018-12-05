import metrics
from keras.layers import Input, Concatenate, Dropout, BatchNormalization, Dense, Lambda
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import random


class CaptionCaptionEvaluator():
    def __init__(self, vae):
        self.vae = vae

    def load_model(self, hidden_size=384):

        sequence_input_1 = Input(
            shape=(self.vae.MAX_SEQUENCE_LENGTH,), dtype='int32')
        sequence_input_2 = Input(
            shape=(self.vae.MAX_SEQUENCE_LENGTH,), dtype='int32')

        sequence_embedding_1 = self.vae.embedding_layer(
            sequence_input_1)
        sequence_embedding_2 = self.vae.embedding_layer(
            sequence_input_2)

        self.vae.encoder_layer.trainable = False
        self.vae.embedding_layer.trainable = False

        def get_mu(sequence_embedding):
            h = self.vae.get_h(
                sequence_embedding)
            return self.vae.z_h_mean_layer(h)

        get_mu_lambda = Lambda(
            get_mu, name='get_mu')
        get_mu_lambda.trainable = False

        sequence_embedding_1 = get_mu_lambda(
            sequence_embedding_1)
        sequence_embedding_2 = get_mu_lambda(
            sequence_embedding_2)

        combined = Concatenate(axis=1)(
            [sequence_embedding_1, sequence_embedding_2])

        hidden = Dropout(0.2)(BatchNormalization()(
            Dense(hidden_size, activation='relu')(combined)))
        predict = Dense(
            1, activation='sigmoid')(hidden)

        self.model = Model(
            [sequence_input_1, sequence_input_2], predict)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',
                                                                                  metrics.recall,
                                                                                  metrics.precision,
                                                                                  metrics.f1])

    def generate_caption_caption_pairs(self, batch_size, data_set_='train'):
        data_set = self.vae.data_loader.data_train
        if data_set_ == 'val':
            data_set = self.vae.data_loader.data_val
        if data_set_ == 'test':
            data_set = self.vae.data_loader.data_test

        while True:
            input_captions_1 = []
            input_captions_2 = []
            matching = []
            for i in range(batch_size):
                #           Choose positive example
                if np.random.uniform() < 0.5:
                    img_key = random.choice(
                        list(data_set.keys()))
                    [caption_1, caption_2] = random.sample(
                        data_set[img_key], 2)
                    input_captions_1.append(
                        caption_1)
                    input_captions_2.append(
                        caption_2)
                    matching.append(1)
                else:
                    [img_key_1, img_key_2] = random.sample(
                        list(data_set.keys()), 2)
                    input_captions_1.append(
                        random.choice(data_set[img_key_1]))
                    input_captions_2.append(
                        random.choice(data_set[img_key_2]))
                    matching.append(0)

            input_captions_1 = self.vae.tokenizer.texts_to_sequences(
                input_captions_1)
            input_captions_1 = pad_sequences(
                input_captions_1, maxlen=self.vae.MAX_SEQUENCE_LENGTH, padding='post')

            input_captions_2 = self.vae.tokenizer.texts_to_sequences(
                input_captions_2)
            input_captions_2 = pad_sequences(
                input_captions_2, maxlen=self.vae.MAX_SEQUENCE_LENGTH, padding='post')

            yield [input_captions_1, input_captions_2], np.asarray(matching)

    def train(self,
              batch_size=64,
              steps_per_epoch=512,
              save_every=5,
              validation=True,
              save_aux=""):

        print("TRAINING CAPTION CAPTION MODEL",
              self.vae.model_name)

        for epoch in range(100):
            self.model.fit_generator(self.generate_caption_caption_pairs(batch_size),
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=save_every,
                                     use_multiprocessing=True)

            print('Validation ',
                  self.model.evaluate_generator(self.generate_caption_caption_pairs(512, 'val'), steps=30))

            save_path = '{}/caption_caption-epoch_{}_{}.hdf5'.format(self.vae.model_name,
                                                                     save_every *
                                                                     (epoch + 1),
                                                                     save_aux)
            self.model.save_weights(
                save_path)
