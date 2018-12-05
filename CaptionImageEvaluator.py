import metrics
from keras.layers import Input, Concatenate, Dropout, BatchNormalization, Dense, Lambda
from keras.models import Model
import tensorflow as tf

from keras import backend as K
import numpy as np
import random

import pickle

from keras.preprocessing.sequence import pad_sequences

from pathlib import Path

from keras.applications.inception_v3 import InceptionV3

from utils import *


class CaptionImageEvaluator():
    def __init__(self, vae, save_aux=None):

        self.vae = vae
        self.save_aux = save_aux
        self.inception_base_model = InceptionV3(
            weights='imagenet', include_top=True)

        for layer in self.inception_base_model.layers:
            layer.trainable = False

    def generate_test_set(self, name):
        test_set = []
        for key in self.vae.data_loader.data_test:
            for captions in self.vae.data_loader.data_test[key]:
                test_set.append(
                    (key, captions, 1))
                other_key = key
                while other_key == key:
                    other_key = random.choice(
                        list(self.vae.data_loader.data_test.keys()))
                test_set.append(
                    (key, random.choice(self.vae.data_loader.data_test[other_key]), 0))

        pickle.dump(test_set, open('{}{}.p'.format(
            self.vae.data_loader.path, name), 'wb'))

    def load_test_set(self, name):
        print('loading test set', name)
        self.test_set = pickle.load(
            open('{}{}.p'.format(self.vae.data_loader.path, name), 'rb'))

    def load_model(self, hidden_size=1024, is_bow = False):
        sequence_input = Input(
            shape=(self.vae.MAX_SEQUENCE_LENGTH,), dtype='int32')

        token_embedding = self.vae.embedding_layer(
            sequence_input)

        self.vae.embedding_layer.trainable=False

        if not is_bow:
          for layer in self.vae.model_vae.layers:
              layer.trainable = False
  
          def get_embedding(sequence_embedding):
              h = self.vae.get_h(
                  sequence_embedding)
              return self.vae.get_embedding(h)
        else:
          def get_embedding(sequence_embedding):
              return K.mean(sequence_embedding, axis=1)
  
        get_embedding_lambda = Lambda(
            get_embedding, name='get_embedding')
        get_embedding_lambda.trainable = False

        sequence_embedding = get_embedding_lambda(
            token_embedding)

        image_z = Dropout(0.2)(
            self.inception_base_model.layers[-2].output)

        combined = Concatenate(axis=1)(
            [sequence_embedding, image_z])

        hidden = Dropout(0.2)(BatchNormalization()(
            Dense(hidden_size, activation='relu')(combined)))
        predict = Dense(
            1, activation='sigmoid')(hidden)

        self.model = Model(
            [sequence_input, self.inception_base_model.input], predict)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',
                                                                                  metrics.recall,
                                                                                  metrics.precision,
                                                                                  metrics.f1])

        loss = K.binary_crossentropy(K.ones(tf.shape(predict)), predict)

        self.input_grads = K.function([sequence_input, self.inception_base_model.input], K.gradients(loss, token_embedding))

        self.embedding_grads = K.function([sequence_embedding, self.inception_base_model.input], K.gradients(loss, sequence_embedding))

    def generate_caption_image_pairs(self, batch_size, data_set_='train'):
        data_set = self.vae.data_loader.data_train
        if data_set_ == 'val':
            data_set = self.vae.data_loader.data_val
        if data_set_ == 'test':
            data_set = self.vae.data_loader.data_test

        while True:
            input_captions = []
            input_images = []
            matching = []

            for i in range(batch_size):
                #           Choose positive example
                if np.random.uniform() < 0.5:
                    img_key = random.choice(
                        list(data_set.keys()))
                    input_images.append(
                        img_key)
                    input_captions.append(
                        random.choice(data_set[img_key]))
                    matching.append(1)
                else:
                    [img_key_1, img_key_2] = random.sample(
                        list(data_set.keys()), 2)
                    input_images.append(
                        img_key_1)
                    input_captions.append(
                        random.choice(data_set[img_key_2]))
                    matching.append(0)

            input_captions = self.vae.tokenizer.texts_to_sequences(
                input_captions)
            input_captions = pad_sequences(
                input_captions, maxlen=self.vae.MAX_SEQUENCE_LENGTH, padding='post')

            input_images = self.vae.data_loader.get_images(
                input_images, 224, 224)
            input_images /= 255.
            input_images -= 0.5
            input_images *= 2

            yield [input_captions, input_images], np.asarray(matching)

    def train(self,
              load_epoch=None,
              load_latest=False,
              batch_size=512,
              steps_per_epoch=512,
              start=0,
              end=100,
              save_every=5,
              validation=True):

        save_aux = self.save_aux

        if load_latest:
            load_epoch = 1
            save_path = '{}/caption_image-epoch_{}_{}.hdf5'.format(self.vae.model_name,
                                                                   save_every * load_epoch,
                                                                   save_aux)
            path = Path(save_path)

            while path.exists():
                load_epoch += 1
                save_path = '{}/caption_image-epoch_{}_{}.hdf5'.format(self.vae.model_name,
                                                                       save_every * load_epoch,
                                                                       save_aux)
                path = Path(save_path)

            load_epoch -= 1
            save_path = '{}/caption_image-epoch_{}_{}.hdf5'.format(self.vae.model_name,
                                                                   save_every * load_epoch,
                                                                   save_aux)
            self.model.load_weights(
                save_path)
            return load_epoch

        if load_epoch:
            save_path = '{}/caption_image-epoch_{}_{}.hdf5'.format(self.vae.model_name,
                                                                   load_epoch,
                                                                   save_aux)
            self.model.load_weights(
                save_path)
            return load_epoch

        print('TRAINING CAPTION IMAGE', '{}/{}-{}-{}-{}'.format(self.vae.model_name,
                                                                self.vae.encoder_name,
                                                                self.vae.latent_name,
                                                                self.vae.decoder_name, save_aux))

        for epoch in range(start, end):
            self.model.fit_generator(self.generate_caption_image_pairs(batch_size),
                                     steps_per_epoch=steps_per_epoch,
                                     epochs=save_every,
                                     use_multiprocessing=True)
            if validation:
                print('Validation ',
                      self.model.evaluate_generator(self.generate_caption_image_pairs(512, 'val'), steps=30))

            save_path = '{}/caption_image-epoch_{}_{}.hdf5'.format(self.vae.model_name,
                                                                   save_every *
                                                                   (epoch + 1),
                                                                   save_aux)
            self.model.save_weights(
                save_path)

    def validate_model(self, start_epoch=5, delta_epoch=5):
        save_aux = self.save_aux

        print('VALIDATING CAPTION IMAGE', '{}/{}-{}-{}-{}'.format(self.vae.model_name,
                                                                  self.vae.encoder_name,
                                                                  self.vae.latent_name,
                                                                  self.vae.decoder_name, save_aux))

        current_epoch = start_epoch

        save_path = '{}/caption_image-epoch_{}_{}.hdf5'.format(self.vae.model_name,
                                                               start_epoch,
                                                               save_aux)

        path = Path(save_path)

        while path.exists():

            self.model.load_weights(
                save_path)

            val_score = self.model.evaluate_generator(
                self.generate_caption_image_pairs(512), steps=4, use_multiprocessing=True)

            print('Validation Epoch',
                  current_epoch, ' scores ', val_score)

            current_epoch += delta_epoch
            save_path = '{}/caption_image-epoch_{}_{}.hdf5'.format(self.vae.model_name,
                                                                   current_epoch,
                                                                   save_aux)
            path = Path(save_path)

    def evaluate_test(self):
        save_aux = self.save_aux
        print('EVALUATING ON TEST SET', '{}/{}-{}-{}-{}'.format(self.vae.model_name,
                                                                self.vae.encoder_name,
                                                                self.vae.latent_name,
                                                                self.vae.decoder_name, save_aux))

        def test_generator(l, size):
            start = 0
            while start + size < len(l):
                img_keys = [
                    x[0] for x in l[start:start + size]]
                input_captions = [
                    x[1] for x in l[start:start + size]]
                matching = [
                    x[2] for x in l[start:start + size]]

                input_images = self.vae.data_loader.get_images(
                    img_keys, 224, 224)
                input_images /= 255.
                input_images -= 0.5
                input_images *= 2

                input_captions = self.vae.tokenizer.texts_to_sequences(
                    input_captions)
                input_captions = pad_sequences(
                    input_captions, maxlen=self.vae.MAX_SEQUENCE_LENGTH, padding='post')

                yield [input_captions, input_images], np.asarray(matching)

                start += size

            img_keys = [x[0]
                        for x in l[start:]]
            input_captions = [
                x[1] for x in l[start:]]
            matching = [x[2]
                        for x in l[start:]]

            input_images = self.vae.data_loader.get_images(
                img_keys, 224, 224)
            input_images /= 255.
            input_images -= 0.5
            input_images *= 2

            input_captions = self.vae.tokenizer.texts_to_sequences(
                input_captions)
            input_captions = pad_sequences(
                input_captions, maxlen=self.vae.MAX_SEQUENCE_LENGTH, padding='post')

            yield [input_captions, input_images], np.asarray(matching)

        print(self.model.evaluate_generator(test_generator(self.test_set, 512),
                                            steps=1 +
                                            (len(
                                                self.test_set) // 512),
                                            use_multiprocessing=True))
