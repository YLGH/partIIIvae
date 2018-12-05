from abc import ABC, abstractmethod
from VAE_base_keras_imports import *

import tensorflow as tf


class VAE_base(ABC):
    def __init__(self,
                 data_loader,
                 model_name,
                 kl_beta=1.0,
                 word_embedding_dim=300,
                 max_sequence_length=30,
                 max_num_words=10000,
                 epsilon_std=1,
                 use_glove_embedding=False):

        path = Path(
            '{}/'.format(model_name))
        if not path.exists():
            path.mkdir()

        self.data_loader = data_loader
        self.MAX_SEQUENCE_LENGTH = max_sequence_length
        self.MAX_NUM_WORDS = max_num_words
        self.EMBEDDING_DIM = word_embedding_dim

        self.kl_beta = kl_beta

        self.epsilon_std = epsilon_std
        self.KL_error = K.variable(0.0)
        self.model_name = model_name

        self.all_captions_train = []
        self.all_captions_val = []
        for img_id in self.data_loader.data_train:
            self.all_captions_train += self.data_loader.data_train[img_id]
        for img_id in self.data_loader.data_val:
            self.all_captions_val += self.data_loader.data_val[img_id]

        self.tokenizer = Tokenizer(num_words=self.MAX_NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True)

        self.tokenizer.fit_on_texts(
            self.all_captions_train)
        self.word_index = self.tokenizer.word_index
        self.word_index['</e>'] = 0
        self.word_index['</s>'] = len(
            self.tokenizer.word_index)
        self.index_word = {}

        for word in self.word_index:
            self.index_word[self.word_index[word]] = word

        if use_glove_embedding:
            self.EMBEDDING_DIM = 300

            GLOVE_DIR = 'Glove'
            embeddings_index = {}

            f = open(os.path.join(
                GLOVE_DIR, 'glove.6B.300d.txt'), 'rb')

            for line in f:
                values = str(
                    line.decode()).rstrip().split(" ")
                word = values[0]
                coefs = np.asarray(
                    values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()
            print('Found %s word vectors.' % len(
                embeddings_index))

            self.embedding_matrix = np.zeros(
                (len(self.word_index), self.EMBEDDING_DIM))

            for word, i in self.word_index.items():
                embedding_vector = embeddings_index.get(
                    word)
                if embedding_vector is not None:
                    self.embedding_matrix[i] = embedding_vector

            self.embedding_matrix[self.word_index['</e>']
                                  ] = np.ones(self.EMBEDDING_DIM)
            self.embedding_matrix[self.word_index['</s>']
                                  ] = np.zeros(self.EMBEDDING_DIM)


            self.embedding_layer = Embedding(input_dim=len(self.word_index),
                                             output_dim=self.EMBEDDING_DIM,
                                             input_length=self.MAX_SEQUENCE_LENGTH,
                                             weights=[
                                                 self.embedding_matrix],
                                             trainable=True)

        else:
            self.embedding_layer = Embedding(input_dim=len(self.word_index),
                                             output_dim=self.EMBEDDING_DIM,
                                             input_length=self.MAX_SEQUENCE_LENGTH,
                                             trainable=True)

        print('Found %s unique tokens.' %
              len(self.word_index))
        self.NUM_WORDS = len(
            self.word_index)

#   Sets hidden state that is passed onto latent layer
    @abstractmethod
    def define_encoder(self):
        pass

#   Samples latent z that is passed onto decoder.
    @abstractmethod
    def define_latent(self):
        pass

#   Sets decoder outputs
    @abstractmethod
    def define_decoder(self):
        pass

    def auxiliary(self):
        pass

    def define_model(self):
        self.define_encoder()
        self.define_latent()
        self.define_decoder()
        self.auxiliary()

        def vae_loss(x, x_cat):

            prob_loss = K.sum(
                objectives.categorical_crossentropy(x, x_cat), axis=-1)

            loss = prob_loss + \
                self.KL_error * \
                (self.kl_loss)

            return loss

        self.model_vae = Model(
            [self.sequence_input, self.decoder_input], self.decoder_distribution_outputs)
        self.model_vae.compile(optimizer=keras.optimizers.adam(
            lr=1e-3, beta_1=0.5), loss=[vae_loss])

        if self.latent_type == 'gaussian':
            self.model_vae_params = Model(
                self.sequence_input, self.z_h_mean)
            self.get_params = K.function([self.sequence_input], [
                self.z_h_mean, self.z_h_log_var])

        elif self.latent_type == 'gumbel':
            self.model_vae_params = Model(
                self.sequence_input, self.logits)
            self.get_params = K.function([self.sequence_input], [self.z_h])


        self.generate = K.function([self.z_h, self.decoder_input], [
                                   self.decoder_distribution_outputs])

    def generator_text_in(self, batch_size):
        while True:
            input_text_data = random.sample(
                self.all_captions_train, batch_size)
            batch_input_text_data = self.tokenizer.texts_to_sequences(
                input_text_data)
            batch_input_text_data = pad_sequences(
                batch_input_text_data, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')

            batch_decoder_inputs = np.roll(
                batch_input_text_data, 1)
            batch_decoder_inputs[:,
                                 0] = self.word_index['</s>']

            batch_decoder_sequence_data = np.zeros(
                (batch_size, self.MAX_SEQUENCE_LENGTH, len(self.word_index)))

            for i in range(batch_size):
                for token_index, word_index in enumerate(
                        batch_input_text_data[i]):
                    word_index = int(
                        word_index)
                    batch_decoder_sequence_data[i][token_index][word_index] = 1.0

            yield [batch_input_text_data, batch_decoder_inputs], batch_decoder_sequence_data

    def load_epoch(self, load_epoch):
        save_path = '{}/{}-{}-{}-epoch_{}.hdf5'.format(self.model_name,
                                                       self.encoder_name,
                                                       self.latent_name,
                                                       self.decoder_name,
                                                       load_epoch)
        self.model_vae.load_weights(
            save_path)

    def train_model_vae(self,
                        batch_size=512,
                        start_epoch=0,
                        end_epoch=10,
                        kl_anneal_rate=0.1,
                        tau_anneal_rate=0.0003,
                        min_temperature=0.5):

        print('Training ', '{}/{}-{}-{}'.format(self.model_name,
                                                self.encoder_name,
                                                self.latent_name,
                                                self.decoder_name))

        num_batches_per_epoch = int(
            50 * 800 / batch_size)

        self.kl_anneal_rate = kl_anneal_rate

        kl_value = 0.0

        for epoch in range(start_epoch, end_epoch):
            print('EPOCH : ', epoch)
            kl_value = min(
                self.kl_beta, self.kl_anneal_rate * epoch)
            print(
                'KL WEIGHT ', kl_value)
            K.set_value(
                self.KL_error, kl_value)

            if self.latent_type == 'gumbel':
                next_tau = np.max([K.get_value(self.tau) *
                                   np.exp(
                                       -tau_anneal_rate * (100 + epoch)),
                                   min_temperature])

                print('TAU', next_tau)

                K.set_value(
                    self.tau, next_tau)

            self.model_vae.fit_generator(self.generator_text_in(batch_size),
                                         steps_per_epoch=num_batches_per_epoch,
                                         epochs=1)

            if (epoch + 1) % 10 == 0:
                save_path = '{}/{}-{}-{}-epoch_{}.hdf5'.format(self.model_name,
                                                               self.encoder_name,
                                                               self.latent_name,
                                                               self.decoder_name,
                                                               epoch + 1)
                self.model_vae.save_weights(
                    save_path)

    def validate_model(self, start_epoch=10, end_epoch=490, delta_epoch=10):

        print('VALIDATING ', '{}/{}-{}-{}'.format(self.model_name,
                                                  self.encoder_name,
                                                  self.latent_name,
                                                  self.decoder_name))

        current_epoch = start_epoch

        while current_epoch < end_epoch:
            save_path = '{}/{}-{}-{}-epoch_{}.hdf5'.format(self.model_name,
                                                           self.encoder_name,
                                                           self.latent_name,
                                                           self.decoder_name,
                                                           current_epoch)
            self.model_vae.load_weights(
                save_path)

            val_score = self.model_vae.evaluate_generator(
                self.generator_text_in(1024), steps=4, use_multiprocessing=True)
            print('Validation Epoch',
                  current_epoch, ' scores ', val_score)

            current_epoch += delta_epoch

    def generate_from_mu(self, mu):
        decoder_in = np.ones(
            (self.MAX_SEQUENCE_LENGTH), dtype=int)
        next_index = self.word_index['</s>']
        decoded_sentence = ""
        for index in range(self.MAX_SEQUENCE_LENGTH):
            decoder_in[index] = next_index
            predict = self.generate(
                [[mu], [decoder_in]])[0][0][index]
            most_likely_word_index = np.argmax(
                predict)
            next_index = most_likely_word_index
            decoded_sentence += self.index_word[most_likely_word_index] + " "
        return decoded_sentence

    def sentence_to_params(self, sentences):
        batch_input_text_data = self.tokenizer.texts_to_sequences(
            sentences)
        batch_input_text_data = pad_sequences(
            batch_input_text_data, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')

        return np.asarray(self.get_params([batch_input_text_data]))

    def decode_sentence(self, sentences):

        batch_input_text_data = self.tokenizer.texts_to_sequences(
            sentences)
        batch_input_text_data = pad_sequences(
            batch_input_text_data, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')

        batch_decoder_inputs = np.roll(
            batch_input_text_data, 1)
        batch_decoder_inputs[:,
                             0] = self.word_index['</s>']

        params = self.model_vae.predict(
            [batch_input_text_data, batch_decoder_inputs])

        for sentence_index, sentence in enumerate(params):
            print('Actual Sentence : ',
                  sentences[sentence_index])
            decoded_sentence = ""
            for word_in_sentence_index in sentence:
                most_likely_word_index = np.argmax(
                    word_in_sentence_index)
                decoded_sentence += self.index_word[most_likely_word_index] + " "
            print(
                "Decoded Sentence : ", decoded_sentence)
            print(
                "================================")
