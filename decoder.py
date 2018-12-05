from keras.layers import Input, LSTM, RepeatVector, Concatenate, TimeDistributed, Dense, BatchNormalization, Conv1D, Dropout
import keras


def decoder_LSTM(self, lstm_size):
    self.decoder_input = Input(
        shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32', name='decoder_Input')

    decoder_inputs = RepeatVector(
        self.MAX_SEQUENCE_LENGTH)(self.z_h)

    embedded_decoder_input = self.embedding_layer(
        self.decoder_input)

    decoder_inputs = Concatenate(
        axis=-1)([decoder_inputs, embedded_decoder_input])

    decoder_lstm = LSTM(lstm_size,
                        return_sequences=True,
                        return_state=True,
                        dropout=0.2,
                        name='decoder_lstm')

    decoder_outputs, _, _ = decoder_lstm(
        decoder_inputs)

    self.decoder_distribution_outputs = TimeDistributed(Dense(len(self.word_index),
                                                              activation='softmax',
                                                            ))(decoder_inputs)

    self.decoder_name = '{}_{}'.format(
        'LSTM', lstm_size)


def decoder_dilated_conv_small(self, dropout = 0.0):
    self.decoder_input = Input(
        shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32', name='decoder_input')

    decoder_inputs = RepeatVector(
        self.MAX_SEQUENCE_LENGTH)(self.z_h)
    embedded_decoder_input = self.embedding_layer(
        self.decoder_input)

    embedded_decoder_input = Dropout(dropout)(embedded_decoder_input)

    decoder_inputs = Concatenate(
        axis=-1)([decoder_inputs, embedded_decoder_input])

    self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=512,
                                                                 kernel_size=1,
                                                                 dilation_rate=1,
                                                                 activation='relu',
                                                                 padding='causal')(decoder_inputs)))

    self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=512,
                                                                 kernel_size=2,
                                                                 dilation_rate=1,
                                                                 activation='relu',
                                                                 padding='causal')(self.decoder_conv)))

    self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=512,
                                                                 kernel_size=1,
                                                                 dilation_rate=1,
                                                                 activation='relu',
                                                                 padding='causal')(self.decoder_conv)))

    self.decoder_conv = Concatenate(
        axis=-1)([decoder_inputs, self.decoder_conv])

    self.decoder_distribution_outputs = TimeDistributed(Dense(len(self.word_index),
                                                              activation='softmax',
                                                              ))(self.decoder_conv)

    self.decoder_name = 'dilated_conv_small'


def decoder_dilated_conv_medium(self, dropout = 0.0):
    self.decoder_input = Input(
        shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32', name='decoder_input')
    decoder_inputs = RepeatVector(
        self.MAX_SEQUENCE_LENGTH)(self.z_h)
    embedded_decoder_input = self.embedding_layer(
        self.decoder_input)


    embedded_decoder_input = Dropout(dropout)(embedded_decoder_input)

    decoder_inputs = Concatenate(
        axis=-1)([decoder_inputs, embedded_decoder_input])

    self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=512,
                                                                 kernel_size=1,
                                                                 dilation_rate=1,
                                                                 activation='relu',
                                                                 padding='causal')(decoder_inputs)))

    self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=512,
                                                                 kernel_size=2,
                                                                 dilation_rate=1,
                                                                 activation='relu',
                                                                 padding='causal')(self.decoder_conv)))

    self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=512,
                                                                 kernel_size=2,
                                                                 dilation_rate=4,
                                                                 activation='relu',
                                                                 padding='causal')(self.decoder_conv)))

    self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=512,
                                                                 kernel_size=1,
                                                                 dilation_rate=1,
                                                                 activation='relu',
                                                                 padding='causal')(self.decoder_conv)))

    self.decoder_conv = Concatenate(
        axis=-1)([decoder_inputs, self.decoder_conv])

    self.decoder_distribution_outputs = TimeDistributed(Dense(len(self.word_index),
                                                              activation='softmax',
                                                              ))(self.decoder_conv)

    self.decoder_name = 'dilated_conv_medium'


def decoder_dilated_conv_large(self, dropout = 0.0):
    self.decoder_input = Input(
        shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32', name='decoder_input')
    decoder_inputs = RepeatVector(
        self.MAX_SEQUENCE_LENGTH)(self.z_h)
    embedded_decoder_input = self.embedding_layer(
        self.decoder_input)

    embedded_decoder_input = Dropout(dropout)(embedded_decoder_input)

    decoder_inputs = Concatenate(
        axis=-1)([decoder_inputs, embedded_decoder_input])

    self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=512,
                                                                 kernel_size=1,
                                                                 dilation_rate=1,
                                                                 activation='relu',
                                                                 padding='causal')(decoder_inputs)))

    self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=512,
                                                                 kernel_size=2,
                                                                 dilation_rate=1,
                                                                 activation='relu',
                                                                 padding='causal')(self.decoder_conv)))

    self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=512,
                                                                 kernel_size=3,
                                                                 dilation_rate=2,
                                                                 activation='relu',
                                                                 padding='causal')(self.decoder_conv)))

    self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=512,
                                                                 kernel_size=2,
                                                                 dilation_rate=4,
                                                                 activation='relu',
                                                                 padding='causal')(self.decoder_conv)))

    self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=512,
                                                                 kernel_size=2,
                                                                 dilation_rate=8,
                                                                 activation='relu',
                                                                 padding='causal')(self.decoder_conv)))

    self.decoder_conv = Dropout(0.1)(BatchNormalization()(Conv1D(filters=512,
                                                                 kernel_size=1,
                                                                 dilation_rate=1,
                                                                 activation='relu',
                                                                 padding='causal')(self.decoder_conv)))

    self.decoder_conv = Concatenate(
        axis=-1)([decoder_inputs, self.decoder_conv])

    self.decoder_distribution_outputs = TimeDistributed(Dense(len(self.word_index),
                                                              activation='softmax',
                                                              ))(self.decoder_conv)

    self.decoder_name = 'dilated_conv_large'

