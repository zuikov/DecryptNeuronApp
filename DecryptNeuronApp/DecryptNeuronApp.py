
# Python program to implement a simple crypto conversion
# of 8-bit plaintext to ciphertext
# and to train Neuron Network to decode message without knowing the key than

import tensorflow as tf
from tensorflow import keras
import itertools
import numpy as np
print("TensorFlow version:", tf.__version__)

PLAIN_TEXT_LENGTH = 8;

# Class to code 8-bit plaintext message
class CryptoWorker():
    HALF_PLAIN_TEXT_LENGTH = int(PLAIN_TEXT_LENGTH / 2);
    ROUND_KEY = [1, 0, 1, 0, 1, 1, 1, 0];
    # ordered s-box-table can't be used, becasuse result obviously correlates to the initial plain text
    # don't do:
    # S_BOX_TABLE = [[[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1]], [[0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1]], [[1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1]], [[1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]]];
    # use unordered table
    S_BOX_TABLE = [[[0,1,0,0], [0,1,0,1], [0,1,1,1], [1,1,1,1]], [[1,1,1,0], [0,0,0,1], [0,0,1,0], [0,0,1,1]], [[1,1,0,0], [1,1,0,1], [0,0,0,0], [1,0,0,0]], [[0,1,1,0], [1,0,0,1], [1,0,1,0], [1,0,1,1]]];
    INDEXES_MAP = ['00', '01', '10', '11'];
    SHIFT_ITERATIONS = 5;

    def print_message(self, description, *value):
        # Accept a list of arguments after description
        print (description, *value);
        print ();

    def bisect_message(self, message):
        # Check that message has an appropriate length
        if len(message) != PLAIN_TEXT_LENGTH:
           self.print_message('plaintext message has an incorrect length');
           return

        left_half_message = message[0 : self.HALF_PLAIN_TEXT_LENGTH];
        right_half_message = message[self.HALF_PLAIN_TEXT_LENGTH : PLAIN_TEXT_LENGTH];

        return [left_half_message, right_half_message];

    def concatenate_message(self, left_message, right_message):
        return [*left_message, *right_message]

    # s-box substitution
    def substitute_message(self, message):
        # Check that initial message has an appropriate length
        if len(message) != self.HALF_PLAIN_TEXT_LENGTH:
           self.print_message('s-box message has an incorrect length');
           return

        # Treat S_BOX_TABLE as a two dimensional 4x4 array, every lowest array of which is an output text
        row_index_string = str(message[0]) + str(message[self.HALF_PLAIN_TEXT_LENGTH - 1]);
        column_index_string = str(message[1]) + str(message[2]);
        row_index = self.INDEXES_MAP.index(row_index_string);
        column_index = self.INDEXES_MAP.index(column_index_string);

        return self.S_BOX_TABLE[row_index][column_index]

    # round key mutation
    def mutate_message(self, message):
        # Check that message has an appropriate length
        if len(message) != PLAIN_TEXT_LENGTH:
           self.print_message("can't apply round key because of inappropriate message length");
           return

        cipher_text = []
        
        for index, round_key_bit in enumerate(self.ROUND_KEY):
            mutated_bit = message[index] ^ round_key_bit;
            cipher_text.append(mutated_bit);

        return cipher_text;

    # cycle shift left
    def shift_message(self, message):
        # Check that cipher text is ready for shifting
        cipher_text_length = len(message);
        if cipher_text_length != PLAIN_TEXT_LENGTH:
           self.print_message('cipher text length ', cipher_text_length);
           self.print_message('you have to prepare cipher text first');
           return

        cipher_text = [*message];

        for _iteration in range(self.SHIFT_ITERATIONS):
            bit_buffer = cipher_text.pop(0);
            cipher_text.append(bit_buffer);

        return cipher_text;

    # crypto conversion
    def code_text(self, plain_text):
        left_half_message, right_half_message = self.bisect_message(plain_text);
        left_substitution = self.substitute_message(left_half_message);
        right_substitution = self.substitute_message(right_half_message);
        concatenated_message = self.concatenate_message(left_substitution, right_substitution);
        mutated_message = self.mutate_message(concatenated_message);

        return self.shift_message(mutated_message);


# Class to generate dataset pairs for Neuron Network
class GetDatasetWorker():
    cipher_dataset = [];
    cryptoWorker = CryptoWorker();

    plain_dataset_list = list(itertools.product([0, 1], repeat=PLAIN_TEXT_LENGTH));
    plain_dataset = [[*dataset] for dataset in plain_dataset_list];
    print('plain_dataset_list', plain_dataset_list);

    for plain_set in plain_dataset:
        print('plain_set', plain_set);
        cipher_dataset.append(cryptoWorker.code_text(plain_set));

    print('plain_dataset', plain_dataset);
    print('plain_dataset length', len(plain_dataset));
    print('cipher_dataset', cipher_dataset);
    print('cipher_dataset length', len(cipher_dataset));


# Class to format dataset pairs for Neuron Network
class FormatDatasetWorker():
    print()


# Keras Sequential Neuron Network
class SequentialDecryptoNN():
    # Sequential model with 3 layers
    nn_model = keras.Sequential(name='SequentialDecryptoModel')
    nn_input = keras.layers.Input(shape=(8,))
    nn_layer_1 = keras.layers.Dense(8, activation='relu', name='layer1')
    nn_layer_2 = keras.layers.Dense(8, activation='relu', name='layer2')
    nn_layer_3 = keras.layers.Dense(8, activation='relu', name='layer3')

    nn_model.add(nn_input)
    nn_model.add(nn_layer_1)
    nn_model.add(nn_layer_2)
    nn_model.add(nn_layer_3)

    nn_model.summary()
    

    # Call model on a test input
    x = tf.ones((1,8))
    y = nn_model(x)

    print('y', y)
    print('nn_layer_1 weights', nn_layer_1.get_weights())



# Driver Code
if __name__ == "__main__":

    plain_text = [0, 0, 1, 1, 1, 0, 1, 1];

    cryptoWorker = CryptoWorker();

    cipher_text = cryptoWorker.code_text(plain_text);

    
    print ('Plain text  ', plain_text);
    print ('Cipher text ', cipher_text);
    print ();
 
    sequentialDecryptoNN = SequentialDecryptoNN()


