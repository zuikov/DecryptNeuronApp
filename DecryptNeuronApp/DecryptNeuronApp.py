
# Python program to implement a simple crypto conversion
# of 8-bit plaintext to ciphertext
 
# Class to code 8-bit plaintext message
class CryptoWorker():
    ROUND_KEY = [1, 0, 1, 0, 1, 1, 1, 0];
    S_BOX_TABLE = [[[0,0,0,0], [0,0,0,1], [0,0,1,0], [0,0,1,1]], [[0,1,0,0], [0,1,0,1], [0,1,1,0], [0,1,1,1]], [[1,0,0,0], [1,0,0,1], [1,0,1,0], [1,0,1,1]], [[1,1,0,0], [1,1,0,1], [1,1,1,0], [1,1,1,1]]];
    INDEXES_MAP = ['00', '01', '10', '11'];
    HALF_MESSAGE_LENGTH = 4;
    SHIFT_ITERATIONS = 5;

    plain_text = [];
    left_half_message = [];
    right_half_message = [];
    cipher_text = []

    def set_plain_text (self, plain_text):
        self.plain_text = plain_text;

    def print_message(self, description, *value):
        # Accept a list of arguments after description
        print (description, *value);
        print ();

    def bisect_message(self):
        # Check that message has an appropriate length
        if len(self.plain_text) != self.HALF_MESSAGE_LENGTH * 2:
           self.print_message('plaintext message has an incorrect length');
           return

        self.left_half_message = self.plain_text[0 : self.HALF_MESSAGE_LENGTH];
        self.right_half_message = self.plain_text[self.HALF_MESSAGE_LENGTH : self.HALF_MESSAGE_LENGTH * 2];
        # self.print_message('left_half_message ', self.left_half_message);
        # self.print_message('right_half_message ', self.right_half_message);

    def concatenate_message(self, left_message, right_message):
        return [*left_message, *right_message]

    # s-box substitution
    def substitute_message(self, message):
        # Check that initial message has an appropriate length
        if len(message) != self.HALF_MESSAGE_LENGTH:
           self.print_message('s-box message has an incorrect length');
           return

        # Treat S_BOX_TABLE as a two dimensional 4x4 array, every lowest array of which is an output text
        row_index_string = str(message[0]) + str(message[self.HALF_MESSAGE_LENGTH - 1]);
        column_index_string = str(message[1]) + str(message[2]);
        row_index = self.INDEXES_MAP.index(row_index_string);
        column_index = self.INDEXES_MAP.index(column_index_string);

        return self.S_BOX_TABLE[row_index][column_index]

    # round key mutation
    def mutate_message(self, message):
        # Check that message has an appropriate length
        if len(message) != self.HALF_MESSAGE_LENGTH * 2:
           self.print_message("can't apply round key because of inappropriate message length");
           return
        
        for index, round_key_bit in enumerate(self.ROUND_KEY):
            # This works correctly only for one round mutation
            mutated_bit = message[index] ^ round_key_bit;
            self.cipher_text.append(mutated_bit);

    # cycle shift left
    def shift_message(self):
        # Check that cipher text is ready for shifting
        if len(self.cipher_text) != self.HALF_MESSAGE_LENGTH * 2:
           self.print_message('you have to prepare cipher text first');
           return

        for _iteration in range(self.SHIFT_ITERATIONS):
            bit_buffer = self.cipher_text.pop(0);
            self.cipher_text.append(bit_buffer);

        # self.print_message('ciphertext processing finished');
        return self.cipher_text;

    # crypto conversion
    def code_text(self, plain_text):
        # self.print_message('plaintext processing starts');
        self.set_plain_text(plain_text);
        self.bisect_message();
        left_substitution = self.substitute_message(self.left_half_message);
        right_substitution =self.substitute_message(self.right_half_message);
        concatenated_message = self.concatenate_message(left_substitution, right_substitution);
        self.mutate_message(concatenated_message);
        return self.shift_message();


# Driver Code
if __name__ == "__main__":

    plain_text = [0, 0, 1, 1, 1, 0, 1, 1];

    cryptoWorker = CryptoWorker();

    cipher_text = cryptoWorker.code_text(plain_text);

    
    print ('Plain text  ', plain_text);
    print ('Cipher text ', cipher_text);
    print ();
 


