import numpy as np
import random

class InputHandle:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.num_paths = len(input_param['paths'])
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.output_data_type = input_param.get('output_data_type', 'float32')
        self.minibatch_size = input_param['minibatch_size']
        self.is_output_sequence = input_param['is_output_sequence']
        self.data = {}
        self.indices = {}
        self.current_position = 0
        self.current_batch_size = 0
        self.current_batch_indices = []
        self.current_input_length = 0
        self.current_output_length = 0
        self.load()

    def load(self):
        dat_1 = np.load(self.paths[0])
        for key in dat_1.keys():
            self.data[key] = dat_1[key]
        if self.num_paths == 2:
            dat_2 = np.load(self.paths[1])
            num_clips_1 = dat_1['clips'].shape[1]
            dat_2['clips'][:,:,0] += num_clips_1
            self.data['clips'] = np.concatenate(
                (dat_1['clips'], dat_2['clips']), axis=1)
            self.data['input_raw_data'] = np.concatenate(
                (dat_1['input_raw_data'], dat_2['input_raw_data']), axis=0)
            self.data['output_raw_data'] = np.concatenate(
                (dat_1['output_raw_data'], dat_2['output_raw_data']), axis=0)
        for key in self.data.keys():
            print(key)
            print(self.data[key].shape)

    def total(self):
        return self.data['clips'].shape[1]

    def begin(self, do_shuffle = True):
        self.indices = np.arange(self.total(),dtype="int32")
        if do_shuffle:
            random.shuffle(self.indices)
        self.current_position = 0
        if self.current_position + self.minibatch_size <= self.total():
            self.current_batch_size = self.minibatch_size
        else:
            self.current_batch_size = self.total() - self.current_position
        self.current_batch_indices = self.indices[
            self.current_position:self.current_position + self.current_batch_size]