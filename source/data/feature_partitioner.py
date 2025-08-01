import torch
import numpy as np
import struct
import os

class FeaturePartitoner:
    def __init__(self, input_file, output_path, block_width, total_nodes):
        self.input = input_file
        self.output_path = output_path
        self.block_width = block_width
        self.total_nodes = total_nodes
        
    def partition_and_save(self):
        feature_output = self.output_path + '/feature'
        
        if not os.path.exists(feature_output):
            os.makedirs(feature_output)
            
        block_cnt = int(np.ceil(self.total_nodes / self.block_width))
        
        feature = torch.load(self.input)
        feature = feature.numpy()
        
        for i in range(block_cnt):
            start = i * self.block_width
            end = min((i+1) * self.block_width, self.total_nodes)
            block_feature = feature[start:end]
            block_feature.tofile(f'{feature_output}/feature_{i}.bin')
