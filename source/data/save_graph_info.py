import numpy as np
import torch
import struct

def save_graph_info(input_file, output_path, cnt, width):
    graph_info = torch.load(input_file)
    
    with open(f'{output_path}/graph_info.dat', "wb") as file:
        
        # Save the basic information for the graph
        file.write(struct.pack('Q', graph_info['num_nodes']))
        file.write(struct.pack('Q', graph_info['num_edges']))
        file.write(struct.pack('I', graph_info['input_dim']))
        file.write(struct.pack('I', graph_info['num_classes']))
        file.write(struct.pack('I', cnt))
        file.write(struct.pack('I', width))
        
        # Save the labels
        labels = graph_info['labels'].numpy()
        file.write(struct.pack(f'{len(labels)}i', *labels))
        del labels
        
        # Save the masks
        train_mask = graph_info['train_mask'].numpy()
        file.write(struct.pack(f'{len(train_mask)}?', *train_mask))
        del train_mask
        
        val_mask = graph_info['val_mask'].numpy()
        file.write(struct.pack(f'{len(val_mask)}?', *val_mask))
        del val_mask
        
        test_mask = graph_info['test_mask'].numpy()
        file.write(struct.pack(f'{len(test_mask)}?', *test_mask))
        del test_mask
        
        file.close()
        
    return graph_info['num_nodes']
        