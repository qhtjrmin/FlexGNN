import pandas as pd
import numpy as np
import os
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import time
from collections import defaultdict
import json
import struct
import argparse
import multiprocessing as mp
import pickle

class EdgePartitioner:
    def __init__(self, input_file, output_path, block_width, block_cnt=0):
        self.input = input_file
        self.output_path = output_path
        self.block_width = block_width
        self.block_cnt = block_cnt
        
        self.graph_info = None
        self.blocks_info = None
        self.indeg = None
        self.outdeg = None
        self.total_nodes = 0
    
    def load_from_npy(self):
        
        input_npy = np.load(self.input)
        src_ids = input_npy[:, 0]
        dst_ids = input_npy[:, 1]
        
        return src_ids, dst_ids
    
    def load_from_csv(self):
        
        #output path generation when it does not exist
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)        
        
        if not os.path.exists(self.input):
            raise FileNotFoundError('Input file does not exist')
        else:
            print(f'Input file: {self.input}')
        
        input_csv = pd.read_csv(self.input,header=None,names=['src_id','dst_id'])
        src_ids = input_csv['src_id'].values
        dst_ids = input_csv['dst_id'].values
        
        return src_ids, dst_ids
    
    def load_from_coo(self):
        import torch
        input_coo = torch.load(self.input)
        src_ids = input_coo[0].numpy()
        dst_ids = input_coo[1].numpy()
        
        return src_ids, dst_ids
    
    def load_from_feather(self):
        print("load_from_feather")
        data = pd.read_feather(self.input)
        return data['src_id'].values, data['dst_id'].values
    
    def load_from_txt(self):
        # Assuming the txt file has two columns: src_id and dst_id separated by whitespace
        data = pd.read_csv(self.input, delim_whitespace=True, header=None, names=['src_id', 'dst_id'])
        data.to_feather(f'{self.input}.feather')
        return data['src_id'].values, data['dst_id'].values
    
    def add_self_loop(self, src_ids, dst_ids, max_node_id):
        
        all_nodes = np.arange(max_node_id + 1)
        self_loop_src = all_nodes
        self_loop_dst = all_nodes
        
        return np.concatenate([src_ids, self_loop_src]), np.concatenate([dst_ids, self_loop_dst])
    
    def save_data(self, block_data):
        output_file = self.output_path + '/tmp/blocks.bin'
        
        with open(output_file, 'wb') as f:         
            pickle.dump(self.blocks_info, f)
            pickle.dump(self.graph_info, f)
            pickle.dump(block_data, f)
            
    def load_data(self):
        input_file = self.output_path + '/tmp/blocks.bin'
        
        with open(input_file, 'rb') as f:
            self.blocks_info = pickle.load(f)
            self.graph_info = pickle.load(f)
            block_data = pickle.load(f)
            
        return block_data

    def partition(self):
        if self.input.endswith('.csv'):
            src_ids, dst_ids = self.load_from_csv()
        elif self.input.endswith('.coo'):
            src_ids, dst_ids = self.load_from_coo()
        elif self.input.endswith('.feather'):
            src_ids, dst_ids = self.load_from_feather()
        elif self.input.endswith('.npy'):
            src_ids, dst_ids = self.load_from_npy()
        else:
            src_ids, dst_ids = self.load_from_txt()

        src_ids = np.asarray(src_ids)
        dst_ids = np.asarray(dst_ids)

        # ensure IDs are in 64-bit integer format
        if src_ids.dtype.kind != 'i' and src_ids.dtype.kind != 'u':
            src_ids = src_ids.astype(np.int64, copy=False)
            dst_ids = dst_ids.astype(np.int64, copy=False)
        else:
            src_ids = src_ids.astype(np.int64, copy=False)
            dst_ids = dst_ids.astype(np.int64, copy=False)

        max_nod_id = int(max(np.max(src_ids), np.max(dst_ids)))
        total_nodes = max_nod_id + 1
        self.total_nodes = total_nodes
        if total_nodes < self.block_width:
            self.block_width = int(total_nodes)

        print('total_nodes:', total_nodes, 'block_cnt:', self.block_cnt, 'block_width:', self.block_width)

        # set up degree arrays
        self.indeg = np.bincount(dst_ids, minlength=total_nodes).astype(np.uint32, copy=False)
        self.outdeg = np.bincount(src_ids, minlength=total_nodes).astype(np.uint32, copy=False)

        blocks_data = {}
        for i in range(self.block_cnt):
            for j in range(self.block_cnt):
                blocks_data[(i, j)] = {'row': [], 'col': [], 'data': []}

        self.blocks_info = {}
        for i in range(self.block_cnt):
            for j in range(self.block_cnt):
                self.blocks_info[(i, j)] = {
                    'edges': 0,
                    'row_check': set(),
                    'col_check': set(),
                    'num_rows': 0,
                    'num_cols': 0
                }

        self.graph_info = {
            'block_cnt': self.block_cnt,
            'block_width': self.block_width
        }

        # vectorized partitioning
        start = time.time()

        src_block = src_ids // self.block_width
        dst_block = dst_ids // self.block_width
        block_keys = src_block * self.block_cnt + dst_block

        local_src = src_ids - src_block * self.block_width  # (src_ids % block_width)
        local_dst = dst_ids - dst_block * self.block_width  # (dst_ids % block_width)

        # sort the edges by block keys
        order = np.argsort(block_keys, kind='mergesort')
        sorted_keys = block_keys[order]
        sorted_lsrc = local_src[order]
        sorted_ldst = local_dst[order]

        uniq_keys, idx_start, counts = np.unique(
            sorted_keys, return_index=True, return_counts=True
        )

        # initialize blocks_data and blocks_info
        for k, s, c in zip(uniq_keys, idx_start, counts):
            e = s + c
            rows = sorted_lsrc[s:e]
            cols = sorted_ldst[s:e]

            # calculate block indices
            i = int(k // self.block_cnt)
            j = int(k % self.block_cnt)
            block_id = (i, j)

            # update blocks_data
            blocks_data[block_id]['row'] = rows.tolist()
            blocks_data[block_id]['col'] = cols.tolist()
            # blocks_data[block_id]['data']  # when weights are present

            # blocks_info
            self.blocks_info[block_id]['edges'] = int(c)

            # update row and column checks
            ur = np.unique(rows)
            uc = np.unique(cols)
            rset = set(ur.tolist())
            cset = set(uc.tolist())

            self.blocks_info[block_id]['row_check'] = rset
            self.blocks_info[block_id]['col_check'] = cset
            self.blocks_info[block_id]['num_rows'] = len(rset)
            self.blocks_info[block_id]['num_cols'] = len(cset)

        end = time.time()
        print('partitioning time:', round(end - start, 2))

        return blocks_data

    def update_blocks_info(self):
        for block_id, data in self.blocks_info.items():
            #update for row
            self.blocks_info[block_id]['num_rows'] = len(data['row_check'])
            self.blocks_info[block_id]['row_indices'] = sorted(data['row_check'])
            del self.blocks_info[block_id]['row_check']
            
            #update for col
            self.blocks_info[block_id]['num_cols'] = len(data['col_check'])
            self.blocks_info[block_id]['col_indices'] = sorted(data['col_check'])
            del self.blocks_info[block_id]['col_check']
            
            print(f'block_id: {block_id}, edges: {data["edges"]}, num_rows: {data["num_rows"]}, num_cols: {data["num_cols"]}')
            
    def save_edge_files(self, blocks_data):
        for block_id, data in blocks_data.items():
            row = np.array(data['row'])
            col = np.array(data['col'])
            # data_arr = np.array(data['data'])  # when weights are present
            
            csr = csr_matrix((np.ones(len(row)), (row, col)), shape=(self.block_width, self.block_width))
            
            with open(f'{self.output_path}/edge/csr/block_{block_id[0]}_{block_id[1]}.dat', "wb") as file:
                
                # store metadata
                file.write(struct.pack('II?', block_id[0], block_id[1], True))
                file.write(struct.pack('IIQ', self.graph_info['block_width'], self.blocks_info[block_id]['num_rows'], self.blocks_info[block_id]['edges']))
                
                # store list/vectors
                row_indices = self.blocks_info[block_id]['row_indices']
                file.write(struct.pack(f'{len(row_indices)}I', *row_indices))
                file.write(struct.pack(f'{len(csr.indptr)}I', *csr.indptr))
                file.write(struct.pack(f'{len(csr.indices)}I', *csr.indices))
                file.write(struct.pack(f'{len(csr.data)}f', *csr.data))
        
                file.close()
            
            del csr
            del self.blocks_info[block_id]['row_indices']
            
            # if data is directed, save csc format as well
            csc = csc_matrix((np.ones(len(col)), (row, col)), shape=(self.block_width, self.block_width))
            with open(f'{self.output_path}/edge/csc/block_{block_id[0]}_{block_id[1]}.dat', "wb") as file:
                
                # store metadata
                file.write(struct.pack('II?', block_id[0], block_id[1], False))
                file.write(struct.pack('IIQ', self.graph_info['block_width'], self.blocks_info[block_id]['num_cols'], self.blocks_info[block_id]['edges']))
                
                # store list/vectors
                col_indices = self.blocks_info[block_id]['col_indices']
                file.write(struct.pack(f'{len(col_indices)}I', *col_indices))
                file.write(struct.pack(f'{len(csc.indptr)}I', *csc.indptr))
                file.write(struct.pack(f'{len(csc.indices)}I', *csc.indices))
                file.write(struct.pack(f'{len(csc.data)}f', *csc.data))
                
                file.close()
                 
            del csc
            del self.blocks_info[block_id]['col_indices']
        
    def save_degree_file(self):
        if not os.path.exists(f'{self.output_path}/edge'):
            os.makedirs(f'{self.output_path}/edge')
        self.indeg.tofile(f'{self.output_path}/edge/indeg.bin')
        self.outdeg.tofile(f'{self.output_path}/edge/outdeg.bin')
    
    def save_blocks_info(self):
        with open(f'{self.output_path}/edge/blocks_info.txt', 'w') as output_file:
            for block_id, data in self.blocks_info.items():
                print(f'block_id: {block_id}, edges: {data["edges"]}')
                output_file.write(f'{data["edges"]}\n')
        
    def save_graph_info(self):
        data = {}
        data['block_cnt'] = int(self.graph_info['block_cnt'])
        data['block_width'] = int(self.graph_info['block_width'])
        with open(f'{self.output_path}/graph_info.json', 'w') as output_file:
            json.dump(data, output_file)
        
    def partition_and_save(self):
        
        start = time.time()
        blocks_data = self.partition()
        end = time.time()
        print('Partitioning complete:', round(end-start, 2))
        
        if not os.path.exists(f'{self.output_path}/edge'):
            os.makedirs(f'{self.output_path}/edge')
        if not os.path.exists(f'{self.output_path}/edge/csr'):
            os.makedirs(f'{self.output_path}/edge/csr')
        if not os.path.exists(f'{self.output_path}/edge/csc'):
            os.makedirs(f'{self.output_path}/edge/csc')
        
        start = time.time()
        self.save_degree_file()
        end = time.time()
        print('Save degree files complete:', round(end-start, 2))
        
        start = time.time()
        self.update_blocks_info()
        end = time.time()
        print('Update blocks info complete:', round(end-start, 2))
        
        start = time.time()
        self.save_blocks_info()
        end = time.time()
        print('Save blocks info complete:', round(end-start, 2))
        
        start = time.time()
        self.save_edge_files(blocks_data)
        end = time.time()
        print('Save edge files complete:', round(end-start, 2))
        
        return self.graph_info['block_cnt'], self.graph_info['block_width']
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--block_width', type=int, default=16777216)
    parser.add_argument('--block_cnt', type=int, default=4)
    
    args = parser.parse_args()
    pg = EdgePartitioner(args.input, args.output_path, args.block_width, args.block_cnt)
    
    start = time.time()
    cnt, width = pg.partition_and_save()
    end = time.time()
    print('Total elapsed time: ', end - start)

    print(f'Partitioning complete: {cnt} blocks, {width} width')
    
if __name__ == '__main__':
    main()