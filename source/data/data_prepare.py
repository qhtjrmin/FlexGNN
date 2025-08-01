import os
import torch
import argparse
import struct
import time
import ogb.nodeproppred
import pymetis
import numpy as np

from edge_partitioner import EdgePartitioner
from feature_partitioner import FeaturePartitoner
from metis_partitoner import MetisPartitioner
import save_graph_info

def save_dataset(output_path, edge_index, features, labels, train_mask, val_mask, test_mask, num_nodes, num_edges, num_classes, name, updated_partition_width=None):
    path = os.path.join(output_path, name)
    os.makedirs(path, exist_ok=True)

    tmp_path = os.path.join(path, 'tmp')
    os.makedirs(tmp_path, exist_ok=True)
    
    edge_path = os.path.join(tmp_path, name+'.coo')
    torch.save(edge_index, edge_path) #store coo for graph partitioning

    features_path = os.path.join(tmp_path, name+'.feat')
    torch.save(features, features_path)
    
    dataset_info_path = os.path.join(tmp_path, name+'.info')
    torch.save({'labels': labels.char(),
                'train_mask': train_mask.bool(), 'val_mask': val_mask.bool(), 'test_mask': test_mask.bool(),
                'num_nodes': num_nodes, 'num_edges': num_edges, 
                'input_dim': features.size(1), 'num_classes': num_classes}, dataset_info_path)
    
    return num_nodes, updated_partition_width, edge_path, features_path, dataset_info_path

def prepare_dgl(data_name, output_path, root_path):
    import dgl
    dataset_sources = {'cora': dgl.data.CoraGraphDataset}
    dgl_dataset: dgl.data.DGLDataset = dataset_sources[data_name](raw_dir=root_path)
    print('dgl dataset', data_name, 'load complete')
    data = dgl_dataset[0]
    edge_index = torch.stack(data.adj_sparse('coo'))
    return save_dataset(output_path, edge_index, data.ndata['feat'], data.ndata['label'],
                 data.ndata['train_mask'], data.ndata['val_mask'], data.ndata['test_mask'],
                 data.num_nodes(), data.num_edges(), dgl_dataset.num_classes, data_name)


def prepare_pyg(data_name, output_path, root_path):
    import torch_geometric
    dataset_sources = {'reddit': torch_geometric.datasets.Reddit,
                       'flickr': torch_geometric.datasets.Flickr,
                       'yelp': torch_geometric.datasets.Yelp,
                        'amazon-products': torch_geometric.datasets.AmazonProducts,
                        }
    pyg_dataset: torch_geometric.data.Dataset = dataset_sources[data_name](root=os.path.join(root_path, data_name))
    print('pyg dataset', data_name, 'load complete')
    data: torch_geometric.data.Data = pyg_dataset[0]
    return save_dataset(output_path, data.edge_index, data.x, data.y,
                 data.train_mask, data.val_mask, data.test_mask,
                 data.num_nodes, data.num_edges, pyg_dataset.num_classes, data_name)

def prepare_pyg_with_metis(data_name, output_path, root_path, num_parts):
    import torch_geometric
    dataset_sources = {'reddit': torch_geometric.datasets.Reddit,
                       'flickr': torch_geometric.datasets.Flickr,
                       'yelp': torch_geometric.datasets.Yelp,
                        'amazon-products': torch_geometric.datasets.AmazonProducts,
                        }
    pyg_dataset: torch_geometric.data.Dataset = dataset_sources[data_name](root=os.path.join(root_path, data_name))
    print('pyg dataset', data_name, 'load complete')
    data: torch_geometric.data.Data = pyg_dataset[0]

    data.edge_index = torch_geometric.utils.to_undirected(data.edge_index, num_nodes=data.num_nodes)
    data.edge_index, _ = torch_geometric.utils.add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)

    # METIS partitioning
    part = MetisPartitioner()
    print(f'[METIS] start partitioning into {num_parts} parts...')
    st = time.time()
    part_id = part.partition(data.edge_index, data.num_nodes, num_parts)
    et = time.time()
    print(f'[METIS] partitioning complete in {et - st:.2f} seconds.')

    # old_id to new_id mapping
    new_id_of_old, offsets, new_partition_width, counts = part.build_relabel_maps(part_id)

    # relabel and padding
    (edge_index_new, x_new, y_new,
     train_new, valid_new, test_new,
     num_nodes_new, num_edges_new) = part.relabel_and_pad_all(
        edge_index=data.edge_index,
        x=data.x,
        y_1d=data.y,
        train_mask=data.train_mask,
        valid_mask=data.val_mask,
        test_mask=data.test_mask,
        new_id_of_old=new_id_of_old,
        num_parts=num_parts,
        W=new_partition_width,
        pad_label_value=-1,
    )

    return save_dataset(
        output_path,
        edge_index_new,
        x_new,
        y_new,
        train_new,
        valid_new,
        test_new,
        num_nodes_new,
        num_edges_new,
        pyg_dataset.num_classes,
        data_name,
        updated_partition_width=new_partition_width
    )

def prepare_ogbn(data_name, output_path, root_path):
    import torch_geometric
    dataset_source =  ogb.nodeproppred.PygNodePropPredDataset
    dataset = dataset_source(root=os.path.join(root_path, data_name), name=data_name)
    print('ogbn dataset', data_name, 'load complete')
    data: torch_geometric.data.Data = dataset[0]
    
    #make symmetric and add self-loops
    data.edge_index = torch_geometric.utils.to_undirected(data.edge_index, num_nodes=data.num_nodes)
    data.edge_index, _ = torch_geometric.utils.add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)
    
    # split masks
    split_idx = dataset.get_idx_split()
    bool_mask = torch.zeros(data.num_nodes).bool()
    make_mask = lambda name: bool_mask.index_fill(0, split_idx[name], True)
    
    if data.y.dim()==2 and data.y.size(1)==1:
        label_1d = torch.reshape(data.y, [-1])
    return save_dataset(output_path, data.edge_index, data.x, label_1d,
                 make_mask('train'), make_mask('valid'), make_mask('test'),
                 data.num_nodes, data.num_edges, dataset.num_classes, data_name)

def prepare_ogbn_with_metis(data_name, output_path, root_path, num_parts):
    import torch_geometric
    dataset_source =  ogb.nodeproppred.PygNodePropPredDataset
    dataset = dataset_source(root=os.path.join(root_path, data_name), name=data_name)
    print('ogbn dataset', data_name, 'load complete')
    data: torch_geometric.data.Data = dataset[0]

    #make symmetric and add self-loops
    data.edge_index = torch_geometric.utils.to_undirected(data.edge_index, num_nodes=data.num_nodes)
    data.edge_index, _ = torch_geometric.utils.add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)
    
    # split masks
    split_idx = dataset.get_idx_split()
    bool_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    make_mask = lambda name: bool_mask.index_fill(0, split_idx[name], True)

    if data.y.dim()==2 and data.y.size(1)==1:
        label_1d = torch.reshape(data.y, [-1])

    # METIS partitioning
    part = MetisPartitioner()
    print(f'[METIS] start partitioning into {num_parts} parts...')
    st = time.time()
    part_id = part.partition(data.edge_index, data.num_nodes, num_parts)
    et = time.time()
    print(f'[METIS] partitioning complete in {et - st:.2f} seconds.')

    # old_id to new_id mapping
    new_id_of_old, offsets, new_partition_width, counts = part.build_relabel_maps(part_id)
    
    # relabel and padding
    (edge_index_new, x_new, y_new,
     train_new, valid_new, test_new,
     num_nodes_new, num_edges_new) = part.relabel_and_pad_all(
        edge_index=data.edge_index,
        x=data.x,
        y_1d=label_1d,
        train_mask=make_mask('train'),
        valid_mask=make_mask('valid'),
        test_mask=make_mask('test'),
        new_id_of_old=new_id_of_old,
        num_parts=num_parts,
        W=new_partition_width,
        pad_label_value=-1,
    )

    return save_dataset(
        output_path,
        edge_index_new,
        x_new,
        y_new,
        train_new,
        valid_new,
        test_new,
        num_nodes_new,
        num_edges_new,
        dataset.num_classes,
        data_name,
        updated_partition_width=new_partition_width
    )


def prepare_dataset(data_name, output_path, root_path, num_parts=1):
    if data_name == 'reddit' or data_name == 'flickr' or data_name == 'yelp' or data_name == 'amazon-products':
        if num_parts > 1:
            return prepare_pyg_with_metis(data_name, output_path, root_path, num_parts)
        else:
            return prepare_pyg(data_name, output_path, root_path)
    elif data_name == 'cora':
        return prepare_dgl(data_name, output_path, root_path)
    elif data_name == 'ogbn-products' or data_name == 'ogbn-arxiv' or data_name == 'ogbn-papers100M':
        if num_parts > 1:
            return prepare_ogbn_with_metis(data_name, output_path, root_path, num_parts)
        else:
            return prepare_ogbn(data_name, output_path, root_path)
    else:
        print('no such dataset', data_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ogbn-products')
    parser.add_argument('--output_path', type=str, default='/data/')
    parser.add_argument('--partition_cnt', type=int, default=4)
    parser.add_argument('--metis', action='store_true')
    # args.dataset example: 'reddit', 'ogbn-products', 'ogbn-arxiv', 'ogbn-papers'

    args = parser.parse_args()
    output_path = args.output_path
    root = os.path.join(output_path, 'raw_data')
    for path in [output_path, root]:
        os.makedirs(path, exist_ok=True)
    
    print("Dataset: ", args.dataset)
    
    num_nodes, new_partition_width, coo_path, feature_path, graph_info_path = prepare_dataset(args.dataset, output_path, root, num_parts=args.partition_cnt if args.metis else 1)
    print("file download and save complete")
    
    path = os.path.join(output_path, args.dataset)

    if args.metis and new_partition_width is not None:
        partition_width = new_partition_width
    else:
        partition_width = int(num_nodes / args.partition_cnt + 1)
    
    pg = EdgePartitioner(coo_path, path, partition_width, args.partition_cnt)
    cnt, width = pg.partition_and_save()
    print("graph partitioning complete")
    
    total_nodes = save_graph_info.save_graph_info(graph_info_path, path, cnt, width)
    print("data info save complete")

    FeaturePartitoner(feature_path, path, partition_width, total_nodes).partition_and_save()
    print("feature partitioning complete")

if __name__ == '__main__':
    main()
    
