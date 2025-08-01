# metis_partitioner.py
import torch
import numpy as np

# METIS partitioning utility for graph data.
class MetisPartitioner:

    def __init__(self, backend: str = "auto"):

        self.backend = backend

    def partition(self, edge_index, num_nodes, num_parts):
        backend = self._choose_backend()
        if backend == "pyg":
            return self._partition_with_pyg(edge_index, num_nodes, num_parts)
        elif backend == "pymetis":
            return self._partition_with_pymetis(edge_index, num_nodes, num_parts)
        else:
            raise RuntimeError("No METIS backend available.")

    def build_relabel_maps(self, part_id: torch.Tensor):
        """
        part_id[v] = p \in {0..P-1}
        return:
          - new_id_of_old (torch.LongTensor, shape [N]): old -> new
          - offsets (list[int], len=P): start index of each partition
          - W (int): max width of partitions
          - counts (np.ndarray, shape [P]): the number of nodes in each partition
        """
        device = part_id.device
        num_nodes = int(part_id.numel())
        num_parts = int(part_id.max().item() + 1)

        idx_per_part = [torch.nonzero(part_id == p, as_tuple=False).flatten() for p in range(num_parts)]
        counts = np.array([int(len(idx)) for idx in idx_per_part], dtype=np.int64)
        W = int(counts.max()) if len(counts) > 0 else 0
        offsets = [p * W for p in range(num_parts)]

        new_id_of_old = torch.empty(num_nodes, dtype=torch.long, device=device)
        for p, old_ids in enumerate(idx_per_part):
            if old_ids.numel() == 0:
                continue
            local_rank = torch.arange(old_ids.numel(), device=device, dtype=torch.long)
            new_ids = offsets[p] + local_rank
            new_id_of_old[old_ids] = new_ids

        return new_id_of_old, offsets, W, counts

    def relabel_and_pad_all(self, edge_index, x, y_1d, train_mask, valid_mask, test_mask, new_id_of_old, num_parts, W, pad_label_value=-1):

        device = x.device if x is not None else torch.device('cpu')
        num_nodes_old = int(new_id_of_old.numel())
        num_nodes_new = int(num_parts * W)

        # 1) edge relabel
        ei = edge_index.to('cpu')
        src_new = new_id_of_old[ei[0]]
        dst_new = new_id_of_old[ei[1]]
        edge_index_new = torch.stack([src_new, dst_new], dim=0)

        # 2) x relabel/pad
        if x is not None:
            feat_dim = x.size(1)
            x_new = torch.zeros((num_nodes_new, feat_dim), dtype=x.dtype, device=device)
            x_new[new_id_of_old.to(device)] = x.to(device)
        else:
            x_new = None

        # 3) y relabel/pad (1D)
        y_dtype = y_1d.dtype if y_1d is not None else torch.long
        y_new = torch.full((num_nodes_new,), fill_value=pad_label_value, dtype=y_dtype, device=device)
        if y_1d is not None:
            y_new[new_id_of_old.to(device)] = y_1d.to(device)

        # 4) mask relabel/pad
        def _remap_mask(mask):
            out = torch.zeros((num_nodes_new,), dtype=torch.bool, device=device)
            if mask is not None and mask.numel() == num_nodes_old:
                out[new_id_of_old.to(device)] = mask.to(device)
            return out

        train_new = _remap_mask(train_mask)
        valid_new = _remap_mask(valid_mask)
        test_new  = _remap_mask(test_mask)

        num_edges_new = int(edge_index_new.size(1))
        return edge_index_new, x_new, y_new, train_new, valid_new, test_new, num_nodes_new, num_edges_new

    # utilities    
    def _choose_backend(self):
        if self.backend in ("pyg", "pymetis"):
            return self.backend
        # auto
        try:
            from torch_geometric.utils import metis as _
            return "pyg"
        except Exception:
            pass
        try:
            import pymetis
            return "pymetis"
        except Exception:
            pass
        return "none"

    def _partition_with_pyg(self, edge_index, num_nodes, num_parts):
        from torch_geometric.utils import metis as pyg_metis
        part_id = pyg_metis(edge_index, num_nodes=num_nodes, num_parts=num_parts)
        if not isinstance(part_id, torch.Tensor):
            part_id = torch.as_tensor(part_id, dtype=torch.long)
        return part_id.to(torch.long)

    def _partition_with_pymetis(self, edge_index, num_nodes, num_parts):
        import pymetis
        ei = edge_index.cpu()
        src = ei[0].numpy()
        dst = ei[1].numpy()

        neighbors = [set() for _ in range(num_nodes)]
        for u, v in zip(src, dst):
            if u == v:
                continue
            neighbors[int(u)].add(int(v))
            neighbors[int(v)].add(int(u)) # undirected

        adjacency = [list(nb) for nb in neighbors]
        _, membership = pymetis.part_graph(num_parts, adjacency=adjacency)
        return torch.as_tensor(membership, dtype=torch.long)
