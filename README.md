# FlexGNN
FlexGNN: A High-Performance, Large-Scale Full-Graph GNN System with Best-Effort Training Plan Optimization

## Requirements
To run FlexGNN, you need the following:
- Docker **v24.0.2** or later
- CUDA Toolkit **v11.6** or later
- Nvida GPU with driver **v510** or later

## Getting started
#### 1. Clone the repository
```bash
$cd ~/
$git clone https://github.com/qhtjrmin/FlexGNN.git
$cd FlexGNN
```

#### 2. Prepare LibTorch (v1.12.0+)
Download the version of LibTorch compatible with your CUDA version:
```
$wget https://download.pytorch.org/libtorch/cu116/libtorch-cxx11-abi-shared-with-deps-1.12.0%2Bcu116.zip
$unzip libtorch-cxx11-abi-shared-with-deps-1.12.0+cu116.zip
```
Make sure the `libtorch` directory is placed in the project root.

#### 3. Prepare dataset
You can either:
- Download a **pre-partitioned** version of the Reddit or ogbn-products datasets (here), **or**
- Generate your own using `source/data/prepare_data.py`
> The `prepare_data.py` supports downloading and partitioning datasets.  
> Note: preprocessing large datasets like ogbn-papers may take significant time.

Examle for downloadng ogbn-products:
``` bash
$mkdir -p dataset/products
$cd dataset/products

# Download pre-partitioned data (example command)
$wget [URL]
$unzip [filename].zip
```

#### 4. Build & Run with Docker
```
$cd ~/FlexGNN/
$docker build --tag flexgnn .
$./scripts/run.sh
```
The `run.sh` script launches the container and executes training.

## Configurations
FlexGNN uses configuration files in `.ini` format (via the [inih](https://github.com/benhoyt/inih) library).

- You can find sample configs under `configs/`
- Modify parameters as needed for your setup
- Update the `scripts/run.sh` file with:
  - Number of epochs
  - Model type (e.g., GCN, GraphSage, GAT)
  - Path to the config file
 
## License
This project is licensed under the [GNU General Public License v3.0](./LICENSE).
