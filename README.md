# CryoTEN: Efficiently Enhancing Cryo-EM Density Maps using Transformers
We introduce CryoTEN - a three-dimensional U-Net style transformer to improve cryo-EM maps effectively. CryoTEN is trained using a diverse set of 1,295 cryo-EM maps as inputs and their corresponding simulated maps generated from known protein structures as targets. An independent test set containing 150 maps is used to evaluate CryoTEN, and the results demonstrate that it can robustly enhance the quality of cryo-EM density maps. In addition, the automatic de novo protein structure modeling shows that the protein structures built from the density maps processed by CryoTEN have substantially better quality than those built from the original maps. Compared to the existing state-of-the-art deep learning methods for enhancing cryo-EM density maps, CryoTEN ranks second in improving the quality of density maps, while running >10 times faster and requiring much less GPU memory than them.

# Overview of CryoTEN
![Network Architecture](<Network-Architecture.png>)

# Installation
### Clone project
```
git clone https://github.com/jianlin-cheng/cryoten
cd cryoten
```
### Download the the trained model 
```
wget https://zenodo.org/records/12693785/files/cryoten.ckpt
# or
curl https://zenodo.org/records/12693785/files/cryoten.ckpt -o cryoten.ckpt
```
### Setup conda environment
```
conda env create -f environment.yaml
conda activate cryoten_env
```

# Usage
### Run CryoTEN on sample EMDB map
```
python eval.py /path/to/input/emd_9311.map /path/to/output/emd_9311_cryoten.mrc
```

# Evaluate CryoTEN on the test dataset
### Download Test data
```
python scripts/collect_data.py --csv data/testset.csv --collection_dir data/collection

# Optionally download half-maps, if present in EMDB for maps in testset. 
python scripts/collect_half_maps.py --csv data/testset.csv --collection_dir data/collection
```
### Run CryoTEN on the Test dataset
```
python eval_testset.py --csv data/testset.csv --collection_dir data/collection --output_dir data/experiments/test
python eval_testset_half_maps.py --csv data/testset.csv --collection_dir data/collection --output_dir data/experiments/test_half_maps
```

# How to train the Model yourself
### Collect EMDB map, PDB and metadata required for creating dataset
```
python scripts/collect_data.py --csv data/dataset.csv --collection_dir data/collection
```
### Create the dataset
```
python scripts/create_dataset.py --csv data/trainvalset.csv --collection_dir data/collection --dataset_dir data/dataset
```
### Train the Model
Customize the training configuration in the config file present in `configs/cryoten.yaml`
```
python main.py fit
...
...
... # if you dont want to connect to wandb for viewing the results, you can select 3 here.
wandb: (1) Create a W&B account
wandb: (2) Use an existing W&B account
wandb: (3) Don't visualize my results
wandb: Enter your choice: 3
...
Trainable params: 29.5 M
Non-trainable params: 0
Total params: 29.5 M
Total estimated model params size (MB): 118
Epoch 0:   0%|        | 0/31 [00:00<?, ?it/s
...
```

### Run the newly trained Model on testset
```
python eval_testset.py --csv data/testset.csv --collection_dir data/collection --output_dir data/experiments/new_model_test --ckpt_path=logs/enhance-cryoem-map/vuf1k5if/checkpoints/last.ckpt
```
