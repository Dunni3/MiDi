# Sampling Made Easy

The original MiDi code is not optimal for enabling you to sample ligands. Even though there are checkpoints, the model class needs access to dataset statistics which are not included in the checkpoints. Therefore, if you wanted to sample molecules, you need to download and process the dataset. This isn't so bad for QM9, but the processing code for the GEOM dataset uses an extremely large amount of memory, unless you can get access to a machine with extremely large memory, you're not going to be able to process the dataset without rewriting the processing code yourself. 

In this repo, I've done the dataset processing for you and pickled the object containing datset statistics! Now you can sample in a few easy steps, no need to even download the datasets! The dataset info files are `model_reconstruction/`. Here are the steps to sample molecules using this method:

1. Clone this repo
2. Build the environment as described in the [readme](README.MD)
2. Download the checkpoints as described in [the readme](README.MD). So far I've only processed the QM9 and GEOM datasets with explicit hydrogens. Implicit hydrogen checkpoints are not supported at the moment.
3. Run `sample.py`

Here is an example command:

```console
python sample.py checkpoints/checkpoint_qm9_h_adaptive.ckpt model_reconstruction/qm9_infos.pkl output_file.pkl --n_mols=1000 --max_batch_size=300
```

# commands used for generating the dataset infos

QM9 
```console
python collect_dataset_infos.py dataset=qm9 dataset.remove_h=False +experiment=qm9_with_h_adaptive general.test_only=/home/ian/projects/mol_diffusion/MiDi/checkpoints/checkpoint_qm9_h_adaptive.ckpt
```
GEOM
```console
python collect_dataset_infos.py dataset=geom dataset.remove_h=False +experiment=geom_with_h_adaptive general.test_only=checkpoints/geom_with_h_adaptive.ckpt
```