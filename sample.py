import argparse
import torch
from midi.diffusion_model import FullDenoisingDiffusion
import pickle
import rdkit.Chem as Chem
import math
import time
from pathlib import Path

def parse_args():

    p = argparse.ArgumentParser(description='Sample script')
    p.add_argument('ckpt_path', type=str, help='Path to the checkpoint')
    p.add_argument('infos_path', type=str, help='Path to the dataset infos')
    p.add_argument('output_file', type=Path, help='Path to the output file')
    p.add_argument('--n_mols', type=int, default=10, help='Number of molecules to sample')
    p.add_argument('--max_batch_size', type=int, default=100, help='Maximum batch size')
    # p.add_argument('--max_batch_size', type=int, default=100, help='Maximum batch size')

    args = p.parse_args()
    return args

if __name__ == "__main__":

    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset_infos = pickle.load(open(args.infos_path, 'rb'))
    model = FullDenoisingDiffusion.load_from_checkpoint(args.ckpt_path, dataset_infos=dataset_infos, train_smiles=[])
    model = model.to(device)
    model.eval()

    n_nodes = model.node_dist.sample_n(args.n_mols, model.device)

    # sampled_mols = model.sample_batch(n_nodes=n_nodes, number_chain_steps=model.number_chain_steps)
    sampled_mols = []
    n_batches = math.ceil(args.n_mols / args.max_batch_size)
    
    start = time.time()
    for i in range(n_batches):
        n_mols_need = args.n_mols - len(sampled_mols)
        batch_size = min(n_mols_need, args.max_batch_size)
        n_nodes = model.node_dist.sample_n(batch_size, model.device)
        sampled_mols += model.sample_batch(n_nodes=n_nodes, number_chain_steps=model.number_chain_steps)

    end = time.time()
    sampling_time = end - start




    sampled_rdkit_mols = [ m.rdkit_mol for m in sampled_mols ]

    # write sampled_rdkit_mols to output_file
    with open(args.output_file, 'wb') as f:
        pickle.dump(sampled_rdkit_mols, f)

    # write sampling time to output_file
    sampling_time_file = args.output_file.parent / f'{args.output_file.stem}_sampling_time.pkl'
    with open(sampling_time_file, 'wb') as f:
        pickle.dump({'sampling_time': sampling_time, 'n_mols': args.n_mols}, f)

