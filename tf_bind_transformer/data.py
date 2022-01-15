from Bio import SeqIO
from random import choice
from pathlib import Path
import functools
import polars as pl
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tf_bind_transformer.protein_utils import parse_gene_name
from enformer_pytorch import FastaInterval

def exists(val):
    return val is not None

# fetch protein sequences by gene name and uniprot id

class FactorProteinDatasetByUniprotID(Dataset):
    def __init__(self, folder):
        super().__init__()
        fasta_paths = [*Path(folder).glob('*.fasta')]
        assert len(fasta_paths) > 0, f'no fasta files found at {folder}'
        self.paths = fasta_paths
        self.index_by_id = dict()

        for path in fasta_paths:
            gene, uniprotid, *_ = path.stem.split('.')
            self.index_by_id[uniprotid] = path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, uid):
        index = self.index_by_id

        if uid not in index:
            return None

        entry = index[uid]
        fasta = SeqIO.read(entry, 'fasta')
        return str(fasta.seq)

# fetch

class FactorProteinDataset(Dataset):
    def __init__(self, folder, return_tuple_only = False):
        super().__init__()
        fasta_paths = [*Path(folder).glob('*.fasta')]
        assert len(fasta_paths) > 0, f'no fasta files found at {folder}'
        self.paths = fasta_paths
        self.index_by_gene = defaultdict(list)
        self.return_tuple_only = return_tuple_only # whether to return tuple even if there is only one subunit

        for path in fasta_paths:
            gene, uniprotid, *_ = path.stem.split('.')
            self.index_by_gene[gene].append(path)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, unparsed_gene_name):
        index = self.index_by_gene

        genes = parse_gene_name(unparsed_gene_name)
        seqs = []

        for gene in genes:
            entry = index[gene]

            if len(entry) == 0:
                print(f'no entries for {gene}')
                continue

            path = choice(entry) if isinstance(entry, list) else entry

            fasta = SeqIO.read(path, 'fasta')
            seqs.append(str(fasta.seq))

        seqs = tuple(seqs)

        if len(seqs) == 1 and not self.return_tuple_only:
            return seqs[0]

        return seqs

# remap dataframe functions

CHR_IDS = set([*range(1, 23), 'X', 'Y'])

def get_chr_names(ids):
    return set(map(lambda t: f'chr{t}', ids))

def remap_df_add_experiment_target_cell_(df, col = 'column_4'):
    exp_id = df.select([pl.col(col).str.extract(r"(\w+)\.[\w+]\.[\w+]")])
    targets = df.select([pl.col(col).str.extract(r"[\w+]\.(\w+)\.[\w+]")])
    cell_type = df.select([pl.col(col).str.extract(r"[\w+]\.[\w+]\.(\w+)")])

    exp_id = exp_id.rename({col: 'experiment'}).to_series(0)
    targets = targets.rename({col: 'target'}).to_series(0)
    cell_type = exp_id.rename({col: 'cell_type'}).to_series(0)

    df.insert_at_idx(3, exp_id)
    df.insert_at_idx(3, targets)
    df.insert_at_idx(3, exp_id)

def pl_isin(col, arr):
    equalities = list(map(lambda t: pl.col(col) == t, arr))
    return functools.reduce(lambda a, b: a | b, equalities)

# dataset for remap data - all peaks

class RemapAllPeakDataset(Dataset):
    def __init__(
        self,
        *,
        bed_file,
        factor_fasta_folder,
        filter_chromosome_ids = None,
        **kwargs
    ):
        super().__init__()
        df = pl.read_csv(bed_file, sep = '\t', has_headers = False)

        dataset_chr_ids = CHR_IDS

        if exists(filter_chromosome_ids):
            dataset_chr_ids = dataset_chr_ids.intersection(set(filter_chromosome_ids))

        df = df.filter(pl_isin('column_1', get_chr_names(dataset_chr_ids)))

        assert len(df) > 0, 'dataset is empty by filter criteria'

        self.df = df
        self.factor_ds = FactorProteinDataset(factor_fasta_folder)
        self.fasta = FastaInterval(**kwargs)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        chr_name, begin, end, experiment_target_cell_type, reading, *_ = self.df.row(ind)
        experiment, target, cell_type = experiment_target_cell_type.split('.')

        seq = self.fasta(chr_name, begin, end)
        aa_seq = self.factor_ds[target]
        context_str = f'{cell_type} | {experiment}'

        value = torch.Tensor([reading])
        label = torch.Tensor([1.])

        return seq, aa_seq, context_str, value, label

# dataloader related functions

def collate_fn(data):
    seq, aa_seq, context_str, values, labels = list(zip(*data))
    return torch.stack(seq), tuple(aa_seq), tuple(context_str), torch.stack(values, dim = 0), torch.cat(labels, dim = 0)

def get_dataloader(ds, **kwargs):
    dl = DataLoader(ds, collate_fn = collate_fn, **kwargs)
    return dl
