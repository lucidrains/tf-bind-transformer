from Bio import SeqIO
from random import choice, randrange
from pathlib import Path
import functools
import polars as pl
from collections import defaultdict

import os
import json
import shutil
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from tf_bind_transformer.gene_utils import parse_gene_name
from enformer_pytorch import FastaInterval

from pyfaidx import Fasta
import pybedtools

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def find_first_index(cond, arr):
    for ind, el in enumerate(arr):
        if cond(el):
            return ind
    return -1

def cast_list(val = None):
    if not exists(val):
        return []
    return [val] if not isinstance(val, (tuple, list)) else val

def read_bed(path):
    return pl.read_csv(path, sep = '\t', has_headers = False)

def save_bed(df, path):
    df.to_csv(path, sep = '\t', has_header = False)

def parse_exp_target_cell(exp_target_cell):
    experiment, target, *cell_type = exp_target_cell.split('.')
    cell_type = '.'.join(cell_type) # handle edge case where cell type contains periods
    return experiment, target, cell_type

# fetch index of datasets, for providing the sequencing reads
# for auxiliary read value prediction

def fetch_experiments_index(path):
    if not exists(path):
        return dict()

    exp_path = Path(path)
    assert exp_path.exists(), 'path to experiments json must exist'

    root_json = json.loads(exp_path.read_text())
    experiments = root_json['experiments']

    index = {}
    for experiment in experiments:
        exp_id = experiment['accession']

        if 'details' not in experiment:
            continue

        details = experiment['details']

        if 'datasets' not in details:
            continue

        datasets = details['datasets']

        for dataset in datasets:
            dataset_name = dataset['dataset_name']
            index[dataset_name] = dataset['peaks_NR']

    return index

# fetch protein sequences by gene name and uniprot id

class FactorProteinDatasetByUniprotID(Dataset):
    def __init__(
        self,
        folder,
        species_priority = ['human', 'mouse']
    ):
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
    def __init__(
        self,
        folder,
        species_priority = ['human', 'mouse', 'unknown'],
        return_tuple_only = False
    ):
        super().__init__()
        fasta_paths = [*Path(folder).glob('*.fasta')]
        assert len(fasta_paths) > 0, f'no fasta files found at {folder}'
        self.paths = fasta_paths

        index_by_gene = defaultdict(list)
        self.return_tuple_only = return_tuple_only # whether to return tuple even if there is only one subunit

        for path in fasta_paths:
            gene, uniprotid, *_ = path.stem.split('.')
            index_by_gene[gene].append(path)

        # prioritize fasta files of certain species
        # but allow for appropriate fallback, by order of species_priority

        get_species_from_path = lambda p: p.stem.split('_')[-1].lower() if '_' in p.stem else 'unknown'

        filtered_index_by_gene = defaultdict(list)

        for gene, gene_paths in index_by_gene.items():
            species_count = list(map(lambda specie: len(list(filter(lambda p: get_species_from_path(p) == specie, gene_paths))), species_priority))
            species_ind_non_zero = find_first_index(lambda t: t > 0, species_count)

            if species_ind_non_zero == -1:
                continue

            species = species_priority[species_ind_non_zero]
            filtered_index_by_gene[gene] = list(filter(lambda p: get_species_from_path(p) == species, gene_paths))

        self.index_by_gene = filtered_index_by_gene

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

def get_chr_names(ids):
    return set(map(lambda t: f'chr{t}', ids))

CHR_IDS = set([*range(1, 23), 'X'])
CHR_NAMES = get_chr_names(CHR_IDS)

def remap_df_add_experiment_target_cell(df, col = 'column_4'):
    df = df.clone()

    exp_id = df.select([pl.col(col).str.extract(r"^([\w\-]+)\.*")])
    exp_id = exp_id.rename({col: 'experiment'}).to_series(0)
    df.insert_at_idx(3, exp_id)

    targets = df.select([pl.col(col).str.extract(r"[\w\-]+\.([\w\-]+)\.[\w\-]+")])
    targets = targets.rename({col: 'target'}).to_series(0)
    df.insert_at_idx(3, targets)

    cell_type = df.select([pl.col(col).str.extract(r"^.*\.([\w\-]+)$")])
    cell_type = cell_type.rename({col: 'cell_type'}).to_series(0)
    df.insert_at_idx(3, cell_type)

    return df

def pl_isin(col, arr):
    equalities = list(map(lambda t: pl.col(col) == t, arr))
    return functools.reduce(lambda a, b: a | b, equalities)

def pl_notin(col, arr):
    equalities = list(map(lambda t: pl.col(col) != t, arr))
    return functools.reduce(lambda a, b: a & b, equalities)

def filter_by_col_isin(df, col, arr, chunk_size = 25):
    """
    polars seem to have a bug
    where OR more than 25 conditions freezes (for pl_isin)
    do in chunks of 25 and then concat instead
    """
    dataframes = []
    for i in range(0, len(arr), chunk_size):
        sub_arr = arr[i:(i + chunk_size)]
        filtered_df = df.filter(pl_isin(col, sub_arr))
        dataframes.append(filtered_df)
    return pl.concat(dataframes)

def filter_bed_file_by_(bed_file_1, bed_file_2, output_file):
    # generated by OpenAI Codex

    bed_file_1_bedtool = pybedtools.BedTool(bed_file_1)
    bed_file_2_bedtool = pybedtools.BedTool(bed_file_2)
    bed_file_1_bedtool_intersect_bed_file_2_bedtool = bed_file_1_bedtool.intersect(bed_file_2_bedtool, v = True)
    bed_file_1_bedtool_intersect_bed_file_2_bedtool.saveas(output_file)

def filter_df_by_tfactor_fastas(df, folder):
    files = [*Path(folder).glob('**/*.fasta')]
    present_target_names = set([f.stem.split('.')[0] for f in files])
    all_df_targets = df.get_column('target').unique().to_list()

    all_df_targets_with_parsed_name = [(target, parse_gene_name(target)) for target in all_df_targets]
    unknown_targets = [target for target, parsed_target_name in all_df_targets_with_parsed_name for parsed_target_name_sub_el in parsed_target_name if parsed_target_name_sub_el not in present_target_names]

    if len(unknown_targets) > 0:
        df = df.filter(pl_notin('target', unknown_targets))
    return df

def generate_random_ranges_from_fasta(
    fasta_file,
    *,
    output_filename = 'random-ranges.bed',
    context_length,
    filter_bed_files = [],
    num_entries_per_key = 10,
    keys = None,
):
    fasta = Fasta(fasta_file)
    tmp_file = f'/tmp/{output_filename}'

    with open(tmp_file, 'w') as f:
        for chr_name in sorted(CHR_NAMES):
            print(f'generating ranges for {chr_name}')

            if chr_name not in fasta:
                print(f'{chr_name} not found in fasta file')
                continue

            chromosome = fasta[chr_name]
            chromosome_length = len(chromosome)

            start = np.random.randint(0, chromosome_length - context_length, (num_entries_per_key,))
            end = start + context_length
            start_and_end = np.stack((start, end), axis = -1)

            for row in start_and_end.tolist():
                start, end = row
                f.write('\t'.join((chr_name, str(start), str(end))) + '\n')

    for file in filter_bed_files:
        filter_bed_file_by_(tmp_file, file, tmp_file)

    shutil.move(tmp_file, f'./{output_filename}')

    print('success')

# context string creator class

class ContextDataset(Dataset):
    def __init__(
        self,
        *,
        biotypes_metadata_path = None,
        include_biotypes_metadata_in_context = False,
        include_biotypes_metadata_columns = [],
        biotypes_metadata_delimiter = ' | ',
    ):
        self.include_biotypes_metadata_in_context = include_biotypes_metadata_in_context
        self.include_biotypes_metadata_columns = include_biotypes_metadata_columns
        self.biotypes_metadata_delimiter = biotypes_metadata_delimiter

        if include_biotypes_metadata_in_context:
            assert len(self.include_biotypes_metadata_columns) > 0, 'must have more than one biotype metadata column to include'
            assert exists(biotypes_metadata_path), 'biotypes metadata path must be supplied if to be included in context string'

            p = Path(biotypes_metadata_path)

            if p.suffix == '.csv':
                sep = ','
            elif p.suffix == '.tsv':
                sep = '\t'
            else:
                raise ValueError(f'invalid suffix {p.suffix} for biotypes')

            self.df = pl.read_csv(str(p), sep = sep)

    def __len__():
        return len(self.df) if self.include_biotypes_metadata_in_context else -1

    def __getitem__(self, biotype):
        if not self.include_biotypes_metadata_in_context:
            return biotype

        col_indices = list(map(self.df.columns.index, self.include_biotypes_metadata_columns))
        filtered = self.df.filter(pl.col('biotype') == biotype)

        if len(filtered) == 0:
            print(f'no rows found for {biotype} in biotype metadata file')
            return biotype

        row = filtered.row(0)
        columns = list(map(lambda t: row[t], col_indices))

        context_string = self.biotypes_metadata_delimiter.join([biotype, *columns])
        return context_string

# dataset for remap data - all peaks

class RemapAllPeakDataset(Dataset):
    def __init__(
        self,
        *,
        factor_fasta_folder,
        bed_file = None,
        remap_df = None,
        filter_chromosome_ids = None,
        exclude_targets = None,
        include_targets = None,
        exclude_cell_types = None,
        include_cell_types = None,
        remap_df_frac = 1.,
        experiments_json_path = None,
        include_biotypes_metadata_in_context = False,
        biotypes_metadata_path = None,
        include_biotypes_metadata_columns = [],
        biotypes_metadata_delimiter = ' | ',
        balance_sampling_by_target = False,
        **kwargs
    ):
        super().__init__()
        assert exists(remap_df) ^ exists(bed_file), 'either remap bed file or remap dataframe must be passed in'

        if not exists(remap_df):
            remap_df = read_bed(bed_file)

        if remap_df_frac < 1:
            remap_df = remap_df.sample(frac = remap_df_frac)

        dataset_chr_ids = CHR_IDS

        if exists(filter_chromosome_ids):
            dataset_chr_ids = dataset_chr_ids.intersection(set(filter_chromosome_ids))

        remap_df = remap_df.filter(pl_isin('column_1', get_chr_names(dataset_chr_ids)))
        remap_df = filter_df_by_tfactor_fastas(remap_df, factor_fasta_folder)

        self.factor_ds = FactorProteinDataset(factor_fasta_folder)

        # filter dataset by inclusion and exclusion list of targets
        # (<all available targets> intersect <include targets>) subtract <exclude targets>

        include_targets = cast_list(include_targets)
        exclude_targets = cast_list(exclude_targets)

        if include_targets:
            remap_df = remap_df.filter(pl_isin('target', include_targets))

        if exclude_targets:
            remap_df = remap_df.filter(pl_notin('target', exclude_targets))

        # filter dataset by inclusion and exclusion list of cell types
        # same logic as for targets

        include_cell_types = cast_list(include_cell_types)
        exclude_cell_types = cast_list(exclude_cell_types)

        if include_cell_types:
            remap_df = remap_df.filter(pl_isin('cell_type', include_cell_types))

        if exclude_cell_types:
            remap_df = remap_df.filter(pl_notin('cell_type', exclude_cell_types))

        assert len(remap_df) > 0, 'dataset is empty by filter criteria'

        self.df = remap_df
        self.fasta = FastaInterval(**kwargs)

        self.experiments_index = fetch_experiments_index(experiments_json_path)

        # balanced target sampling logic

        self.balance_sampling_by_target = balance_sampling_by_target

        if self.balance_sampling_by_target:
            self.df_indexed_by_target = []

            for target in self.df.get_column('target').unique().to_list():
                df_by_target = self.df.filter(pl.col('target') == target)
                self.df_indexed_by_target.append(df_by_target)

        # context string creator

        self.context_ds = ContextDataset(
            include_biotypes_metadata_in_context = include_biotypes_metadata_in_context,
            biotypes_metadata_path = biotypes_metadata_path,
            include_biotypes_metadata_columns = include_biotypes_metadata_columns,
            biotypes_metadata_delimiter = biotypes_metadata_delimiter
        )

    def __len__(self):
        if self.balance_sampling_by_target:
            return len(self.df_indexed_by_target)
        else:
            return len(self.df)

    def __getitem__(self, ind):
        # if balancing by target, randomly draw sample from indexed dataframe

        if self.balance_sampling_by_target:
            filtered_df = self.df_indexed_by_target[ind]
            rand_ind = randrange(0, len(filtered_df))
            sample = filtered_df.row(rand_ind)
        else:
            sample = self.df.row(ind)

        chr_name, begin, end, _, _, _, experiment_target_cell_type, reading, *_ = sample

        # now aggregate all the data

        experiment, target, cell_type = parse_exp_target_cell(experiment_target_cell_type)

        seq = self.fasta(chr_name, begin, end)
        aa_seq = self.factor_ds[target]
        context_str = self.context_ds[cell_type]

        read_value = torch.Tensor([reading])

        peaks_nr = self.experiments_index.get(experiment_target_cell_type, 0.)
        peaks_nr = torch.Tensor([peaks_nr])

        label = torch.Tensor([1.])

        return seq, aa_seq, context_str, peaks_nr, read_value, label

# filter functions for exp-target-cells based on heldouts

def filter_exp_target_cell(
    arr,
    *,
    exclude_targets = None,
    include_targets = None,
    exclude_cell_types = None,
    include_cell_types = None,
):
    out = []

    for el in arr:
        experiment, target, cell_type = parse_exp_target_cell(el)

        if exists(include_targets) and len(include_targets) > 0 and target not in include_targets:
            continue

        if exists(exclude_targets) and target in exclude_targets:
            continue

        if exists(include_cell_types) and len(include_cell_types) > 0 and cell_type not in include_cell_types:
            continue

        if exists(exclude_cell_types) and cell_type in exclude_cell_types:
            continue

        out.append(el)

    return out


# dataset for negatives scoped to a specific exp-target-celltype

class ScopedNegativePeakDataset(Dataset):
    def __init__(
        self,
        *,
        fasta_file,
        factor_fasta_folder,
        numpy_folder_with_scoped_negatives,
        exts = '.bed.bool.npy',
        remap_bed_file = None,
        remap_df = None,
        filter_chromosome_ids = None,
        experiments_json_path = None,
        exclude_targets = None,
        include_targets = None,
        exclude_cell_types = None,
        include_cell_types = None,
        include_biotypes_metadata_in_context = False,
        biotypes_metadata_path = None,
        include_biotypes_metadata_columns = [],
        biotypes_metadata_delimiter = ' | ',
        balance_sampling_by_target = False,
        **kwargs
    ):
        super().__init__()
        assert exists(remap_df) ^ exists(remap_bed_file), 'either remap bed file or remap dataframe must be passed in'

        if not exists(remap_df):
            remap_df = read_bed(remap_bed_file)

        dataset_chr_ids = CHR_IDS

        if exists(filter_chromosome_ids):
            dataset_chr_ids = dataset_chr_ids.intersection(set(filter_chromosome_ids))

        filter_map_df = remap_df.with_column(pl.when(pl_isin('column_1', get_chr_names(dataset_chr_ids))).then(True).otherwise(False).alias('mask'))
        mask = filter_map_df.get_column('mask').to_numpy()

        num_scoped_negs = mask.sum()

        print(f'{num_scoped_negs} scoped negative rows found for training')

        assert num_scoped_negs > 0, 'all remap rows filtered out for scoped negative peak dataset'

        self.df = remap_df
        self.chromosome_mask = mask

        # get dictionary with exp-target-cell to boolean numpy indicating which ones are negatives

        npys_paths = [*Path(numpy_folder_with_scoped_negatives).glob('**/*.npy')]
        exp_target_cell_negatives = [(path.name.rstrip(exts), path) for path in npys_paths]

        exp_target_cells = [el[0] for el in exp_target_cell_negatives]

        exp_target_cells = filter_exp_target_cell(
            exp_target_cells,
            include_targets = include_targets,
            exclude_targets = exclude_targets,
            include_cell_types = include_cell_types,
            exclude_cell_types = exclude_cell_types
        )

        filtered_exp_target_cell_negatives = list(filter(lambda el: el[0] in exp_target_cells, exp_target_cell_negatives))

        self.exp_target_cell_negatives = filtered_exp_target_cell_negatives
        assert len(self.exp_target_cell_negatives) > 0, 'no experiment-target-cell scoped negatives to select from after filtering'

        # balanced target sampling

        self.balance_sampling_by_target = balance_sampling_by_target

        if balance_sampling_by_target:
            self.exp_target_cell_by_target = defaultdict(list)

            for exp_target_cell, filepath in self.exp_target_cell_negatives:
                _, target, *_ = parse_exp_target_cell(exp_target_cell)
                self.exp_target_cell_by_target[target].append((exp_target_cell, filepath))

        # tfactor dataset

        self.factor_ds = FactorProteinDataset(factor_fasta_folder)

        self.fasta = FastaInterval(fasta_file = fasta_file, **kwargs)
        self.experiments_index = fetch_experiments_index(experiments_json_path)

        # context string creator

        self.context_ds = ContextDataset(
            include_biotypes_metadata_in_context = include_biotypes_metadata_in_context,
            biotypes_metadata_path = biotypes_metadata_path,
            include_biotypes_metadata_columns = include_biotypes_metadata_columns,
            biotypes_metadata_delimiter = biotypes_metadata_delimiter
        )

    def __len__(self):
        if self.balance_sampling_by_target:
            return len(self.exp_target_cell_by_target)
        else:
            return len(self.exp_target_cell_negatives)

    def __getitem__(self, idx):
        if self.balance_sampling_by_target:
            negatives = list(self.exp_target_cell_by_target.values())[idx]
            sample = choice(negatives)
        else:
            sample = self.exp_target_cell_negatives[idx]

        exp_target_cell, bool_numpy_path = sample
        experiment, target, cell_type = parse_exp_target_cell(exp_target_cell)

        # load boolean numpy array
        # and select random peak that is a negative

        np_arr = np.load(str(bool_numpy_path))
        np_arr_noised = np_arr.astype(np.float32) + np.random.uniform(low = -1e-1, high = 1e-1, size = np_arr.shape[0])

        # mask with chromosomes allowed

        np_arr_noised *= self.chromosome_mask.astype(np.float32)

        # select random negative peak

        random_neg_peak_index = np_arr_noised.argmax()

        chr_name, begin, end, *_ = self.df.row(random_neg_peak_index)
        seq = self.fasta(chr_name, begin, end)

        aa_seq = self.factor_ds[target]
        context_str = self.context_ds[cell_type]

        peaks_nr = self.experiments_index.get(exp_target_cell, 0.)
        peaks_nr = torch.Tensor([peaks_nr])

        read_value = torch.Tensor([0.])

        label = torch.Tensor([0.])

        return seq, aa_seq, context_str, peaks_nr, read_value, label

# dataset for hard negatives (negatives to all peaks)

class NegativePeakDataset(Dataset):
    def __init__(
        self,
        *,
        factor_fasta_folder,
        negative_bed_file = None,
        remap_bed_file = None,
        remap_df = None,
        negative_df = None,
        filter_chromosome_ids = None,
        exclude_targets = None,
        include_targets = None,
        exclude_cell_types = None,
        include_cell_types = None,
        exp_target_cell_column = 'column_4',
        experiments_json_path = None,
        include_biotypes_metadata_in_context = False,
        biotypes_metadata_path = None,
        include_biotypes_metadata_columns = [],
        biotypes_metadata_delimiter = ' | ',
        balance_sampling_by_target = False,
        **kwargs
    ):
        super().__init__()
        assert exists(remap_df) ^ exists(remap_bed_file), 'either remap bed file or remap dataframe must be passed in'
        assert exists(negative_df) ^ exists(negative_bed_file), 'either negative bed file or negative dataframe must be passed in'

        # instantiate dataframes if not passed in

        if not exists(remap_df):
            remap_df = read_bed(remap_bed_file)

        neg_df = negative_df
        if not exists(negative_df):
            neg_df = read_bed(negative_bed_file)

        # filter remap dataframe

        remap_df = filter_df_by_tfactor_fastas(remap_df, factor_fasta_folder)

        dataset_chr_ids = CHR_IDS

        if exists(filter_chromosome_ids):
            dataset_chr_ids = dataset_chr_ids.intersection(set(filter_chromosome_ids))

        neg_df = neg_df.filter(pl_isin('column_1', get_chr_names(dataset_chr_ids)))

        assert len(neg_df) > 0, 'dataset is empty by filter criteria'

        self.neg_df = neg_df

        # get all exp-target-cells and filter by above

        exp_target_cells = remap_df.get_column(exp_target_cell_column).unique().to_list()

        self.filtered_exp_target_cells = filter_exp_target_cell(
            exp_target_cells,
            include_targets = include_targets,
            exclude_targets = exclude_targets,
            include_cell_types = include_cell_types,
            exclude_cell_types = exclude_cell_types
        )

        assert len(self.filtered_exp_target_cells), 'no experiment-target-cell left for hard negative set'

        # balanced sampling of targets

        self.balance_sampling_by_target = balance_sampling_by_target

        if balance_sampling_by_target:
            self.exp_target_cell_by_target = defaultdict(list)

            for exp_target_cell in self.filtered_exp_target_cells:
                _, target, *_ = parse_exp_target_cell(exp_target_cell)
                self.exp_target_cell_by_target[target].append(exp_target_cell)

        # factor ds

        self.factor_ds = FactorProteinDataset(factor_fasta_folder)
        self.fasta = FastaInterval(**kwargs)

        self.experiments_index = fetch_experiments_index(experiments_json_path)

        # context string creator

        self.context_ds = ContextDataset(
            include_biotypes_metadata_in_context = include_biotypes_metadata_in_context,
            biotypes_metadata_path = biotypes_metadata_path,
            include_biotypes_metadata_columns = include_biotypes_metadata_columns,
            biotypes_metadata_delimiter = biotypes_metadata_delimiter
        )

    def __len__(self):
        return len(self.neg_df)

    def __getitem__(self, ind):
        chr_name, begin, end = self.neg_df.row(ind)

        if self.balance_sampling_by_target:
            rand_ind = randrange(0, len(self.exp_target_cell_by_target))
            exp_target_cell_by_target_list = list(self.exp_target_cell_by_target.values())
            random_exp_target_cell_type = choice(exp_target_cell_by_target_list[rand_ind])
        else:
            random_exp_target_cell_type = choice(self.filtered_exp_target_cells)

        experiment, target, cell_type = parse_exp_target_cell(random_exp_target_cell_type)

        seq = self.fasta(chr_name, begin, end)
        aa_seq = self.factor_ds[target]
        context_str = self.context_ds[cell_type]

        read_value = torch.Tensor([0.])

        peaks_nr = self.experiments_index.get(random_exp_target_cell_type, 0.)
        peaks_nr = torch.Tensor([peaks_nr])

        label = torch.Tensor([0.])

        return seq, aa_seq, context_str, peaks_nr, read_value, label

# dataloader related functions

def collate_fn(data):
    seq, aa_seq, context_str, peaks_nr, read_values, labels = list(zip(*data))
    return torch.stack(seq), tuple(aa_seq), tuple(context_str), torch.stack(peaks_nr, dim = 0), torch.stack(read_values, dim = 0), torch.cat(labels, dim = 0)

def collate_dl_outputs(*dl_outputs):
    outputs = list(zip(*dl_outputs))
    ret = []
    for entry in outputs:
        if isinstance(entry[0], torch.Tensor):
            entry = torch.cat(entry, dim = 0)
        else:
            entry = (sub_el for el in entry for sub_el in el)
        ret.append(entry)
    return tuple(ret)

def cycle(loader):
    while True:
        for data in loader:
            yield data

def get_dataloader(ds, cycle_iter = False, **kwargs):
    dataset_len = len(ds)
    batch_size = kwargs.get('batch_size')
    drop_last = dataset_len > batch_size

    dl = DataLoader(ds, collate_fn = collate_fn, drop_last = drop_last, **kwargs)
    wrapper = cycle if cycle_iter else iter
    return wrapper(dl)
