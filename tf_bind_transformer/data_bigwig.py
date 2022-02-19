from pathlib import Path
import polars as pl
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from tf_bind_transformer.data import FactorProteinDataset, ContextDataset, cast_list, filter_df_by_tfactor_fastas
from tf_bind_transformer.data import pl_isin, pl_notin, fetch_experiments_index, parse_exp_target_cell, read_bed, cycle
from tf_bind_transformer.data import CHR_IDS, CHR_NAMES, get_chr_names
from enformer_pytorch import FastaInterval

try:
    import pyBigWig
except ImportError:
    print('pyBigWig needs to be installed - conda install pyBigWig')
    exit()

def exists(val):
    return val is not None

def chip_atlas_add_experiment_target_cell(
    df,
    col_target = 'column_4',
    col_cell_type = 'column_5'
):
    df = df.clone()

    targets = df.select(col_target)
    targets = targets.rename({col_target: 'target'}).to_series(0)
    df.insert_at_idx(2, targets)

    cell_type = df.select(col_cell_type)
    cell_type = cell_type.rename({col_cell_type: 'cell_type'}).to_series(0)
    df.insert_at_idx(2, cell_type)

    return df

# dataset for CHIP ATLAS - all peaks

class BigWigDataset(Dataset):
    def __init__(
        self,
        *,
        factor_fasta_folder,
        bigwig_folder,
        enformer_loci_path,
        fasta_file,
        annot_file = None,
        filter_chromosome_ids = None,
        exclude_targets = None,
        include_targets = None,
        exclude_cell_types = None,
        include_cell_types = None,
        df_frac = 1.,
        experiments_json_path = None,
        include_biotypes_metadata_in_context = False,
        biotypes_metadata_path = None,
        filter_sequences_by = None,
        include_biotypes_metadata_columns = [],
        biotypes_metadata_delimiter = ' | ',
        only_ref = ['mm10', 'hg38'],
        downsample_factor = 128,
        target_length = 896,
        bigwig_reduction_type = 'sum',
        **kwargs
    ):
        super().__init__()
        assert exists(annot_file) 

        bigwig_folder = Path(bigwig_folder)
        assert bigwig_folder.exists(), 'bigwig folder does not exist'

        bw_experiments = [p.stem for p in bigwig_folder.glob('*.bw')]
        assert len(bw_experiments) > 0, 'no bigwig files found in bigwig folder'

        loci = read_bed(enformer_loci_path)
        annot_df = pl.read_csv(annot_file, sep = "\t", has_headers = False, columns = list(map(lambda i: f'column_{i + 1}', range(17))))

        annot_df = annot_df.filter(pl_isin('column_2', only_ref))

        # :TODO find out why this step is taking forever
        annot_df = annot_df.filter(pl_isin('column_1', bw_experiments))

        if df_frac < 1:
            annot_df = annot_df.sample(frac = df_frac)

        dataset_chr_ids = CHR_IDS

        if exists(filter_chromosome_ids):
            dataset_chr_ids = dataset_chr_ids.intersection(set(filter_chromosome_ids))

        # filtering loci by chromosomes
        # as well as training or validation

        loci = loci.filter(pl_isin('column_1', get_chr_names(dataset_chr_ids)))

        if exists(filter_sequences_by):
            col_name, col_val = filter_sequences_by
            loci = loci.filter(pl.col(col_name) == col_val)

        self.factor_ds = FactorProteinDataset(factor_fasta_folder)

        annot_df = chip_atlas_add_experiment_target_cell(annot_df)
        annot_df = filter_df_by_tfactor_fastas(annot_df, factor_fasta_folder)

        # filter dataset by inclusion and exclusion list of targets
        # (<all available targets> intersect <include targets>) subtract <exclude targets>

        include_targets = cast_list(include_targets)
        exclude_targets = cast_list(exclude_targets)

        if include_targets:
            annot_df = annot_df.filter(pl_isin('target', include_targets))

        if exclude_targets:
            annot_df = annot_df.filter(pl_notin('target', exclude_targets))

        # filter dataset by inclusion and exclusion list of cell types
        # same logic as for targets

        include_cell_types = cast_list(include_cell_types)
        exclude_cell_types = cast_list(exclude_cell_types)

        # :TODO reformulate this
        # Cell_type should probably be column_6
        if include_cell_types:
            annot_df = annot_df.filter(pl_isin('cell_type', include_cell_types))

        if exclude_cell_types:
            annot_df = annot_df.filter(pl_notin('cell_type', exclude_cell_types))

        self.fasta = FastaInterval(fasta_file = fasta_file, **kwargs)

        self.df = loci
        self.annot = annot_df
        self.ntargets = self.annot.shape[0]

        # bigwigs

        self.bigwigs = [pyBigWig.open(str(bigwig_folder / f'{str(i)}.bw')) for i in self.annot.get_column("column_1")]
        
        self.downsample_factor = downsample_factor
        self.target_length = target_length

        self.bigwig_reduction_type = bigwig_reduction_type

    def __len__(self):
        return len(self.df) * self.ntargets

    def __getitem__(self, ind):
        # TODO return all targets from an individual enformer loci
        chr_name, begin, end, _ = self.df.row(ind % self.df.shape[0])

        targets = self.annot.select('target').to_series(0)
        cell_types = self.annot.select('cell_type').to_series(0)

        ix_target = ind // self.df.shape[0]
    
        #experiment, target, cell_type = parse_exp_target_cell(experiment_target_cell_type)

        target = targets[ix_target]
        context_str = cell_types[ix_target]
        exp_bw = self.bigwigs[ix_target]

        # figure out ref and fetch appropriate sequence

        aa_seq = self.factor_ds[target]
        seq = self.fasta(chr_name, begin, end)

        # calculate bigwig
        # properly downsample and then crop

        output = np.array(exp_bw.values(chr_name, begin, end))
        output = output.reshape((-1, self.downsample_factor))

        if self.bigwig_reduction_type == 'mean':
            om = np.nanmean(output, axis = 1)
        elif self.bigwig_reduction_type == 'sum':
            om = np.nansum(output, axis = 1)
        else:
            raise ValueError(f'unknown reduction type {self.bigwig_reduction_type}')

        output_length = output.shape[0]

        if output_length < self.target_length:
            assert f'target length {self.target_length} cannot be less than the {output_length}'

        trim = (output.shape[0] - self.target_length) // 2
        om = om[trim:-trim]

        np.nan_to_num(om, copy = False)

        label = torch.Tensor(om)
        return seq, aa_seq, context_str, label

# data loader

def bigwig_collate_fn(data):
    seq, aa_seq, context_str, labels = list(zip(*data))
    return torch.stack(seq), tuple(aa_seq), tuple(context_str), torch.stack(labels)

def get_bigwig_dataloader(ds, cycle_iter = False, **kwargs):
    dataset_len = len(ds)
    batch_size = kwargs.get('batch_size')
    drop_last = dataset_len > batch_size

    dl = DataLoader(ds, collate_fn = bigwig_collate_fn, drop_last = drop_last, **kwargs)
    wrapper = cycle if cycle_iter else iter
    return wrapper(dl)
