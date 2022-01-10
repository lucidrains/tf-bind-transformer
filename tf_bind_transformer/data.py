from Bio import SeqIO
from random import choice
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset
from tf_bind_transformer.protein_utils import parse_gene_name

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
