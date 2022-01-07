from Bio import SeqIO
from random import choice
from pathlib import Path
from collections import defaultdict
from torch.utils.data import Dataset

class FactorProteinDataset(Dataset):
    def __init__(self, folder):
        super().__init__()
        fasta_paths = [*Path(folder).glob('*.fasta')]
        assert len(fasta_paths) > 0, f'no fasta files found at {folder}'
        self.paths = fasta_paths
        self.index_by_gene = defaultdict(list)
        self.index_by_id = dict()

        for path in fasta_paths:
            gene, uniprotid, *_ = path.stem.split('.')
            self.index_by_gene[gene].append(path)
            self.index_by_id[f'{gene}.{uniprotid}'] = path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ind):
        is_id = '.' in ind
        index = self.index_by_id if is_id else self.index_by_gene

        if ind not in index:
            return None

        entry = index[ind]

        if len(entry) == 0:
            return None

        path = choice(entry) if isinstance(entry, list) else entry

        fasta = SeqIO.read(path, 'fasta')
        return str(fasta.seq)
