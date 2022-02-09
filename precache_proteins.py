import click
from tqdm import tqdm
from pathlib import Path
from Bio import SeqIO
from tf_bind_transformer.protein_utils import get_protein_embedder

@click.command()
@click.option('--model-name', default = 'protalbert', help = 'Protein model name')
@click.option('--fasta-folder', help = 'Path to factor fastas', required = True)
def cache_embeddings(
    model_name,
    fasta_folder
):
    fn = get_protein_embedder(model_name)['fn']
    fastas = [*Path(fasta_folder).glob('**/*.fasta')]

    assert len(fastas) > 0, f'no fasta files found at {fasta_folder}'

    for fasta in tqdm(fastas):
        seq = SeqIO.read(fasta, 'fasta')
        seq_str = str(seq.seq)
        fn([seq_str], device = 'cpu')

if __name__ == '__main__':
    cache_embeddings()
