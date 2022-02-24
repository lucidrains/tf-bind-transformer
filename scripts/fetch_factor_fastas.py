import requests
from pathlib import Path
import click
import polars as pl
from tqdm import tqdm
from tf_bind_transformer.gene_utils import parse_gene_name
from tf_bind_transformer.data import read_bed

# constants

UNIPROT_URL = 'http://www.uniprot.org'

DEFAULT_REMAP_PATH = dict(
    HUMAN = './remap2022_crm_macs2_hg38_v1_0.bed',
    MOUSE = './remap2022_crm_macs2_mm10_v1_0.bed',
)

GENE_NAME_TO_ID_OVERRIDE = {
    'SS18-SSX': ['Q8IZH1'],
    'TFIIIC': ['A6ZV34']        # todo: figure out where the human entry is in Uniprot
}

# helper functions

def uniprot_mapping(fromtype, totype, identifier):
    params = {
        'from': fromtype,
        'to': totype,
        'format': 'tab',
        'query': identifier,
    }

    response = requests.get(f'{UNIPROT_URL}/mapping', params = params)
    return response.text

# main functions

@click.command()
@click.option('--species', help = 'Species', default = 'human', type = click.Choice(['human', 'mouse']))
@click.option('--remap-bed-path', help = 'Path to species specific remap file')
@click.option('--fasta-folder', help = 'Path to factor fastas', default = './tfactor.fastas')
def fetch_factors(
    species,
    remap_bed_path,
    fasta_folder
):
    species = species.upper()

    if remap_bed_path is None:
        remap_bed_path = DEFAULT_REMAP_PATH[species]

    remap_bed_path = Path(remap_bed_path)

    assert remap_bed_path.exists(), f'remap file does not exist at {str(remap_bed_path)}'

    # load bed file and get all unique targets from column 3

    df = read_bed(remap_bed_path)
    genes = set([target for targets in df[:, 3] for target in targets.split(',')])

    print(f'{len(genes)} factors found')

    # load all saved fasta files, so can resume gracefully

    fasta_files = [str(path) for path in Path('./').glob('*.fasta')]
    processed_genes = set([*map(lambda t: str(t).split('.')[0], fasta_files)])

    results_folder = Path(fasta_folder)
    results_folder.mkdir(exist_ok = True, parents = True)

    for unparsed_gene_name in tqdm(genes):
        for gene_name in parse_gene_name(unparsed_gene_name):

            if gene_name in processed_genes:
                continue

            # fetch uniprot id based on gene id

            if gene_name not in GENE_NAME_TO_ID_OVERRIDE:
                uniprot_resp = uniprot_mapping('GENENAME', 'ID', gene_name)

                # only get the human ones (todo: make species agnostic)

                entries = list(filter(lambda t: f'_{species}' in t, uniprot_resp.split('\n')))
                entries = list(map(lambda t: t.split('\t')[1], entries))
            else:
                entries = GENE_NAME_TO_ID_OVERRIDE[gene_name]

            if len(entries) == 0:
                print(f'no entries found for {gene_name}')
                continue

            # save all hits

            for entry in entries:
                response = requests.get(f'{UNIPROT_URL}/uniprot/{entry}.fasta')

                if response.status_code != 200:
                    print(f'<{response.status_code}> error fetching fasta file from gene {gene_name} {entry}')
                    continue

                fasta_path = str(results_folder / f'{gene_name}.{entry}.fasta')

                with open(fasta_path, 'w') as f:
                    f.write(response.text)

            print(f'gene {gene_name} written')

# main function

if __name__ == '__main__':
    fetch_factors()
