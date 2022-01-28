import requests
from pathlib import Path
import polars as pl
from tf_bind_transformer.protein_utils import parse_gene_name
from tf_bind_transformer.data import read_bed

UNIPROT_URL = 'http://www.uniprot.org'
REMAP_BED_PATH = './remap2022_crm_macs2_hg38_v1_0.bed'

GENE_NAME_TO_ID_OVERRIDE = {
    'SS18-SSX': ['Q8IZH1'],
    'TFIIIC': ['A6ZV34']        # todo: figure out where the human entry is in Uniprot
}

def uniprot_mapping(fromtype, totype, identifier):
    params = {
        'from': fromtype,
        'to': totype,
        'format': 'tab',
        'query': identifier,
    }

    response = requests.get(f'{UNIPROT_URL}/mapping', params = params)
    return response.text

if __name__ == '__main__':
    # load bed file and get all unique targets from column 3

    df = read_bed(REMAP_BED_PATH)
    genes = set([target for targets in df[:, 3] for target in targets.split(',')])

    print(f'{len(genes)} factors found')

    # load all saved fasta files, so can resume gracefully

    fasta_files = [str(path) for path in Path('./').glob('*.fasta')]
    processed_genes = set([*map(lambda t: str(t).split('.')[0], fasta_files)])

    for unparsed_gene_name in genes:
        if gene_name in processed_genes:
            continue

        for gene_name in parse_gene_name(unparsed_gene_name):
            # fetch uniprot id based on gene id

            if gene_name not in GENE_NAME_TO_ID_OVERRIDE:
                uniprot_resp = uniprot_mapping('GENENAME', 'ID', gene_name)

                # only get the human ones (todo: make species agnostic)

                entries = list(filter(lambda t: '_HUMAN' in t, uniprot_resp.split('\n')))
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

                with open(f'./{gene_name}.{entry}.fasta', 'w') as f:
                    f.write(response.text)

            print(f'gene {gene_name} written')
