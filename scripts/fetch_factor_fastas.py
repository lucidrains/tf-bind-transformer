import requests
from pathlib import Path
import pandas as pd

UNIPROT_URL = 'http://www.uniprot.org'
REMAP_BED_PATH = './remap2022_crm_macs2_hg38_v1_0.bed'

GENE_NAME_TO_ID_OVERRIDE = {
    'SS18-SSX': ['Q8IZH1'],
    'TFIIIC': ['A6ZV34']        # todo: figure out where the human entry is in Uniprot
}

GENE_IDENTIFIER_MAP = {
    'RXR': 'RXRA'
}

NAMES_WITH_HYPHENS = {
    'NKX3-1',
    'NKX2-1',
    'NKX2-5',
    'SS18-SSX'
}

def parse_gene_name(name):
    if '-' not in name or name in NAMES_WITH_HYPHENS:
        return (name,)

    first, *rest = name.split('-')

    parsed_rest = []

    for name in rest:
        if len(name) == 1:
            name = f'{first[:-1]}{name}'
        parsed_rest.append(name)

    return tuple([first, *parsed_rest])

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

    df = pd.read_csv(REMAP_BED_PATH, sep = '\t', header = None)
    genes = set([target for targets in df[3] for target in targets.split(',')])

    print(f'{len(genes)} factors found')

    # load all saved fasta files, so can resume gracefully

    fasta_files = [str(path) for path in Path('./').glob('*.fasta')]
    processed_genes = set([*map(lambda t: str(t).split('.')[0], fasta_files)])

    for unparsed_gene_name in genes:
        if gene_name in processed_genes:
            continue

        for gene_name in parse_gene_name(unparsed_gene_name):
            # fetch uniprot id based on gene id

            gene_name = GENE_IDENTIFIER_MAP.get(gene_name, gene_name)

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
