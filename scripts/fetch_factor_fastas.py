import urllib
import urllib.parse
from urllib.request import urlopen
import requests
from pathlib import Path
import pandas as pd

UNIPROT_URL = 'http://www.uniprot.org'
REMAP_BED_PATH = './remap2022_crm_macs2_hg38_v1_0.bed'

def uniprot_mapping(fromtype, totype, identifier):
    tool = 'mapping'

    params = {
        'from': fromtype,
        'to': totype,
        'format': 'tab',
        'query': identifier,
    }

    data = urllib.parse.urlencode(params)
    response = urlopen(f'{UNIPROT_URL}/{tool}?{data}')
    return response.read().decode('utf-8')

# load bed file and get all unique targets from column 3

df = pd.read_csv(REMAP_BED_PATH, sep = '\t', header = None)
genes = set([target for targets in df[3] for target in targets.split(',')])

print(f'{len(genes)} factors found')

# load all saved fasta files, so can resume gracefully

fasta_files = [str(path) for path in Path('./').glob('*.fasta')]
processed_genes = set([*map(lambda t: str(t).split('.')[0], fasta_files)])

for gene_name in genes:
    if gene_name in processed_genes:
        continue

    # fetch uniprot id based on gene id

    uniprot_resp = uniprot_mapping('GENENAME', 'ID', gene_name)

    # only get the human ones (todo: make species agnostic)

    entries = list(filter(lambda t: '_HUMAN' in t, uniprot_resp.split('\n')))
    entries = list(map(lambda t: t.split('\t')[1], entries))

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
