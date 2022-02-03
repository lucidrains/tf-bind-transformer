import json
import tqdm
import requests

NCBI_TAX_ID = dict(
    human = 9606,
    mouse = 10090
)

SPECIES = 'human'
API_URL = 'https://remap.univ-amu.fr/api/v1/'

def get_json(url, params = dict()):
    headers = dict(Accept = 'application/json')
    resp = requests.get(url, params = params, headers = headers)
    return resp.json()

def get_experiments(species):
    assert species in NCBI_TAX_ID
    taxid = NCBI_TAX_ID[species]
    experiments = get_json(f'{API_URL}list/experiments/taxid={taxid}')
    return experiments

def get_experiment(experiment_id, species):
    assert species in NCBI_TAX_ID
    taxid = NCBI_TAX_ID[species]
    experiment = get_json(f'http://remap.univ-amu.fr/api/v1/info/byExperiment/experiment={experiment_id}&taxid={taxid}')
    return experiment

experiments = get_experiments(SPECIES)

for experiment in tqdm.tqdm(experiments['experiments']):
    experiment_details = get_experiment(experiment['accession'], SPECIES)
    experiment['details'] = experiment_details

with open('data/experiments.json', 'w+') as f:
    contents = json.dumps(experiments, indent = 4, sort_keys = True)
    f.write(contents)

print('success')
