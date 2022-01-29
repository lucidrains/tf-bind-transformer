# for fetching transcription factor sequences

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
        name = GENE_IDENTIFIER_MAP.get(name, name)

        if '_' in name:
            # for now, if target with modification
            # just search for the target factor name to the left of the underscore
            name, *_ = name.split('_')

        return (name,)

    first, *rest = name.split('-')

    parsed_rest = []

    for name in rest:
        if len(name) == 1:
            name = f'{first[:-1]}{name}'
        parsed_rest.append(name)

    return tuple([first, *parsed_rest])
