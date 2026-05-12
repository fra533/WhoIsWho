import os
import json
import codecs
from os.path import join


def check_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dump_json(obj, wfname, indent=None):
    with codecs.open(wfname, 'w', encoding='utf-8') as wf:
        json.dump(obj, wf, ensure_ascii=False, indent=indent)


def save_results(args, names, pubs, results):
    result_dict = {}

    for name in names:
        if isinstance(pubs[name], dict):
            paper_ids = []
            for aid in pubs[name]:
                paper_ids.extend(pubs[name][aid])
        elif isinstance(pubs[name], list):
            paper_ids = pubs[name]
        else:
            print(f"Warning: unexpected format for {name}")
            paper_ids = []

        clusters = results[name]
        if not isinstance(clusters, list):
            print(f"Warning: {name} has wrong format, converting")
            clusters = [[clusters]] if clusters else []

        result_dict[name] = clusters

    output_dir  = 'out'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'res.json')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")
    return output_file