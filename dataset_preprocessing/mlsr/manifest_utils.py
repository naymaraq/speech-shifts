import json


def read_manifest(manifest_filepath):
    samples = []
    with open(manifest_filepath, 'r') as f:
        for line in f.readlines():
            samples.append(json.loads(line))
    return samples


def write_manifest(items, manifest_filepath):
    with open(manifest_filepath, 'w') as fout:
        for item in items:
            json.dump(item, fout, ensure_ascii=False)
            fout.write('\n')
            fout.flush()
