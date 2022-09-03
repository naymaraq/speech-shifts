import os

from manifest_utils import read_manifest, write_manifest


def get_dur_mapping(data):
    mapping = {}
    for sample in data:
        label = sample["speaker"]
        if label not in mapping:
            mapping[label] = 0
        mapping[label] += sample["duration"]
    return mapping


def find_candidate_speakers(data_manifest_paths, min_dur, max_dur):
    data = []
    for manifest_path in data_manifest_paths:
        data.extend(read_manifest(manifest_path))

    dur_mapping = get_dur_mapping(data)
    candidates = []
    for item in data:
        label = item["speaker"]
        if dur_mapping[label] < min_dur or dur_mapping[label] > max_dur:
            candidates.append(label)
    return set(candidates)


def filter_out(data_manifest_paths, min_dur, max_dur, lang, out_folder):
    os.makedirs(out_folder, exist_ok=True)

    bad_speakers = find_candidate_speakers(data_manifest_paths, min_dur, max_dur)
    print("Candidate speaker count for {} = {}".format(lang, len(bad_speakers)))
    for manifest_path in data_manifest_paths:
        items = read_manifest(manifest_path)
        filtered_data = [item for item in items if item["speaker"] not in bad_speakers]
        path = manifest_path.replace(".json", f"-{lang}.json")
        path = os.path.join(out_folder, os.path.split(path)[1])
        write_manifest(filtered_data, path)
