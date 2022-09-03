import numpy as np
from manifest_utils import read_manifest
from tqdm import tqdm

np.random.seed(2022)

rel_path = lambda x: "/".join(x.split("/")[-3:])

def get_collection(data):
    collection = {}
    for sample in data:
        gender = sample["gender"]
        age = sample["age"]
        label = sample["speaker"]
        if gender and age:
            key = f"{gender}_{age}"
            if key not in collection:
                collection[key] = {}
            if label not in collection[key]:
                collection[key][label] = []
            collection[key][label].append(sample)
    return collection

def remove_bad_speakers(collection):
    new_collection = {}
    for key in collection:
        if len(collection[key]) >= 20:
            for label in collection[key]:
                z = len(collection[key][label])
                if z >= 2:
                    if key not in new_collection:
                        new_collection[key] = {}
                    new_collection[key][label] = collection[key][label]
    return new_collection

def remove_duplicate_trials(trials):
    bag = set()
    unique_trials = []
    for y, p0, p1 in trials:
        a1 = p0["audio_filepath"]
        a2 = p1["audio_filepath"]
        a = a1+a2
        if a not in bag:
           bag |= {a}
           unique_trials.append((y, p0, p1))

    return unique_trials

def get_trials(data, hard, n_trials):    
    collection = get_collection(data)
    collection = remove_bad_speakers(collection)

    trials = []
    categories = sorted(list(collection.keys()))
    for i in tqdm(range(0, n_trials)):
        category = categories[i%len(categories)]
        samples_given_category = collection[category]
        labels_given_category = [l for l in samples_given_category]
        
        if i%2==0:
            #positive pair
            anchor_label = np.random.choice(labels_given_category)
            pair = np.random.choice(samples_given_category[anchor_label], 2, replace=False)
            y = 1
        else:
            #negative pair
            y = 0
            if hard:
                anchor_label, neg_label = np.random.choice(labels_given_category, 2, replace=False)
                anchor = np.random.choice(samples_given_category[anchor_label])
                neg = np.random.choice(samples_given_category[neg_label])
            else:
                anchor_label = np.random.choice(labels_given_category)
                anchor = np.random.choice(samples_given_category[anchor_label])
                
                while 1:
                    samples_given_category = collection[np.random.choice(categories)]
                    labels_given_category = [l for l in samples_given_category if l != anchor_label]
                    if len(labels_given_category)>0:
                        break

                neg_label = np.random.choice(labels_given_category)
                neg = np.random.choice(samples_given_category[neg_label])
            pair = [anchor, neg]
        
        trials.append((y, pair[0], pair[1]))

    unique_trials = remove_duplicate_trials(trials)
    unique_trials = [(y, rel_path(a1["audio_filepath"]), rel_path(a2["audio_filepath"])) 
                        for y, a1, a2 in unique_trials
                    ]
    return unique_trials


def generate_trials(manifest_paths):
    data = []
    for manifest_path in manifest_paths:
        data.extend(read_manifest(manifest_path))
    
    hard_trials = get_trials(data, hard=True, n_trials=50000)
    easy_trials = get_trials(data, hard=False, n_trials=50000)
    hard_trials = hard_trials[:30000]
    easy_trials = easy_trials[:30000]

    return easy_trials, hard_trials


def save_trial(trials, trials_path):
    with open(trials_path, "w") as f:
        for y, p1, p2 in trials:
            f.write(f"{y} {p1} {p2}\n")

if __name__ == "__main__":
    dataset2manifests = {
        "rw": [
            "cv-corpus-wav/manifests/dev-rw.json",
            "cv-corpus-wav/manifests/test-rw.json",
            "cv-corpus-wav/manifests/train-rw.json"
        ],
        "es": [
            "cv-corpus-wav/manifests/dev-es.json",
            "cv-corpus-wav/manifests/test-es.json",
            "cv-corpus-wav/manifests/train-es.json"

        ],
        "zh-CN": [
            "cv-corpus-wav/manifests/dev-zh-CN.json",
            "cv-corpus-wav/manifests/test-zh-CN.json",
            "cv-corpus-wav/manifests/train-zh-CN.json"
        ],
        "de": [
            "cv-corpus-wav/manifests/dev-de.json",
            "cv-corpus-wav/manifests/test-de.json",
            "cv-corpus-wav/manifests/train-de.json"

        ]
    }

    out_dir = "cv-corpus-wav/manifests"
    for lang, manifests in dataset2manifests.items():
        easy_trials, hard_trials = generate_trials(manifests)

        easy_trial_path = f"{out_dir}/{lang}-easy.txt"
        hard_trial_path = f"{out_dir}/{lang}-hard.txt"

        save_trial(easy_trials, easy_trial_path)
        save_trial(hard_trials, hard_trial_path)


