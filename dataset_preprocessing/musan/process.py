import os
import glob
import librosa
import json
from wav_spliter import WavSpliter

SAMPLE_RATE = 16000
CUT_SIZE_IN_SEC = 10

def get_duration(filename):
    return librosa.get_duration(filename=filename)

def process_dataset(root_dir, key, out_dir):
    cut_dir_for_key = os.path.join(out_dir, key)
    os.makedirs(cut_dir_for_key, exist_ok=True)

    folder_path = os.path.join(root_dir, key)
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder {folder_path} does not exist.")
    files = glob.glob(os.path.join(folder_path, "**/*.wav"), recursive=True)

    items_for_split = []
    for f in files:
        duration  = get_duration(f)
        start = 0
        items = []
        while start < duration:
            end = min(start+CUT_SIZE_IN_SEC, duration)
            _, tail = os.path.split(f)
            tail = tail.replace(".wav", "")
            cut_wav_filename = f"{cut_dir_for_key}/{tail}-{start}-{end}.wav"
            items.append({ "wav_filename": cut_wav_filename,
                           "start": start,
                           "end": end,
                           "label": ""
            })
            start = end
        items_for_split.append((f, items))

    spliter = WavSpliter()
    samples = spliter.parallel_split(items_for_split)

    json_path = os.path.join(out_dir, f"musan-{key}.json")
    with open(json_path, 'w') as fout:
        for item in samples:
            duration = get_duration(item['wav_filename'])
            metadata = {
                    "audio_filepath": item['wav_filename'],
                    "duration": duration
            }
            if item["label"]:
                metadata.update({"label": item['label']})
            json.dump(metadata, fout)
            fout.write('\n')
            fout.flush()

    return [json_path]

if __name__ == "__main__":

    root_dir = "/data/mlsr-data/musan"
    out_dir  = "/data/mlsr-data/musan/musan-proccesed"
    os.makedirs(out_dir, exist_ok=True)

    for key in ["music", "speech", "noise"]:
        process_dataset(root_dir, key, out_dir)
