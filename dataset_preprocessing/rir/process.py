import os
import glob
import librosa
import json
from tqdm import tqdm

SAMPLE_RATE = 16000

def get_duration(filename):
    return librosa.get_duration(filename=filename)

def process_dataset(root_dir, key, out_dir):
    
    folder_path = os.path.join(root_dir, key)
    if not os.path.isdir(folder_path):
        raise ValueError(f"The folder {folder_path} does not exist.")
    files = glob.glob(os.path.join(folder_path, "**/*.wav"), recursive=True)
    samples = []
    for f in files:
        samples.append({"wav_filename": f,
                "label": ""}
        )
    json_path = os.path.join(out_dir, f"rir-{key}.json")
    with open(json_path, 'w') as fout:
        for item in tqdm(samples):
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

if __name__ == "__main__":

    root_dir = "/data/mlsr-data/RIRS_NOISES/"
    out_dir  = "/data/mlsr-data/RIRS_NOISES/rir-proccesed"
    os.makedirs(out_dir, exist_ok=True)

    for key in ["pointsource_noises", "real_rirs_isotropic_noises", "simulated_rirs"]:
        process_dataset(root_dir, key, out_dir)
