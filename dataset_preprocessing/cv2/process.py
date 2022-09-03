import csv
import os
import sys
from multiprocessing import Pool

import librosa
import progressbar
import sox

from filter import filter_out
from manifest_utils import write_manifest

csv.field_size_limit(sys.maxsize)
CHANNELS = 1
SAMPLE_RATE = 16000
ONE_HOUR_IN_SEC = 3600

MIN_UTTERANCE_DUR = 0
MAX_UTTERANCE_DUR = 30
MIN_SPEAKER_DUR = 5
MAX_SPEAKER_DUR = 5000


def _maybe_convert_wav(mp3_filename, wav_filename):
    if not os.path.exists(wav_filename):
        transformer = sox.Transformer()
        transformer.convert(samplerate=SAMPLE_RATE, n_channels=CHANNELS)
        try:
            transformer.build(mp3_filename, wav_filename)
        except sox.core.SoxError:
            pass


def one_sample(sample):
    mp3_filename = sample[0]
    out_audio_dir = sample[-1]
    wav_filename = os.path.splitext(mp3_filename)[0] + ".wav"
    wav_filename = os.path.split(wav_filename)[-1]
    wav_filename = os.path.join(out_audio_dir, wav_filename)
    _maybe_convert_wav(mp3_filename, wav_filename)

    file_size = -1
    if os.path.exists(wav_filename):
        file_size = os.path.getsize(wav_filename)
    rows = [(os.path.split(wav_filename)[-1],
             file_size,
             sample[1],
             sample[2],
             sample[3],
             sample[4])
            ]
    return rows


def process_dataset(root_dir, lang, out_dir):
    tsv_dir = os.path.join(root_dir, lang)
    lang_out_dir = os.path.join(out_dir, lang)
    os.makedirs(lang_out_dir, exist_ok=True)

    exclude = []
    json_files = []
    for dataset in ["test", "dev", "validated"]:
        set_samples, json_file = _maybe_convert_set(tsv_dir,
                                                    lang_out_dir,
                                                    dataset=dataset,
                                                    lang=lang
                                                    )
        if dataset in ["test", "dev"]:
            exclude += set_samples
        if dataset == "validated":
            _, json_file = _maybe_convert_set(tsv_dir,
                                              lang_out_dir,
                                              dataset="train",
                                              rows=set_samples,
                                              exclude=exclude,
                                              lang=lang)
        json_files.append(json_file)

    filter_out(json_files,
               MIN_SPEAKER_DUR,
               MAX_SPEAKER_DUR,
               lang,
               os.path.join(out_dir, "manifests"))


def _maybe_convert_set(tsv_dir,
                       out_dir,
                       dataset,
                       rows=None,
                       exclude=None,
                       lang=None):
    audio_dir = os.path.join(tsv_dir, "clips")
    out_audio_dir = os.path.join(out_dir, "clips")
    os.makedirs(out_audio_dir, exist_ok=True)

    exclude_filenames = set()
    if exclude is not None:
        for sample in exclude:
            exclude_filenames.add(sample[0])

    if rows is None:
        rows = []
        input_tsv = os.path.join(os.path.abspath(tsv_dir), dataset + ".tsv")

        samples = []
        with open(input_tsv, encoding="utf-8") as input_tsv_file:
            reader = csv.DictReader(input_tsv_file, delimiter="\t")
            for row in reader:
                samples.append(
                    (os.path.join(audio_dir, row["path"]),
                     row["sentence"],
                     row["client_id"],
                     row["age"],
                     row["gender"],
                     out_audio_dir
                     )
                )

        num_samples = len(samples)
        pool = Pool()
        bar = progressbar.ProgressBar(max_value=num_samples)
        for i, processed in enumerate(pool.imap_unordered(one_sample, samples), start=1):
            rows += processed
            bar.update(i)
        bar.update(num_samples)
        pool.close()
        pool.join()

    output_json = os.path.join(out_dir, f"{dataset}.json")
    filtered_rows = []
    bar = progressbar.ProgressBar(max_value=len(rows))
    filtered_samples_count = 0
    fitered_duaration = 0
    total_duration = 0
    for filename, file_size, transcript, speaker, age, gender in bar(rows):
        if filename in exclude_filenames:
            continue
        wav_filename = os.path.join(out_audio_dir, filename)
        duration = librosa.core.get_duration(filename=wav_filename, sr=SAMPLE_RATE)
        total_duration += duration
        if MIN_UTTERANCE_DUR <= duration <= MAX_UTTERANCE_DUR:
            filtered_rows.append({"audio_filepath": os.path.abspath(wav_filename),
                                  "duration": duration,
                                  "transcript": transcript,
                                  "speaker": speaker,
                                  "lang": lang,
                                  "age": age,
                                  "gender": gender})
        else:
            filtered_samples_count += 1
            fitered_duaration += duration

    print(f"Filtered  {filtered_samples_count}/{len(rows)} files for {lang}")
    print(f"Filtered  {fitered_duaration}/{round(total_duration / ONE_HOUR_IN_SEC, 2)}h audio for {lang}")

    write_manifest(filtered_rows, output_json)
    return rows, output_json


if __name__ == "__main__":

    root_dir = "cv-corpus-10.0-2022-07-04"
    out_dir = "cv-corpus-wav"
    os.makedirs(out_dir, exist_ok=True)
    for lang in ["zh-CN", "it", "be", "ca", "de", "eo", "es", "fr", "rw", "en"]:
        process_dataset(root_dir, lang, out_dir)