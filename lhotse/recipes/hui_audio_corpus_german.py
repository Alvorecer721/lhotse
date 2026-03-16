"""
HUI-Audio-Corpus-German is an open German speech dataset with ~100 speakers
reading audiobooks. Each speaker has multiple books, each with a metadata.csv
(pipe-separated: clip_id|text) and a wavs/ directory.

Structure: corpus_dir/{speaker}/{book}/metadata.csv + wavs/*.wav

~96k clips, ~260 hours. Released under CC-BY 4.0.

See https://opendata.iisys.de/opendata/Datasets/HUI-Audio-Corpus-German/
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def prepare_hui_audio_corpus_german(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None, **kwargs
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: a Dict with key 'all' and value Dicts with 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Discover all metadata.csv files: speaker/book/metadata.csv
    metadata_files = sorted(corpus_dir.rglob("metadata.csv"))
    assert len(metadata_files) > 0, f"No metadata.csv found under {corpus_dir}"
    logging.info(f"Found {len(metadata_files)} metadata.csv files")

    recordings = []
    supervisions = []
    for meta_path in tqdm(metadata_files, desc="Processing HUI books"):
        book_dir = meta_path.parent
        speaker = book_dir.parent.name
        wavs_dir = book_dir / "wavs"

        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                clip_id, text = line.split("|", maxsplit=1)
                audio_path = wavs_dir / f"{clip_id}.wav"
                if not audio_path.is_file():
                    logging.warning(f"No such file: {audio_path}")
                    continue
                recording = Recording.from_file(audio_path)
                segment = SupervisionSegment(
                    id=clip_id,
                    recording_id=clip_id,
                    start=0.0,
                    duration=recording.duration,
                    channel=0,
                    language="German",
                    speaker=speaker,
                    text=text,
                )
                recordings.append(recording)
                supervisions.append(segment)

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    if output_dir is not None:
        supervision_set.to_file(
            output_dir / "hui_audio_corpus_german_supervisions_all.jsonl.gz"
        )
        recording_set.to_file(
            output_dir / "hui_audio_corpus_german_recordings_all.jsonl.gz"
        )

    return {"clean": {"recordings": recording_set, "supervisions": supervision_set}}
