"""
Thorsten-DE is an open German speech dataset containing 22,672 short audio clips
of a single male speaker reading sentences. Each clip has an accompanying transcription.

The dataset was created by Thorsten Müller and is released under CC0 (public domain).

See https://www.thorsten-voice.de/ for more details.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def prepare_thorsten_de(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None, **kwargs
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions.

    :param corpus_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :return: The RecordingSet and SupervisionSet with the keys 'recordings' and 'supervisions'.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv_path = corpus_dir / "metadata.csv"
    assert metadata_csv_path.is_file(), f"No such file: {metadata_csv_path}"

    lines = []
    with open(metadata_csv_path) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)

    recordings = []
    supervisions = []
    for line in tqdm(lines, desc="Scanning WAV files"):
        recording_id, text = line.split("|", maxsplit=1)
        audio_path = corpus_dir / "wavs" / f"{recording_id}.wav"
        if not audio_path.is_file():
            logging.warning(f"No such file: {audio_path}")
            continue
        recording = Recording.from_file(audio_path)
        segment = SupervisionSegment(
            id=recording_id,
            recording_id=recording_id,
            start=0.0,
            duration=recording.duration,
            channel=0,
            language="German",
            gender="male",
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
            output_dir / "thorsten_de_supervisions_all.jsonl.gz"
        )
        recording_set.to_file(output_dir / "thorsten_de_recordings_all.jsonl.gz")

    return {"all": {"recordings": recording_set, "supervisions": supervision_set}}
