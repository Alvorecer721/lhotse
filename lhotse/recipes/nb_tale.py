"""
NB-Tale is a Norwegian speech corpus with clip-level transcriptions.

Expected layout:
    corpus_dir/
      manifest.tsv
      part_1/.../*.wav
      part_2/.../*.wav
      part_3/.../*.wav

The manifest is tab-separated with columns:
    audio_path, text, speaker, duration
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional, Union

from tqdm.auto import tqdm

from lhotse import fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike


def _resolve_audio_path(audio_path_raw: str, corpus_dir: Path) -> Optional[Path]:
    audio_path = Path(audio_path_raw)
    candidates = []

    if audio_path.is_absolute():
        candidates.append(audio_path)

        # Allow manifest paths created elsewhere by re-rooting under corpus_dir.
        parts = audio_path.parts
        if "nb-tale" in parts:
            idx = parts.index("nb-tale")
            if idx + 1 < len(parts):
                candidates.append(corpus_dir / Path(*parts[idx + 1 :]))
    else:
        candidates.append(corpus_dir / audio_path)
        candidates.append(audio_path)

    for p in candidates:
        if p.is_file():
            return p
    return None


def _next_recording_id(base_id: str, seen_counts: Dict[str, int]) -> str:
    count = seen_counts.get(base_id, 0)
    seen_counts[base_id] = count + 1
    if count == 0:
        return base_id
    return f"{base_id}_{count + 1}"


def prepare_nb_tale(
    corpus_dir: Pathlike, output_dir: Optional[Pathlike] = None, **kwargs
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Prepare NB-Tale manifests.

    :param corpus_dir: Path to dataset root containing manifest.tsv.
    :param output_dir: Optional path where manifests are written.
    :return: Dict with split key 'all' and values {'recordings', 'supervisions'}.
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = corpus_dir / "manifest.tsv"
    assert manifest_path.is_file(), f"No such file: {manifest_path}"

    recordings = []
    supervisions = []
    seen_ids: Dict[str, int] = {}

    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fieldnames = reader.fieldnames or []
        required = {"audio_path", "text", "speaker"}
        missing = sorted(required - set(fieldnames))
        assert not missing, (
            f"manifest.tsv missing required columns: {missing}; "
            f"found: {fieldnames}"
        )

        for row in tqdm(reader, desc="Scanning NB-Tale rows"):
            audio_path_raw = (row.get("audio_path") or "").strip()
            text = (row.get("text") or "").strip()
            speaker = (row.get("speaker") or "").strip() or None

            if not audio_path_raw:
                logging.warning("Skipping row with empty audio_path")
                continue
            if not text:
                logging.warning("Skipping row with empty text for audio_path=%s", audio_path_raw)
                continue

            resolved_audio = _resolve_audio_path(audio_path_raw, corpus_dir)
            if resolved_audio is None:
                logging.warning("Audio file not found for row path: %s", audio_path_raw)
                continue

            base_id = resolved_audio.stem
            recording_id = _next_recording_id(base_id, seen_ids)

            try:
                recording = Recording.from_file(resolved_audio, recording_id=recording_id)
            except Exception as e:
                logging.warning("Failed to load audio %s: %s", resolved_audio, e)
                continue

            segment = SupervisionSegment(
                id=recording_id,
                recording_id=recording_id,
                start=0.0,
                duration=recording.duration,
                channel=0,
                language="Norwegian",
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
        supervision_set.to_file(output_dir / "nb_tale_supervisions_all.jsonl.gz")
        recording_set.to_file(output_dir / "nb_tale_recordings_all.jsonl.gz")

    return {"all": {"recordings": recording_set, "supervisions": supervision_set}}
