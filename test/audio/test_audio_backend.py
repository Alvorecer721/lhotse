import sys
import types
from io import BytesIO

import numpy as np
import pytest
import torch

import lhotse
from lhotse.audio import AudioLoadingError
from lhotse.audio.backend import (
    CompositeAudioBackend,
    LibsndfileBackend,
    TorchaudioDefaultBackend,
    TorchaudioFFMPEGBackend,
    TorchcodecBackend,
    check_torchaudio_version_gt,
    torchaudio_ffmpeg_backend_available,
    torchcodec_load,
    torchaudio_soundfile_supports_format,
)
from lhotse.testing.random import deterministic_rng
from lhotse.utils import INT16MAX, is_torchcodec_available, is_torchaudio_available


def test_default_audio_backend():
    lhotse.audio.backend.CURRENT_AUDIO_BACKEND = None
    b = lhotse.get_current_audio_backend()
    assert isinstance(b, CompositeAudioBackend)


def test_list_available_audio_backends():
    assert lhotse.available_audio_backends() == [
        "default",
        "AudioreadBackend",
        "CompositeAudioBackend",
        "FfmpegSubprocessOpusBackend",
        "FfmpegTorchaudioStreamerBackend",
        "LibsndfileBackend",
        "Sph2pipeSubprocessBackend",
        "TorchaudioDefaultBackend",
        "TorchaudioFFMPEGBackend",
        "TorchcodecBackend",
    ]


@pytest.mark.parametrize("backend", ["LibsndfileBackend", LibsndfileBackend()])
def test_audio_backend_contextmanager(backend):
    lhotse.audio.backend.CURRENT_AUDIO_BACKEND = None
    assert isinstance(lhotse.get_current_audio_backend(), CompositeAudioBackend)
    with lhotse.audio_backend(backend):
        assert isinstance(lhotse.get_current_audio_backend(), LibsndfileBackend)
    assert isinstance(lhotse.get_current_audio_backend(), CompositeAudioBackend)


@pytest.fixture()
def backend_set_via_env_var(monkeypatch):
    lhotse.audio.backend.CURRENT_AUDIO_BACKEND = None
    monkeypatch.setenv("LHOTSE_AUDIO_BACKEND", "LibsndfileBackend")
    yield
    lhotse.set_current_audio_backend("default")


def test_envvar_audio_backend(backend_set_via_env_var):
    b = lhotse.get_current_audio_backend()
    assert isinstance(b, LibsndfileBackend)


@pytest.mark.skipif(
    not torchaudio_soundfile_supports_format(), reason="Requires torchaudio v0.9.0+"
)
@pytest.mark.parametrize("backend", lhotse.available_audio_backends())
@pytest.mark.parametrize("format", ["wav", "flac", "opus"])
def test_save_and_load(deterministic_rng, tmp_path, backend, format):

    if (
        "Torchaudio" in backend
        and format == "opus"
        and not check_torchaudio_version_gt("2.1.0")
    ):
        return
    if backend == "CompositeAudioBackend":
        return

    path = tmp_path / f"test.{format}"

    with lhotse.audio_backend(backend) as backend_inst:
        if not backend_inst.supports_save():
            return
        if not backend_inst.is_applicable(path):
            return

        audio = (np.random.randint(0, INT16MAX, size=(1, 16000)) / INT16MAX).astype(
            np.float32
        )
        lhotse.audio.backend.save_audio(path, audio, sampling_rate=16000, format=format)
        restored, sr = lhotse.audio.backend.read_audio(path)

        if format != "opus":
            np.testing.assert_allclose(audio, restored)
        else:
            if backend in ("LibsndfileBackend", "default"):
                # only libnsdfile auto-resamples OPUS
                assert restored.shape == audio.shape
            else:
                assert restored.shape == (1, 48000)


@pytest.mark.skipif(
    not check_torchaudio_version_gt("2.1.0"),
    reason="Older torchaudio versions don't support most configurations in this test.",
)
@pytest.mark.parametrize(
    ["backend_save", "backend_read"],
    [
        pytest.param(
            LibsndfileBackend,
            TorchaudioFFMPEGBackend,
            marks=[
                pytest.mark.skipif(
                    not torchaudio_ffmpeg_backend_available(),
                    reason="Requires Torchaudio + FFMPEG",
                ),
            ],
        ),
        pytest.param(LibsndfileBackend, TorchaudioDefaultBackend),
        pytest.param(TorchaudioDefaultBackend, LibsndfileBackend),
        pytest.param(
            TorchaudioDefaultBackend,
            TorchaudioFFMPEGBackend,
            marks=[
                pytest.mark.skipif(
                    not torchaudio_ffmpeg_backend_available(),
                    reason="Requires Torchaudio + FFMPEG",
                ),
            ],
        ),
        pytest.param(
            TorchaudioFFMPEGBackend,
            LibsndfileBackend,
            marks=pytest.mark.skipif(
                not torchaudio_ffmpeg_backend_available(),
                reason="Requires Torchaudio + FFMPEG",
            ),
        ),
        pytest.param(
            TorchaudioFFMPEGBackend,
            TorchaudioDefaultBackend,
            marks=pytest.mark.skipif(
                not torchaudio_ffmpeg_backend_available(),
                reason="Requires Torchaudio + FFMPEG",
            ),
        ),
    ],
)
def test_save_load_opus_different_backends(
    deterministic_rng, tmp_path, backend_save, backend_read
):
    with lhotse.audio_backend(backend_save):
        audio = (np.random.randint(0, INT16MAX, size=(1, 16000)) / INT16MAX).astype(
            np.float32
        )
        path = tmp_path / "test.opus"
        lhotse.audio.backend.save_audio(path, audio, sampling_rate=16000, format="opus")

        recording_old = lhotse.Recording.from_file(path)

    with lhotse.audio_backend(backend_read):
        # Raw read/save utilities work across backends
        restored, sr = lhotse.audio.backend.read_audio(path)
        if backend_read == LibsndfileBackend:
            assert restored.shape == (1, 16000)
        else:
            assert restored.shape == (1, 48000)

        # lhotse Recording doesn't work when it was created with backend1 and is read with incompatible backend2
        if backend_save == LibsndfileBackend or backend_read == LibsndfileBackend:
            with pytest.raises(AudioLoadingError):
                restored2 = recording_old.load_audio()
        else:
            # but works otherwise (e.g. using different torchaudio versions)
            restored2 = recording_old.load_audio()
            np.testing.assert_allclose(restored2, restored)
            recording_old.to_cut().truncate(duration=0.2).move_to_memory(
                audio_format="wav"
            ).load_audio()

        # lhotse Recording works when created with the same backend but data saved using different backend
        recording_new = lhotse.Recording.from_file(path)
        restored3 = recording_new.load_audio()
        np.testing.assert_allclose(restored3, restored)
        # transcoding does not raise exception
        recording_new.to_cut().truncate(duration=0.2).move_to_memory(
            audio_format="wav"
        ).load_audio()


@pytest.mark.parametrize("backend", ["default", LibsndfileBackend])
def test_audio_info_from_bytes_io(backend):
    audio_filelike = BytesIO(open("test/fixtures/mono_c0.wav", "rb").read())
    with lhotse.audio_backend(backend):
        meta = lhotse.audio.info(audio_filelike)
        assert meta.duration == 0.5
        assert meta.frames == 4000
        assert meta.samplerate == 8000
        assert meta.channels == 1


def test_bytesio_special_case_routing_preserves_position(monkeypatch):
    import lhotse.audio.backend as backend_mod

    monkeypatch.setattr(backend_mod, "_torchcodec_runtime_available", lambda: True)

    wav = BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    wav.seek(5)
    assert LibsndfileBackend().handles_special_case(wav)
    assert not TorchcodecBackend().handles_special_case(wav)
    assert wav.tell() == 5

    ogg = BytesIO(b"OggS\x00\x02\x00\x00\x00\x00\x00\x00")
    ogg.seek(3)
    assert TorchcodecBackend().handles_special_case(ogg)
    assert not LibsndfileBackend().handles_special_case(ogg)
    assert ogg.tell() == 3

    mp3 = BytesIO(b"ID3\x04\x00\x00\x00\x00\x00\x21")
    mp3.seek(4)
    assert TorchcodecBackend().handles_special_case(mp3)
    assert not LibsndfileBackend().handles_special_case(mp3)
    assert mp3.tell() == 4


def test_composite_info_routes_bytesio_special_cases_without_fallbacks(monkeypatch):
    import lhotse.audio.backend as backend_mod

    monkeypatch.setattr(backend_mod, "_torchcodec_runtime_available", lambda: True)

    backend = CompositeAudioBackend([LibsndfileBackend(), TorchcodecBackend()])
    wav_meta = object()
    ogg_meta = object()

    def fake_libsndfile_info(self, path_or_fd, force_opus_sampling_rate=None):
        assert path_or_fd.tell() == 0
        return wav_meta

    def fake_torchcodec_info(self, path_or_fd, force_opus_sampling_rate=None):
        assert path_or_fd.tell() == 0
        return ogg_meta

    monkeypatch.setattr(LibsndfileBackend, "info", fake_libsndfile_info)
    monkeypatch.setattr(TorchcodecBackend, "info", fake_torchcodec_info)

    wav = BytesIO(b"RIFF\x24\x00\x00\x00WAVEfmt ")
    ogg = BytesIO(b"OggS\x00\x02\x00\x00\x00\x00\x00\x00")

    assert backend.info(wav) is wav_meta
    assert backend.info(ogg) is ogg_meta


@pytest.mark.skipif(not is_torchcodec_available(), reason="Requires torchcodec")
@pytest.mark.parametrize(
    "path",
    [
        "test/fixtures/mono_c0.wav",
        "test/fixtures/stereo.wav",
        "test/fixtures/mono_c0.opus",
        "test/fixtures/stereo.opus",
        "test/fixtures/common_voice_en_651325.mp3",
    ],
)
def test_torchcodec_info_and_read(path):
    backend = TorchcodecBackend()

    meta = backend.info(path)
    assert meta.samplerate > 0
    assert meta.channels > 0
    assert meta.duration > 0

    audio, sr = backend.read_audio(path)
    assert sr == meta.samplerate
    assert audio.shape[0] == meta.channels
    assert audio.dtype == np.float32


@pytest.mark.skipif(not is_torchcodec_available(), reason="Requires torchcodec")
@pytest.mark.parametrize(
    "path", ["test/fixtures/mono_c0.wav", "test/fixtures/stereo.wav"]
)
@pytest.mark.parametrize("offset", [0, 0.1])
@pytest.mark.parametrize("duration", [None, 0.1])
def test_torchcodec_read_with_offset_duration(path, offset, duration):
    backend = TorchcodecBackend()
    audio, sr = backend.read_audio(path, offset=offset, duration=duration)
    assert audio.ndim == 2
    if duration is not None:
        expected_samples = int(round(duration * sr))
        # Allow 1 sample tolerance for rounding
        assert abs(audio.shape[1] - expected_samples) <= 1


def test_torchcodec_not_applicable_for_sph():
    backend = TorchcodecBackend()
    assert not backend.is_applicable("test/fixtures/stereo.sph")


@pytest.mark.skipif(not is_torchcodec_available(), reason="Requires torchcodec")
@pytest.mark.parametrize("format", ["wav", "flac", "mp3", "m4a"])
def test_torchcodec_save_and_load(tmp_path, format):
    backend = TorchcodecBackend()
    assert backend.supports_save()
    audio = np.clip(np.random.randn(1, 16000).astype(np.float32) * 0.5, -1.0, 1.0)
    path = tmp_path / f"test.{format}"
    backend.save_audio(path, audio, sampling_rate=16000, format=format)
    restored, sr = backend.read_audio(path)
    assert sr == 16000
    assert restored.shape[0] == 1
    if format in ("wav", "flac"):
        np.testing.assert_allclose(audio, restored, atol=1e-4)


@pytest.mark.skipif(not is_torchcodec_available(), reason="Requires torchcodec")
def test_torchcodec_audio_backend_contextmanager():
    with lhotse.audio_backend("TorchcodecBackend") as b:
        assert isinstance(b, TorchcodecBackend)


@pytest.mark.skipif(
    not is_torchcodec_available() or not is_torchaudio_available(),
    reason="Requires torchcodec and torchaudio",
)
def test_torchcodec_load_falls_back_to_torchaudio_for_fileobj(monkeypatch):
    import torchaudio

    def failing_decoder(_):
        raise RuntimeError("torchcodec cannot decode this file-like object")

    expected = torch.arange(0, 320, dtype=torch.float32).reshape(2, 160)

    def fake_load(path_or_fd):
        assert path_or_fd.tell() == 0
        return expected.clone(), 16000

    fake_torchcodec = types.ModuleType("torchcodec")
    fake_decs = types.ModuleType("torchcodec.decoders")
    fake_decs.AudioDecoder = failing_decoder
    fake_torchcodec.decoders = fake_decs
    monkeypatch.setitem(sys.modules, "torchcodec", fake_torchcodec)
    monkeypatch.setitem(sys.modules, "torchcodec.decoders", fake_decs)
    monkeypatch.setattr(torchaudio, "load", fake_load)

    audio, sr = torchcodec_load(BytesIO(b"not-a-real-ogg"), offset=0.002, duration=0.004)

    assert sr == 16000
    np.testing.assert_equal(audio, expected.numpy()[:, 32:96])
