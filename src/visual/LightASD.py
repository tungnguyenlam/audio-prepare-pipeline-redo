"""Light-ASD active-speaker detection on tracked faces.

Architecture and scoring follow
https://github.com/Junhua-Liao/Light-ASD (CVPR 2023). Weights are cloned
into ``.data/light-asd`` on first ``load()``. Face identity is handled
separately; this class only answers whether a visible face is speaking.
"""

from __future__ import annotations

import gc
import logging
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.base.model import ManagedModel
from src.data_paths import DATA_DIR
from src.diarization.schemas import DiarizationModelInfo
from src.utils.AudioClass import Audio
from src.visual.Video import Video
from src.visual.schemas import ASDFrameScore, ASDResult, FaceTrack, FaceTrackSet

logger = logging.getLogger(__name__)

LIGHT_ASD_GIT_URL = "https://github.com/Junhua-Liao/Light-ASD.git"
DEFAULT_WEIGHT_NAME = "pretrain_AVA_CVPR.model"
ASD_FPS = 25
ASD_SAMPLE_RATE = 16000
CROP_SCALE = 0.40
DEFAULT_DURATION_SET = (1, 2, 3, 4, 5, 6)


class LightASDError(RuntimeError):
    """Raised when Light-ASD weights cannot be loaded or scoring fails."""


class LightASD(ManagedModel):
    """Score face tracks for audible active speech.

    Input face crops are grayscale 112×112 at 25 fps with 13-dim MFCCs at
    100 Hz, matching the upstream inference path. The speaking threshold is
    ``0`` on the class-1 logit (the official demo convention).

    Example::

        asd = LightASD(device="cuda")
        with asd:
            scores = asd.score(video, audio, tracks)
    """

    def __init__(
        self,
        *,
        device: str = "auto",
        weights_path: str | Path | None = None,
        repo_dir: str | Path | None = None,
        active_threshold: float = 0.0,
        duration_set: tuple[int, ...] = DEFAULT_DURATION_SET,
        crop_scale: float = CROP_SCALE,
    ) -> None:
        """Initialize Light-ASD.

        Args:
            device: ``"auto"``, ``"cuda"``, or ``"cpu"``.
            weights_path: Optional explicit ``.model`` checkpoint. Defaults to
                ``.data/light-asd/weight/pretrain_AVA_CVPR.model``.
            repo_dir: Checkout used to obtain official weights when missing.
            active_threshold: Minimum logit to mark a frame as speaking.
            duration_set: Chunk lengths in seconds averaged at inference,
                following the upstream ensemble.
            crop_scale: Bounding-box expansion used when cropping faces.
        """
        ManagedModel.__init__(self)
        if not duration_set or any(item < 1 for item in duration_set):
            raise ValueError("duration_set must contain positive integers")
        self.device = str(device)
        self.repo_dir = (
            Path(repo_dir).expanduser()
            if repo_dir is not None
            else DATA_DIR / "light-asd"
        )
        self.weights_path = (
            Path(weights_path).expanduser()
            if weights_path is not None
            else self.repo_dir / "weight" / DEFAULT_WEIGHT_NAME
        )
        self.active_threshold = float(active_threshold)
        self.duration_set = tuple(int(item) for item in duration_set)
        self.crop_scale = float(crop_scale)
        self._net: Any | None = None
        self._torch_device: Any | None = None

    def _load(self) -> None:
        """Clone official weights if needed and load the network."""
        path = self._ensure_weights()
        if self.device == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            torch_device = torch.device(self.device)
        net = _LightASDNet()
        state = torch.load(path, map_location="cpu", weights_only=False)
        _load_compatible_state(net, state)
        net.to(torch_device)
        net.eval()
        self._net = net
        self._torch_device = torch_device

    def _unload(self) -> None:
        """Release the network and CUDA cache."""
        self._net = None
        self._torch_device = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score(
        self,
        video: Video,
        audio: Audio,
        tracks: FaceTrackSet,
    ) -> ASDResult:
        """Score every face track for active audible speech.

        Args:
            video: File-backed video whose faces were tracked.
            audio: File-backed audio aligned to the same clock.
            tracks: Output of :meth:`FaceAnalyzer.analyze`.

        Returns:
            Per-track 25 Hz scores with ``speaking`` flags.

        Raises:
            RuntimeError: If the model is not loaded.
            FileNotFoundError: If video or audio is missing.
            LightASDError: If audio features cannot be extracted.
        """
        if not self.is_loaded or self._net is None or self._torch_device is None:
            raise RuntimeError(
                "LightASD is not loaded. Call load() before score(), "
                "or use it as a context manager."
            )
        video_path = Path(video.path)
        audio_path = Path(audio.path)
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {video_path}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file does not exist: {audio_path}")

        import cv2
        import numpy as np

        waveform = _load_mono16k(audio_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise LightASDError(f"Could not open video: {video_path}")
        native_fps = float(tracks.fps or video.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if native_fps <= 0:
            native_fps = 25.0

        needed: dict[str, list[tuple[int, tuple[float, float, float, float]]]] = {
            track.track_id: _sample_track_frames(track, native_fps=native_fps)
            for track in tracks.tracks
        }
        by_frame: dict[int, list[tuple[str, tuple[float, float, float, float]]]] = {}
        for track_id, samples in needed.items():
            for native_index, bbox in samples:
                by_frame.setdefault(native_index, []).append((track_id, bbox))
        crops: dict[str, dict[int, Any]] = {track_id: {} for track_id in needed}
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                for track_id, bbox in by_frame.get(frame_index, []):
                    crop = _crop_face(frame, bbox, crop_scale=self.crop_scale)
                    if crop is not None:
                        crops[track_id][frame_index] = crop
                frame_index += 1
        finally:
            cap.release()

        scores: list[ASDFrameScore] = []
        for track in tracks.tracks:
            samples = needed[track.track_id]
            if not samples:
                continue
            visual = []
            times = []
            last_crop = None
            for native_index, _bbox in samples:
                crop = crops[track.track_id].get(native_index, last_crop)
                if crop is None:
                    continue
                last_crop = crop
                visual.append(crop)
                times.append(native_index / native_fps)
            if len(visual) < 2:
                continue
            visual_arr = np.stack(visual, axis=0)
            start_s = times[0]
            end_s = times[-1] + (1.0 / ASD_FPS)
            audio_slice = _slice_waveform(waveform, start_s, end_s)
            try:
                mfcc = _mfcc16k(audio_slice)
            except Exception as exc:
                raise LightASDError(
                    f"MFCC extraction failed for track {track.track_id}: {exc}"
                ) from exc
            track_scores = self._infer_track(mfcc, visual_arr)
            count = min(len(track_scores), len(times))
            smoothed = _smooth_scores(track_scores[:count])
            for time_s, score in zip(times[:count], smoothed):
                scores.append(
                    ASDFrameScore(
                        time_s=float(time_s),
                        track_id=track.track_id,
                        score=float(score),
                        speaking=bool(score >= self.active_threshold),
                    )
                )

        return ASDResult(
            video_id=video.source_id,
            scores=tuple(scores),
            active_threshold=self.active_threshold,
            model=DiarizationModelInfo(
                backend="light-asd",
                model_id=DEFAULT_WEIGHT_NAME,
            ),
        )

    def _infer_track(self, mfcc: Any, visual: Any) -> list[float]:
        import numpy as np

        length = min((mfcc.shape[0] - mfcc.shape[0] % 4) / 100.0, visual.shape[0] / ASD_FPS)
        if length <= 0:
            return []
        audio_feat = mfcc[: int(round(length * 100)), :]
        video_feat = visual[: int(round(length * ASD_FPS)), :, :]
        duration_scores: list[list[float]] = []
        for duration in self.duration_set:
            usable = min(duration, max(1, int(math.ceil(length))))
            batch_size = int(math.ceil(length / usable))
            scores: list[float] = []
            with torch.no_grad():
                for index in range(batch_size):
                    audio_chunk = audio_feat[
                        index * usable * 100 : (index + 1) * usable * 100, :
                    ]
                    video_chunk = video_feat[
                        index * usable * ASD_FPS : (index + 1) * usable * ASD_FPS, :, :
                    ]
                    if audio_chunk.shape[0] < 4 or video_chunk.shape[0] < 1:
                        continue
                    input_a = (
                        torch.from_numpy(audio_chunk.astype("float32"))
                        .unsqueeze(0)
                        .to(self._torch_device)
                    )
                    input_v = (
                        torch.from_numpy(video_chunk.astype("float32"))
                        .unsqueeze(0)
                        .to(self._torch_device)
                    )
                    embed_a = self._net.model.forward_audio_frontend(input_a)
                    embed_v = self._net.model.forward_visual_frontend(input_v)
                    out = self._net.model.forward_audio_visual_backend(embed_a, embed_v)
                    chunk_scores = self._net.loss_av.forward(out, labels=None)
                    scores.extend(float(value) for value in np.asarray(chunk_scores).ravel())
            if scores:
                duration_scores.append(scores)
        if not duration_scores:
            return []
        min_len = min(len(item) for item in duration_scores)
        stacked = np.stack([item[:min_len] for item in duration_scores], axis=0)
        return [float(value) for value in np.mean(stacked, axis=0)]

    def _ensure_weights(self) -> Path:
        path = self.weights_path
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path
        repo = self.repo_dir
        candidate = repo / "weight" / DEFAULT_WEIGHT_NAME
        if candidate.is_file() and candidate.stat().st_size > 1_000_000:
            return candidate
        _clone_light_asd(repo)
        if candidate.is_file() and candidate.stat().st_size > 1_000_000:
            return candidate
        raise LightASDError(
            f"Light-ASD weights not found at {path} or {candidate}. "
            f"Clone {LIGHT_ASD_GIT_URL} into {repo} (git-lfs may be required)."
        )


def _clone_light_asd(root: Path) -> None:
    if (root / "weight").is_dir():
        return
    if shutil.which("git") is None:
        raise LightASDError(
            f"Light-ASD checkout not found at {root} and git is not available "
            f"to clone {LIGHT_ASD_GIT_URL}."
        )
    root.parent.mkdir(parents=True, exist_ok=True)
    if root.exists():
        shutil.rmtree(root)
    logger.info("Cloning Light-ASD into %s", root)
    completed = subprocess.run(
        ["git", "clone", "--depth", "1", LIGHT_ASD_GIT_URL, str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise LightASDError(
            f"git clone of Light-ASD failed: {detail[:1000] or 'No error output'}"
        )


def _load_compatible_state(net: nn.Module, state: dict[str, Any]) -> None:
    self_state = net.state_dict()
    loaded = 0
    for name, param in state.items():
        key = name
        if key not in self_state and key.startswith("lossAV."):
            key = key.replace("lossAV.", "loss_av.", 1)
        if key not in self_state:
            key = name.replace("module.", "")
        if key not in self_state:
            continue
        if self_state[key].shape != param.shape:
            continue
        self_state[key].copy_(param)
        loaded += 1
    if loaded == 0:
        raise LightASDError("Checkpoint did not match the Light-ASD architecture")
    net.load_state_dict(self_state)


def _load_mono16k(path: Path) -> Any:
    import librosa
    import numpy as np

    waveform, _sample_rate = librosa.load(str(path), sr=ASD_SAMPLE_RATE, mono=True)
    return np.clip(waveform * 32768.0, -32768, 32767).astype(np.int16)


def _slice_waveform(waveform: Any, start_s: float, end_s: float) -> Any:
    start = max(0, int(round(start_s * ASD_SAMPLE_RATE)))
    end = min(len(waveform), int(round(end_s * ASD_SAMPLE_RATE)))
    if end <= start:
        return waveform[:0]
    return waveform[start:end]


def _mfcc16k(waveform: Any) -> Any:
    from python_speech_features import mfcc

    if waveform.size < ASD_SAMPLE_RATE // 10:
        raise LightASDError("audio slice is too short for MFCC")
    return mfcc(waveform, ASD_SAMPLE_RATE, numcep=13, winlen=0.025, winstep=0.010)


def _sample_track_frames(
    track: FaceTrack,
    *,
    native_fps: float,
) -> list[tuple[int, tuple[float, float, float, float]]]:
    start_s = track.start_s
    end_s = track.end_s
    if end_s <= start_s:
        return []
    samples = []
    index = 0
    while True:
        time_s = start_s + index / ASD_FPS
        if time_s > end_s + 1e-6:
            break
        samples.append((max(0, int(round(time_s * native_fps))), _bbox_at(track, time_s)))
        index += 1
    return samples


def _bbox_at(track: FaceTrack, time_s: float) -> tuple[float, float, float, float]:
    observations = track.observations
    if time_s <= observations[0].time_s:
        return observations[0].bbox_xyxy
    if time_s >= observations[-1].time_s:
        return observations[-1].bbox_xyxy
    for left, right in zip(observations, observations[1:]):
        if left.time_s <= time_s <= right.time_s:
            span = right.time_s - left.time_s
            t = 0.0 if span <= 0 else (time_s - left.time_s) / span
            return tuple(
                left.bbox_xyxy[i] + t * (right.bbox_xyxy[i] - left.bbox_xyxy[i])
                for i in range(4)
            )
    return observations[-1].bbox_xyxy


def _crop_face(
    frame: Any,
    bbox: tuple[float, float, float, float],
    *,
    crop_scale: float,
) -> Any | None:
    import cv2
    import numpy as np

    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bs = max(x2 - x1, y2 - y1) / 2.0
    if bs < 1:
        return None
    pad = int(bs * (1 + 2 * crop_scale))
    padded = np.pad(
        frame,
        ((pad, pad), (pad, pad), (0, 0)),
        mode="constant",
        constant_values=110,
    )
    my = cy + pad
    mx = cx + pad
    y_a = int(my - bs)
    y_b = int(my + bs * (1 + 2 * crop_scale))
    x_a = int(mx - bs * (1 + crop_scale))
    x_b = int(mx + bs * (1 + crop_scale))
    face = padded[max(0, y_a) : y_b, max(0, x_a) : x_b]
    if face.size == 0:
        return None
    face = cv2.resize(face, (224, 224))
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    return gray[56:168, 56:168]


def _smooth_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    smoothed = []
    for index, _score in enumerate(scores):
        window = scores[max(0, index - 2) : min(len(scores), index + 3)]
        smoothed.append(sum(window) / len(window))
    return smoothed


class _AudioBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        self.m_3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False
        )
        self.bn_m_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_3 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False
        )
        self.bn_t_3 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.m_5 = nn.Conv2d(
            in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0), bias=False
        )
        self.bn_m_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.t_5 = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2), bias=False
        )
        self.bn_t_5 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)
        self.last = nn.Conv2d(
            out_channels, out_channels, kernel_size=(1, 1), padding=(0, 0), bias=False
        )
        self.bn_last = nn.BatchNorm2d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_3 = self.relu(self.bn_m_3(self.m_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))
        x_5 = self.relu(self.bn_m_5(self.m_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))
        return self.relu(self.bn_last(self.last(x_3 + x_5)))


class _VisualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, is_down: bool = False) -> None:
        super().__init__()
        self.relu = nn.ReLU()
        stride = (1, 2, 2) if is_down else (1, 1, 1)
        self.s_3 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 3, 3),
            stride=stride,
            padding=(0, 1, 1),
            bias=False,
        )
        self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.t_3 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0),
            bias=False,
        )
        self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.s_5 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 5, 5),
            stride=stride,
            padding=(0, 2, 2),
            bias=False,
        )
        self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.t_5 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(5, 1, 1),
            padding=(2, 0, 0),
            bias=False,
        )
        self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)
        self.last = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=(1, 1, 1),
            padding=(0, 0, 0),
            bias=False,
        )
        self.bn_last = nn.BatchNorm3d(out_channels, momentum=0.01, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_3 = self.relu(self.bn_s_3(self.s_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))
        x_5 = self.relu(self.bn_s_5(self.s_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))
        return self.relu(self.bn_last(self.last(x_3 + x_5)))


class _VisualEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = _VisualBlock(1, 32, is_down=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block2 = _VisualBlock(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.block3 = _VisualBlock(64, 128)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = x.transpose(1, 2)
        batch, frames, channels, width, height = x.shape
        x = x.reshape(batch * frames, channels, width, height)
        x = self.maxpool(x)
        return x.view(batch, frames, channels)


class _AudioEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block1 = _AudioBlock(1, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.block2 = _AudioBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))
        self.block3 = _AudioBlock(64, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.block1(x))
        x = self.pool2(self.block2(x))
        x = self.block3(x)
        x = torch.mean(x, dim=2, keepdim=True)
        return x.squeeze(2).transpose(1, 2)


class _BGRU(nn.Module):
    def __init__(self, channel: int) -> None:
        super().__init__()
        self.gru_forward = nn.GRU(
            input_size=channel,
            hidden_size=channel,
            num_layers=1,
            bidirectional=False,
            bias=True,
            batch_first=True,
        )
        self.gru_backward = nn.GRU(
            input_size=channel,
            hidden_size=channel,
            num_layers=1,
            bidirectional=False,
            bias=True,
            batch_first=True,
        )
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gru_forward(x)
        x = self.gelu(x)
        x = torch.flip(x, dims=[1])
        x, _ = self.gru_backward(x)
        x = torch.flip(x, dims=[1])
        return self.gelu(x)


class _ASDModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.visualEncoder = _VisualEncoder()
        self.audioEncoder = _AudioEncoder()
        self.GRU = _BGRU(128)

    def forward_visual_frontend(self, x: torch.Tensor) -> torch.Tensor:
        _batch, frames, width, height = x.shape
        x = x.view(_batch, 1, frames, width, height)
        x = (x / 255 - 0.4161) / 0.1688
        return self.visualEncoder(x)

    def forward_audio_frontend(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1).transpose(2, 3)
        return self.audioEncoder(x)

    def forward_audio_visual_backend(
        self, audio_embed: torch.Tensor, visual_embed: torch.Tensor
    ) -> torch.Tensor:
        x = self.GRU(audio_embed + visual_embed)
        return torch.reshape(x, (-1, 128))

    def forward_visual_backend(self, x: torch.Tensor) -> torch.Tensor:
        return torch.reshape(x, (-1, 128))


class _LossAV(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCELoss()
        self.FC = nn.Linear(128, 2)

    def forward(self, x: torch.Tensor, labels: torch.Tensor | None = None, r: float = 1):
        x = x.squeeze(1)
        x = self.FC(x)
        if labels is None:
            pred = x[:, 1].reshape(-1).detach().cpu().numpy()
            return pred
        x1 = F.softmax(x / r, dim=-1)[:, 1]
        nloss = self.criterion(x1, labels.float())
        pred_score = F.softmax(x, dim=-1)
        pred_label = torch.round(F.softmax(x, dim=-1))[:, 1]
        correct = (pred_label == labels).sum().float()
        return nloss, pred_score, pred_label, correct


class _LightASDNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _ASDModel()
        self.lossAV = _LossAV()
        self.loss_av = self.lossAV
