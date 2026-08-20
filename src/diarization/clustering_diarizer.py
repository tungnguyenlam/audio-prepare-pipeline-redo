"""Offline SOTA Clustering Speaker Diarizer (SpeechBrain ECAPA-TDNN & Spectral / Spherical Clustering).

Zero-token offline diarization fallback:
1. Fast Energy & Spectral Voice Activity Detection (VAD).
2. Uniform contiguous non-overlapping speech chunking (prevents stuttering and turn collision).
3. SpeechBrain ECAPA-TDNN 192-d embedding extraction with L2 normalization.
4. Spectral Clustering / Spherical KMeans (Normalized Cut) to isolate balanced speaker groups.
5. Centroid Refinement & 1D Median Filtering to eliminate rapid speaker jitter.
6. Clean turn aggregation and duration metrics.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import scipy.signal
import soundfile as sf
import torch

from src.diarization.base import BaseDiarizer, SpeakerTurn, refine_and_merge_turns

logger = logging.getLogger(__name__)

# Configure local project HF & Torch cache directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCAL_HF_CACHE = PROJECT_ROOT / ".cache" / "huggingface"
LOCAL_HF_CACHE.mkdir(parents=True, exist_ok=True)
TORCH_CACHE = PROJECT_ROOT / ".cache" / "torch"
TORCH_CACHE.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(LOCAL_HF_CACHE)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(LOCAL_HF_CACHE / "hub")
os.environ["HF_HUB_CACHE"] = str(LOCAL_HF_CACHE / "hub")
os.environ["TORCH_HOME"] = str(TORCH_CACHE)
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


class OfflineClusteringDiarizer(BaseDiarizer):
    """SpeechBrain ECAPA-TDNN Offline Spectral / Spherical Clustering Speaker Diarizer."""

    def __init__(self, device: str = "cuda"):
        self.device = device if (torch.cuda.is_available() and device == "cuda") else "cpu"
        self._classifier = None

    def _init_model(self):
        """Lazy load ECAPA-TDNN model with local project cache."""
        if self._classifier is None:
            from speechbrain.inference.speaker import EncoderClassifier

            local_cache = LOCAL_HF_CACHE / "speechbrain_ecapa"
            local_cache.mkdir(parents=True, exist_ok=True)

            logger.info("Initializing SpeechBrain ECAPA-TDNN on %s...", self.device)
            # Use 'cuda:0' or 'cpu' to avoid torch unpack errors
            run_device = "cuda:0" if self.device.startswith("cuda") else "cpu"
            self._classifier = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir=str(local_cache),
                run_opts={"device": run_device}
            )
        return self._classifier

    def _vad_segments(
        self,
        audio_data: np.ndarray,
        sr: int,
        frame_ms: float = 30.0,
        energy_threshold_db: float = -42.0,
        min_speech_ms: float = 400.0,
        min_silence_ms: float = 300.0,
    ) -> List[tuple[float, float]]:
        """Fast energy & spectral voice activity detection (VAD)."""
        frame_len = int(sr * (frame_ms / 1000.0))
        hop_len = frame_len // 2
        num_frames = (len(audio_data) - frame_len) // hop_len + 1

        if num_frames <= 0:
            return [(0.0, round(len(audio_data) / sr, 2))]

        # Calculate frame RMS energy in dB
        energies = []
        for i in range(num_frames):
            start = i * hop_len
            frame = audio_data[start:start + frame_len]
            rms = np.sqrt(np.mean(frame ** 2) + 1e-12)
            db = 20.0 * np.log10(rms + 1e-12)
            energies.append(db)

        energies = np.array(energies)
        max_db = np.max(energies) if len(energies) > 0 else -100.0
        dyn_thresh = max(energy_threshold_db, max_db - 35.0)
        is_speech = energies > dyn_thresh

        # Morphological smoothing (closing short pauses)
        min_sil_frames = max(1, int((min_silence_ms / 1000.0) / (hop_len / sr)))
        gap_count = 0
        for i in range(len(is_speech)):
            if not is_speech[i]:
                gap_count += 1
            else:
                if 0 < gap_count <= min_sil_frames:
                    is_speech[i - gap_count:i] = True
                gap_count = 0

        # Extract continuous speech chunks
        segments = []
        in_segment = False
        seg_start = 0

        for i, val in enumerate(is_speech):
            if val and not in_segment:
                in_segment = True
                seg_start = i * hop_len
            elif not val and in_segment:
                in_segment = False
                seg_end = i * hop_len + frame_len
                dur_ms = ((seg_end - seg_start) / sr) * 1000.0
                if dur_ms >= min_speech_ms:
                    segments.append((round(seg_start / sr, 2), round(min(len(audio_data) / sr, seg_end / sr), 2)))

        if in_segment:
            seg_end = len(audio_data)
            dur_ms = ((seg_end - seg_start) / sr) * 1000.0
            if dur_ms >= min_speech_ms:
                segments.append((round(seg_start / sr, 2), round(seg_end / sr, 2)))

        return segments if segments else [(0.0, round(len(audio_data) / sr, 2))]

    def diarize(
        self,
        audio_path: Union[str, Path],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        filter_overlap: bool = True,
        min_duration_s: float = 0.5,
    ) -> List[SpeakerTurn]:
        """Perform offline speaker diarization with Spectral Clustering & ECAPA-TDNN."""
        from sklearn.cluster import KMeans, SpectralClustering
        from sklearn.metrics import silhouette_score

        audio_path = Path(audio_path).resolve()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        info = sf.info(str(audio_path))
        orig_sr = info.samplerate
        data, _ = sf.read(str(audio_path), dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Resample to 16kHz for ECAPA-TDNN
        target_sr = 16000
        if orig_sr != target_sr:
            data_16k = scipy.signal.resample(data, int(len(data) * target_sr / orig_sr))
        else:
            data_16k = data

        total_duration = len(data_16k) / target_sr
        classifier = self._init_model()

        # Step 1: Extract speech segments via VAD
        vad_chunks = self._vad_segments(data_16k, target_sr)
        logger.info("VAD detected %d speech segments in %s", len(vad_chunks), audio_path.name)

        if not vad_chunks:
            return [SpeakerTurn(start_s=0.0, end_s=total_duration, speaker_id="SPEAKER_00")]

        # Step 2: Slice speech segments into contiguous, non-overlapping chunks (2.0s - 3.0s)
        sub_chunks: List[tuple[float, float]] = []
        max_subchunk_s = 3.0
        for start_s, end_s in vad_chunks:
            dur = end_s - start_s
            if dur <= max_subchunk_s:
                sub_chunks.append((start_s, end_s))
            else:
                n_splits = int(np.ceil(dur / 2.5))
                split_dur = dur / n_splits
                cur = start_s
                for _ in range(n_splits):
                    nxt = min(end_s, cur + split_dur)
                    if (nxt - cur) >= 0.4:
                        sub_chunks.append((round(cur, 2), round(nxt, 2)))
                    cur = nxt

        # Step 3: Extract ECAPA-TDNN Embeddings
        embeddings = []
        valid_chunks: List[tuple[float, float]] = []
        run_device = "cuda:0" if self.device.startswith("cuda") else "cpu"

        for start_s, end_s in sub_chunks:
            start_idx = int(start_s * target_sr)
            end_idx = int(end_s * target_sr)
            chunk_data = data_16k[start_idx:end_idx]

            if len(chunk_data) < int(0.3 * target_sr):
                continue

            tensor_chunk = torch.from_numpy(chunk_data.astype(np.float32)).unsqueeze(0).to(run_device)
            with torch.no_grad():
                emb = classifier.encode_batch(tensor_chunk).squeeze().cpu().detach().numpy()
            norm = np.linalg.norm(emb) + 1e-8
            embeddings.append(emb / norm)
            valid_chunks.append((start_s, end_s))

        if not embeddings:
            return [SpeakerTurn(start_s=0.0, end_s=total_duration, speaker_id="SPEAKER_00")]

        embeddings_np = np.array(embeddings)
        n_samples = len(embeddings_np)

        # Step 4: Spectral / KMeans Clustering
        if num_speakers is not None and num_speakers > 0:
            k = min(int(num_speakers), n_samples)
            if k == 1 or n_samples <= 1:
                labels = np.zeros(n_samples, dtype=int)
            else:
                n_neighbors = min(20, max(5, n_samples // 4))
                try:
                    spectral = SpectralClustering(
                        n_clusters=k,
                        affinity="nearest_neighbors",
                        n_neighbors=n_neighbors,
                        assign_labels="kmeans",
                        random_state=42
                    )
                    labels = spectral.fit_predict(embeddings_np)
                except Exception:
                    kmeans = KMeans(n_clusters=k, n_init=20, random_state=42)
                    labels = kmeans.fit_predict(embeddings_np)
        else:
            # Auto estimate optimal k using Silhouette Score on Spherical KMeans / Spectral
            if n_samples <= 2:
                labels = np.zeros(n_samples, dtype=int)
            else:
                min_k = max(2, min_speakers) if min_speakers else 2
                max_k = min(n_samples - 1, max_speakers if max_speakers else 8)

                best_k = 1
                best_score = -1.0
                best_labels = np.zeros(n_samples, dtype=int)

                if min_k <= max_k:
                    for cand_k in range(min_k, max_k + 1):
                        try:
                            kmeans = KMeans(n_clusters=cand_k, n_init=15, random_state=42)
                            cand_labels = kmeans.fit_predict(embeddings_np)
                            score = silhouette_score(embeddings_np, cand_labels, metric="cosine")
                            if score > best_score:
                                best_score = score
                                best_k = cand_k
                                best_labels = cand_labels
                        except Exception:
                            continue

                if best_score >= 0.10:
                    labels = best_labels
                    logger.info("Auto clustering selected %d speakers with Silhouette score %.3f", best_k, best_score)
                else:
                    logger.info("Low silhouette score (%.3f); using fallback 2-speaker KMeans", best_score)
                    kmeans = KMeans(n_clusters=2, n_init=20, random_state=42)
                    labels = kmeans.fit_predict(embeddings_np)

        # Step 5: Centroid Refinement & Temporal Smoothing (Median Filter)
        unique_labels = sorted(list(set(labels)))
        num_clusters = len(unique_labels)

        if num_clusters > 1:
            # Compute normalized centroid vector for each cluster
            centroids = np.zeros((num_clusters, embeddings_np.shape[1]), dtype=np.float32)
            for idx, lbl in enumerate(unique_labels):
                mask = (labels == lbl)
                c = np.mean(embeddings_np[mask], axis=0)
                centroids[idx] = c / (np.linalg.norm(c) + 1e-8)

            # Re-assign segments based on highest cosine similarity with cluster centroids
            sim_matrix = np.dot(embeddings_np, centroids.T)  # (N, num_clusters)
            reassigned_indices = np.argmax(sim_matrix, axis=1)
            reassigned_labels = np.array([unique_labels[i] for i in reassigned_indices])

            # Apply running 1D median filter (kernel size 3) to smooth rapid speaker jitter
            filtered_labels = reassigned_labels.copy()
            for i in range(1, len(filtered_labels) - 1):
                prev_lbl = filtered_labels[i - 1]
                next_lbl = filtered_labels[i + 1]
                if prev_lbl == next_lbl and filtered_labels[i] != prev_lbl:
                    filtered_labels[i] = prev_lbl
            labels = filtered_labels

        # Step 6: Merge consecutive contiguous segments of the same speaker & trim boundary collar
        initial_turns: List[SpeakerTurn] = []
        for (start_s, end_s), label in zip(valid_chunks, labels):
            spk_name = f"SPEAKER_{label:02d}"
            initial_turns.append(SpeakerTurn(
                start_s=start_s,
                end_s=end_s,
                speaker_id=spk_name,
                is_overlap=False
            ))

        final_turns = refine_and_merge_turns(
            turns=initial_turns,
            max_merge_gap_s=1.0,
            boundary_collar_s=0.08,
            min_duration_s=min_duration_s,
        )
        return final_turns if final_turns else initial_turns
