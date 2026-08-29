"""InsightFace SCRFD detection, ArcFace embeddings, and face tracking."""

from __future__ import annotations

import gc
import io
import logging
import math
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from src.base.model import ManagedModel
from src.data_paths import DATA_DIR
from src.diarization.schemas import DiarizationModelInfo
from src.visual.Video import Video, VideoError
from src.visual.schemas import FaceObservation, FaceTrack, FaceTrackSet

logger = logging.getLogger(__name__)

BUFFALO_L_URL = (
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
)
DEFAULT_DET_MODEL = "det_10g.onnx"
DEFAULT_REC_MODEL = "w600k_r50.onnx"
ARCFACE_DST = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


class FaceAnalyzerError(RuntimeError):
    """Raised when face models cannot be loaded or a video cannot be analyzed."""


class FaceAnalyzer(ManagedModel):
    """Detect, track, and embed faces in a file-backed video.

    Uses InsightFace buffalo_l ONNX weights (SCRFD-10G + ArcFace ResNet50)
    downloaded into ``.data/insightface/models/buffalo_l`` on first ``load()``.
    Tracking is IoU linking with ArcFace re-identification across gaps and
    shot changes. Face and voice embeddings are never compared directly.

    Example::

        analyzer = FaceAnalyzer(device="cuda")
        with analyzer:
            tracks = analyzer.analyze(video)
    """

    def __init__(
        self,
        *,
        device: str = "auto",
        models_dir: str | Path | None = None,
        det_size: tuple[int, int] = (640, 640),
        det_threshold: float = 0.5,
        detection_fps: float = 12.5,
        iou_threshold: float = 0.4,
        max_lost_s: float = 0.4,
        min_track_s: float = 0.4,
        reid_threshold: float = 0.45,
        embed_top_k: int = 8,
    ) -> None:
        """Initialize the analyzer.

        Args:
            device: ``"auto"``, ``"cuda"``, or ``"cpu"``.
            models_dir: Directory containing buffalo_l ONNX files. Defaults to
                ``.data/insightface/models/buffalo_l``.
            det_size: SCRFD square input size.
            det_threshold: Minimum detection score.
            detection_fps: Face-detection sampling rate. Bounding boxes are
                interpolated onto every video frame between detections.
            iou_threshold: Minimum IoU to continue a track to the next
                detection.
            max_lost_s: Maximum gap without a detection before a track ends.
            min_track_s: Drop tracks shorter than this duration.
            reid_threshold: Cosine similarity for merging tracks of the same
                person across cuts. Temporally overlapping tracks are never
                merged.
            embed_top_k: Highest-scoring detections averaged into a track
                centroid.
        """
        ManagedModel.__init__(self)
        if detection_fps <= 0:
            raise ValueError("detection_fps must be positive")
        if not 0 < iou_threshold <= 1:
            raise ValueError("iou_threshold must be in (0, 1]")
        if max_lost_s < 0 or min_track_s < 0:
            raise ValueError("max_lost_s and min_track_s must be non-negative")
        if not -1 <= reid_threshold <= 1:
            raise ValueError("reid_threshold must be between -1 and 1")
        if embed_top_k < 1:
            raise ValueError("embed_top_k must be at least 1")
        self.device = str(device)
        self.models_dir = (
            Path(models_dir).expanduser()
            if models_dir is not None
            else DATA_DIR / "insightface" / "models" / "buffalo_l"
        )
        self.det_size = (int(det_size[0]), int(det_size[1]))
        self.det_threshold = float(det_threshold)
        self.detection_fps = float(detection_fps)
        self.iou_threshold = float(iou_threshold)
        self.max_lost_s = float(max_lost_s)
        self.min_track_s = float(min_track_s)
        self.reid_threshold = float(reid_threshold)
        self.embed_top_k = int(embed_top_k)
        self._detector: Any | None = None
        self._recognizer: Any | None = None

    def _load(self) -> None:
        """Download buffalo_l if needed and open ONNX sessions."""
        self._ensure_models()
        import onnxruntime as ort

        providers = _ort_providers(self.device)
        det_path = self.models_dir / DEFAULT_DET_MODEL
        rec_path = self.models_dir / DEFAULT_REC_MODEL
        if not det_path.is_file() or not rec_path.is_file():
            raise FaceAnalyzerError(
                f"buffalo_l models missing under {self.models_dir}. "
                f"Expected {DEFAULT_DET_MODEL} and {DEFAULT_REC_MODEL}."
            )
        self._detector = _SCRFD(
            det_path,
            session=ort.InferenceSession(str(det_path), providers=providers),
            det_thresh=self.det_threshold,
            input_size=self.det_size,
        )
        self._recognizer = _ArcFace(
            rec_path,
            session=ort.InferenceSession(str(rec_path), providers=providers),
        )

    def _unload(self) -> None:
        """Release ONNX sessions."""
        self._detector = None
        self._recognizer = None
        gc.collect()

    def analyze(self, video: Video) -> FaceTrackSet:
        """Detect, track, and embed faces in ``video``.

        Args:
            video: File-backed video.

        Returns:
            Tracks with interpolated observations and ArcFace centroids.

        Raises:
            RuntimeError: If the analyzer is not loaded.
            FileNotFoundError: If the video file is missing.
            FaceAnalyzerError: If the video cannot be read.
        """
        self._require_loaded()
        path = Path(video.path)
        if not path.is_file():
            raise FileNotFoundError(f"Video file does not exist: {path}")

        import cv2
        import numpy as np

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FaceAnalyzerError(f"Could not open video: {path}")

        fps = float(video.fps or cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 0:
            fps = 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = float(
            video.duration_s
            if video.duration_s is not None
            else (frame_count / fps if frame_count > 0 else 0.0)
        )
        step = max(1, int(round(fps / self.detection_fps)))
        detections: dict[int, list[dict[str, Any]]] = {}
        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_index % step == 0:
                    bboxes, keypoints = self._detector.detect(frame)
                    faces: list[dict[str, Any]] = []
                    for index, bbox in enumerate(bboxes):
                        kps = (
                            keypoints[index]
                            if keypoints is not None and index < len(keypoints)
                            else None
                        )
                        embedding = None
                        if kps is not None:
                            embedding = self._recognizer.embed(frame, kps)
                        faces.append(
                            {
                                "bbox": tuple(float(value) for value in bbox[:4]),
                                "det_score": float(bbox[4]),
                                "landmarks": (
                                    tuple(tuple(float(v) for v in point) for point in kps)
                                    if kps is not None
                                    else None
                                ),
                                "embedding": embedding,
                            }
                        )
                    detections[frame_index] = faces
                frame_index += 1
        finally:
            cap.release()

        if frame_count <= 0:
            frame_count = frame_index
        if duration_s <= 0 and fps > 0:
            duration_s = frame_count / fps

        raw_tracks = _link_detections(
            detections,
            fps=fps,
            iou_threshold=self.iou_threshold,
            max_lost_frames=max(1, int(round(self.max_lost_s * fps))),
        )
        tracks = [
            self._finish_track(track_id, raw, fps)
            for track_id, raw in enumerate(raw_tracks)
        ]
        tracks = [track for track in tracks if track.duration_s >= self.min_track_s]
        tracks = _reid_merge(tracks, threshold=self.reid_threshold)
        return FaceTrackSet(
            video_id=video.source_id,
            tracks=tuple(tracks),
            fps=fps,
            duration_s=duration_s,
            model=DiarizationModelInfo(
                backend="insightface-buffalo_l",
                model_id="buffalo_l",
            ),
            video_path=str(path),
        )

    def embed_images(self, paths: list[Path]) -> Any:
        """Return the L2-normalized mean ArcFace embedding of face images.

        Each image should contain a clearly visible target face. The largest
        detection is used.

        Args:
            paths: Face image files (JPEG/PNG).

        Returns:
            A 1-D numpy float64 vector.

        Raises:
            RuntimeError: If the analyzer is not loaded.
            FaceAnalyzerError: If no face can be embedded.
        """
        self._require_loaded()
        import cv2
        import numpy as np

        vectors = []
        failures: list[str] = []
        for path in paths:
            image_path = Path(path)
            if not image_path.is_file():
                failures.append(f"{image_path.name}: missing file")
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                failures.append(f"{image_path.name}: unreadable")
                continue
            bboxes, keypoints = self._detector.detect(image)
            if bboxes.shape[0] == 0 or keypoints is None:
                failures.append(f"{image_path.name}: no face")
                continue
            areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
            best = int(np.argmax(areas))
            vector = self._recognizer.embed(image, keypoints[best])
            if vector is None:
                failures.append(f"{image_path.name}: embed failed")
                continue
            vectors.append(vector)
        if not vectors:
            detail = "; ".join(failures) or "no images"
            raise FaceAnalyzerError(f"Could not embed any face images. {detail}")
        centroid = np.mean(np.stack(vectors), axis=0)
        norm = float(np.linalg.norm(centroid))
        if not math.isfinite(norm) or norm == 0:
            raise FaceAnalyzerError("Face enrollment produced a zero centroid")
        return centroid / norm

    def _finish_track(
        self,
        index: int,
        raw: list[dict[str, Any]],
        fps: float,
    ) -> FaceTrack:
        import numpy as np

        observations = []
        scored_embeddings: list[tuple[float, Any]] = []
        for item in raw:
            frame_index = int(item["frame_index"])
            observations.append(
                FaceObservation(
                    frame_index=frame_index,
                    time_s=frame_index / fps,
                    bbox_xyxy=tuple(item["bbox"]),
                    det_score=float(item["det_score"]),
                    landmarks=item.get("landmarks"),
                )
            )
            if item.get("embedding") is not None:
                scored_embeddings.append((float(item["det_score"]), item["embedding"]))
        scored_embeddings.sort(key=lambda pair: pair[0], reverse=True)
        chosen = [vector for _, vector in scored_embeddings[: self.embed_top_k]]
        embedding = None
        if chosen:
            centroid = np.mean(np.stack(chosen), axis=0)
            norm = float(np.linalg.norm(centroid))
            if math.isfinite(norm) and norm > 0:
                embedding = tuple(float(value) for value in (centroid / norm).tolist())
        return FaceTrack(
            track_id=f"face_{index:03d}",
            observations=tuple(observations),
            embedding=embedding,
        )

    def _ensure_models(self) -> None:
        det_path = self.models_dir / DEFAULT_DET_MODEL
        rec_path = self.models_dir / DEFAULT_REC_MODEL
        if det_path.is_file() and rec_path.is_file():
            return
        self.models_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading InsightFace buffalo_l into %s", self.models_dir)
        try:
            with urllib.request.urlopen(BUFFALO_L_URL, timeout=120) as response:
                payload = response.read()
        except Exception as exc:
            raise FaceAnalyzerError(
                f"Failed to download buffalo_l from {BUFFALO_L_URL}. "
                f"Place {DEFAULT_DET_MODEL} and {DEFAULT_REC_MODEL} in "
                f"{self.models_dir}."
            ) from exc
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(self.models_dir)
        # Some zips nest files in a buffalo_l/ folder.
        if not det_path.is_file():
            nested = list(self.models_dir.rglob(DEFAULT_DET_MODEL))
            rec_nested = list(self.models_dir.rglob(DEFAULT_REC_MODEL))
            if nested:
                nested[0].replace(det_path)
            if rec_nested:
                rec_nested[0].replace(rec_path)
        if not det_path.is_file() or not rec_path.is_file():
            raise FaceAnalyzerError(
                f"buffalo_l download did not contain the expected ONNX files in "
                f"{self.models_dir}"
            )

    def _require_loaded(self) -> None:
        if not self.is_loaded or self._detector is None or self._recognizer is None:
            raise RuntimeError(
                "FaceAnalyzer is not loaded. Call load() before analyze(), "
                "or use it as a context manager."
            )


def _ort_providers(device: str) -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers: list[str] = []
    want_cuda = device == "cuda" or (
        device == "auto" and "CUDAExecutionProvider" in available
    )
    if want_cuda and "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _iou(box_a: tuple[float, ...], box_b: tuple[float, ...]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _link_detections(
    detections: dict[int, list[dict[str, Any]]],
    *,
    fps: float,
    iou_threshold: float,
    max_lost_frames: int,
) -> list[list[dict[str, Any]]]:
    frames = sorted(detections)
    active: list[list[dict[str, Any]]] = []
    finished: list[list[dict[str, Any]]] = []
    for frame_index in frames:
        unused = list(detections[frame_index])
        matched: set[int] = set()
        for track in active:
            last = track[-1]
            if frame_index - int(last["frame_index"]) > max_lost_frames:
                continue
            best_i = -1
            best_iou = iou_threshold
            for index, face in enumerate(unused):
                if index in matched:
                    continue
                overlap = _iou(last["bbox"], face["bbox"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_i = index
            if best_i >= 0:
                chosen = dict(unused[best_i])
                chosen["frame_index"] = frame_index
                track.append(chosen)
                matched.add(best_i)
        still_active = []
        for track in active:
            last_frame = int(track[-1]["frame_index"])
            if frame_index - last_frame > max_lost_frames:
                finished.append(track)
            else:
                still_active.append(track)
        active = still_active
        for index, face in enumerate(unused):
            if index in matched:
                continue
            started = dict(face)
            started["frame_index"] = frame_index
            active.append([started])
    finished.extend(active)
    return finished


def _reid_merge(tracks: list[FaceTrack], *, threshold: float) -> list[FaceTrack]:
    import numpy as np

    remaining = list(tracks)
    merged: list[FaceTrack] = []
    while remaining:
        current = remaining.pop(0)
        changed = True
        while changed:
            changed = False
            keep: list[FaceTrack] = []
            for other in remaining:
                if current.overlaps(other.start_s, other.end_s):
                    keep.append(other)
                    continue
                if current.embedding is None or other.embedding is None:
                    keep.append(other)
                    continue
                similarity = float(
                    np.dot(np.asarray(current.embedding), np.asarray(other.embedding))
                )
                if similarity < threshold:
                    keep.append(other)
                    continue
                current = _concat_tracks(current, other)
                changed = True
            remaining = keep
        merged.append(current)
    renamed = []
    for index, track in enumerate(merged):
        renamed.append(
            FaceTrack(
                track_id=f"face_{index:03d}",
                observations=track.observations,
                embedding=track.embedding,
            )
        )
    return renamed


def _concat_tracks(left: FaceTrack, right: FaceTrack) -> FaceTrack:
    import numpy as np

    observations = tuple(
        sorted(
            (*left.observations, *right.observations),
            key=lambda item: (item.frame_index, item.time_s),
        )
    )
    embedding = left.embedding
    if left.embedding is not None and right.embedding is not None:
        centroid = np.mean(
            np.stack(
                [np.asarray(left.embedding), np.asarray(right.embedding)]
            ),
            axis=0,
        )
        norm = float(np.linalg.norm(centroid))
        if math.isfinite(norm) and norm > 0:
            embedding = tuple(float(value) for value in (centroid / norm).tolist())
    elif right.embedding is not None:
        embedding = right.embedding
    return FaceTrack(
        track_id=left.track_id,
        observations=observations,
        embedding=embedding,
    )


class _SCRFD:
    """Minimal InsightFace SCRFD runtime for buffalo_l ``det_10g``."""

    def __init__(
        self,
        model_file: Path,
        *,
        session: Any,
        det_thresh: float,
        input_size: tuple[int, int],
    ) -> None:
        self.session = session
        self.det_thresh = det_thresh
        self.input_size = input_size
        self.nms_thresh = 0.4
        self.input_mean = 127.5
        self.input_std = 128.0
        inputs = session.get_inputs()
        self.input_name = inputs[0].name
        self.output_names = [item.name for item in session.get_outputs()]
        outputs = session.get_outputs()
        self.batched = len(outputs[0].shape) == 3
        self.use_kps = len(outputs) in {9, 15}
        if len(outputs) in {6, 9}:
            self.fmc = 3
            self.feat_stride_fpn = [8, 16, 32]
            self.num_anchors = 2
        else:
            self.fmc = 5
            self.feat_stride_fpn = [8, 16, 32, 64, 128]
            self.num_anchors = 1
        self.center_cache: dict[tuple[int, int, int], Any] = {}

    def detect(self, image: Any) -> tuple[Any, Any]:
        import cv2
        import numpy as np

        input_size = self.input_size
        im_ratio = float(image.shape[0]) / float(image.shape[1])
        model_ratio = float(input_size[1]) / float(input_size[0])
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / float(image.shape[0])
        resized = cv2.resize(image, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized
        blob = cv2.dnn.blobFromImage(
            det_img,
            1.0 / self.input_std,
            input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        net_outs = self.session.run(self.output_names, {self.input_name: blob})
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_height = blob.shape[2]
        input_width = blob.shape[3]
        for idx, stride in enumerate(self.feat_stride_fpn):
            if self.batched:
                scores = net_outs[idx][0]
                bbox_preds = net_outs[idx + self.fmc][0] * stride
                kps_preds = (
                    net_outs[idx + self.fmc * 2][0] * stride if self.use_kps else None
                )
            else:
                scores = net_outs[idx]
                bbox_preds = net_outs[idx + self.fmc] * stride
                kps_preds = (
                    net_outs[idx + self.fmc * 2] * stride if self.use_kps else None
                )
            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)
            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self.num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self.num_anchors, axis=1
                    ).reshape((-1, 2))
                self.center_cache[key] = anchor_centers
            scores = scores.reshape(-1)
            pos_inds = np.where(scores >= self.det_thresh)[0]
            bboxes = _distance2bbox(anchor_centers, bbox_preds.reshape(-1, 4))
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            if self.use_kps and kps_preds is not None:
                kpss = _distance2kps(anchor_centers, kps_preds.reshape(-1, 10))
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kpss_list.append(kpss[pos_inds])
        if not scores_list or all(item.size == 0 for item in scores_list):
            empty = np.empty((0, 5), dtype=np.float32)
            kps_empty = np.empty((0, 5, 2), dtype=np.float32) if self.use_kps else None
            return empty, kps_empty
        scores = np.concatenate(scores_list, axis=0)
        bboxes = np.concatenate(bboxes_list, axis=0) / det_scale
        order = scores.argsort()[::-1]
        pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32)
        pre_det = pre_det[order]
        kpss = None
        if self.use_kps and kpss_list:
            kpss = np.concatenate(kpss_list, axis=0) / det_scale
            kpss = kpss[order]
        keep = _nms(pre_det, self.nms_thresh)
        det = pre_det[keep]
        if kpss is not None:
            kpss = kpss[keep]
        return det, kpss


class _ArcFace:
    """ArcFace recognition on 5-point-aligned 112×112 crops."""

    def __init__(self, model_file: Path, *, session: Any) -> None:
        self.session = session
        self.input_mean = 127.5
        self.input_std = 127.5
        inputs = session.get_inputs()
        self.input_name = inputs[0].name
        self.input_size = tuple(inputs[0].shape[2:4][::-1])
        self.output_names = [item.name for item in session.get_outputs()]

    def embed(self, image: Any, landmarks: Any) -> Any | None:
        import cv2
        import numpy as np

        aligned = _norm_crop(image, landmarks, image_size=int(self.input_size[0]))
        if aligned is None:
            return None
        blob = cv2.dnn.blobFromImage(
            aligned,
            1.0 / self.input_std,
            self.input_size,
            (self.input_mean, self.input_mean, self.input_mean),
            swapRB=True,
        )
        vector = self.session.run(self.output_names, {self.input_name: blob})[0]
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm == 0:
            return None
        return vector / norm


def _norm_crop(image: Any, landmarks: Any, *, image_size: int) -> Any | None:
    import cv2
    import numpy as np

    src = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if src.shape[0] < 5:
        return None
    dst = np.asarray(ARCFACE_DST, dtype=np.float32)
    if image_size != 112:
        dst = dst * (image_size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(src[:5], dst, method=cv2.LMEDS)
    if matrix is None:
        return None
    return cv2.warpAffine(image, matrix, (image_size, image_size), borderValue=0.0)


def _distance2bbox(points: Any, distance: Any) -> Any:
    import numpy as np

    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: Any, distance: Any) -> Any:
    import numpy as np

    preds = []
    for index in range(0, distance.shape[1], 2):
        px = points[:, index % 2] + distance[:, index]
        py = points[:, index % 2 + 1] + distance[:, index + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def _nms(dets: Any, thresh: float) -> list[int]:
    import numpy as np

    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        index = int(order[0])
        keep.append(index)
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1.0)
        height = np.maximum(0.0, yy2 - yy1 + 1.0)
        overlap = width * height / (areas[index] + areas[order[1:]] - width * height)
        remaining = np.where(overlap <= thresh)[0]
        order = order[remaining + 1]
    return keep
