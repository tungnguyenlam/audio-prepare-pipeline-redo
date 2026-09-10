"""Align supplied turns using explicit word timings or a selected ASR/alignment engine."""
from __future__ import annotations

import contextlib
import gc
import json
import logging
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Sequence
import numpy as np
import soundfile as sf
from _manifest import arguments, load, save
from _common.files import ROOT, identity, probe, progress, read_json
logger = logging.getLogger(__name__)

def _ensure_torch_hub_trusted() -> None:
    """Ensure torch.hub trusts known repositories non-interactively (e.g. Silero VAD)."""
    try:
        import torch.hub
        if '_TRUSTED_REPO_OWNERS' in torch.hub and isinstance(torch.hub._TRUSTED_REPO_OWNERS, tuple):
            if 'snakers4' not in torch.hub._TRUSTED_REPO_OWNERS:
                torch.hub._TRUSTED_REPO_OWNERS = (*torch.hub._TRUSTED_REPO_OWNERS, 'snakers4')
        hub_dir = torch.hub.get_dir()
        os.makedirs(hub_dir, exist_ok=True)
        trusted_file = os.path.join(hub_dir, 'trusted_list')
        existing: set[str] = set()
        if os.path.exists(trusted_file):
            with open(trusted_file, 'r', encoding='utf-8') as f:
                existing = {line.strip() for line in f if line.strip()}
        needed = {'snakers4_silero-vad', 'snakers4_silero-vad_master', 'snakers4/silero-vad'}
        to_add = [name for name in needed if name not in existing]
        if to_add:
            with open(trusted_file, 'a', encoding='utf-8') as f:
                for name in to_add:
                    f.write(f'{name}\n')
        if not torch.hub.get('_trust_repo_patched', False):
            orig_load = torch.hub.load

            def _patched_hub_load(repo_or_dir, model, *args, **kwargs):
                if kwargs.get('trust_repo') in (None, 'check'):
                    kwargs['trust_repo'] = True
                return orig_load(repo_or_dir, model, *args, **kwargs)
            torch.hub.load = _patched_hub_load
            torch.hub._trust_repo_patched = True
    except Exception as exc:
        logger.debug('Failed to pre-trust torch.hub repositories: %s', exc)

def _lock_turns_with_words(turns: Sequence[dict], words: list[dict[str, Any]], policy: str='whisper_word_lock', audio_duration_s: float | None=None, competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None=None) -> tuple[list[dict], list[dict[str, Any]]]:
    """Keep incoming bounds; reject only when a word completion hits a speaker-safe wall.

    Inter-turn gaps are not trusted same-speaker regions, so this function does
    not expand into them. Completing a recognized word that would enter a
    competitor or adjacent turn is rejected. ASR timestamps that merely overlap
    an edge into an undiarized gap are not enough to reject or to repair.
    Clamp bounds:
    1. Do not overlap preceding or succeeding turns (or competitor speakers).
    2. Do not exceed audio duration [0.0, audio_duration_s].
    3. Do not leave consensus bounds into disputed speech.
    4. Do not encroach into raw competitor intervals from primary or secondary diarizers.
    """
    if not turns:
        return ([], [])
    aligned_turns: list[dict] = []
    audits: list[dict[str, Any]] = []
    n = len(turns)
    sorted_order = sorted(range(n), key=lambda idx: (turns[idx]['start_s'], turns[idx]['end_s']))
    safe_bounds: dict[int, tuple[float, float]] = {}
    for rank, orig_idx in enumerate(sorted_order):
        t = turns[orig_idx]
        prev_t = turns[sorted_order[rank - 1]] if rank > 0 else None
        next_t = turns[sorted_order[rank + 1]] if rank < n - 1 else None
        min_s = prev_t['end_s'] if prev_t is not None else 0.0
        max_e = next_t['start_s'] if next_t is not None else audio_duration_s if audio_duration_s is not None else float('inf')
        if audio_duration_s is not None:
            max_e = min(max_e, audio_duration_s)
            min_s = max(0.0, min_s)
        if '_consensus_start_s' in t:
            min_s = max(min_s, t['_consensus_start_s'])
        if '_consensus_end_s' in t:
            max_e = min(max_e, t['_consensus_end_s'])
        if competitor_intervals_by_speaker:
            comp_ivs = competitor_intervals_by_speaker.get(t['speaker_id'], [])
            for c_s, c_e in comp_ivs:
                if c_e <= t['start_s'] + 0.0001:
                    min_s = max(min_s, c_e)
                if c_s >= t['end_s'] - 0.0001:
                    max_e = min(max_e, c_s)
        safe_bounds[orig_idx] = (min_s, max_e)
    for i, turn in enumerate(turns):
        orig_start = turn.get('_original_start_s', turn['start_s'])
        orig_end = turn.get('_original_end_s', turn['end_s'])
        raw_start = turn.get('_raw_start_s', turn['start_s'])
        raw_end = turn.get('_raw_end_s', turn['end_s'])
        safe_min, safe_max = safe_bounds.get(i, (0.0, audio_duration_s or float('inf')))
        wanted_start = turn['start_s']
        wanted_end = turn['end_s']
        for word in words:
            word_start = float(word['start'])
            word_end = float(word['end'])
            if word_start < turn['end_s'] < word_end:
                wanted_end = max(wanted_end, word_end)
            if word_start < turn['start_s'] < word_end:
                wanted_start = min(wanted_start, word_start)
        clamped_start = max(safe_min, wanted_start)
        clamped_end = min(safe_max, wanted_end)
        expansion_blocked = wanted_start < turn['start_s'] - 0.0001 and clamped_start > wanted_start + 0.0001 or (wanted_end > turn['end_s'] + 0.0001 and clamped_end < wanted_end - 0.0001)
        new_start = max(safe_min, turn['start_s'])
        new_end = min(safe_max, turn['end_s'])
        new_start_s = round(new_start, 4)
        new_end_s = round(new_end, 4)
        shrunk = new_start_s > turn['start_s'] + 0.0001 or new_end_s < turn['end_s'] - 0.0001
        blocked_after_shrink = shrunk and any((float(word['start']) < edge < float(word['end']) for word in words for edge in (new_start_s, new_end_s)))
        overlapping_words = [w for w in words if float(w['end']) > new_start_s and float(w['start']) < new_end_s]
        if new_start_s >= new_end_s or expansion_blocked or blocked_after_shrink or (not overlapping_words):
            audits.append({'raw_start_s': raw_start, 'raw_end_s': raw_end, 'original_start_s': orig_start, 'original_end_s': orig_end, 'start_s': turn['start_s'], 'end_s': turn['end_s'], 'policy': policy, 'action': 'reject', 'error': 'word_boundary_conflicts_with_safe_bounds' if expansion_blocked or blocked_after_shrink else 'no_complete_words_in_safe_interval'})
            continue
        raw_words_in_turn = [w for w in words if float(w['start']) >= new_start_s and float(w['end']) <= new_end_s]
        words_in_turn = [str(w.get('text', '')).strip() for w in raw_words_in_turn if w.get('text')]
        transcript = ' '.join([wt for wt in words_in_turn if wt]) if words_in_turn else None
        delta_start = round((new_start_s - raw_start) * 1000.0, 1)
        delta_end = round((new_end_s - orig_end) * 1000.0, 1)
        tail_rescued = new_end_s > orig_end
        has_lock = bool(words_in_turn and (new_start_s != turn['start_s'] or new_end_s != turn['end_s'] or transcript))
        applied_policy = policy if has_lock else turn.get('_boundary_policy', 'standard')
        refined_turn = dict(speaker_id=turn['speaker_id'], start_s=new_start_s, end_s=new_end_s, confidence=turn['confidence'])
        refined_turn['_original_start_s'] = orig_start
        refined_turn['_original_end_s'] = orig_end
        refined_turn['_raw_start_s'] = raw_start
        refined_turn['_raw_end_s'] = raw_end
        refined_turn['_delta_start_ms'] = delta_start
        refined_turn['_delta_end_ms'] = delta_end
        refined_turn['_boundary_policy'] = applied_policy
        refined_turn['_tail_rescued'] = tail_rescued
        refined_turn['_transcript'] = transcript
        refined_turn['_words'] = raw_words_in_turn
        if '_consensus_start_s' in turn:
            refined_turn['_consensus_start_s'] = turn['_consensus_start_s']
        if '_consensus_end_s' in turn:
            refined_turn['_consensus_end_s'] = turn['_consensus_end_s']
        aligned_turns.append(refined_turn)
        audits.append({'raw_start_s': raw_start, 'raw_end_s': raw_end, 'original_start_s': orig_start, 'original_end_s': orig_end, 'start_s': new_start_s, 'end_s': new_end_s, 'delta_start_ms': delta_start, 'delta_end_ms': delta_end, 'policy': applied_policy, 'tail_rescued': tail_rescued, 'transcript': transcript})
    return (aligned_turns, audits)

def _extract_words_from_asr_result(res_obj: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract flat list of word dictionaries from Whisper transcription output."""
    extracted = []
    for seg in res_obj.get('segments', []):
        for w in seg.get('words', []):
            text = w.get('text') or w.get('word') or ''
            start_val = w.get('start')
            end_val = w.get('end')
            conf = w.get('confidence', 1.0)
            if text and start_val is not None and (end_val is not None):
                extracted.append({'text': str(text).strip(), 'start': float(start_val), 'end': float(end_val), 'confidence': float(conf)})
    return extracted

def _transcribe_words_with_whisper(audio: Path, *, model_name: str='vinai/PhoWhisper-small', language: str='vi', device: str='cpu') -> list[dict[str, Any]]:
    """Transcribe audio using whisper-timestamped (or openai-whisper) and return word timestamps.

    Supports CPU inference and automatically frees VRAM with CPU fallback upon CUDA OOM.
    """
    import torch
    device_str = 'cpu'
    if device and device != 'cpu' and (device != 'same'):
        if device.startswith('cuda:'):
            device_str = device if torch.cuda.is_available() else 'cpu'
        elif device in {'auto', 'cuda'}:
            device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        elif device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device_str = 'mps'
    try:
        import whisper_timestamped as whisperts
    except ImportError:
        whisperts = None
    whisper_mod = None
    if whisperts is None:
        try:
            import whisper as whisper_mod
        except ImportError:
            raise RuntimeError("Package 'whisper-timestamped' (or 'openai-whisper') is required for Whisper alignment. Please run: uv pip install whisper-timestamped")
    model = None
    words: list[dict[str, Any]] = []
    try:
        _ensure_torch_hub_trusted()
        logger.info("Loading Whisper alignment model '%s' on %s (lang=%s)...", model_name, device_str, language)
        if whisperts is not None:
            model = whisperts.load_model(model_name, device=device_str)
            transcribe_opts = {'language': language or 'vi', 'vad': True, 'detect_disfluencies': True}
            try:
                res = whisperts.transcribe(model, str(audio), **transcribe_opts)
            except Exception as vad_exc:
                err_msg = str(vad_exc).lower()
                if 'silero' in err_msg or 'vad' in err_msg or 'untrusted' in err_msg:
                    logger.warning('Silero VAD pre-segmentation failed (%s). Retrying Whisper alignment with vad=False...', vad_exc)
                    transcribe_opts['vad'] = False
                    res = whisperts.transcribe(model, str(audio), **transcribe_opts)
                else:
                    raise
        else:
            model = whisper_mod.load_model(model_name, device=device_str)
            res = model.transcribe(str(audio), language=language or 'vi', word_timestamps=True)
        words = _extract_words_from_asr_result(res)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if ('out of memory' in str(exc).lower() or isinstance(exc, torch.cuda.OutOfMemoryError)) and device_str != 'cpu':
            logger.warning('CUDA OOM on device %s during Whisper alignment (%s). Freeing GPU VRAM and retrying on CPU.', device_str, exc)
            try:
                del model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            device_str = 'cpu'
            logger.info('Executing Whisper alignment on CPU fallback...')
            if whisperts is not None:
                cpu_model = whisperts.load_model(model_name, device='cpu')
                try:
                    res = whisperts.transcribe(cpu_model, str(audio), language=language or 'vi', vad=True, detect_disfluencies=True)
                except Exception as vad_exc:
                    err_msg = str(vad_exc).lower()
                    if 'silero' in err_msg or 'vad' in err_msg or 'untrusted' in err_msg:
                        logger.warning('Silero VAD pre-segmentation failed on CPU fallback (%s). Retrying with vad=False...', vad_exc)
                        res = whisperts.transcribe(cpu_model, str(audio), language=language or 'vi', vad=False, detect_disfluencies=True)
                    else:
                        raise
                del cpu_model
            else:
                cpu_model = whisper_mod.load_model(model_name, device='cpu')
                res = cpu_model.transcribe(str(audio), language=language or 'vi', word_timestamps=True)
                del cpu_model
            words = _extract_words_from_asr_result(res)
        else:
            raise
    finally:
        try:
            del model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    return words

def _run_whisper_timestamped_alignment(audio: Path, turns: Sequence[dict], *, model_name: str='vinai/PhoWhisper-small', language: str='vi', device: str='cpu', competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None=None) -> tuple[list[dict], list[dict[str, Any]]]:
    """Align turns using whisper-timestamped with Vietnamese or multilingual models."""
    words = _transcribe_words_with_whisper(audio, model_name=model_name, language=language, device=device)
    return _lock_turns_with_words(turns, words, policy=f'whisper_lock_{Path(model_name).name}', audio_duration_s=probe(audio)['duration_s'], competitor_intervals_by_speaker=competitor_intervals_by_speaker)

def _run_remote_whisper_alignment(audio: Path, turns: Sequence[dict], *, endpoint: str, language: str='vi', competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None=None) -> tuple[list[dict], list[dict[str, Any]]]:
    """Query a remote OpenAI-compatible Whisper transcription server for word timestamps."""
    import json
    import urllib.parse
    import urllib.request
    import uuid
    boundary = f'----WebKitFormBoundary{uuid.uuid4().hex}'
    body = bytearray()

    def add_field(name: str, value: str):
        body.extend(f'--{boundary}\r\n'.encode('utf-8'))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode('utf-8'))
        body.extend(f'{value}\r\n'.encode('utf-8'))
    add_field('response_format', 'verbose_json')
    add_field('timestamp_granularities[]', 'word')
    if language:
        add_field('language', language)
    with open(str(audio), 'rb') as f:
        file_bytes = f.read()
    body.extend(f'--{boundary}\r\n'.encode('utf-8'))
    body.extend(f'Content-Disposition: form-data; name="file"; filename="{Path(audio).name}"\r\n'.encode('utf-8'))
    body.extend(b'Content-Type: audio/wav\r\n\r\n')
    body.extend(file_bytes)
    body.extend(b'\r\n')
    body.extend(f'--{boundary}--\r\n'.encode('utf-8'))
    req = urllib.request.Request(endpoint, data=body, headers={'Content-Type': f'multipart/form-data; boundary={boundary}'})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode('utf-8'))
    words: list[dict[str, Any]] = []
    for w in data.get('words', []):
        text = w.get('word') or w.get('text') or ''
        s = w.get('start')
        e = w.get('end')
        if text and s is not None and (e is not None):
            words.append({'text': str(text).strip(), 'start': float(s), 'end': float(e), 'confidence': float(w.get('confidence', 1.0))})
    return _lock_turns_with_words(turns, words, policy='remote_whisper_lock', audio_duration_s=probe(audio)['duration_s'], competitor_intervals_by_speaker=competitor_intervals_by_speaker)

def _run_mms_fa_alignment(audio: Path, turns: Sequence[dict], *, device: str='cpu', competitor_intervals_by_speaker: dict[str, list[tuple[float, float]]] | None=None) -> tuple[list[dict], list[dict[str, Any]]]:
    """Run PyTorch MMS forced alignment with CPU support and CUDA OOM fallback."""
    import torch
    import torchaudio
    device_str = 'cpu'
    if device and device != 'cpu' and (device != 'same'):
        if device.startswith('cuda:'):
            device_str = device if torch.cuda.is_available() else 'cpu'
        elif device in {'auto', 'cuda'}:
            device_str = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    bundle = torchaudio.pipelines.MMS_FA
    model = None
    audio_data, sr = sf.read(str(audio), dtype='float32', always_2d=True)
    waveform = torch.from_numpy(audio_data.T)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, bundle.sample_rate)
        waveform = resampler(waveform)
        sr = bundle.sample_rate
    emission = None
    try:
        model = bundle.get_model().to(device_str)
        model.eval()
        with torch.inference_mode():
            emission, _ = model(waveform.to(device_str))
            emission = torch.log_softmax(emission, dim=-1)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if ('out of memory' in str(exc).lower() or isinstance(exc, torch.cuda.OutOfMemoryError)) and device_str != 'cpu':
            logger.warning('CUDA OOM during MMS-FA on %s (%s). Falling back to CPU.', device_str, exc)
            try:
                del model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            device_str = 'cpu'
            model = bundle.get_model().to('cpu')
            model.eval()
            with torch.inference_mode():
                emission, _ = model(waveform.to('cpu'))
                emission = torch.log_softmax(emission, dim=-1)
        else:
            raise
    finally:
        try:
            del model
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    aligned_turns: list[dict] = []
    audits: list[dict[str, Any]] = []
    num_frames = emission.shape[1]
    total_duration = waveform.shape[1] / sr
    frame_to_sec = total_duration / num_frames
    blank_prob = torch.exp(emission[0, :, 0]).cpu().numpy()
    speech_prob = 1.0 - blank_prob
    n = len(turns)
    sorted_order = sorted(range(n), key=lambda idx: (turns[idx]['start_s'], turns[idx]['end_s']))
    safe_bounds: dict[int, tuple[float, float]] = {}
    for rank, orig_idx in enumerate(sorted_order):
        t = turns[orig_idx]
        prev_t = turns[sorted_order[rank - 1]] if rank > 0 else None
        next_t = turns[sorted_order[rank + 1]] if rank < n - 1 else None
        min_s = prev_t['end_s'] if prev_t is not None else 0.0
        max_e = next_t['start_s'] if next_t is not None else total_duration
        max_e = min(max_e, total_duration)
        min_s = max(0.0, min_s)
        if '_consensus_start_s' in t:
            min_s = max(min_s, t['_consensus_start_s'])
        if '_consensus_end_s' in t:
            max_e = min(max_e, t['_consensus_end_s'])
        if competitor_intervals_by_speaker:
            comp_ivs = competitor_intervals_by_speaker.get(t['speaker_id'], [])
            for c_s, c_e in comp_ivs:
                if c_e <= t['start_s'] + 0.0001:
                    min_s = max(min_s, c_e)
                if c_s >= t['end_s'] - 0.0001:
                    max_e = min(max_e, c_s)
        safe_bounds[orig_idx] = (min_s, max_e)
    for i, turn in enumerate(turns):
        orig_start = turn.get('_original_start_s', turn['start_s'])
        orig_end = turn.get('_original_end_s', turn['end_s'])
        raw_start = turn.get('_raw_start_s', turn['start_s'])
        raw_end = turn.get('_raw_end_s', turn['end_s'])
        safe_min, safe_max = safe_bounds.get(i, (0.0, total_duration))
        start_f = max(0, int(round(turn['start_s'] / frame_to_sec)))
        end_f = min(num_frames - 1, int(round(turn['end_s'] / frame_to_sec)))
        search_f = int(round(0.3 / frame_to_sec))
        new_end_f = end_f
        if speech_prob[end_f] > 0.3:
            for f in range(end_f, min(num_frames - 1, end_f + search_f)):
                if speech_prob[f] < 0.2:
                    new_end_f = f
                    break
        new_end_s = round(new_end_f * frame_to_sec, 4)
        new_start_s = round(start_f * frame_to_sec, 4)
        new_start_s = max(safe_min, new_start_s)
        new_end_s = min(safe_max, new_end_s)
        if new_start_s >= new_end_s:
            new_start_s = turn['start_s']
            new_end_s = turn['end_s']
        delta_start = round((new_start_s - raw_start) * 1000.0, 1)
        delta_end = round((new_end_s - orig_end) * 1000.0, 1)
        tail_rescued = new_end_s > orig_end
        has_lock = new_end_f != end_f or tail_rescued
        applied_policy = 'syllable_word_lock' if has_lock else turn.get('_boundary_policy', 'standard')
        refined_turn = dict(speaker_id=turn['speaker_id'], start_s=new_start_s, end_s=new_end_s, confidence=turn['confidence'])
        refined_turn['_original_start_s'] = orig_start
        refined_turn['_original_end_s'] = orig_end
        refined_turn['_raw_start_s'] = raw_start
        refined_turn['_raw_end_s'] = raw_end
        refined_turn['_delta_start_ms'] = delta_start
        refined_turn['_delta_end_ms'] = delta_end
        refined_turn['_boundary_policy'] = applied_policy
        refined_turn['_tail_rescued'] = tail_rescued
        if '_consensus_start_s' in turn:
            refined_turn['_consensus_start_s'] = turn['_consensus_start_s']
        if '_consensus_end_s' in turn:
            refined_turn['_consensus_end_s'] = turn['_consensus_end_s']
        aligned_turns.append(refined_turn)
        audits.append({'raw_start_s': raw_start, 'raw_end_s': raw_end, 'original_start_s': orig_start, 'original_end_s': orig_end, 'start_s': new_start_s, 'end_s': new_end_s, 'delta_start_ms': delta_start, 'delta_end_ms': delta_end, 'policy': applied_policy, 'tail_rescued': tail_rescued})
    return (aligned_turns, audits)


def main() -> int:
    p = arguments(__doc__)
    p.add_argument('--engine', choices=('whisper_timestamped', 'remote_whisper', 'mms_fa', 'words'), default='whisper_timestamped', help='Alignment engine choice: "whisper_timestamped", "remote_whisper", "mms_fa", or "words"')
    p.add_argument('--model', default='vinai/PhoWhisper-small', help='Model name or Hugging Face repository for alignment')
    p.add_argument('--language', default='vi', help='Audio language code (e.g. "vi", "en")')
    p.add_argument('--device', default='cpu', help='Inference device ("cpu", "cuda", or "hip")')
    p.add_argument('--endpoint', help='Remote endpoint URL for remote_whisper engine')
    p.add_argument('--words-file', type=Path, help='Path to JSON file containing precomputed word timestamps')
    args = p.parse_args()
    if args.concurrency < 1:
        p.error('--concurrency must be at least 1')
    if args.batch_size < 1:
        p.error('--batch-size must be at least 1')
    if args.engine == 'words' and not args.words_file:
        p.error('--words-file is required for the words engine')
    if args.engine == 'remote_whisper' and not args.endpoint:
        p.error('--endpoint is required for remote_whisper')
    manifest, source = load(args)
    values = {'engine': args.engine, 'model': args.model, 'language': args.language,
              'device': args.device, 'endpoint': args.endpoint,
              'words_file': identity(args.words_file) if args.words_file else None}
    os.environ.setdefault('TORCH_HOME', str(ROOT / '.data/torch'))
    progress('ALIGN_START', f'Running alignment engine={args.engine} on {source.name}')
    with contextlib.redirect_stdout(sys.stderr):
        if args.engine == 'words':
            words = read_json(args.words_file)['words']
            turns, audits = _lock_turns_with_words(manifest['turns'], words, audio_duration_s=probe(source)['duration_s'])
        elif args.engine == 'remote_whisper':
            turns, audits = _run_remote_whisper_alignment(source, manifest['turns'], endpoint=args.endpoint, language=args.language)
        elif args.engine == 'mms_fa':
            turns, audits = _run_mms_fa_alignment(source, manifest['turns'], device=args.device)
        else:
            turns, audits = _run_whisper_timestamped_alignment(source, manifest['turns'], model_name=args.model, language=args.language, device=args.device)
    progress('ALIGN_DONE', f'Produced {len(turns)} aligned turns')
    save(args, manifest, source, turns, 'align', values, audits=audits)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
