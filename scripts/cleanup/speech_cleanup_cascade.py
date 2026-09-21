"""Run independent speech-cleanup launchers in sequence, each in its own venv.

This does not chain crawl → separate → diarize → mix. It only shells out to the
standalone cleanup commands. Default steps are denoise → enhance → VAD gate.
VoiceFixer restore is opt-in because it can color the voice. Overlap separation
writes two stems and must be run as its own command, not mixed into this list.
"""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common.files import (  # noqa: E402
    ROOT, batch, completed, convert, destinations, identity, parser, positive_int,
    probe, progress, publish, request,
)

LINEAR_STEPS = ('denoise', 'enhance', 'restore', 'vad_gate')
DEFAULT_STEPS = ('denoise', 'enhance', 'vad_gate')
STEP_LAUNCHERS = {
    'denoise': ROOT / 'scripts/cleanup/denoise_deepfilternet.sh',
    'enhance': ROOT / 'scripts/cleanup/enhance_clearvoice.sh',
    'restore': ROOT / 'scripts/cleanup/restore_voicefixer.sh',
    'vad_gate': ROOT / 'scripts/cleanup/vad_gate_silero.sh',
}


def main() -> int:
    p = parser(__doc__, 'cleanup', 'speech_cleanup_cascade')
    p.add_argument('--steps', default=','.join(DEFAULT_STEPS),
                   help='Comma-separated linear steps: denoise, enhance, restore, vad_gate '
                        f'(default: {",".join(DEFAULT_STEPS)})')
    p.add_argument('-d', '--device', default='auto',
                   help='Forwarded to each step (auto, cpu, cuda:0)')
    p.add_argument('--atten-lim-db', type=float, default=12.0,
                   help='Forwarded to denoise (default: 12)')
    p.add_argument('--no-atten-lim', action='store_true',
                   help='Forwarded to denoise: unlimited attenuation')
    p.add_argument('--enhance-model', default='MossFormer2_SE_48K',
                   help='Forwarded to enhance as --model')
    p.add_argument('--vad-threshold', type=float, default=0.5,
                   help='Forwarded to vad_gate as --threshold')
    p.add_argument('--vad-pad-ms', type=float, default=200.0,
                   help='Forwarded to vad_gate as --pad-ms')
    p.add_argument('--restore-mode', type=int, choices=(0, 1, 2), default=0,
                   help='Forwarded to restore as --mode')
    p.add_argument('-sr', '--sample-rate', type=positive_int, default=None,
                   help='Output sample rate in Hz (default: preserve source)')
    p.add_argument('-ch', '--channels', type=int, choices=(1, 2), default=1,
                   help='Output channel layout (1=mono, 2=stereo)')
    args = p.parse_args()
    try:
        steps = _parse_steps(args.steps)
    except ValueError as exc:
        p.error(str(exc))
    if not steps:
        p.error('--steps must list at least one of denoise, enhance, restore, vad_gate')
    pairs = destinations(args, '_cleanup')
    args.work_dir.mkdir(parents=True, exist_ok=True)

    def process(src: Path, dest: Path) -> None:
        rate = args.sample_rate or probe(src)['sample_rate']
        parameters = {
            'steps': list(steps),
            'sample_rate': rate,
            'channels': args.channels,
            'device': args.device,
            'atten_lim_db': None if args.no_atten_lim else args.atten_lim_db,
            'enhance_model': args.enhance_model,
            'vad_threshold': args.vad_threshold,
            'vad_pad_ms': args.vad_pad_ms,
            'restore_mode': args.restore_mode,
        }
        metadata = request(identity(src), 'speech_cleanup_cascade', parameters, 'cascade')
        if completed(dest, metadata, args.overwrite):
            return
        with tempfile.TemporaryDirectory(dir=args.work_dir) as directory:
            work = Path(directory)
            current = src
            produced: list[Path] = []
            for index, step in enumerate(steps):
                step_dest = work / f'{index:02d}_{step}.wav'
                extra = _step_args(step, args)
                cmd = [
                    'bash', str(STEP_LAUNCHERS[step]),
                    '--input-file', str(current),
                    '--output-file', str(step_dest),
                    '--work-dir', str(work / step),
                    '--device', args.device,
                    '--sample-rate', str(rate),
                    '--channels', str(args.channels),
                    *extra,
                ]
                if args.overwrite:
                    cmd.append('--overwrite')
                progress('CASCADE_STEP', f'{src.name} {step}')
                artifact = _run_launcher(cmd)
                if artifact.resolve() != step_dest.resolve() and artifact.is_file():
                    step_dest = artifact
                current = step_dest
                produced.append(current)
            staged = work / 'output.wav'
            convert(current, staged, rate, args.channels)
            publish(staged, dest, {**metadata, 'intermediates': [identity(path) for path in produced]})

    return batch(pairs, process, concurrency=args.concurrency, batch_size=args.batch_size)


def _parse_steps(raw: str) -> tuple[str, ...]:
    steps: list[str] = []
    seen: set[str] = set()
    for part in raw.split(','):
        key = part.strip().lower().replace('-', '_')
        if not key:
            continue
        aliases = {'vad': 'vad_gate', 'voicefixer': 'restore', 'deepfilternet': 'denoise',
                   'clearvoice': 'enhance'}
        key = aliases.get(key, key)
        if key == 'overlap' or key == 'separate_overlap':
            raise ValueError(
                'Overlap separation writes two stems; run scripts/cleanup/separate_overlap_clearvoice.sh '
                'as its own command instead of mixing it into --steps'
            )
        if key not in STEP_LAUNCHERS:
            known = ', '.join(LINEAR_STEPS)
            raise ValueError(f'Unknown cleanup step {part!r}. Choose from: {known}')
        if key in seen:
            continue
        steps.append(key)
        seen.add(key)
    return tuple(steps)


def _step_args(step: str, args) -> list[str]:
    if step == 'denoise':
        extra = ['--atten-lim-db', str(args.atten_lim_db)]
        if args.no_atten_lim:
            extra.append('--no-atten-lim')
        return extra
    if step == 'enhance':
        return ['--model', args.enhance_model]
    if step == 'restore':
        return ['--mode', str(args.restore_mode)]
    if step == 'vad_gate':
        return ['--threshold', str(args.vad_threshold), '--pad-ms', str(args.vad_pad_ms)]
    return []


def _run_launcher(cmd: list[str]) -> Path:
    result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=None, text=True)
    stdout = result.stdout or ''
    if result.returncode != 0:
        detail = stdout.strip().splitlines()[-1] if stdout.strip() else f'exit {result.returncode}'
        raise RuntimeError(f'Cleanup step failed ({detail}): {" ".join(cmd[:4])}')
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f'Cleanup step printed no artifact path: {" ".join(cmd[:4])}')
    return Path(lines[-1])


if __name__ == '__main__':
    raise SystemExit(main())
