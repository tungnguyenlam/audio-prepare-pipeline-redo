#!/usr/bin/env python3
"""Run a configurable speech-cleanup cascade using separate Python environments.

Expected input is preferably the VOCALS stem already produced by Mel/BS-RoFormer,
HTDemucs or MVSEP. Default 'balanced' path:
    DeepFilterNet -> ClearVoice enhancement -> Silero VAD gate
Aggressive mode additionally runs VoiceFixer at the end.

Each heavy model may live in its own virtualenv. Pass the corresponding Python
interpreter with --df-python / --cv-python / --vf-python / --vad-python.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print('+', ' '.join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-i', '--input', type=Path, required=True, help='Prefer a vocal stem, not raw music mixture')
    p.add_argument('-o', '--output', type=Path, required=True, help='Final output file')
    p.add_argument('--work-dir', type=Path, default=Path('.speech_cleanup_work'))
    p.add_argument('--mode', choices=('conservative', 'balanced', 'aggressive'), default='balanced')
    p.add_argument('--df-python', default=sys.executable)
    p.add_argument('--cv-python', default=sys.executable)
    p.add_argument('--vad-python', default=sys.executable)
    p.add_argument('--vf-python', default=sys.executable)
    p.add_argument('--clearvoice-model', default='MossFormer2_SE_48K',
                   choices=('MossFormer2_SE_48K', 'FRCRN_SE_16K', 'MossFormerGAN_SE_16K'))
    p.add_argument('--device', default='auto')
    p.add_argument('--df-atten-lim-db', type=float, default=None)
    p.add_argument('--keep-intermediate', action='store_true')
    args = p.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    s1 = args.work_dir / '01_deepfilter.wav'
    s2 = args.work_dir / '02_clearvoice.wav'
    s3 = args.work_dir / '03_vad.wav'
    s4 = args.work_dir / '04_voicefixer.wav'

    df_cmd = [args.df_python, str(HERE / 'denoise_deepfilternet.py'), '-i', str(args.input), '-o', str(s1)]
    if args.df_atten_lim_db is not None:
        df_cmd += ['--atten-lim-db', str(args.df_atten_lim_db)]
    run(df_cmd)

    if args.mode == 'conservative':
        current = s1
    else:
        run([args.cv_python, str(HERE / 'enhance_clearvoice.py'), '-i', str(s1), '-o', str(s2),
             '--model', args.clearvoice_model, '--device', args.device, '--preserve-sr'])
        current = s2

    run([args.vad_python, str(HERE / 'vad_gate_silero.py'), '-i', str(current), '-o', str(s3),
         '--speech-pad-ms', '80', '--fade-ms', '8'])
    current = s3

    if args.mode == 'aggressive':
        vf_cmd = [args.vf_python, str(HERE / 'restore_voicefixer.py'), '-i', str(current), '-o', str(s4),
                  '--mode', '0', '--preserve-sr']
        if args.device.startswith('cuda'):
            vf_cmd.append('--cuda')
        run(vf_cmd)
        current = s4

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(current.read_bytes())
    print(f'FINAL -> {args.output}')

    if not args.keep_intermediate:
        for pth in (s1, s2, s3, s4):
            if pth.exists() and pth != current:
                pth.unlink()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
