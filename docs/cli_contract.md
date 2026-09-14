# CLI contract

[← Index](README.md) · [Cookbook](commands.md) · [Data contract](data_contract.md)

## Global rules

- **Inputs.** `--input-file` takes precedence over `--input-dir`. Directory input is
  scanned recursively in sorted order (audio suffixes only), excluding the output
  subtree; identical input/output roots are rejected. Relative paths resolve against
  the caller's working directory.
- **Outputs.** For single-output commands `--output-file` is the exact destination
  and requires `--input-file`. Diarizers and manifest writers take `--output-dir` /
  `--output-manifest`, never `--output-file`. `info.py` prints JSON Lines to stdout.
- **Names.** Output stems and directories pass through `safe_name`: only
  `[a-zA-Z0-9_-]`, spaces and dots become `-`, diacritics are transliterated
  (`đ → d`). Subdirectories of an input tree are mirrored.
- **Audio family.** Downloads are named `<video_id>_<title10>-<sample_rate>.wav`;
  the family key `<video_id>_<title10>` is inferred downstream from the filename or
  sidecar. Default outputs are `.data/<operation>[/<model>]/<family>/`
  (e.g. `.data/download/<family>/`, `.data/separate/htdemucs/<family>/`,
  `.data/diarize/sortformer/<family>/segments.json`,
  `.data/purity/<stage>/<family>/segments.json`,
  `.data/speaker/<stage>/<family>/segments.json`,
  `.data/agent/<backend>/<family>/`,
  `.data/agent/verifier/<backend>/<family>/`; Gemini adds
  `<model>/<reasoning-effort>/` after `gemini`). `--work-dir` defaults to
  `.data/<operation>[/<model>]/work`.
- **Idempotency.** A destination whose sidecar matches the request and whose output
  hash is intact is skipped; a matching but incomplete record is retried; anything
  else needs `--overwrite`. Audio is never written in place.
- **Concurrency.** Every command accepts `--concurrency` and `--batch-size`
  (default 1/1). Models load once per invocation; thread pools parallelize I/O and
  requests. Gemini commands default to the provider Batch API (`--inference-mode
  batch`, ≤ 10 requests per job); `--inference-mode standard` is synchronous.
- **Logging.** Parsed configuration and timestamped progress go to stderr;
  successful output paths go to stdout; any per-item failure yields a nonzero exit
  after the batch finishes. `-h` on `.sh` or `.py` prints every flag and default.

## Command matrix

Key flags only; run `-h` for the full list. All commands also accept
`--overwrite`, `--concurrency`, `--batch-size`.

| Group | Command | Key flags | Output |
|---|---|---|---|
| Download | `download/youtube` | `--url` \| `--url-file`, `--output-file`, `--output-dir`, `--sample-rate` (48000), `--cookie-file` | mono WAV + `.json` |
| | `download/playlist`, `download/channel` | `--url`, `--limit`/`--max-items`, `--sample-rate`, `--output-dir`, `--cookie-file` | WAVs + `.json` |
| Separate | `separate/htdemucs`, `htdemucs_ft` | `--stem` (vocals/instrumental/drums/bass/other), `--device`, `--sample-rate`, `--channels`, `--shifts`, `--overlap`, `--segment` | `<stem>_<model>.wav` + `.json` |
| | `separate/bs_roformer`, `mel_roformer` | `--model`, `--backend`, `--device`, `--stem`, `--model-sample-rate`, `--sample-rate`, `--channels` | `<stem>_<model>.wav` + `.json` |
| | `separate/mvsep_mdx23` | `--device`, `--stem`, `--overlap-large/-small`, `--single-onnx`, `--large-gpu`, `--use-kim-model-1`, `--chunk-size`, `--max-segment-seconds`, `--repo-dir` | `<stem>_mvsep_mdx23.wav` + `.json` |
| Diarize | `diarize/sortformer` | `--model-id`, `--device`, `--window-duration-s`, `--overlap-duration-s`, `--onset/--offset`, `--enable-speaker-similarity`, `--min/--max-duration-s` (2/15) | `<stem>/segments.json` + `segments.raw.json` + clips + `timeline*.png` |
| | `diarize/pyannote_community1`, `pyannote_31` (`pyannote.sh` = community1) | `--num-speakers`, `--min/--max-speakers`, `--device`, `--min/--max-duration-s` | `<stem>/segments.json` + `segments.raw.json` + clips + `timeline*.png` |
| | `diarize/clustering` | `--num-speakers`, `--max-num-speakers`, `--vad-model`, `--speaker-model`, `--vad-*`, `--min/--max-duration-s` | `<stem>/segments.json` + `segments.raw.json` + clips + `timeline*.png` |
| | `diarize/threed_speaker` | `--num-speakers`, `--include-overlap`, `--chunk-duration-s`, `--chunk-step-s`, `--min/--max-duration-s` | `<stem>/segments.json` + `segments.raw.json` + clips + `timeline*.png` |
| | `diarize/diarizen` | `--model`, `--num/--min/--max-speakers`, `--segmentation-step`, `--binarize-onset/-offset`, `--min/--max-duration-s` | `<stem>/segments.json` + `segments.raw.json` + clips + `timeline*.png` |
| Audio | `audio/info` | `--input-file` \| `--input-dir` | JSON Lines on stdout |
| | `audio/convert` | `--sample-rate`, `--channels` | WAV + `.json` |
| | `audio/cut` | `--start`, `--end` (seconds), `--sample-rate`, `--channels` | `<stem>_cut.wav` + `.json` |
| | `audio/export_segments` | `--input-manifest`, `--output-dir`, `--input-file` (source override), `--sample-rate`, `--channels`, `--min/--max-duration-s` | clips + updated `segments.json` + three plots (default dir `.data/audio/clips/<family>`) |
| | `audio/compare_waveforms`, `compare_spectrograms` | `--input-file` \| `--input-dir`, `--reference-file`, `--sample-rate` (16000); spectrograms add `--n-mels`, `--hop-length`, `--fmax`, `--top-db` | PNG + `.json` |
| Speaker | `speaker/enroll` | `--name`, `--clip` (repeatable) \| `--clip-dir`, `--profiles-dir`, `--add`, `--channel-id/-name/-url` | `profile.json` + copied clips |
| | `speaker/score` | `--input-manifest`, `--profile`, `--profiles-dir`, `--model-id`, `--device`, `--output-manifest` | scored `segments.json` |
| | `speaker/filter` | `--input-manifest`, `--threshold`, `--min-duration-s`, `--exclude-overlap`, `--output-manifest` | filtered `segments.json` |
| | `speaker/purity` | `--input-manifest`, `--profile`, `--similarity-threshold`, `--window-duration-s`, `--window-hop-s`, `--max-overlap-duration-s`, `--output-manifest` | verified `segments.json` |
| Purity | `purity/consensus` | `--input-manifest`, `--secondary-manifest` | `segments.json` |
| | `purity/cleanup` | `--min-turn-duration-s`, `--merge-same-speaker-gap-s`, `--boundary-collar-s`, `--jitter-max-duration-s` | `segments.json` |
| | `purity/collar` | `--collar-s`, `--context-aware`, `--transition-exclusion-s`, `--min-duration-s` | `segments.json` |
| | `purity/snap` | `--search-window-s`, `--energy-floor-db`, `--frame-len-ms`, `--hop-len-ms` | `segments.json` |
| | `purity/align` | `--engine`, `--model`, `--language`, `--device`, `--endpoint`, `--words-file` | word-locked `segments.json` |
| | `purity/merge` | `--input-manifest`, `--input-file`, `--output-manifest`, `--max-gap-s` (1), `--silence-threshold-dbfs` (-40), `--frame-ms` (20) | unfiltered `segments.json` + merge audit |
| | `purity/segment` | `--words-file`, `--max/--min-duration-s`, `--min-pause-s` | duration-bounded `segments.json` |
| Agent | `agent/gemini` | `--prompt-file` (required), `--system-prompt-file`, `--model` (gemini-3.8-flash), `--reasoning-effort` (medium), `--inference-mode` (batch), `--max-tokens`, `--temperature`, `--top-p/-k`, `--batch-timeout-s` | `<stem>_gemini.txt` + `.json` |
| | `agent/endpoint` | `--prompt-file`, `--endpoint`, `--model`, `--timeout-s`, `--max-tokens`, `--temperature` | `<stem>_endpoint.txt` + `.json` |
| | `agent/hf` | `--prompt-file`, `--model-id`, `--device`, `--adapter-path`, `--load-in-4bit/-8bit` | `<stem>_hf.txt` + `.json` |
| Verifier | `agent/verifier/gemini` | as `agent/gemini`; `--prompt-file` optional (default `prompts/acoustic_defect-3.txt`) | `.txt` + verdict/transcript `.json` |
| | `agent/verifier/hf` | `--prompt-file`, `--model-id`, `--device`, `--adapter-path`, `--max-new-tokens` (1024), `--load-in-4bit/-8bit` | `.txt` + verdict/transcript `.json` |
| | `agent/verifier/endpoint`, `unsloth`, `vllm`, `moss`, `minicpm`, `kimi` | `--prompt-file` (optional) plus backend options (`--model-id`/`--model`, `--endpoint`, `--device`, `--adapter-path`, `--gguf-variant`, `--dtype`, …) | `.txt` + verdict/transcript `.json` |
| | `agent/verifier/vibevoice` | `--model-id`, `--device`, `--max-new-tokens`, `--min-secondary-speech-s` | `.txt` + speaker-count verdict `.json` |
| | `agent/verifier/analysis` (`analyze` alias) | `--input-dir` (`--verdict-dir` alias), `--input-manifest`, `--manifest-dir`, `--output-dir` | `<input-dir>/plot/` with transcript CSV/statistics |
| | `agent/verifier/compare` | `--reference-dir`, `--candidates-dir` (repeatable), `--output-dir`, `--title`, `--no-plots` | comparison directory |
| | `agent/verifier/evaluate_verifier` | `--predictions-dir`, `--reference-dir`, `--output-file`, `--title` | accuracy / reject recall / F1 / FRR / latency JSON |
| | `agent/verifier/scaffold_experiment` | `--name` | `.data/agent/verifier/experiments/<name>/` |
| Mix/eval | `mix/mix` | `--speech`, `--music`, `--smr-db`, `--seed`, `--sample-rate`, `--channels`, `--peak-ceiling-dbfs`, `--output-dir` | mixture, references, metadata |
| | `evaluate/separation` | `--input-file`, `--reference-file`, `--mixture-file`, `--sample-rate`, `--output-file` | SI-SDR / SDR JSON |
| | `evaluate/diarization` | `--input-manifest`, `--reference-manifest`, `--duration`, `--collar`, `--skip-overlap`, `--output-file` | DER / JER / confusion JSON |
| | `evaluate/plot_diarization` | `--input-manifest`, `--reference-manifest`, `--title`, `--output-file`, `--bin-width` (0.25) | Gantt PNG plus sibling `_duration.png` and `_cutoff.png` |
| | `evaluate/plot_metrics` | `--metrics-file` (repeatable), `--title`, `--output-file` | bar plot PNG |
| Dataset | `dataset/index` | `--input-dir`, `--output-manifest`, `--tag` | manifest JSON |
| | `dataset/filter` | `--input-manifest`, `--output-manifest`, `--tag`, `--exclude-tag`, `--min/--max-duration` | manifest JSON |
| | `dataset/export` | `--input-manifest`, `--output-file`, `--format` (jsonl/csv) | JSONL / CSV |
| | `dataset/bundle` | `--input-manifest`, `--output-file` | ZIP |

`purity/merge` processes turns in timeline order with bounded frame reads. It
accepts the shared concurrency/batch flags but runs sequentially. Duration
filtering belongs to the subsequent `audio/export_segments` invocation.
