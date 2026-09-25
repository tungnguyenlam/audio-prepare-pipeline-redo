# Debugging Gemini Batch verifier failures

Read-only snippets for diagnosing a Gemini Batch run where jobs finish as
`BATCH_STATE_SUCCEEDED` but items fail. Run them from the repository root on the
machine that ran the batch. None of them submits generation work except the
optional one-token quota probe in step 4. They read `GEMINI_API_KEY` from the
environment or `.env` and never print it.

Default Gemini runtime paths are per model and reasoning level. The printed
configuration shows `.data/s4-agent/verifier/gemini/work`, but Batch state lives in
`.data/s4-agent/verifier/gemini/<model>/<reasoning>/work/batch_jobs/`
(for example `gemini-3-8-flash/medium`). The globs below search every variant.

## 1. Rule out broken input audio

```bash
.venvs/verify/bin/python - <<'EOF'
import soundfile as sf, pathlib, collections
bad, c = [], collections.Counter()
for p in sorted(pathlib.Path('.data/s3-diarize/diarizen/thu-am-studio/remaining').rglob('*.wav')):
    try:
        i = sf.info(p); c[(i.samplerate, i.channels, i.subtype)] += 1
        if i.frames == 0: bad.append((p, 'empty'))
    except Exception as e:
        bad.append((p, type(e).__name__))
print(c); print(len(bad), 'bad'); [print(*b) for b in bad[:10]]
EOF
```

Audio bytes are inlined into each request, so a moved folder only matters if the
files themselves are damaged.

## 2. Per-request errors saved in local Batch state

```bash
python3 - <<'EOF'
import json, glob, collections
for p in sorted(glob.glob('.data/s4-agent/verifier/gemini/**/batch_jobs/batch_*.json', recursive=True)):
    s = json.load(open(p)); c = collections.Counter()
    for g in s['groups']:
        for v in (g.get('results') or {}).values():
            e = v.get('error')
            c[json.dumps(e)[:400] if e else ('ok' if isinstance(v.get('response'), dict) else f'keys={sorted(v)}')] += 1
    print(p, '| groups:', len(s['groups'])); [print(' ', n, m) for m, n in c.most_common(5)]
EOF
```

A state file with groups but no counted results has jobs that were never
collected (for example a run interrupted before polling, or one whose input
folder was later moved so continuation no longer matches it).

## 3. Remote job states and per-request outcomes

Job states only (set `CANCEL=1` to cancel jobs still pending or running):

```bash
.venvs/verify/bin/python - <<'EOF'
import json, glob, os, collections, pathlib, httpx
env = pathlib.Path('.env')
for line in env.read_text().splitlines() if env.exists() else []:
    line = line.strip().removeprefix('export ')
    if line and not line.startswith('#') and '=' in line:
        k, _, v = line.partition('='); os.environ.setdefault(k.strip(), v.strip().strip('"\''))
H = {'x-goog-api-key': os.environ['GEMINI_API_KEY']}
cancel = os.environ.get('CANCEL') == '1'
with httpx.Client(headers=H, timeout=60) as c:
    for p in sorted(glob.glob('.data/s4-agent/verifier/gemini/**/batch_jobs/batch_*.json', recursive=True)):
        states = collections.Counter()
        for g in json.load(open(p))['groups']:
            job = c.get(f"https://generativelanguage.googleapis.com/v1beta/{g['job_name']}").json()
            st = (job.get('metadata') or {}).get('state') or job.get('state') or job.get('error', {}).get('status')
            states[st] += 1
            if cancel and st in ('BATCH_STATE_PENDING', 'BATCH_STATE_RUNNING'):
                c.post(f"https://generativelanguage.googleapis.com/v1beta/{g['job_name']}:cancel")
        print(p, dict(states))
EOF
```

Per-request success/error counts inside each remote job:

```bash
.venvs/verify/bin/python - <<'EOF'
import json, glob, os, collections, pathlib, httpx
env = pathlib.Path('.env')
for line in env.read_text().splitlines() if env.exists() else []:
    line = line.strip().removeprefix('export ')
    if line and not line.startswith('#') and '=' in line:
        k, _, v = line.partition('='); os.environ.setdefault(k.strip(), v.strip().strip('"\''))
H = {'x-goog-api-key': os.environ['GEMINI_API_KEY']}
def items(job):
    r = job.get('response') or (job.get('metadata') or {}).get('output') or {}
    r = r.get('inlinedResponses', r)
    return r.get('inlinedResponses', r) if isinstance(r, dict) else r
with httpx.Client(headers=H, timeout=120) as c:
    for p in sorted(glob.glob('.data/s4-agent/verifier/gemini/**/batch_jobs/batch_*.json', recursive=True)):
        req, jobs = collections.Counter(), collections.Counter()
        for i, g in enumerate(json.load(open(p))['groups']):
            job = c.get(f"https://generativelanguage.googleapis.com/v1beta/{g['job_name']}").json()
            if 'error' in job and 'name' not in job:
                jobs['get_error:' + str(job['error'].get('status'))] += 1; continue
            res = items(job) if isinstance(items(job), list) else []
            ok = sum(isinstance(x.get('response'), dict) for x in res)
            errs = collections.Counter(str((x.get('error') or {}).get('code')) for x in res if 'error' in x)
            req['ok'] += ok; req.update({'err_' + k: n for k, n in errs.items()})
            jobs['all_ok' if ok == len(res) and res else 'all_err' if not ok else 'mixed'] += 1
            if i < 3 or ok: print(f"  job {i} {g['job_name']}: ok={ok} err={dict(errs)}")
        print(p, 'jobs:', dict(jobs), 'requests:', dict(req))
EOF
```

Error codes are gRPC codes: `8` is `RESOURCE_EXHAUSTED` (quota), `1` is
`CANCELLED`. A `get_error:INTERNAL` job is a failed status lookup, not a failed job.

## 4. Is only Batch quota exhausted?

A one-token standard request (negligible cost). A 429 body names the exhausted
quota; a normal response means standard generation still works and only Batch
quota is exhausted.

```bash
set -a; . ./.env; set +a
curl -s -H "x-goog-api-key: $GEMINI_API_KEY" -H 'Content-Type: application/json' \
  https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent \
  -d '{"contents":[{"parts":[{"text":"ok"}]}],"generationConfig":{"maxOutputTokens":5}}' | head -c 1500
```

## Interpreting results

| Finding | Action |
|---|---|
| All requests `RESOURCE_EXHAUSTED`, standard probe OK | Batch quota exhausted: wait for reset (check AI Studio rate limits) or use `--inference-mode standard` |
| Flex returns HTTP 503 | Flex capacity shed; retries back off automatically, use `-c 1` off-peak or standard |
| Old state files with pending jobs | Cancel with `CANCEL=1` or let them finish; they consume Batch quota |
| Old jobs with `ok` responses | Paid results never published; recover manually if worthwhile |

Since commit `f2c6b11` the verifier logs each failed Batch request's provider
status and message, stops submitting once a finished job is all quota errors,
and warns about unfinished state files from other runs.
