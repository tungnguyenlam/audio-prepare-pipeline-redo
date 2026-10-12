# Gemini Free Quota & Audio Operational Limits

Empirical measurements conducted on 2026-10-12 using the project's Google AI Studio free tier API key via `scripts/s4-agent/gemini.sh` and direct REST evaluation.

---

## 1. Model Availability & Token Ceilings

Verified via `GET https://generativelanguage.googleapis.com/v1beta/models`:

| Model ID | Input Token Limit | Output Token Limit | Thinking Support | Generation Methods |
|---|---|---|---|---|
| `gemini-3.8-flash` | 1,048,576 (1M) | 65,536 (64k) | Yes (`ThinkingLevel`) | `generateContent`, `batchGenerateContent`, `countTokens`, `createCachedContent` |
| `gemini-3.5-flash-lite` | 1,048,576 (1M) | 65,536 (64k) | Yes | `generateContent`, `batchGenerateContent`, `countTokens`, `createCachedContent` |

---

## 2. Audio Tokenization Scaling

Measured across verified clean anchor audio files (`.data/vicut_bench/anchors/vivos/`):

| Audio Clip | Duration (s) | Audio Tokens | Prompt Text Tokens | Audio Token Rate |
|---|---|---|---|---|
| `vivos_VIVOSDEV01_VIVOSDEV01_R027.wav` | 4.72 s | 118 | 38 (short) / 5,332 (rubric) | **25.0 tokens / s** |
| `vivos_VIVOSSPK30_VIVOSSPK30_118.wav` | 10.78 s | 270 | 38 (short prompt) | **25.04 tokens / s** |

**Tokenization formula:**
$$\text{Audio Tokens} \approx 25 \times \text{Duration (seconds)}$$

- 10-second clip $\approx 250$ audio tokens.
- 60-second clip $\approx 1,500$ audio tokens.
- 1-hour audio file $\approx 90,000$ audio tokens (well within the 1M input token ceiling).

---

## 3. End-to-End Latency & Thinking Token Overhead

Measured with standard synchronous calls:

| Model | Prompt Type | Thinking Tokens | Output Tokens | Latency |
|---|---|---|---|---|
| `gemini-3.5-flash-lite` | Short 1-sentence prompt | 0 | 8 | **1.75 s** |
| `gemini-3.8-flash` (medium) | Short 1-sentence prompt | 110 | 19 | **14.99 s** |
| `gemini-3.8-flash` (low) | Short 1-sentence prompt | 0 | 17 | **14.37 s** |
| `gemini-3.8-flash` (medium) | Full verifier rubric (`prompts/full-tags-prompt.md`) | 4,073 | 105 | **34.14 s** |

**Key observations:**
- `gemini-3.5-flash-lite` is ~8.5× faster than `gemini-3.8-flash` on brief prompts, making it suitable for high-throughput screening.
- With the full verification rubric, `gemini-3.8-flash` thinking tokens (~4,000 tokens) dominate generation time, pushing total per-clip latency to ~34 seconds.

---

## 4. Rate Limits, Bursts, and Transient Errors

1. **Header Inspection:** Google's AI Studio REST API does not emit `x-ratelimit-*` headers in successful HTTP 200 responses.
2. **Burst Sensitivity (503 UNAVAILABLE):** Rapid back-to-back requests (0 s interval) on `gemini-3.8-flash` triggered HTTP 503 `UNAVAILABLE` (*"This model is currently experiencing high demand. Spikes in demand are usually temporary."*).
3. **Recovery & Inter-Call Delay:**
   - A delay of 2–5 seconds between requests reliably resolved the 503 condition.
   - `gemini.py`'s exponential backoff (`RETRYABLE_STATUS = {429, 500, 502, 503, 504}`) successfully handles these transient spikes.

---

## 5. Practical Guidelines for Pipeline Execution

1. **Concurrency:** Always maintain `-c 1` for free-tier Gemini runs.
2. **Throughput Ceiling:** At ~34 seconds per clip with medium reasoning, sequential processing yields $\approx 100$ clips per hour.
3. **Daily Budget:** In accordance with `GOAL.md` §5, keep ad-hoc exploratory audio calls to $\le 20$ calls/day, and trickle larger evaluation batches over days using `--continue`.
