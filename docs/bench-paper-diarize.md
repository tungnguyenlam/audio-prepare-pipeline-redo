I checked the current `src/diarization/` on `main`. You currently have **4 actual diarization systems**; the `*WorkerDiarizer` files are only execution wrappers, not additional models. ([GitHub][1])

### Public DER benchmarks for models currently in the repo

**DER %, lower is better.** `-` = I could not find a published number for that exact system/benchmark.

| Current model in repo                         | AISHELL-4 | AliMeeting | AMI IHM |  AMI SDM | AVA-AVD |           CALLHOME p2 |   DIHARD 3 | MSDWild | REPERE | VoxConverse | Ego4D | RAMC | CH109 |
| --------------------------------------------- | --------: | ---------: | ------: | -------: | ------: | --------------------: | ---------: | ------: | -----: | ----------: | ----: | ---: | ----: |
| **Pyannote Community-1**                      |  **11.7** |       20.3 |    17.0 | **19.9** |    44.6 |                  26.7 |       20.2 |    22.8 |    8.9 |    **11.2** |  46.8 | 20.8 |     - |
| **3D-Speaker (FSMN + CAM++)**                 | **10.30** |  **19.73** |       - |    21.76 |       - |                     - |          - |       - |      - |       11.75 |     - |    - |     - |
| **NeMo Sortformer `diar_sortformer_4spk-v1`** |         - |          - |       - |        - |       - | 6.49 / 10.01 / 14.14* | **16.28†** |       - |      - |           - |     - |    - |  6.27 |
| **NeMo Clustering (MarbleNet + TitaNet)**     |         - |          - |       - |        - |       - |                     - |          - |       - |      - |           - |     - |    - |     - |

Pyannote's current code uses `pyannote/speaker-diarization-community-1`. ([GitHub][2]) Its official model card reports the full benchmark row above under **fully automatic processing, 0 s collar, overlap included**, making that row internally consistent. ([Hugging Face][3])

Your 3D-Speaker implementation uses the upstream **FSMN VAD + CAM++ embedding + spectral clustering** setup, with CAM++ model `speech_campplus_sv_zh_en_16k-common_advanced`. ([GitHub][4]) The upstream 3D-Speaker repo publishes DER for AISHELL-4, AliMeeting, AMI-SDM and VoxConverse. ([GitHub][5])

Your Sortformer is exactly NVIDIA's offline **`nvidia/diar_sortformer_4spk-v1`** checkpoint. ([GitHub][6]) NVIDIA publishes 16.28 DER on its DIHARD3-Eval subset, 6.27 on CH109, and separate CALLHOME part-2 numbers for 2-, 3-, and 4-speaker recordings. ([Hugging Face][7])

* CALLHOME Sortformer = **6.49 / 10.01 / 14.14** for 2 / 3 / 4 speakers respectively.

† Important: Sortformer's **16.28 DIHARD3** is specifically `DIHARD3-Eval` restricted to recordings with ≤4 speakers; Pyannote's **20.2** is reported as `DIHARD 3 (full)`. So **do not conclude 16.28 < 20.2 means Sortformer is definitively better**. The protocols aren't identical. ([Hugging Face][7])

### Why the NeMo clustering row is blank

Your implementation is:

```text
MarbleNet automatic VAD
→ TitaNet-large embeddings
→ spectral clustering
```

which the code confirms directly. ([GitHub][8])

NVIDIA does publish TitaNet clustering benchmarks, but they use **oracle VAD**, meaning perfect ground-truth speech boundaries rather than MarbleNet predictions:

| Published TitaNet clustering condition | AMI Lapel | AMI MixHeadset | CH109 | NIST SRE 2000 |
| -------------------------------------- | --------: | -------------: | ----: | ------------: |
| Oracle VAD, known speaker count        |      1.28 |           1.07 |  0.56 |          5.62 |
| Oracle VAD, unknown speaker count      |      1.28 |           1.40 |  0.88 |          4.33 |

Those numbers look extremely good because VAD errors are removed entirely, so I intentionally **did not put them into the main table as your current model's results**. NVIDIA explicitly says those results use oracle VAD. ([GitHub][9])

### What the table actually tells us

The only genuinely useful head-to-head overlap at the moment is:

| Benchmark   | Pyannote Community-1 | 3D-Speaker |                   Difference |
| ----------- | -------------------: | ---------: | ---------------------------: |
| AISHELL-4   |                 11.7 |  **10.30** | 3D-Speaker better by 1.40 pp |
| AliMeeting  |                 20.3 |  **19.73** | 3D-Speaker better by 0.57 pp |
| AMI SDM     |             **19.9** |      21.76 |   Pyannote better by 1.86 pp |
| VoxConverse |             **11.2** |      11.75 |   Pyannote better by 0.55 pp |

So **Pyannote and 3D-Speaker are actually quite close overall** on common public benchmarks. 3D-Speaker looks better on the Chinese meeting sets; Pyannote slightly better on AMI-SDM and VoxConverse. ([GitHub][5])

And critically, **none of these exact current checkpoints has a published ViYT-Diar number that I could verify**, so the Vietnamese column would currently be:

| Model                    | ViYT-Diar DER |
| ------------------------ | ------------: |
| Pyannote Community-1     |             - |
| 3D-Speaker               |             - |
| Sortformer 4spk-v1       |             - |
| NeMo MarbleNet + TitaNet |             - |

That is therefore probably the benchmark **you should run yourself next**, because it will be much more informative for this project than trying to infer Vietnamese performance from AMI or DIHARD. ViYT-Diar gives 100 manually annotated Vietnamese YouTube files specifically for diarization evaluation. ([Hugging Face][10])

[1]: https://github.com/tungnguyenlam/audio-prepare-pipeline-redo/tree/main/src/diarization "audio-prepare-pipeline-redo/src/diarization at main · tungnguyenlam/audio-prepare-pipeline-redo · GitHub"
[2]: https://github.com/tungnguyenlam/audio-prepare-pipeline-redo/blob/main/src/diarization/PyannoteDiarizer.py "audio-prepare-pipeline-redo/src/diarization/PyannoteDiarizer.py at main · tungnguyenlam/audio-prepare-pipeline-redo · GitHub"
[3]: https://huggingface.co/pyannote/speaker-diarization-community-1?utm_source=chatgpt.com "pyannote/speaker-diarization-community-1 · Hugging Face"
[4]: https://github.com/tungnguyenlam/audio-prepare-pipeline-redo/blob/main/src/diarization/ThreeDSpeakerDiarizer.py "audio-prepare-pipeline-redo/src/diarization/ThreeDSpeakerDiarizer.py at main · tungnguyenlam/audio-prepare-pipeline-redo · GitHub"
[5]: https://github.com/modelscope/3D-Speaker?utm_source=chatgpt.com "GitHub - modelscope/3D-Speaker: A Repository for Single- and Multi-modal Speaker Verification, Speaker Recognition and Speaker Diarization · GitHub"
[6]: https://github.com/tungnguyenlam/audio-prepare-pipeline-redo/blob/main/src/diarization/SortformerDiarizer.py "audio-prepare-pipeline-redo/src/diarization/SortformerDiarizer.py at main · tungnguyenlam/audio-prepare-pipeline-redo · GitHub"
[7]: https://huggingface.co/nvidia/diar_sortformer_4spk-v1?utm_source=chatgpt.com "nvidia/diar_sortformer_4spk-v1 · Hugging Face"
[8]: https://github.com/tungnguyenlam/audio-prepare-pipeline-redo/blob/main/src/diarization/ClusteringDiarizer.py "audio-prepare-pipeline-redo/src/diarization/ClusteringDiarizer.py at main · tungnguyenlam/audio-prepare-pipeline-redo · GitHub"
[9]: https://github.com/NVIDIA-NeMo/Speech/blob/main/examples/speaker_tasks/diarization/README.md?utm_source=chatgpt.com "Speech/examples/speaker_tasks/diarization/README.md at main · NVIDIA-NeMo/Speech · GitHub"
[10]: https://huggingface.co/datasets/tuanduy1612/ViYT-Diar?utm_source=chatgpt.com "tuanduy1612/ViYT-Diar · Datasets at Hugging Face"
s