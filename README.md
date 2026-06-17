# Enhancing training data pipelines with Lance and the multimodal lakehouse

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lancedb/tmls-2026-demo/blob/main/notebooks/colab_textvqa_lance.ipynb)

Hands-on materials for the **Toronto Machine Learning Summit (TMLS) 2026** workshop
**“Enhancing training data pipelines with Lance and the multimodal lakehouse,”**
presented by **Prashanth Rao** and **Sarwar Bhuiyan** (LanceDB).

You'll fine-tune a vision-language model end-to-end on a **single free Colab T4** —
with one **Lance** table backing the whole loop, from raw image bytes to a tuned
adapter. The notebook is **Run-All with zero manual steps**: the data is a public
Hugging Face dataset, so there's nothing to configure.

> **The task — TextVQA:** answer a question about an image where the answer is text
> written *in* the picture (e.g. *“what brand is the sugar?” → “domino”*). The model
> has to read the image, not just recognize objects. Dataset: [textvqa.org](https://textvqa.org/).

## What the notebook does

[`notebooks/colab_textvqa_lance.ipynb`](./notebooks/colab_textvqa_lance.ipynb) runs the
whole loop on a free T4 (~5 GB peak VRAM, 4-bit QLoRA):

1. **Download** a curated Lance subset from Hugging Face —
   [`lance-format/textvqa-lance-colab`](https://huggingface.co/datasets/lance-format/textvqa-lance-colab)
   (600 train / 400 val rows).
2. **Explore** it with LanceDB — column distributions and a cross-modal vector-search
   demo over the shipped CLIP features.
3. **Benchmark** Lance-vs-Parquet read throughput for the dataloader.
4. **Fine-tune** `Qwen2.5-VL-3B-Instruct` with QLoRA, reading precomputed columns
   straight off the Lance table.
5. **Evaluate** before/after accuracy on a held-out curated val split
   (text-dense slice: **0.799 → 0.820**, +2.1 pp).

## The key idea: compute the image embeddings once, store them as a column

A vision-language model answers in two stages: an **image encoder** converts the
picture into visual embeddings, then a **language model** reads those embeddings plus
the question and writes the answer. During supervised fine-tuning (SFT) the image
encoder is **frozen** — for a given image it produces the exact same embeddings every
epoch.

So we compute those embeddings **once** and **store them as a column in the Lance
table** (`vision_tower_hiddens`). Adding that column is a cheap, zero-copy append — the
table isn't rewritten (unlike Parquet/Iceberg) and there are no sidecar files to keep
in sync. The training loop then reads the column straight from the table and loads only
the language model — no image decode, no encoder pass per step:

> **~2× train-step throughput (16.1 vs 7.9 samples/s), −1.3 GB VRAM.**

This is the “add a feature as a column, not a rewrite” property of Lance, applied to a
real training run.

## Repo layout

```
tmls-2026-demo/
├── notebooks/
│   ├── colab_textvqa_lance.ipynb   # the workshop notebook (Run-All on a free T4)
│   └── build_notebook.py           # regenerates the notebook (source of truth)
├── vlm/
│   ├── schema.py                   # the one schema-enforced Lance table
│   ├── ingest.py                   # raw TextVQA → Lance
│   ├── slices.py                   # curation (the text-dense slice)
│   ├── slice_experiment.py         # how the slice was chosen, empirically
│   ├── backfill_direct.py          # compute vision_tower_hiddens + SFT tokens (single process)
│   ├── backfill_geneva.py          # the same backfill, distributed via Geneva
│   ├── geneva_udfs.py              # the feature UDFs
│   ├── colab_prepare.py            # prep entry point: bake + push the curated subset
│   ├── dataloader.py               # LanceDB Permutation-API DataLoader
│   ├── train_qwen25vl_lora.py      # QLoRA training from the columns
│   └── eval.py                     # before/after accuracy + EDA helpers
└── pyproject.toml
```

## Rebake the dataset yourself (optional)

The notebook downloads a pre-baked subset, so you don't need to run prep. To build and
host your own slice you need a GPU box and an `HF_TOKEN` with write access — the prep
pipeline lives in [`vlm/colab_prepare.py`](./vlm/colab_prepare.py):

```bash
python -m vlm.colab_prepare \
    --out data/colab --slice text_dense --train-rows 600 --val-rows 400 \
    --hf-repo <your-org>/textvqa-lance-colab --push      # needs HF_TOKEN
```

It ingests raw TextVQA into a Lance table, curates the text-dense slice, backfills the
`vision_tower_hiddens` column, and pushes the result to the Hub.

## At full scale

This is the Colab-sized slice of a pipeline that runs the full 34,602-row corpus on an
H100 — see [`examples/vlm-textvqa`](https://github.com/lancedb/training/tree/vlm-textvqa/examples/vlm-textvqa)
in the LanceDB training repo.

## Links

- Lance (open format): [github.com/lancedb/lance](https://github.com/lancedb/lance)
- LanceDB: [lancedb.com](https://lancedb.com)
- TextVQA dataset: [textvqa.org](https://textvqa.org/)
