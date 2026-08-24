# TRIM: Temporal Risk Variance Minimization for Dynamic-Graph OOD Generalization

This anonymous artifact contains the PyTorch implementation, data-layout
instructions, representative pretrained models, and reproducibility suites for
TRIM. The objective minimizes mean per-timestep forecasting loss plus a
temporal-risk variance penalty. It supports link prediction (Collab, ACT, and
Synthetic-Collab) and node classification (Aminer and Temporal-Motif).

## Anonymous-artifact contents

| Path | Purpose |
| --- | --- |
| `data/README.md` | Expected data files and placement instructions |
| `pretrained/` | Anonymous, state-dict-only representative checkpoints |
| `revision/code/` | Leakage-safe experiment runners and analysis utility |
| `revision/suites/` | Exact seed/configuration grids for Tables 5 and 6 |
| `results/tables_5_6.md` | Values reported in the revision |

## Environment

The experiments were developed for Python 3.10, PyTorch, PyTorch Geometric,
and a CUDA GPU. Create the environment with either:

```bash
conda env create -f environment.yml
conda activate trim
```

or install a PyTorch build appropriate to the local CUDA driver and then run
`pip install -r requirements.txt`. PyTorch Geometric extension wheels must be
compatible with the installed PyTorch/CUDA pair; consult its installation guide
when installing by pip.

## Data

Obtain the data archive from the anonymous artifact and extract it under
`data/` as described in [data/README.md](data/README.md). Data are excluded
from Git to keep the source repository lightweight and to avoid republishing
third-party data without their original terms.

## Run a standard TRIM experiment

```bash
python main.py --dataset collab --device_id 0
python main.py --dataset Aminer --device_id 0
```

The legacy entry point above matches the original experiment interface. For
revision experiments, use the runners below: they make the objective choice,
variance estimator, seed, split description, and checkpoint selection explicit.

## Reproduce Tables 5 and 6

Run from the repository root after placing the data. These are full training
runs and require a CUDA GPU; set `CUDA_VISIBLE_DEVICES` if the desired device
is not GPU 0.

```bash
bash scripts/run_table5.sh
bash scripts/run_table6.sh
```

Table 5 runs 40 jobs: ERM and TRIM, each under the proposed ST-Attn. encoder
and an independent GCN-GRU encoder, over five seeds on two datasets. The
ST-Attn. control uses λ=1.0 for Collab and λ=10.0 for Aminer, matching the
reported experiment. Table 6 runs 30 jobs: five depth settings over three
seeds on two datasets. Each runner writes per-run summaries, checkpoints, and
an append-only manifest below `outputs/`. To aggregate a completed directory:

```bash
python revision/code/analyze_revision_results.py outputs/table5/st_attn
python revision/code/analyze_revision_results.py outputs/table6
```

The recorded values are in [results/tables_5_6.md](results/tables_5_6.md).

## Pretrained checkpoints

`pretrained/` holds a representative Collab and Aminer TRIM checkpoint. Each
file is deliberately restricted to model parameters and an anonymous
configuration. Their manifest gives the architecture required to reconstruct
the model before loading its `model_state_dict`.

## Double-blind verification

Before pushing to a new anonymous repository, run:

```bash
bash scripts/verify_anonymity.sh
git init
git add .
git commit -m 'Anonymous review artifact'
```

Configure a fresh anonymous local Git identity before committing. Do not reuse
the original repository, remote, commits, releases, or personal account. See
[docs/ANONYMITY_CHECKLIST.md](docs/ANONYMITY_CHECKLIST.md).
