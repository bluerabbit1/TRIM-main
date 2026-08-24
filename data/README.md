# Data placement

The data files are intentionally excluded from Git because they are binary
research artefacts. Obtain the data archive supplied with the anonymous review
artifact, verify its checksum from the artifact page, and extract it here while
preserving this layout:

```
data/
├── collab/processed2-0
├── act/act
├── Aminer/processed_data_128.pt
├── DyMotif-0.4/raw/dymotif_data
└── synthetic/processed2-sythetic2-(P, 0.05, 0.1, 0.0)
```

No data download URL is embedded in this repository, so the double-blind
review process is not linked to an author-controlled account. The loaders and
expected filenames are defined in `data_utils.py`.
