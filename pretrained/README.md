# Pretrained checkpoints

This directory contains two representative TRIM checkpoints: one for Collab
link prediction and one for Aminer node classification. They are state-dict
only files with no user, host, repository, or run-path metadata.

They are supplied for evaluation and inspection; reproducing Tables 5 and 6
trains the models from scratch and does not load a checkpoint. The checkpoint
filenames and configurations are recorded in `manifest.json`.
