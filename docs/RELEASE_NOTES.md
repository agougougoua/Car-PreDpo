# Release contents

Kept: CAR algorithm utilities; deterministic perturbations and RVDI scorer;
Stage 1/2 launchers for 3B and 7B; margin materializers and SimPO launcher;
source-level auxiliary trainers; seven focused test modules; exact Stage 2
split IDs; locally modified LLaMA-Factory source and ZeRO-3 configuration.
The framework source is kept together because its imports and the custom
DPO data/loss changes are required by training.

Omitted: model weights/adapters; raw and benchmark data/images; generated
perturbations; scored datasets; prediction dumps; logs; caches; Git history;
server-specific migration/queue/proxy scripts; temporary backups; manuscript
files and internal planning documents. No files in the original project were
removed or changed.

Portability changes apply only to this release copy: project-relative root,
active Python/torchrun, configurable model/adapter/data paths, portable
checkpoint defaults, parameterized RLHF-V converter, active-interpreter tests,
and explicit GPU export. Stage 1 defaults to all 5,701 pairs with no holdout
evaluation; Stage 2 still uses the image-disjoint 5,131/570 split. The 7B
Stage 2 default learning rate is 1e-6, matching the named lambda020_lr1e6
configuration; the older generic launcher used 2e-6. Set LEARNING_RATE=2e-6
explicitly when reproducing that earlier configuration. Core formulas and
trainer source have not been rewritten for this release.

The optional UE-DPO/MDPO/relation modules are retained as compact algorithm
sources. Server-specific baseline orchestration and protocol manifests are
not part of this release. This is a source release, not a bundle of historical
trained checkpoints or an automated rerun of all comparison experiments.
