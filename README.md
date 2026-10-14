# CISPA Hackathon Championship: Vienna Qualifier

Code of team **!!1337** at the Vienna round of the CISPA European Cybersecurity and AI
Hackathon Championship (University of Vienna, December 13–14, 2025). The team was
**one of the three winning teams** and qualified for the
[Grand Finale](https://github.com/florianzainzinger1/cispa-hackathon-grand-finale-2026),
where it finished 4th overall.

## About the event

The championship is a Europe-wide student competition organised by the CISPA Helmholtz
Center for Information Security (SprintML Lab). In each regional round, teams of up to
four Bachelor's and Master's students solve challenges in trustworthy AI and
machine-learning security within 24 hours. Jobs run on the JURECA supercomputer
(Jülich Supercomputing Centre), and every submission is scored on a hidden leaderboard.
The Vienna round had almost 70 participants from Austria and Germany.

## Tasks and approaches

### 1. Dataset inference ([`task1_dataset_inference/`](task1_dataset_inference))

Decide for 1,000 subsets of 100 images each whether they were part of the training data of a given
ResNet-18 classifier (metric: TPR at 5% FPR).

Mean-based membership signals were too weak, because dropout regularised the model. The final attack
uses **extreme-value features per subset** instead: the maximum NLL, the minimum logit
margin and the 10th-percentile confidence. The hypothesis is that a model trained on a subset
also handles that subset's hardest images better. This raised the leaderboard score from
0.141 to 0.200.

### 2. LLM text attribution ([`task2_text_attribution/`](task2_text_attribution))

Predict which LLM generated a given text (about 85k training samples).

- TF-IDF baseline, then fine-tuned DeBERTa-v3 (base and large) and ModernBERT-large
  classifiers trained on A100 GPUs via Slurm
- Pseudo-labelling of the test set, test-time augmentation and a length-aware ensemble
  of the models

### 3. Image watermark removal ([`task3_watermark_removal/`](task3_watermark_removal))

Remove BitMark watermarks from 100 AI-generated images (512×512) while keeping the
MSE to the original at or below 0.08 (metric: 1 − TPR at 1% FPR of the watermark detector).

The solution is a modular attack pipeline: inpainting-based perturbation ("Fast-Sand"),
re-rendering with 2D Gaussian splatting (Image-GS), diffusion img2img regeneration,
small rotations, Gaussian blur and histogram-based colour correction.

## Repository layout

```
task1_dataset_inference/   feature-based membership inference attack
task2_text_attribution/    transformer classifiers, training + inference scripts, Slurm jobs
task3_watermark_removal/   attack pipeline (src/attacks/*), runner scripts
```

The organisers' datasets, task descriptions, model checkpoints and submission files are
not included. The code documents the approaches and needs the original challenge data to run.
Some of the internal notes in the task folders are written in German.

## Reference

Official event recap: https://cispa.de/en/cispa-hackathon-championship-vienna
