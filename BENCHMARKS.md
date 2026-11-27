# Image forensics benchmark log

## Local tampering localization

| Run | Split | Epoch | Val loss | Pixel accuracy | F1 | IoU | Sampled ROC-AUC |
|---|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Ti baseline | 1,500 train / 400 validation | 5/5 | 0.7531 | 0.7687 | 0.2740 | 0.1620 | 0.7081 |
| RTX 5070 Ti long run | 3,984 train / 997 validation, seed 42 | 28/30 (selected) | 0.2168 | 0.9734 | 0.8192 | 0.7119 | 0.9486 |

Sources: `outputs/log.json` and `outputs/v2-30ep-20261127-181840/log.json`. These runs used different random CASIA v2 splits of tampered image/mask pairs. Local threshold metrics are averaged by batch at 0.5 and ROC-AUC uses sampled pixels. Pixel accuracy can be dominated by unedited background. The validation sets are used for checkpoint selection, so report these as development results.

## Published references

| Task | Reference | Published metric |
|---|---|---|
| Tampering localization | [Forensim, WACV 2026, Table 4](https://openaccess.thecvf.com/content/WACV2026/papers/Nandi_Can_Image_Splicing_and_Copy-Move_Forgery_Be_Detected_by_the_WACV_2026_paper.pdf) | CASIA pixel F1 at fixed 0.5: Forensim 81.8%, TruFor 81.8%, SparseViT 82.7% |
| AI-generated image detection | [GenImage, NeurIPS 2023, §4.2](https://papers.neurips.cc/paper_files/paper/2023/file/f4d4a021f9051a6c18183b059117e8b5-Paper-Datasets_and_Benchmarks.pdf) | 66.9% image accuracy averaged over 64 train-generator/test-generator pairings in the reported experiment |

Keep these references contextual until local and reference models are scored with the same test images, resolution policy, masks, threshold, and aggregation. Image-level accuracy and mask F1 measure different tasks.

For a matched comparison, export floating-point probability maps for the validation and test images as `Au/<image filename>.npy` and `Tp/<image filename>.npy`, aligned to the full image. `evaluate_reference.py --split-manifest MANIFEST --maps MAP_DIR --model-name NAME --output REPORT` scores those maps with the same mask metrics and validation calibration as the local evaluator. It checks the probability range and records input hashes. Published reference models have yet to be run on this split.

The evaluator also accepts the `map` field from [official TruFor inference](https://github.com/grip-unina/TruFor/tree/main/TruFor_train_test) in the equivalent `.npz` paths with `--map-format trufor --training-overlap possible`. The authors list revised CASIA 2.0 among their training datasets, so matching the scoring protocol alone does not establish a clean holdout for the released weights. Use an external dataset excluded from both models' training for a clean comparison. The authors restrict their software to informational and nonprofit purposes.

External comparison candidate: the [Columbia Uncompressed Image Splicing Detection Evaluation Dataset](https://www.ee.columbia.edu/ln/dvmm/downloads/authsplcuncmp/) has 183 authentic and 180 spliced images. It covers uncompressed splices and will complement the CASIA benchmark. Credit the Columbia DVMM Laboratory and Hsu and Chang, *Detecting Image Splicing Using Geometry Invariants and Camera Characteristics Consistency*, ICME 2006. The university's download terms specify noncommercial research use. Keep the archives and images local. Verify the mask interpretation and both models' training provenance before scoring.

Columbia local protocol: all 180 three-color masks match the image dimensions. Score thresholded probability maps against the red camera region, report the better of both prediction polarities, and ignore blue/ambiguous boundary pixels. The CASIA development checkpoint scored 0.8666 mean pixel F1 and 0.6151 mask-area image ROC-AUC. An all-positive map scores 0.8611 mean F1 on this protocol, so the F1 is only marginally above that trivial baseline. This is a local comparison protocol and does not reproduce the published boundary morphology. The TruFor exporter uses the official repository at `ae54475df6f41a491d7615100feb19263dec13f7` and the checksum-verified weights. The complete map export is underway.

## Milestones

`main.py --forensic-augmentations` enables training-only JPEG compression (quality 55–95) and low-amplitude, per-channel Gaussian noise as a sensor-noise proxy. These leave mask labels untouched and do not change validation or inference preprocessing. The flag is recorded for frozen-split runs. Test it as a separate ablation and compare clean and recompressed holdouts before adopting it.

Immediate roadmap experiments: `main.py --loss focal-dice` enables binary focal plus Dice loss (default focal weight 0.5, positive alpha 0.75, gamma 2). `--attention` inserts squeeze-and-excitation gates in the decoder. Existing checkpoints and defaults remain compatible. Benchmark the loss-only and attention-only changes separately before evaluating a combined model. The 30-epoch focal-only run is queued behind the source-disjoint baseline, with the same seed, split, image size and batch size. Compare development mask metrics and false-alarm behavior before choosing a candidate for an untouched confirmation set. Loss values are on different scales across objectives.

Run the frozen protocol in a fresh output directory, then evaluate the checkpoint selected by validation loss after training completes:

```powershell
.\.venv\Scripts\python.exe main.py --epochs 30 --seed 42 --split-manifest outputs/benchmarks/casia-source-v2.json --output-dir outputs/casia-source-v2-run
.\.venv\Scripts\python.exe evaluate_casia.py --split-manifest outputs/benchmarks/casia-source-v2.json --run-dir outputs/casia-source-v2-run
```

The evaluator writes `heldout_eval.json` with checkpoint and manifest hashes. It uses a fixed pixel threshold of 0.5, averages localization F1 and IoU over nonempty masks, and selects the image-level mask area threshold on validation by balanced accuracy. The same image threshold is used for test scoring.

Boundary F1 is averaged over nonempty masks at a two-model-pixel tolerance (square neighborhood). It measures contour agreement at the evaluation resolution and should be compared at a fixed image size. Identical, shifted, disjoint and missing-edge cases were verified on fixtures.

The v2 manifest excludes 17 image/mask pairs whose stored dimensions differ (15 train, 2 test). It records every exclusion and the original manifest hash. Source assignments are preserved. The test partition now contains 381 tampered and 1,115 authentic images. Regenerate v2 with `python audit_casia_split.py outputs/benchmarks/casia-source-v1.json outputs/benchmarks/casia-source-v2.json`.

The frozen `outputs/benchmarks/casia-source-v1.json` manifest uses CASIA's filename-encoded source identifiers. It includes 12,418 records with zero shared source IDs across splits. The training, validation, and test partitions contain 4,214/5,206, 384/1,116, and 383/1,115 tampered/authentic images respectively. Connected source groups constrain the balance. This check covers encoded IDs and does not establish visual deduplication under different IDs. Future models must train on this manifest's training partition before using its test partition for a held-out score.

- [x] Record the initial GPU baseline and verify mask/report output on an external image.
- [x] Finish the longer training run and record its selected checkpoint and final metrics.
- [ ] Freeze a source-disjoint evaluation split, include authentic negatives, and measure image-level false positives and pixel F1/IoU. Tune thresholds on validation and lock them before testing. Compare reference models on the same protocol.
- [ ] Train and evaluate an AI-generated image classifier using held-out generators and authentic images, including recompressed and resized copies. Log balanced accuracy, ROC-AUC, and false-positive rates.
- [ ] Deliver a local app with image upload, separately labeled tampering and AI-generation results, an overlay, and model/version information. Verify it on authentic, edited, generated, and mixed examples.

Desktop smoke test passed on Windows with the epoch-28 tampering checkpoint and the low-FPR AI pilot. The image-selection action loaded the synthetic copy-move fixture, displayed the saved overlay report, and showed the AI abstention. This checks the workflow. Detection quality still needs a labeled app test set and a qualified AI checkpoint.

Launch the Windows preview with `powershell -NoProfile -ExecutionPolicy Bypass -File .\run_desktop.ps1`. Paste an image with Ctrl+V or choose an image file; reports are stored under `outputs/desktop`. The app prefers the source-disjoint CASIA checkpoint with its matching held-out calibration and uses CPU inference by default so it can run during CUDA training. Use `-Weights` to select another checkpoint, `-Device cuda` for CUDA inference, and `-AiWeights` to opt into an experimental AI checkpoint. When calibration is absent or does not match the checkpoint, results are labeled unsure.

Tampering inference now surfaces the low-false-alarm validation cutoff when the held-out report matches the checkpoint hash, input size, and pixel threshold. The image decision uses model-resolution mask area, consistent with benchmark scoring. The display overlay may use an upsampled and filtered mask. This calibrates the image-level decision only; pixel heatmap scores remain uncalibrated. Older checkpoints produce an explicitly uncalibrated report.

The source-disjoint checkpoint's validation-selected low-false-alarm operating point measured a 1.08% authentic false-positive rate and 52.49% tampered recall on its 1,496-image CASIA test split (12/1,115 authentic images flagged). This is an evaluation estimate, not a guarantee on other datasets or individual images.

## AI-generated image baseline

### Cross-generator v1

The first larger run uses 1,920 images, with 320 authentic and 320 generated examples in each partition. Training uses ADM, validation uses BigGAN and Midjourney, and test uses Stable Diffusion 1.5. The manifest was checked for unique source paths and normalized RGB hashes, and every listed file exists. The sample comes from the GenImage mirror, revision `435d4ba`, rather than the official paper's full cross-generator grid.

```powershell
.\.venv\Scripts\python.exe prepare_genimage_shards.py dataset/genimage-shards dataset/genimage-cross-generator-v1 --per-class 320 --train ADM --validation BigGAN Midjourney --test stable_diffusion_v_1_5
.\.venv\Scripts\python.exe ai_detector.py dataset/genimage-cross-generator-v1 outputs/ai-genimage-cross-generator-v1 --epochs 5 --device cpu
```

Epoch 1 validation recorded 0.6887 ROC-AUC and 59.22% balanced accuracy at the cutoff selected under a 5% authentic false-positive cap. Generated recall was 23.44%. This is a development checkpoint. Test metrics will be logged after validation selects a final checkpoint.

Completed after five epochs. Validation selected epoch 3 by ROC-AUC (0.7225). On the 640 held-out images the low-false-alarm cutoff yielded 49.69% balanced accuracy, 0.4636 ROC-AUC, 9/320 generated detections and 11/320 authentic false alarms. JPEG quality 70 yielded 49.53% balanced accuracy, 0.4469 ROC-AUC, 11/320 detections and 14/320 false alarms. The baseline fails to generalize to Stable Diffusion 1.5 under this protocol. The validation-only augmented run compares candidates without reopening this test partition. Select any replacement on validation and confirm it on an untouched generator/sample before app promotion.

The next confirmation set is frozen at `dataset/genimage-glide-holdout-v1/manifest.json`. It has 100 GLIDE and 100 authentic images, excluding source paths and normalized RGB hashes from both previous AI experiments. `evaluate_ai_holdout.py` scores a validation-selected checkpoint at its saved threshold and repeats the assessment after JPEG-70 recompression. GLIDE is absent from the larger run's training and validation. Near-duplicate and source-domain effects remain possible.

Perceptual audit: no pairs at Hamming distance 5 or less using 63 low-frequency grayscale DCT bits, comparing the 200 holdout images against 1,968 earlier AI benchmark images and each other. Results are in `outputs/benchmarks/glide-phash-audit.json`. This audit can miss crops and other transformations.

The augmented checkpoint is eligible for GLIDE confirmation only if its selected validation epoch improves on the unaugmented checkpoint in both ROC-AUC and recall while keeping the observed validation false-positive rate at or below 5%. This is a screening rule, not a statistical significance test. The confirmation report will be written to `outputs/benchmarks/ai-augmented-glide-v1.json` if the rule passes.

Augmented epoch 2 (interim) recorded 0.7521 validation ROC-AUC, 64.06% balanced accuracy, 105/320 generated detections and 15/320 authentic false alarms. The loss and operating-point curves for both runs can be regenerated with `python plot_losses.py --log <run-dir>/epochs.jsonl`.

The augmented run finished five epochs and selected epoch 3 (validation ROC-AUC 0.7573, recall 25.63%, false alarms 5.00%). The independent GLIDE confirmation measured 80.00% balanced accuracy, 0.9090 ROC-AUC, 67/100 generated detections and 7/100 authentic false alarms. JPEG-70 yielded 76.50%, 0.8924, 62/100 and 9/100 respectively. The 5% validation target did not carry over. Keep this checkpoint opt-in pending broader generator coverage and better false-alarm performance. Results are in `outputs/benchmarks/ai-augmented-glide-v1.json`.

Evaluators now include raw event counts and 95% Wilson intervals for image-level recall and false-positive rates. As a scale example, 16 false positives out of 320 images corresponds to 5.00% with a 3.10–7.97% interval under independent sampling. Intervals at a validation-tuned cutoff are descriptive. They do not account for threshold selection or correlated images and do not establish performance on future generators.

### Tiny pilot

Operating-point follow-up: `outputs/ai-genimage-tiny-low-fpr/heldout_eval.json` compares balanced-accuracy selection with a validation false-positive cap of 5%. On the same pilot, the balanced cutoff detected all eight test fakes and falsely flagged six of eight real images. The constrained selection found no useful cutoff and abstains in the app. With eight authentic validation images, the measured rate changes in 12.5-point increments. Replicate on a much larger authentic validation set before interpreting the cap as deployment performance.

`ai_detector.py` expects `train`, `validation`, and `test` folders, each with `real/` and `fake/<generator>/`. Use different fake-generator names across partitions and prepare disjoint real images with matched encoding and size distributions. The loader checks generator folder overlap. Independent deduplication of real images and verification of generator labels are still required.

For unpacked [GenImage archives](https://github.com/GenImage-Dataset/GenImage/blob/main/Readme.md), run `python prepare_genimage.py SOURCE dataset/genimage-prepared --train TRAIN_GENERATOR --validation VALIDATION_GENERATOR --test TEST_GENERATOR`, using the actual generator folder names. The pilot balances samples per class, normalizes both classes to the same JPEG size and quality, removes exact duplicates after RGB normalization, and records provenance. Near-duplicates still require a separate audit. Check the dataset license before using it outside research.

```powershell
.\.venv\Scripts\python.exe ai_detector.py dataset/genimage-prepared outputs/ai-baseline
.\.venv\Scripts\python.exe desktop_app.py --weights outputs/best_model.pth --ai-weights outputs/ai-baseline/best_ai_model.pth
```

The trainer logs each epoch to `epochs.jsonl`, selects the checkpoint by validation ROC-AUC, and reports held-out accuracy, balanced accuracy, ROC-AUC, and authentic false positives in `heldout_eval.json`. A generator-disjoint local run is a separate protocol from the published GenImage multi-pairing average.

The first real-data pilot used 48 images from the [GenImage tiny mirror at revision 435d4ba](https://huggingface.co/datasets/naatee/genimage-subset-9/tree/435d4ba/tiny), with eight real and eight generated images per split. The generator was SD 1.4 for training, ADM for validation, and BigGAN for test. Normalized RGB hashes were distinct across all splits. The selected epoch 1 of 5 scored **62.50% balanced accuracy**, **0.8906 ROC-AUC**, and **75.00% authentic false positives** on 16 test images. Validation ROC-AUC was **0.3125**. This pilot verifies the real-data pipeline and exposes an unacceptable false-positive rate. Use a larger multi-generator benchmark before promoting an AI checkpoint to the app default. Artifacts: `outputs/ai-genimage-tiny-pilot/heldout_eval.json` and `epochs.jsonl`.
