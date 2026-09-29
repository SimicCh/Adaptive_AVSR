# Adaptive AVSR

Code for **Adaptive AVSR: Integrating Speaker and Environmental Embeddings for Robust Audio-Visual Speech Recognition**, INTERSPEECH 2026.

This repository implements a Whisper-based audio-visual speech recognition (AVSR) system and adaptation variants that incorporate speaker or environmental embeddings. The variants explore cross-attention, gated weighting and prefix-based integration to condition recognition on the speaker or acoustic environment.

## Models

All variants use `openai/whisper-base` together with an audio-visual fusion network. Speaker embeddings are extracted from the input waveform using the SpeechBrain X-vector model. Environmental embeddings are extracted from the input mel spectrogram using the supplied noise classifier.

| Variant | Additional embedding | Integration | Config directory |
| --- | --- | --- | --- |
| Baseline | None | Audio-visual fusion | [baseline](configs/baseline/) |
| SpeakerAdapt cross-attention | Speaker | Cross-attention (post-encoder) | [SpeakerAdapt_cross_attention](configs/SpeakerAdapt_cross_attention/) |
| SpeakerAdapt gated weighting | Speaker | Feature gating (post-encoder) | [SpeakerAdapt_gated_weighting](configs/SpeakerAdapt_gated_weighting/) |
| SpeakerAdapt prefix | Speaker | Prefix (post-encoder) | [SpeakerAdapt_prefix](configs/SpeakerAdapt_prefix/) |
| NoiseAdapt cross-attention | Environment | Cross-attention (Fusion Module) | [NoiseAdapt_cross_attention](configs/NoiseAdapt_cross_attention/) |
| NoiseAdapt gated weighting | Environment | Feature gating (Fusion Module) | [NoiseAdapt_gated_weighting](configs/NoiseAdapt_gated_weighting/) |
| NoiseAdapt prefix | Environment | Prefix (Fusion Module) | [NoiseAdapt_prefix](configs/NoiseAdapt_prefix/) |

Each variant has `AVSR_<variant>__finetune.yaml` and `AVSR_<variant>__finetune__test.yaml` configurations. Baseline pretraining additionally uses [AVSR_baseline__pretrain.yaml](configs/baseline/AVSR_baseline__pretrain.yaml).

## Repository layout and available assets

```text
Adaptive_AVSR/
├── configs/                 # Pretraining, finetuning, and evaluation configurations
├── dataset/                 # Audio/video loading and augmentation
├── models/                  # Baseline, adaptation models, and fusion blocks
├── noise_classifier/        # Environmental embedding model implementation
├── pretrained_models/
│   └── noise_classifier/
│       └── epoch3_Final.pth  # Included environmental embedding checkpoint
├── utils/                   # Model loading, data loaders, and scheduler setup
├── requirements.txt
├── training.py
└── testing__prepared_files.py
```

The Git checkout includes the source code, configurations, and noise-classifier checkpoint. Datasets, file lists, prepared test audio, and AVSR checkpoints under `results/` must be supplied separately. Data preprocessing and generation of the prepared noisy test sets are not included in this checkout.

The following local files are excluded by [.gitignore](.gitignore): `00_data/`, `container_env/`, `test_baseline.*` (including the local Slurm script), and all `*.pth` files under `results/`. The commands below therefore use the Python entry points directly.

## Installation

Use Linux with a CUDA-capable NVIDIA GPU and Python 3.10. The scripts use FP16 through Accelerate and include CUDA-specific operations. The pinned packages include PyTorch/Torchaudio 2.5.1, Torchvision 0.20.1, Transformers 4.47.0, Accelerate 1.2.1, and SpeechBrain 1.0.2.

```bash
git clone https://github.com/SimicCh/Adaptive_AVSR.git
cd Adaptive_AVSR

conda create -n adaptive-avsr python=3.10 -y
conda activate adaptive-avsr
```

On Debian/Ubuntu, install the native build tools and audio/video libraries before installing the Python packages. On a managed cluster, provide these libraries through the container or system environment.

```bash
sudo apt-get update
sudo apt-get install -y build-essential libsndfile1 \
    libsm6 libxext6 libxrender1 libglib2.0-0 libxcb1 libgl1

python -m pip install -r requirements.txt
```

The current root `requirements.txt` contains both legacy `PySoundFile` and `soundfile`. They install the same Python module. After installation, remove the legacy package and restore the pinned SoundFile package:

```bash
python -m pip uninstall -y PySoundFile
python -m pip install --force-reinstall --no-deps soundfile==0.12.1
```

SoundFile requires libsndfile; see its [installation documentation](https://python-soundfile.readthedocs.io/en/0.13.1/#installation). Keep Python 3.10 for these instructions: the pinned NumPy 1.23.5 does not support Python 3.12 ([NumPy release notes](https://numpy.org/doc/stable/release/1.23.5-notes.html)).

Check the environment before staging the datasets or submitting a long job:

```bash
python - <<'PY'
import torch
import torchaudio
import soundfile
from utils.load_save_models import prepare_model

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("SoundFile:", soundfile.__version__)
print("Audio backends:", torchaudio.list_audio_backends())
assert torch.cuda.is_available(), "Run this check on a GPU node."
assert "soundfile" in torchaudio.list_audio_backends(), "Check SoundFile/libsndfile."
PY
```

Run all subsequent commands from the repository root; relative paths in YAML files are resolved against the working directory.

## Data setup

The configurations use LRS3 for finetuning and evaluation, a combined LRS3/VoxCeleb2 file list for baseline pretraining, and MUSAN plus speech samples for training-time noise augmentation. Obtain and prepare these datasets separately.

### Audio, video, and transcripts

Provide aligned audio/video clips with matching file IDs. For an ID such as `test/example_speaker/00001`, the dataset loader opens:

```text
<audio_root>/test/example_speaker/00001.wav
<video_root>/test/example_speaker/00001.mp4
```

Use mono audio and prepared mouth-region videos. Audio is resampled to 16 kHz. The supplied configs assume 25 fps video, grayscale 88 × 88 crops, and sequences of at most 30 seconds. The loader crops/normalizes frames; it does not perform face detection or mouth-region extraction from raw video.

File-ID lists contain one relative ID per line, without `.wav` or `.mp4`. Label lists contain one transcript per line in exactly the same order. Finetuning, validation, and testing require transcripts; the pretraining training split uses `label_list: null`.

The configurations reference:

- `LRS3_Vox2_fids_train.list` for baseline pretraining.
- `LRS3_fids_train.list`, `LRS3_fids_valid.list`, and `LRS3_fids_test.list`.
- `LRS3_labels_train.list`, `LRS3_labels_valid.list`, and `LRS3_labels_test.list`.
- MUSAN noise lists under `tsv/{babble_musan,music,noise}/{train,valid,test}.tsv`.

Despite the `.tsv` extension, each noise-list line is read as one audio filename, with no header or additional columns. Paths relative to `--data_dir_noise` must not start with `/`; a leading `/` makes the path absolute and bypasses the supplied noise root. `single_sidespeaker` lists use the same extension-free IDs as the main audio lists.

Replace `<path_to_fids_lists>` and `<path_to_musan>` in each configuration you intend to use. The canonical baseline test config already uses `./00_data/file_lists/` and `./00_data/musan/`; adjust those paths if your layout differs. These placeholders are literal strings, not automatically expanded environment variables.

### Prepared evaluation audio

[testing__prepared_files.py](testing__prepared_files.py) expects pre-generated noisy WAV files. Its `--data_dir_audio` argument is the parent directory of the condition-specific folders:

```text
<prepared_audio_root>/
├── wav_files_LRS3_noiseMixture_clean/
│   └── test/example_speaker/00001.wav
├── wav_files_LRS3_noiseMixture_SNR-5/
│   └── test/example_speaker/00001.wav
├── wav_files_musan_music_SNR0/
│   └── test/example_speaker/00001.wav
└── ...
```

The current script evaluates six categories: `LRS3_noiseMixture`, `musan_babble`, `LRS3_babble`, `musan_music`, `musan_noise`, and `LRS3_sidespeaker`. Its SNR list is `[-5, 0, 5, 10, 50]` dB. Supply `wav_files_<category>_SNR<snr>` for every combination, plus `wav_files_LRS3_noiseMixture_clean`: 31 condition folders in total, each containing every test file ID.

These categories and SNRs are set in the Python script. During evaluation, `SNR_test` is set to 50 to disable additional on-the-fly noise mixing; it does not change the SNR of the prepared audio. The shared dataset constructor still reads the configured test noise lists, so those lists must exist and be nonempty. Test videos are shared across all audio conditions.

## Checkpoints and pretrained backbones

The supplied configs expect this local layout for AVSR weights:

```text
results/
├── baseline/checkpoint/
│   ├── pretrain_epoch3_Final.pth
│   └── best_valid_epoch_finetune.pth
└── <variant>/checkpoint/
    └── best_valid_epoch_finetune.pth
```

Here, `<variant>` is any of the six adaptation names in the model table. These AVSR weights are excluded from Git and are not obtained by cloning the repository. Use separately obtained checkpoints or train the models below, then update the checkpoint fields to the actual output paths. No AVSR-checkpoint download link is currently supplied in this README.

| Config field | Purpose |
| --- | --- |
| `model.model_chkp` | Initialization checkpoint for training; `null` for baseline pretraining |
| `model.model_chkp_test` | AVSR checkpoint selected by the test entry point |
| `model.adaptation_config.checkpoint_path` | Environmental embedding checkpoint for NoiseAdapt |
| `model.adaptation_config.embedding_model_name` | SpeechBrain model for SpeakerAdapt |

NoiseAdapt uses the included `pretrained_models/noise_classifier/epoch3_Final.pth`. The code also loads [Whisper Base](https://huggingface.co/openai/whisper-base), and SpeakerAdapt loads [SpeechBrain X-vector](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb). The first run needs network access unless the required files have already been cached. Set `HF_HOME` to your chosen cache location if necessary; use `HF_HUB_OFFLINE=1` only after preparing the cache. SpeakerAdapt also creates a local `speechbrain/spkrec-xvect-voxceleb/` directory.

Match the model architecture to the selected checkpoint. The canonical configs use 12 fusion layers. Loading uses `strict=False`, so incompatible extra keys can be ignored without stopping the run. Use [AVSR_baseline__finetune__test.yaml](configs/baseline/AVSR_baseline__finetune__test.yaml) for baseline evaluation; alternative copies may use different architecture settings.

## Training

The commands below assume that the corresponding YAML data-list paths have been configured. Replace the example audio, video, and noise roots with your own paths. The legacy `--data_dir` argument is unused; provide the three specific roots shown here.

### Baseline pretraining

For a fresh run, set the following fields in [AVSR_baseline__pretrain.yaml](configs/baseline/AVSR_baseline__pretrain.yaml):

```yaml
training:
  start_epoch: 0
  skip_iterations: null
model:
  model_chkp: null
```

Edit these fields within the existing sections. For a fresh run, leave `skip_iterations` unset (`null`); use a nonzero value only when deliberately skipping batches.

```bash
python training.py \
    --data_dir_audio /path/to/clean_audio \
    --data_dir_video /path/to/mouth_videos \
    --data_dir_noise /path/to/musan \
    --config_fn configs/baseline/AVSR_baseline__pretrain.yaml
```

This stage matches encoder embeddings to a clean-audio Whisper target; its decoder loss weight is zero. The default run saves `results/baseline/pretrain/epoch3_Final.pth`. To use that output for finetuning, set `model.model_chkp` in the desired finetuning YAML to this path, or place it at the default `results/baseline/checkpoint/pretrain_epoch3_Final.pth` location.

### Baseline and adaptive finetuning

For the baseline:

```bash
python training.py \
    --data_dir_audio /path/to/clean_audio \
    --data_dir_video /path/to/mouth_videos \
    --data_dir_noise /path/to/musan \
    --config_fn configs/baseline/AVSR_baseline__finetune.yaml
```

For any adaptation variant, use its finetuning config, for example:

```bash
python training.py \
    --data_dir_audio /path/to/clean_audio \
    --data_dir_video /path/to/mouth_videos \
    --data_dir_noise /path/to/musan \
    --config_fn configs/NoiseAdapt_cross_attention/AVSR_NoiseAdapt_cross_attention__finetune.yaml
```

The adaptation configs initialize from the baseline pretraining checkpoint. Finetuning combines encoder matching and decoder losses. Review `num_epochs`, learning rate, and the encoder/decoder freezing flags for your intended experiment; freezing settings differ across variants. The supplied finetuning configs currently specify five epochs.

### Multiple GPUs and training outputs

Training uses Hugging Face Accelerate. For a single machine with four allocated GPUs, replace `python training.py` with:

```bash
accelerate launch --multi_gpu --num_processes 4 --num_machines 1 \
    --mixed_precision fp16 training.py \
    --data_dir_audio /path/to/clean_audio \
    --data_dir_video /path/to/mouth_videos \
    --data_dir_noise /path/to/musan \
    --config_fn configs/baseline/AVSR_baseline__finetune.yaml
```

See the [Accelerate launching guide](https://huggingface.co/docs/accelerate/v1.2.1/en/basic_tutorials/launch) for other execution environments. The effective batch size is `batch_size_train × number_of_processes × gradient_accumulation_steps`; for example, `4 × 4 × 32 = 512`. Adjust the batch size to available GPU memory and keep `num_worker` positive with the current data loaders.

Logs and checkpoints are written to `training.output_dir`. Epoch checkpoints are named `epoch<N>_Final.pth`. Training does not automatically create `best_valid_epoch_finetune.pth`; select a checkpoint using validation results and set the test config accordingly. Saved files contain model weights, not optimizer/scheduler state, so changing `start_epoch` and loading a checkpoint does not constitute an exact training-state resume.

## Evaluation

Use one GPU for the current evaluation entry point. It rebuilds condition-specific data loaders outside Accelerate's preparation step and does not aggregate metrics across processes.

After supplying the prepared audio, videos, file lists, and matching AVSR checkpoint:

```bash
python testing__prepared_files.py \
    --data_dir_audio /path/to/prepared_test_audio \
    --data_dir_video /path/to/mouth_videos \
    --data_dir_noise /path/to/musan \
    --config_fn configs/baseline/AVSR_baseline__finetune__test.yaml
```

To evaluate an adaptation variant, replace the config with `configs/<variant>/AVSR_<variant>__finetune__test.yaml`. Ensure `model.model_chkp_test` points to that variant's checkpoint.

The baseline writes `results/baseline/test/03_test.log` and `results/baseline/test/03_test.pkl`. Other variants use their own output directories. The pickle groups results by condition (`clean`, `musan_music_0dB`, etc.), then by file ID, with `prediction`, `label`, and `wer` fields for each utterance.



