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

## Repository layout

```text
Adaptive_AVSR/
├── configs/                 # Pretraining, finetuning and evaluation configurations
├── dataset/                 # Audio/video loading and augmentation
├── models/                  # Baseline, adaptation models and fusion blocks
├── noise_classifier/        # Environmental embedding model implementation
├── pretrained_models/
│   └── noise_classifier/
│       └── epoch3_Final.pth  # Included environmental embedding checkpoint
├── utils/                   # Model loading, data loaders and scheduler setup
├── requirements.txt
├── training.py
└── testing__prepared_files.py
```

## Data and model availability

This repository includes the source code, configurations and the small noise-classifier checkpoint.

- **AVSR checkpoints:** Large model checkpoints are not included due to their storage requirements. They can be shared upon request; please contact the authors.
- **Datasets and prepared evaluation files:** We do not distribute these assets because we do not have the rights to redistribute them. Obtain the datasets from their respective providers under the applicable terms and prepare the files locally.
- **File lists and transcripts:** Experiment-specific file-ID lists, label lists and noise-file lists are not included and must be prepared locally.

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

The configurations use LRS3 for finetuning and evaluation, LRS3/VoxCeleb2 for baseline pretraining, and MUSAN plus speech samples for noise augmentation. For data preprocessing, follow the recipe from [AV-Fusion](https://github.com/SimicCh/AVSR-AV_Fusion_Module_for_Pre-Trained_ASR). Data preparation is external to this repository.

The loader expects the following locally prepared inputs:

- Aligned audio and mouth-region video clips at `<audio_root>/<file_id>.wav` and `<video_root>/<file_id>.mp4`. Use mono audio (resampled to 16 kHz); the configs assume 25 fps video, grayscale 88 × 88 crops and sequences of at most 30 seconds. The loader does not extract mouth regions from raw video.
- File-ID lists with one relative ID per line, without a file extension, and label lists with one transcript per line in matching order. Finetuning, validation and testing require transcripts; the pretraining training split uses `label_list: null`.
- Noise-file lists with one audio path per line, without headers or extra columns, despite their `.tsv` extension. Paths relative to `--data_dir_noise` must not start with `/`. The `single_sidespeaker` lists use extension-free audio IDs.

Update the dataset paths in each YAML configuration, including `<path_to_fids_lists>` and `<path_to_musan>`, to point to your local files. These placeholders are not expanded automatically.

## Pretrained models and checkpoint configuration

For AVSR weights obtained on request or produced by training, set the relevant checkpoint paths in the YAML configuration:

| Config field | Purpose |
| --- | --- |
| `model.model_chkp` | Initialization checkpoint for training; `null` for baseline pretraining |
| `model.model_chkp_test` | AVSR checkpoint selected by the test entry point |
| `model.adaptation_config.checkpoint_path` | Environmental embedding checkpoint for NoiseAdapt |
| `model.adaptation_config.embedding_model_name` | SpeechBrain model for SpeakerAdapt |

NoiseAdapt uses the included `pretrained_models/noise_classifier/epoch3_Final.pth`. The third-party backbones [Whisper Base](https://huggingface.co/openai/whisper-base) and [SpeechBrain X-vector](https://huggingface.co/speechbrain/spkrec-xvect-voxceleb) (SpeakerAdapt) are downloaded separately by the code. The first run needs network access unless the required files have already been cached.

Ensure that the model configuration matches the selected checkpoint. Use [AVSR_baseline__finetune__test.yaml](configs/baseline/AVSR_baseline__finetune__test.yaml) for baseline evaluation.

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

This stage matches encoder embeddings to a clean-audio Whisper target; its decoder loss weight is zero. Set `model.model_chkp` in the desired finetuning YAML to the checkpoint produced by pretraining, or to a compatible checkpoint obtained on request.

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

Logs and checkpoints are generated locally in `training.output_dir`. Epoch checkpoints are named `epoch<N>_Final.pth`; select a checkpoint using validation results and set `model.model_chkp_test` to its actual path. Saved files contain model weights, not optimizer/scheduler state, so loading a checkpoint does not constitute an exact training-state resume.

## Evaluation

[testing__prepared_files.py](testing__prepared_files.py) evaluates locally prepared test audio; neither the prepared files nor a script to generate them is included. Follow the folder naming and test conditions defined in the script, or adapt them to your own evaluation data. `--data_dir_audio` must point to the parent directory of those condition-specific folders. The configured test noise lists must also exist and be nonempty.

Use one GPU for the current evaluation entry point; metrics are not aggregated across processes. The following command requires your own prepared audio, videos, file lists and a matching AVSR checkpoint:

```bash
python testing__prepared_files.py \
    --data_dir_audio /path/to/prepared_test_audio \
    --data_dir_video /path/to/mouth_videos \
    --data_dir_noise /path/to/musan \
    --config_fn configs/baseline/AVSR_baseline__finetune__test.yaml
```

To evaluate an adaptation variant, replace the config with `configs/<variant>/AVSR_<variant>__finetune__test.yaml`. Ensure `model.model_chkp_test` points to that variant's checkpoint.

Evaluation generates logs and a prediction pickle in the configured output directory.
