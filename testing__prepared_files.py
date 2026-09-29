
import logging
import os
import torch
import random
from datetime import timedelta
import numpy as np
import pickle as pkl
from jiwer import wer  # Word Error Rate (WER) aus der jiwer-Bibliothek
from tqdm import tqdm
import time
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.cuda.amp import autocast, GradScaler
from safetensors.torch import load_file
import yaml
from peft import get_peft_model, LoraConfig, TaskType
from accelerate import Accelerator
import re
import glob

from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from utils.load_save_models import prepare_model, save_model
from utils.load_datasets_scheduler import load_dataloader_train, load_dataloader_eval, load_dataloader_test, prepare_scheduler



def setup_logger(log_dir, log_filename="training.log"):
    """
    Richtet den Logger ein, der sowohl in die Konsole als auch in eine Datei schreibt.
    """
    # Erstelle das Verzeichnis, falls es nicht existiert
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)

    # Eigenen Logger erstellen
    logger = logging.getLogger("TrainingLogger")
    logger.setLevel(logging.INFO)  # Log-Level setzen

    # Datei-Handler hinzufügen
    file_handler = logging.FileHandler(log_filepath)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Stream-Handler (Konsole) hinzufügen
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(stream_formatter)

    # Handler zum Logger hinzufügen
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.propagate = False  # Verhindert doppelte Logs im Root-Logger

    logger.info("Logger initialized.")

    return logger

def set_seeds(seed: int, logger):
    """
    Setzt alle notwendigen Seeds für deterministische Ergebnisse.
    """
    # Seed für Python
    random.seed(seed)
    
    # Seed für NumPy
    np.random.seed(seed)
    
    # Seed für PyTorch
    torch.manual_seed(seed)
    
    # Seed für CUDA (falls GPU genutzt wird)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Für mehrere GPUs
    
    # Deterministisches Verhalten bei bestimmten Operationen (optional, kann Leistungseinbußen verursachen)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Seeds are set to {seed}!")

def get_base_model(model):
    if isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config



def train_one_epoch(
        model, 
        target_model, 
        dataloader, 
        eval_loader, 
        optimizer, 
        lr_scheduler, 
        whisper_processor, 
        accelerator, 
        device, 
        logger, 
        config,
        logging_freq=125, 
        eval_freq=4000):
    
    # Set model modes
    model.train()
    target_model.eval()

    # Start training
    loss_list = list()
    train_t1 = time.time()
    for iteration, batch in enumerate(dataloader):

        if 'skip_iterations' in config['training']:
            if config['training']['skip_iterations'] is not None:
                iteration += config['training']['skip_iterations']
    
        mel_specs_clean = batch['mel_specs_clean'].to(device)
        mel_specs_noisy = batch['mel_specs_noisy'].to(device)
        attentions_mask_clean = None
        input_videos = batch['video_frames'].to(device)
        video_lens = batch['video_lens'].to(device)
        file_ids = batch['file_ids']
        labels = batch['labels']


        if 'doMasking' in config['model']:
            if config['model']['doMasking']:
                attentions_mask_clean = batch['attentions_mask_clean'].to(device)

        # Prepare environment embeddings if available
        if config['dataset']['type'] == 'AVSR_wSpkEmb':
            embeddings = batch['embeddings'].to(device)
        else:
            embeddings = None
    
        # Prepare labels if label text is available for every sample
        if not any(label is None for label in labels):
            labels = [l.lower() for l in labels]
            labels = whisper_processor(text=labels, return_tensors="pt", padding=True).input_ids.to(device)

        optimizer.zero_grad()
        #with autocast():
        # Target encoder embeddings
        with torch.no_grad():
            encoder_targets = target_model.model.encoder(mel_specs_clean)['last_hidden_state']
        
        if config['training']['dec_embed_loss_weight'] > 0.0:
            mode = 'full'
        else:
            mode = 'encoder_embedding'

        out = model.forward(
                mode=mode,
                calc_loss=True,
                melspec_inputs=mel_specs_noisy,
                embeds=embeddings,
                video_inputs=input_videos,
                vid_lens=video_lens,
                attention_mask=attentions_mask_clean,
                melspec_labels=mel_specs_clean,
                melspec_loss_weight=config['training']['melspec_loss_weight'],
                enc_embed_labels=encoder_targets,
                enc_embed_loss_weight=config['training']['enc_embed_loss_weight'],
                label_tokens=labels,
                dec_embed_loss_weight=config['training']['dec_embed_loss_weight']
                )
        
        loss = out.loss

        # Backward pass and optimizer step
        accelerator.backward(loss)
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        loss_list.append(loss.item())

        # Logging
        if (iteration + 1) % logging_freq == 0:
            if accelerator.is_main_process:
                elapsed_time = time.time() - train_t1
                avg_loss_last = sum(loss_list[-logging_freq:]) / logging_freq  # Durchschnittlicher Loss der letzten 500 Iterationen
                total_iters = len(dataloader)  # Gesamtanzahl der Iterationen

                if 'skip_iterations' in config['training']:
                    if config['training']['skip_iterations'] is not None:
                        total_iters += config['training']['skip_iterations']

                remaining_time = (elapsed_time / (iteration + 1)) * (total_iters - (iteration + 1))
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Iter {iteration + 1}/{total_iters} | "
                    f"Elapsed: {elapsed_time:.2f}s | "
                    f"Remaining: {remaining_time:.2f}s | "
                    f"Avg Loss (last {logging_freq}): {avg_loss_last:.4f} | "
                    f"Curr LR: {current_lr:.8f}", flush=True)
        
        # Inter-training evaluation
        if (iteration + 1) % eval_freq == 0:

            # accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logger.info(f"Save/Eval after {(iteration+1)} iterations ...")
            # eval_losses = evaluate(model, target_model, eval_loader, device, processor, accelerator, train_logger)
            # model.train()
            if accelerator.is_main_process:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Curr LR: {current_lr:.8f}")

            if accelerator.is_main_process:
                # print(model.module.video_embedding.out_CNN[0].weight.squeeze(), flush=True)
                epoch = config['epoch']
                out_fn = os.path.join(config['training']['output_dir'], f'epoch{epoch+1}_iter{(iteration+1)}.pth')
                out_dir = os.path.dirname(out_fn)
                os.makedirs(out_dir, exist_ok=True)
                save_model(get_base_model(model), out_fn)
                # get_base_model(model).save_pretrained(out_fn)
                train_logger.info(f"Model saved at {out_fn}!")
            accelerator.wait_for_everyone()

    duration = time.time() - train_t1
    if accelerator.is_main_process:
        logger.info(f"Train duration: {duration:.2f}s Loss: {np.mean(loss_list):.4f}")
    # accelerator.wait_for_everyone()

    return np.mean(loss_list), duration

def evaluate(model, target_model, config, dataloader, device, whisper_processor, accelerator, logger):
    model.eval()
    target_model.eval()
    encoder_criterion = MSELoss()
    wer_results_dict = dict()
    enc_loss_list = list()
    normalizer = BasicTextNormalizer()
    eval_t1 = time.time()
    with torch.no_grad():
        all_transcriptions = list()
        all_labels = list()
        for batch in dataloader:
            mel_specs_clean = batch['mel_specs_clean'].to(device)
            mel_specs_noisy = batch['mel_specs_noisy'].to(device)
            mel_specs_noisy_no_aug = batch['mel_specs_noisy_no_aug'].to(device)
            attentions_mask_clean = None
            # input_videos = batch['video_frames'].to(device)
            input_videos = batch['video_frames_padded'].to(device)
            video_lens = batch['video_lens'].to(device)
            waveforms_noisy = batch['waveforms_noisy']
            # env_embeddings = batch['env_embeddings'].to(device)
            file_ids = batch['file_ids']
            labels_text = batch['labels']
            labels_text = [l.lower() for l in labels_text]
            labels = whisper_processor(text=labels_text, return_tensors="pt", padding=True).input_ids.to(device) #[:, 1:].to(device)

            if 'doMasking' in config['model']:
                if config['model']['doMasking']:
                    attentions_mask_clean = batch['attentions_mask_clean'].to(device)

                
            # with autocast():
            with torch.no_grad():
                encoder_prediction = get_base_model(model).forward(
                        mode='encoder_embedding',
                        calc_loss=False,
                        melspec_inputs=mel_specs_noisy,
                        melspec_inputs_no_aug=mel_specs_noisy_no_aug,
                        raw_audio_inputs=waveforms_noisy,
                        video_inputs=input_videos,
                        vid_lens=video_lens,
                        attention_mask=attentions_mask_clean,
                        melspec_labels=None,
                        melspec_loss_weight=0.0,
                        enc_embed_labels=None,
                        enc_embed_loss_weight=0.0,
                        label_tokens=None,
                        dec_embed_loss_weight=0.0
                        ).encoder_last_hidden_state
            
                generated_ids = get_base_model(model).generate(
                    melspec_inputs=mel_specs_noisy,
                    melspec_inputs_no_aug=mel_specs_noisy_no_aug,
                    raw_audio_inputs=waveforms_noisy,
                    video_inputs=input_videos,
                    vid_lens=video_lens,
                    task='transcribe',
                    language='en',
                    attention_mask=attentions_mask_clean
                    )
            
            with torch.no_grad():
                encoder_targets = target_model.model.encoder(mel_specs_clean)['last_hidden_state']

            # If shape doesn't match - Calculate offset and cut tensors
            if encoder_prediction.shape[1] > encoder_targets.shape[1]:
                offset = encoder_prediction.shape[1] - encoder_targets.shape[1]
                encoder_prediction = encoder_prediction[:, offset:, :]
            elif encoder_prediction.shape[1] < encoder_targets.shape[1]:
                offset = encoder_targets.shape[1] - encoder_prediction.shape[1]
                encoder_targets = encoder_targets[:, offset:, :]

            enc_loss = encoder_criterion(encoder_prediction, encoder_targets)
            enc_loss_list.append(enc_loss.item())

            transcriptions_prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)
            transcriptions_label = processor.batch_decode(labels, skip_special_tokens=True)
            transcriptions_prediction = [el if el.strip() else "a" for el in transcriptions_prediction]

            for fid, p, l in zip(file_ids, transcriptions_prediction, transcriptions_label):
                wer_score = wer(normalizer(p.lower()).strip(), normalizer(l.lower()).strip())
                wer_results_dict[fid] = {'prediction': p, 'label': l, 'wer': wer_score}
            
            for p, l in zip(transcriptions_prediction, transcriptions_label):
                all_transcriptions.append(normalizer(p.lower()).strip())
                all_labels.append(normalizer(l.lower()).strip())
            
            # accelerator.wait_for_everyone()

    all_transcriptions = [el if el.strip() else "a" for el in all_transcriptions]

    # print("", flush=True)
    # for l, b in zip(all_labels, all_transcriptions):
    #     print(f'{l} - {b}', flush=True)
    # print("", flush=True)
    mean_enc_loss = np.mean(enc_loss_list)
    avg_wer = wer(all_transcriptions, all_labels)

    eval_t2 = time.time()
    elapsed_time = eval_t2 - eval_t1
    # print(f"PreEval duration: {elapsed_time:.2f}s | Loss: {eval_loss[0]:.4f} | Eval WER: {(100*eval_loss[1]):.2f}%", flush=True)
    # print('', flush=True)
    logger.info(f"Eval {device} | duration: {elapsed_time:.2f}s | Encoder Loss: {mean_enc_loss:.4f} | WER: {(100*avg_wer):.2f}%")

    return mean_enc_loss, avg_wer, wer_results_dict



if __name__ == "__main__":

    import argparse
    # Argument parser
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model")
    parser.add_argument("--data_dir", type=str, help="Pfad zum Hauptdatenverzeichnis." )
    parser.add_argument("--data_dir_audio", type=str, help="Pfad zum Audio-Datenverzeichnis." )
    parser.add_argument("--data_dir_video", type=str, help="Pfad zum Video-Datenverzeichnis.")
    parser.add_argument("--data_dir_noise", type=str, help="Pfad zum Noise-Datenverzeichnis.")
    parser.add_argument("--config_fn", type=str, help="Pfad zur Konfigurationsdatei.")
    parser.add_argument("--use_accelerate", type=bool, default=True, help="Ob die Accelerate-Bibliothek verwendet werden soll.")
    args = parser.parse_args()

    # Set device
    if args.use_accelerate:
        accelerator = Accelerator(mixed_precision="fp16")
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = accelerator.device
    else:
        accelerator = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    config = load_config(args.config_fn)
    config['audio_dir'] = args.data_dir_audio
    config['video_dir'] = args.data_dir_video
    config['noise_dir'] = args.data_dir_noise

    if 'model_chkp_test' in config['model']:
        config['model']['model_chkp'] = config['model']['model_chkp_test']
    
    # Setup logger and log config information
    train_logger = setup_logger(config['training']['output_dir'], log_filename=config['training']['log_fn_test'])
    if accelerator is None or accelerator.is_main_process:
        train_logger.info('')
        train_logger.info('Config:')
        for key, value in config.items():
            train_logger.info(f"{key}: {value}")
        train_logger.info('')

    # Set seeds
    set_seeds(config['seed'], train_logger)

    # Load model
    processor, target_model, model = prepare_model(config, accelerator, device, train_logger)

    # Load dataset loader
    # train_loader = load_dataloader_train(config, accelerator, train_logger, seed=config['seed'])
    # valid_loader = load_dataloader_eval(config, accelerator, train_logger, seed=config['seed'])
    test_loader = load_dataloader_test(config, accelerator, train_logger, seed=config['seed'])
        
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=float(config['training']['learning_rate']), weight_decay=float(config['training']['weight_decay']))

    # Accelarator preparation
    if accelerator is not None:
        model, optimizer, test_loader = accelerator.prepare(model, optimizer, test_loader)

    # Prepare scheduler
    # lr_scheduler = prepare_scheduler(config, optimizer, train_loader, accelerator, train_logger)


    # audio_data_dir = '/home/atuin/b196ac/b196ac13/00_data/01_LRS3/prepared_testfiles'
    audio_data_dir = args.data_dir_audio

    result_dicts = dict()

    # Clean test
    config['dataset']['SNR_test'] = 50
    print('Testing with clean audio', flush=True)
    audio_dir = os.path.join(audio_data_dir, f'wav_files_LRS3_noiseMixture_clean')
    
    print('Audio-data dir:', audio_dir, flush=True)
    config['audio_dir'] = audio_dir
    
    test_loader = load_dataloader_test(config, accelerator, train_logger, seed=config['seed'])
    eval_enc_loss_clean, eval_wer_clean, wer_results_dict = evaluate(model, target_model, config, test_loader, device, processor, accelerator, train_logger)
    print('', flush=True)
    result_dicts['clean'] = wer_results_dict


    Noise_categories = [
        'LRS3_noiseMixture',
        'musan_babble',
        'LRS3_babble',
        'musan_music',
        'musan_noise',
        'LRS3_sidespeaker'
        ]
    SNR_values = [-5, 0, 5, 10, 50]

    results = np.zeros((len(Noise_categories), len(SNR_values)))

    for catIdx, cat in enumerate(Noise_categories):
        print('Testing with noise category:', cat, flush=True)
        for snrIdx, snr in enumerate(SNR_values):
            config['dataset']['SNR_test'] = 50

            print('Testing with SNR:', snr, flush=True)
            audio_dir = os.path.join(audio_data_dir, f'wav_files_{cat}_SNR{snr}')
            
            print('Audio-data dir:', audio_dir, flush=True)
            config['audio_dir'] = audio_dir

            test_loader = load_dataloader_test(config, accelerator, train_logger, seed=config['seed'])
            eval_enc_loss, eval_wer, wer_results_dict = evaluate(model, target_model, config, test_loader, device, processor, accelerator, train_logger)
            results[catIdx, snrIdx] = eval_wer
            result_dicts[f'{cat}_{snr}dB'] = wer_results_dict

            print('', flush=True)

        print('', flush=True)

    
    txt = 'Results:'
    train_logger.info(txt)

    txt = f'Clean: {eval_wer_clean:.5f}'
    train_logger.info(txt)

    train_logger.info('')

    txt = 'Noise values: ' + ' '.join([str(snr) for snr in SNR_values])
    train_logger.info(txt)

    for catIdx, cat in enumerate(Noise_categories):
        txt = f'{cat.ljust(20)}' + ' '.join([f'{results[catIdx, snrIdx]:.5f}' for snrIdx in range(len(SNR_values))])
        train_logger.info(txt)
    
    # Save results
    result_fn = config['training']['log_fn_test'].replace('.log', '.pkl')
    results_fn = os.path.join(config['training']['output_dir'], result_fn)
    with open(results_fn, 'wb') as f:
        pkl.dump(result_dicts, f)
    
    train_logger.info(f"Results saved to {results_fn}")

    print('Done.', flush=True)
