import torch
import fnmatch
from peft import LoraConfig, get_peft_model
from transformers import WhisperConfig, WhisperForConditionalGeneration, AutoProcessor


from models.CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion import CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion

from models.CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptCAttn import CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptCAttn
from models.CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptFeatureGating import CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptFeatureGating
from models.CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wNoiseAdaptPrefix_KV import CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wNoiseAdaptPrefix_KV

from models.CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptPrefix import CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptPrefix
from models.CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptCAttn import CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptCAttn
from models.CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptFeatureGating import CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptFeatureGating


def load_model__CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion(config, accelerator, logger):
    """
    Load Custom Whisper Model like ASRU for AVSR with SAttn w Concat AV Fusion
    """ 
    if accelerator is None or accelerator.is_main_process:
        logger.info('')
        logger.info('Load CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion model')
    # Load Whisper config
    whisper_config = WhisperConfig.from_pretrained(config['model']['whisper_name'])
    # Set embedding config parameter
    whisper_config.freeze_embedding = config['model']['freeze_embedding']
    whisper_config.freeze_encoder = config['model']['freeze_encoder']
    whisper_config.freeze_decoder = config['model']['freeze_decoder']
    # Define embedding_config
    embedding_config = config['model']['embedding_config']
    # Load model
    model = CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion(
            model_name_or_path=config['model']['whisper_name'],
            model_config=whisper_config,
            embedding_config=embedding_config
        )

    model_parts = {
        "Whisper.encoder": model.afm.model.encoder,
        "Whisper.decoder": model.afm.model.decoder,
        "fusion": model.fusion,
    }

    return model, model_parts



def load_model__CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptCAttn(config, accelerator, logger):
    """
    Load Custom Whisper Model like ASRU for AVSR with SAttn w Concat AV Fusion and Noise Adaptation with Cross-Attention
    """ 
    if accelerator is None or accelerator.is_main_process:
        logger.info('')
        logger.info('Load CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptCAttn model')
    # Load Whisper config
    whisper_config = WhisperConfig.from_pretrained(config['model']['whisper_name'])
    # Set embedding config parameter
    whisper_config.freeze_embedding = config['model']['freeze_embedding']
    whisper_config.freeze_encoder = config['model']['freeze_encoder']
    whisper_config.freeze_decoder = config['model']['freeze_decoder']
    whisper_config.enc_embed_loss_type = config['model'].get('enc_embed_loss_type', 'mse')
    whisper_config.loss_config = config['model'].get('loss_config', {}) #['loss_config']
    # Define embedding_config
    embedding_config = config['model']['embedding_config']
    # Define adptation config
    adaptation_config = config['model']['adaptation_config']
    # Load model
    model = CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptCAttn(
            model_name_or_path=config['model']['whisper_name'],
            model_config=whisper_config,
            embedding_config=embedding_config,
            adaptation_config=adaptation_config
        )

    model_parts = {
        'noise_classifier': model.noise_classifier,
        'embedding_mapping_net': model.embedding_mapping_net,
        "Whisper.encoder": model.afm.model.encoder,
        "Whisper.decoder": model.afm.model.decoder,
        "fusion": model.fusion,
    }

    return model, model_parts

def load_model__CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptFeatureGating(config, accelerator, logger):
    """
    Load Custom Whisper Model like ASRU for AVSR with SAttn w Concat AV Fusion and Noise Adaptation with Feature Gating
    """ 
    if accelerator is None or accelerator.is_main_process:
        logger.info('')
        logger.info('Load CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptFeatureGating model')
    # Load Whisper config
    whisper_config = WhisperConfig.from_pretrained(config['model']['whisper_name'])
    # Set embedding config parameter
    whisper_config.freeze_embedding = config['model']['freeze_embedding']
    whisper_config.freeze_encoder = config['model']['freeze_encoder']
    whisper_config.freeze_decoder = config['model']['freeze_decoder']
    whisper_config.enc_embed_loss_type = config['model'].get('enc_embed_loss_type', 'mse')
    whisper_config.loss_config = config['model'].get('loss_config', {}) #['loss_config']
    # Define embedding_config
    embedding_config = config['model']['embedding_config']
    # Define adptation config
    adaptation_config = config['model']['adaptation_config']
    # Load model
    model = CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptFeatureGating(
            model_name_or_path=config['model']['whisper_name'],
            model_config=whisper_config,
            embedding_config=embedding_config,
            adaptation_config=adaptation_config
        )

    model_parts = {
        'noise_classifier': model.noise_classifier,
        'embedding_mapping_net': model.embedding_mapping_net,
        "Whisper.encoder": model.afm.model.encoder,
        "Whisper.decoder": model.afm.model.decoder,
        "fusion": model.fusion,
    }

    return model, model_parts

def load_model__CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wNoiseAdaptPrefix_KV(config, accelerator, logger):
    """
    Load Custom Whisper Model like ASRU for AVSR with SAttn w Concat AV Fusion and Noise Adaptation with Prefix on Key and Value
    """ 
    if accelerator is None or accelerator.is_main_process:
        logger.info('')
        logger.info('Load CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wNoiseAdaptPrefix_KV model')
    # Load Whisper config
    whisper_config = WhisperConfig.from_pretrained(config['model']['whisper_name'])
    # Set embedding config parameter
    whisper_config.freeze_embedding = config['model']['freeze_embedding']
    whisper_config.freeze_encoder = config['model']['freeze_encoder']
    whisper_config.freeze_decoder = config['model']['freeze_decoder']
    whisper_config.enc_embed_loss_type = config['model'].get('enc_embed_loss_type', 'mse')
    whisper_config.loss_config = config['model'].get('loss_config', {}) #['loss_config']
    # Define embedding_config
    embedding_config = config['model']['embedding_config']
    # Define adptation config
    adaptation_config = config['model']['adaptation_config']
    # Load model
    model = CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wNoiseAdaptPrefix_KV(
            model_name_or_path=config['model']['whisper_name'],
            model_config=whisper_config,
            embedding_config=embedding_config,
            adaptation_config=adaptation_config
        )

    model_parts = {
        'noise_classifier': model.noise_classifier,
        'embedding_mapping_net': model.embedding_mapping_net,
        "Whisper.encoder": model.afm.model.encoder,
        "Whisper.decoder": model.afm.model.decoder,
        "fusion": model.fusion,
    }

    return model, model_parts


def load_model__CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptPrefix(config, accelerator, logger, device):
    """
    Load Custom Whisper Model like ASRU for AVSR with SAttn w Concat AV Fusion and Speaker Adaptation with Prefix
    """ 
    if accelerator is None or accelerator.is_main_process:
        logger.info('')
        logger.info('Load CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptPrefix model')
    # Load Whisper config
    whisper_config = WhisperConfig.from_pretrained(config['model']['whisper_name'])
    # Set embedding config parameter
    whisper_config.freeze_embedding = config['model']['freeze_embedding']
    whisper_config.freeze_encoder = config['model']['freeze_encoder']
    whisper_config.freeze_decoder = config['model']['freeze_decoder']
    whisper_config.enc_embed_loss_type = config['model'].get('enc_embed_loss_type', 'mse')
    whisper_config.loss_config = config['model'].get('loss_config', {}) #['loss_config']
    # Define embedding_config
    embedding_config = config['model']['embedding_config']
    # Define adptation config
    adaptation_config = config['model']['adaptation_config']
    adaptation_config['device'] = device
    # Load model
    model = CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptPrefix(
            model_name_or_path=config['model']['whisper_name'],
            model_config=whisper_config,
            embedding_config=embedding_config,
            adaptation_config=adaptation_config
        )

    model_parts = {
        'spk_embedding_model': model.spk_embedding_model,
        'embedding_mapping_net_spk': model.embedding_mapping_net_spk,
        "Whisper.encoder": model.afm.model.encoder,
        "Whisper.decoder": model.afm.model.decoder,
        "fusion": model.fusion,
    }

    return model, model_parts

def load_model__CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptCAttn(config, accelerator, logger, device):
    """
    Load Custom Whisper Model like ASRU for AVSR with SAttn w Concat AV Fusion and Speaker Adaptation with Cross-Attention
    """ 
    if accelerator is None or accelerator.is_main_process:
        logger.info('')
        logger.info('Load CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptCAttn model')
    # Load Whisper config
    whisper_config = WhisperConfig.from_pretrained(config['model']['whisper_name'])
    # Set embedding config parameter
    whisper_config.freeze_embedding = config['model']['freeze_embedding']
    whisper_config.freeze_encoder = config['model']['freeze_encoder']
    whisper_config.freeze_decoder = config['model']['freeze_decoder']
    whisper_config.enc_embed_loss_type = config['model'].get('enc_embed_loss_type', 'mse')
    whisper_config.loss_config = config['model'].get('loss_config', {}) #['loss_config']
    # Define embedding_config
    embedding_config = config['model']['embedding_config']
    # Define adptation config
    adaptation_config = config['model']['adaptation_config']
    adaptation_config['device'] = device
    # Load model
    model = CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptCAttn(
            model_name_or_path=config['model']['whisper_name'],
            model_config=whisper_config,
            embedding_config=embedding_config,
            adaptation_config=adaptation_config
        )

    model_parts = {
        'spk_embedding_model': model.spk_embedding_model,
        'embedding_cattn': model.embedding_cattn,
        'embedding_mapping_net_spk': model.embedding_mapping_net_spk,
        "Whisper.encoder": model.afm.model.encoder,
        "Whisper.decoder": model.afm.model.decoder,
        "fusion": model.fusion,
    }

    return model, model_parts

def load_model__CustomWhisper_CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptFeatureGating(config, accelerator, logger, device):
    """
    Load Custom Whisper Model like ASRU for AVSR with SAttn w Concat AV Fusion and Speaker Adaptation with Feature Gating
    """ 
    if accelerator is None or accelerator.is_main_process:
        logger.info('')
        logger.info('Load CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptFeatureGating model')
    # Load Whisper config
    whisper_config = WhisperConfig.from_pretrained(config['model']['whisper_name'])
    # Set embedding config parameter
    whisper_config.freeze_embedding = config['model']['freeze_embedding']
    whisper_config.freeze_encoder = config['model']['freeze_encoder']
    whisper_config.freeze_decoder = config['model']['freeze_decoder']
    whisper_config.enc_embed_loss_type = config['model'].get('enc_embed_loss_type', 'mse')
    whisper_config.loss_config = config['model'].get('loss_config', {}) #['loss_config']
    # Define embedding_config
    embedding_config = config['model']['embedding_config']
    # Define adptation config
    adaptation_config = config['model']['adaptation_config']
    adaptation_config['device'] = device
    # Load model
    model = CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptFeatureGating(
            model_name_or_path=config['model']['whisper_name'],
            model_config=whisper_config,
            embedding_config=embedding_config,
            adaptation_config=adaptation_config
        )

    model_parts = {
        'spk_embedding_model': model.spk_embedding_model,
        'embedding_mapping_net_spk': model.embedding_mapping_net_spk,
        "Whisper.encoder": model.afm.model.encoder,
        "Whisper.decoder": model.afm.model.decoder,
        "fusion": model.fusion,
    }

    return model, model_parts



def prepare_peft_model(model, config, accelerator, logger):
    # Prepare PEFT Model
    patterns = config['model']['lora_config']['lora_targets']
    target_modules = [name for name, _ in model.named_modules() if any(fnmatch.fnmatch(name, pattern) for pattern in patterns)]
    lora_config = LoraConfig(
                r=config['model']['lora_config']['lora_r'],
                lora_alpha=config['model']['lora_config']['lora_alpha'],
                target_modules=target_modules,
                # target_modules=config['model']['lora_config']['lora_targets'],
                # target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
                lora_dropout=config['model']['lora_config']['lora_dropout'],
                use_dora=config['model']['lora_config']['use_dora'],
            )
    model = get_peft_model(model, lora_config)
    if accelerator is None or accelerator.is_main_process:
        logger.info('')
        logger.info(f'[MODEL] Prepare PEFT Model:')
        logger.info(f'[MODEL] Lora r:'+str(config['model']['lora_config']['lora_r']))
        logger.info(f'[MODEL] Lora alpha:'+str(config['model']['lora_config']['lora_alpha']))
        logger.info(f'[MODEL] Lora target_modules:'+str(config['model']['lora_config']['lora_targets']))
        logger.info(f'[MODEL] Lora dropout:'+str(config['model']['lora_config']['lora_dropout']))
        logger.info(f'[MODEL] Lora use_dora:'+str(config['model']['lora_config']['use_dora']))
    return model

def load_model_checkpoint(model, config, accelerator, logger):
    """
    Load model checkpoint
    """
    if accelerator is None or accelerator.is_main_process:
        logger.info('[MODEL] Load checkpoint from ' + str(config['model']['model_chkp']))
    chkp_state_dict = torch.load(config['model']['model_chkp'])
    model_state_dict = model.state_dict()
    model.load_state_dict(chkp_state_dict, strict=False)
    missing_keys = [key for key in chkp_state_dict.keys() if key not in model_state_dict]
    if len(missing_keys) > 0:
        if accelerator is None or accelerator.is_main_process:
            logger.info('[MODEL] Keys that doesnt match model state_dict:')
            logger.info(missing_keys)
    return model


def count_parameters(model_part):

    if isinstance(model_part, torch.nn.Module):
        # total_params = sum(p.numel() for p in model_part.parameters())
        total_params = sum(p.numel() for p in model_part.parameters())
        trainable_params = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
    elif isinstance(model_part, torch.nn.Parameter):
        total_params = model_part.numel()
        if model_part.requires_grad:
            trainable_params = model_part.numel()
        else:
            trainable_params = 0

    # total_params = sum(p.numel() for p in model_part.parameters())
    # trainable_params = sum(p.numel() for p in model_part.parameters() if p.requires_grad)
    return total_params, trainable_params

def count_parameters_for_model_parts(model_parts, accelerator, logger):
    """
    Count the number of parameters in a model
    """
    if accelerator is None or accelerator.is_main_process:
        logger.info('')
        logger.info('Parameter:')
        all_total = 0
        all_trainable = 0
        for name, part in model_parts.items():
            total, trainable = count_parameters(part)
            all_total += total
            all_trainable += trainable
            logger.info(f"{name}: Total Parameters = {total:,}, Trainable Parameters = {trainable:,}")
        logger.info(f"Sum Total Parameters = {all_total:,}, Trainable Parameters = {all_trainable:,}")
        logger.info('')

def set_init_decoder_token_ids(processor, accelerator, logger):
    """
    Set the initial decoder token ids
    """
    forced_decoder_ids = processor.get_decoder_prompt_ids(language='en', task='transcribe')
    if accelerator is None or accelerator.is_main_process:
        # "Set" forced_decoder_ids and check init Tokens
        logger.info('')
        logger.info('Check Tokens:')

        logger.info(f'forced_decoder_ids {forced_decoder_ids}')

        text = "BUT YOU JUST HAVEN'T FOUND IT YET"
        text = text.lower().strip()
        labels = processor(text=text, return_tensors="pt", padding=True).input_ids
        logger.info(f'labels {labels}')
        logger.info('')
        logger.info('')
    return processor, forced_decoder_ids

def prepare_model(config, accelerator, device, logger):

    processor = AutoProcessor.from_pretrained(config['model']['whisper_name'])
    target_model = WhisperForConditionalGeneration.from_pretrained(config['model']['whisper_name'])
    
    if config['model']['model_name'] == 'CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion':
        model, model_parts = load_model__CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion(config, accelerator, logger)
    elif config['model']['model_name'] == 'CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptCAttn':
        model, model_parts = load_model__CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptCAttn(config, accelerator, logger)
    elif config['model']['model_name'] == 'CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptFeatureGating':
        model, model_parts = load_model__CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion___wNoiseAdaptFeatureGating(config, accelerator, logger)
    elif config['model']['model_name'] == 'CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wNoiseAdaptPrefix_KV':
        model, model_parts = load_model__CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wNoiseAdaptPrefix_KV(config, accelerator, logger)
    elif config['model']['model_name'] == 'CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptPrefix':
        model, model_parts = load_model__CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptPrefix(config, accelerator, logger, device)
    elif config['model']['model_name'] == 'CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptCAttn':
        model, model_parts = load_model__CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptCAttn(config, accelerator, logger, device)
    elif config['model']['model_name'] == 'CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptFeatureGating':
        model, model_parts = load_model__CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptFeatureGating(config, accelerator, logger, device)
    else:
        raise ValueError(f"Model {config['model']['model_name']} not implemented!")


    if config['model']['training_mode'] == 'peft':
        # Prepare PEFT Model
        model = prepare_peft_model(model, config, accelerator, logger)
        # Make custom layers trainable in case PEFT froze all layers except the ones in the lora_config
        model.base_model.make_custom_layers_trainable()

    # Load state_dict
    if config['model']['model_chkp'] is not None:
        model = load_model_checkpoint(model, config, accelerator, logger)
    
    # Shift model to device
    target_model = target_model.to(device)
    model = model.to(device)

    # Count model parameters
    count_parameters_for_model_parts(model_parts, accelerator, logger)

    # Log model information and decoder init tokens
    processor, forced_decoder_ids = set_init_decoder_token_ids(processor, accelerator, logger)

    return processor, target_model, model

def save_model(model, out_fn):
    # Get state_dict
    full_state_dict = model.state_dict()
    # Save state_dict
    torch.save(full_state_dict, out_fn)
    return
