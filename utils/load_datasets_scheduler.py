from transformers import get_scheduler
from torch.utils.data import DataLoader

from dataset.dataset_AVSR import AVSR_Dataset


def load_dataloader_train(config, accelerator, logger, seed):

    if config['dataset']['type'] == 'AVSR_standard':
        ds = AVSR_Dataset
        if accelerator is None or accelerator.is_main_process:
            logger.info('[DATASET] Use standard AVSR dataset for training.')

    # Load dataset
    train_dataset = ds(
                ds_config=config['dataset'],
                audio_dir=config['audio_dir'],
                video_dir=config['video_dir'],
                noise_dir=config['noise_dir'],
                stage='train',           # 'train' vs 'test'/'valid'
                seed=seed,
            )
    # train_dataset.file_ids = train_dataset.file_ids[:1024]
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size_train'],
        shuffle=False,
        num_workers=config['training']['num_worker'],
        prefetch_factor=2,
        collate_fn=train_dataset.collate_fn
        )
    
    return train_loader

def load_dataloader_eval(config, accelerator, logger, seed):

    if config['dataset']['type'] == 'AVSR_standard':
        ds = AVSR_Dataset
        if accelerator is None or accelerator.is_main_process:
            logger.info('[DATASET] Use standard AVSR dataset for eval.')

    # Load dataset
    eval_dataset = ds(
                ds_config=config['dataset'],
                audio_dir=config['audio_dir'],
                video_dir=config['video_dir'],
                noise_dir=config['noise_dir'],
                stage='valid',           # 'train' vs 'test'/'valid'
                seed=seed,
            )
    if config['training']['num_eval_samples'] is not None:
        eval_dataset.file_ids = eval_dataset.file_ids[:config['training']['num_eval_samples']]
    
    # DataLoader
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config['training']['batch_size_eval'],
        shuffle=False,
        num_workers=config['training']['num_worker'],
        prefetch_factor=2,
        collate_fn=eval_dataset.collate_fn
        )
    
    return eval_loader

def load_dataloader_test(config, accelerator, logger, seed):

    if config['dataset']['type'] == 'AVSR_standard':
        ds = AVSR_Dataset
        if accelerator is None or accelerator.is_main_process:
            logger.info('[DATASET] Use standard AVSR dataset for testing.')

    # Load dataset
    test_dataset = ds(
                ds_config=config['dataset'],
                audio_dir=config['audio_dir'],
                video_dir=config['video_dir'],
                noise_dir=config['noise_dir'],
                stage='test',           # 'train' vs 'test'/'valid'
                seed=seed,
            )
    if 'num_samples_test' in config['training'] and config['training']['num_samples_test'] is not None:
        test_dataset.file_ids = test_dataset.file_ids[:config['training']['num_samples_test']]
    
    # DataLoader
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['training']['batch_size_test'],
        shuffle=False,
        num_workers=config['training']['num_worker'],
        prefetch_factor=2,
        collate_fn=test_dataset.collate_fn
        )
    
    return test_loader

def prepare_scheduler(config, optimizer, train_loader, accelerator, logger):

    if config['training']['lr_scheduler_type'] is not None:
        # Define scheduler
        num_batches = len(train_loader)
        max_train_steps = int(num_batches * (config['training']['num_epochs'] - config['training']['start_epoch']))
        if 'skip_iterations' in config['training']:
            if config['training']['skip_iterations'] is not None:
                max_train_steps = max_train_steps - config['training']['skip_iterations']
        if accelerator is None or accelerator.is_main_process:
            logger.info('')
            logger.info('LR_scheduler:')
            logger.info('name: '+str(config['training']['lr_scheduler_type']))
            logger.info('num_warmup_steps: '+str(config['training']['lr_scheduler_num_warmup_steps']))
            logger.info('num_training_steps: '+str(max_train_steps))
            logger.info('')
    
    # Define scheduler
        lr_scheduler = get_scheduler(
            name=config['training']['lr_scheduler_type'],
            optimizer=optimizer,
            num_warmup_steps=int(config['training']['lr_scheduler_num_warmup_steps']),
            num_training_steps=int(max_train_steps),
        )
    else:
        lr_scheduler = None
    
    return lr_scheduler
