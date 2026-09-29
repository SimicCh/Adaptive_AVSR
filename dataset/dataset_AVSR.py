import os
import random
import numpy as np
import cv2
import torch
from torchvision import transforms
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchvision.transforms import functional as F
from collections import defaultdict

from transformers import AutoProcessor


# Torchvision transformation components
class Normalize_DivideBy255(torch.nn.Module):
    def forward(self, tensor):
        return tensor / 255.0

class Normalize_minmax(torch.nn.Module):
    def forward(self, tensor):
        tensor = tensor / 255.0
        tensor = tensor - 0.5
        tensor = tensor / 0.5
        return tensor

class RandomFrameMasking(torch.nn.Module):
    """
    Führe Frame Masking durch
        :param mask_len: Anzahl der Frames die gemasked werden sollen
    """

    def __init__(self, mask_len, frame_rate=25):
        super().__init__()
        self.mask_len = mask_len
        self.frame_rate = frame_rate

    def forward(self, tensor):
        """
        Führe Frame Masking durch
            :param tensor: Input Video Frames von shape [batch, sq_len, channels, H, W]
            :return: Video Frames mit Maskierung
        """

        # Bestimme shape des Tensors
        sequ_len, channels, height, width = tensor.shape
            
        # Der Wert mit dem gemasked werden soll ist der Mittelwert der Frames dieses batches
        tensor = tensor.float()
        mask_value = torch.mean(tensor, dim=(0,2,3))
        # Alle jeweils self.frame_rate frames soll randomisiert gemasked werden
        for j in range(int(sequ_len/self.frame_rate)):
            # Falls self.mask_len ein tuple ist, wähle zufällig eine Anzahl von Frames zwischen den beiden Werten
            if isinstance(self.mask_len, tuple):
                mask_len_sel = torch.randint(self.mask_len[0], self.mask_len[1], (1,)).item()
            elif isinstance(self.mask_len, list):
                mask_len_sel = torch.randint(self.mask_len[0], self.mask_len[1], (1,)).item()
            else:
                mask_len_sel = self.mask_len
            
            # Bestimme randomisiert start und end Indizes für die Maske für diesen 25 Frame Block
            start_idx = torch.randint(0, self.frame_rate-mask_len_sel, (1,)).item() + j*self.frame_rate
            end_idx = start_idx + mask_len_sel
            # Überschreibe alle frames von start_idx bis end_idx mit mask_value dieses batches
            for channel in range(channels):
                tensor[start_idx:end_idx, channel, :, :] = mask_value[channel]
        
        return tensor

class RandomFrameMasking_singleFrames(torch.nn.Module):
    """
    Führe Frame Masking durch
        :param mask_len: Anzahl der Frames die gemasked werden sollen
    """

    def __init__(self, mask_probability):
        super().__init__()
        self.mask_probability = mask_probability

    def forward(self, tensor):
        """
        Führe Frame Masking durch
            :param tensor: Input Video Frames von shape [batch, sq_len, channels, H, W]
            :return: Video Frames mit Maskierung
        """

        # Bestimme shape des Tensors
        sequ_len, channels, height, width = tensor.shape
            
        # Der Wert mit dem gemasked werden soll ist der Mittelwert der Frames dieses batches
        tensor = tensor.float()
        mask_value = torch.mean(tensor, dim=(0,2,3))

        for framesIdx in range(sequ_len):
            if torch.rand(1) < self.mask_probability:
                for channel in range(channels):
                    tensor[framesIdx, channel, :, :] = mask_value[channel]
        
        return tensor

class RandomFrameCropping(torch.nn.Module):
    """
    Führe Frame Masking durch
        :param mask_len: Anzahl der Frames die gemasked werden sollen
    """

    def __init__(self, cropping_prob=0.2, cropping_maxAreaShare=0.3):
        super().__init__()
        self.cropping_prob = cropping_prob
        self.cropping_maxAreaShare = cropping_maxAreaShare

    def forward(self, tensor):
        """
        Führe Frame Masking durch
            :param tensor: Input Video Frames von shape [batch, sq_len, 88, 88, 3]
            :return: Video Frames mit Maskierung
        """

        # Bestimme shape des Tensors
        sequ_len, channels, height, width = tensor.shape

        # Der Wert mit dem gemasked werden soll ist der Mittelwert der Frames dieses batches
        mask_value = torch.mean(tensor, dim=(0,2,3))
        # Loop über die gesamte sequ_len, bei jedem Frame wird mit einer Wahrscheinlichkeit von self.cropping_prob ein Teil des Frames gecropped
        for j in range(sequ_len):
            # Bestimme randomisiert ob gecropped wird
            if torch.rand(1) < self.cropping_prob:
                # Get area of crop
                crop_area = torch.randint(1, int(height*width*self.cropping_maxAreaShare), (1,)).item()
                min_crop_height = min(max(int(crop_area/width),1), height-1)
                max_crop_height = min(crop_area, height)
                crop_height = torch.randint(min_crop_height, max_crop_height+1, (1,)).item()
                crop_width = min(max(int(crop_area / crop_height),1), width-1)
                # # Bestimme die Anzahl der Pixel die gecropped werden
                # crop_height = int(height * torch.rand(1) * self.cropping_maxAreaShare)
                # crop_width = int(width * torch.rand(1) * self.cropping_maxAreaShare)
                # Bestimme die Start und End Indizes für das Cropping
                start_h = torch.randint(0, height-crop_height+1, (1,)).item()
                end_h = start_h + crop_height
                start_w = torch.randint(0, width-crop_width+1, (1,)).item()
                end_w = start_w + crop_width
                # Überschreibe die Pixel mit mask_value
                for channel in range(channels):
                    tensor[j, channel, start_h:end_h, start_w:end_w] = mask_value[channel]

        return tensor

class RandomHorizontalFlip_mod(torch.nn.Module):
    """Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, tensor):
        """
        tensor hat shape [batch, sq_len, 3, 88, 88]

        Args:
            tensor (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """

        # Überprüfe ob shape korrekt ist
        # assert tensor.shape[2:] == (3, 88, 88), f'Input shape is {tensor.shape}, but should be [batch, sq_len, 3, 88, 88]'

        # Bestimme shape des Tensors
        # batch_size, sequ_len, channels, height, width = tensor.shape
        # for i in range(batch_size):
        #     if torch.rand(1) < self.p:
        #         tensor[i] = F.hflip(tensor[i])

        if torch.rand(1) < self.p:
            tensor = F.hflip(tensor)
                
        return tensor

class Normalize_mod(torch.nn.Module):

    def __init__(self, do_normalize: bool = False, mean: list = None, std: list = None):
        super().__init__()
        self.do_normalize = do_normalize
        self.mean = mean
        self.std = std
    
    def forward(self, tensor):
        
        # # Bestimme shape des Tensors
        # batch_size, sequ_len, channels, height, width = tensor.shape
        # 
        # if self.do_normalize:
        #     for i in range(batch_size):
        #         tensor[i] = F.normalize(tensor[i], self.mean, self.std)
        
        if self.do_normalize:
            tensor = F.normalize(tensor, self.mean, self.std)
                
        return tensor

class GaussianBlur_mod(torch.nn.Module):
    def __init__(self, kernel_size, sigma_range):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_range = sigma_range

    def forward(self, tensor):
        # # tensor shape: [B, T, C, H, W]
        # batch_size, seq_len, channels, height, width = tensor.shape
        # sigma = torch.FloatTensor(1).uniform_(*self.sigma_range).item()
        # 
        # # Anwenden von GaussianBlur auf jeden Frame
        # for b in range(batch_size):
        #     for t in range(seq_len):
        #         tensor[b, t] = F.gaussian_blur(tensor[b, t], kernel_size=self.kernel_size, sigma=sigma)
        # return tensor
    
        # tensor shape: [B, T, C, H, W]
        seq_len, channels, height, width = tensor.shape
        sigma = torch.FloatTensor(1).uniform_(*self.sigma_range).item()

        # Anwenden von GaussianBlur auf jeden Frame
        for t in range(seq_len):
            tensor[t] = F.gaussian_blur(tensor[t], kernel_size=self.kernel_size, sigma=sigma)
        return tensor

class ResizeVideo(torch.nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.resize = transforms.Resize(size)

    def forward(self, tensor):
        # tensor shape: [B, T, C, H, W]
        # batch_size, seq_len, channels, height, width = tensor.shape
        # seq_len, channels, height, width = tensor.shape
        # resized_frames = []

        # Iteriere über Batch und Zeitdimension
        # for b in range(batch_size):
        #     resized_seq = [self.resize(tensor[b, t]) for t in range(seq_len)]
        #     resized_frames.append(torch.stack(resized_seq, dim=0))
        resized_frames = self.resize(tensor)

        # Rückgabe als Tensor
        # return torch.stack(resized_frames, dim=0)
        return resized_frames



class AVSR_Dataset(Dataset):
    def __init__(
                self, 
                ds_config, 
                audio_dir,
                video_dir,
                noise_dir,
                stage = 'train',           # 'train' vs 'test'/'valid'
                seed = 0,
                noise_categories = None
            ):
        self.ds_config = ds_config
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.noise_dir = noise_dir
        self.stage = stage
        self.seed = seed
        self.noise_categories = noise_categories
        if 'max_duration' in ds_config:
            self.max_duration = ds_config['max_duration']
        else:
            self.max_duration = 30

        if self.stage == 'train':
            self.SNR = ds_config['SNR_train']
        elif self.stage=='valid':
            self.SNR = ds_config['SNR_valid']
        elif self.stage=='test':
            self.SNR = ds_config['SNR_test']

        self.specaug_maxSequLen = ds_config['specaug_maxSequLen']
        self.specaug_maxlenratio = ds_config['specaug_maxlenratio']
        self.specaug_maxchannels = ds_config['specaug_maxchannels']
        
        # Set whisper processor for melspec generation
        self.whisper_processor = AutoProcessor.from_pretrained(ds_config['whisper_modelname'])

        if self.stage=='train':
            self.fid_list_path = ds_config['train_fids']['fid_list']
            self.label_list_path = ds_config['train_fids']['label_list']
            self.noise_musan_babble = ds_config['train_fids']['noise_fid_lists']['musan_babble']
            self.noise_musan_music = ds_config['train_fids']['noise_fid_lists']['musan_music']
            self.noise_musan_noise = ds_config['train_fids']['noise_fid_lists']['musan_noise']
            self.noise_single_sidespeaker = ds_config['train_fids']['noise_fid_lists']['single_sidespeaker']
        elif self.stage=='valid':
            self.fid_list_path = ds_config['valid_fids']['fid_list']
            self.label_list_path = ds_config['valid_fids']['label_list']
            self.noise_musan_babble = ds_config['valid_fids']['noise_fid_lists']['musan_babble']
            self.noise_musan_music = ds_config['valid_fids']['noise_fid_lists']['musan_music']
            self.noise_musan_noise = ds_config['valid_fids']['noise_fid_lists']['musan_noise']
            self.noise_single_sidespeaker = ds_config['valid_fids']['noise_fid_lists']['single_sidespeaker']
        elif self.stage=='test':
            self.fid_list_path = ds_config['test_fids']['fid_list']
            self.label_list_path = ds_config['test_fids']['label_list']
            self.noise_musan_babble = ds_config['test_fids']['noise_fid_lists']['musan_babble']
            self.noise_musan_music = ds_config['test_fids']['noise_fid_lists']['musan_music']
            self.noise_musan_noise = ds_config['test_fids']['noise_fid_lists']['musan_noise']
            self.noise_single_sidespeaker = ds_config['test_fids']['noise_fid_lists']['single_sidespeaker']

        # Lade fids
        with open(self.fid_list_path, 'r') as f:
            self.file_ids = [line.strip() for line in f]

        if self.label_list_path is not None:
            with open(self.label_list_path, 'r') as f:
                self.labels = [line.strip() for line in f]
        else:
            self.labels = None
        

        # Load noise file IDs
        self.available_noises = list()

        if self.noise_musan_babble is not None:
            with open(self.noise_musan_babble, 'r') as f:
                self.fids_musan_babble = [line.strip() for line in f]
                self.fids_musan_babble = [os.path.join(self.noise_dir, fid) for fid in self.fids_musan_babble]
                self.available_noises.append('musan_babble')

        if self.noise_musan_music is not None:
            with open(self.noise_musan_music, 'r') as f:
                self.fids_musan_music = [line.strip() for line in f]
                self.fids_musan_music = [os.path.join(self.noise_dir, fid) for fid in self.fids_musan_music]
                self.available_noises.append('musan_music')

        if self.noise_musan_noise is not None:
            with open(self.noise_musan_noise, 'r') as f:
                self.fids_musan_noise = [line.strip() for line in f]
                self.fids_musan_noise = [os.path.join(self.noise_dir, fid) for fid in self.fids_musan_noise]
                self.available_noises.append('musan_noise')

        if self.noise_single_sidespeaker is not None:
            with open(self.noise_single_sidespeaker, 'r') as f:
                self.fids_single_sidespeaker = [line.strip() for line in f]
                self.fids_single_sidespeaker = [os.path.join(self.audio_dir, fid+'.wav') for fid in self.fids_single_sidespeaker]
                self.available_noises.append('single_sidespeaker')

        if self.noise_categories is not None:
            self.available_noises = self.noise_categories

        # Set all seeds
        self.set_seeds(seed)

        # Shuffle the file_ids list and accordingly the labels
        index_list = list(range(len(self.file_ids)))
        random.shuffle(index_list)

        if self.stage=='train':
            self.file_ids = [self.file_ids[i] for i in index_list]
            if self.labels is not None:
                self.labels = [self.labels[i] for i in index_list]

        fids_len = len(self.file_ids)

        # Dedicated snr values for training
        if isinstance(self.SNR, list):
            self.didicated__snr_list = np.random.uniform(low=self.SNR[0], high=self.SNR[1], size=fids_len)
        else:
            self.didicated__snr_list = np.ones([fids_len]) * self.SNR
        
        # Dedicated noise samples for training
        self.random_noises_list = random.choices(self.available_noises, k=fids_len)
        if self.noise_musan_babble is not None:
            self.dedicated_fids_musan_babble = random.choices(self.fids_musan_babble, k=fids_len)
        if self.noise_musan_music is not None:
            self.dedicated_fids_musan_music = random.choices(self.fids_musan_music, k=fids_len)
        if self.noise_musan_noise is not None:
            self.dedicated_fids_musan_noise = random.choices(self.fids_musan_noise, k=fids_len)
        if self.noise_single_sidespeaker is not None:
            self.dedicated_fids_single_sidespeaker = random.choices(self.fids_single_sidespeaker, k=fids_len)

        self.didicated__noise_fid_list = list()
        for idx in range(fids_len):
            selected_noise = self.random_noises_list[idx]
            attribute_name = f"dedicated_fids_{selected_noise}"
            selected_list = getattr(self, attribute_name, None)
            self.didicated__noise_fid_list.append(selected_list[idx])
        
        # Pre define start index offset for noise
        self.noise_start_offset = np.random.uniform(low=0.0, high=1.0, size=fids_len)

        # Transformation # input must bis [L, 3, H, W] 
        transform_seq = list()
        if self.stage == 'train':
            # Resize inputs
            if ds_config['video_transform_resize_mode'] == 'resize':
                seq_resize = ResizeVideo(ds_config['video_transform_input_shape'])
                transform_seq.append(seq_resize)
            elif ds_config['video_transform_resize_mode'] == 'crop':
                seq_resize = transforms.RandomCrop(ds_config['video_transform_input_shape'])
                transform_seq.append(seq_resize)
            else:
                assert 'No correct video_transform_resize_mode - ' + ds_config['video_transform_resize_mode']
            # Random horizontal flip
            if ds_config['video_transform_do_horizontal_flip'] == True:
                seq_flip = RandomHorizontalFlip_mod(p=ds_config['video_transform_horizontal_flip_prob'])
                transform_seq.append(seq_flip)
            # Random gaussian blur
            if ds_config['video_transform_do_GaussianBlur'] == True:
                # seq_blur = transforms.GaussianBlur(kernel_size=ds_config['video_transform_GaussianBlur_kernelsize'], sigma=tuple(ds_config['video_transform_GaussianBlur_sigma']))
                seq_blur = GaussianBlur_mod(kernel_size=ds_config['video_transform_GaussianBlur_kernelsize'], sigma_range=tuple(ds_config['video_transform_GaussianBlur_sigma']))
                transform_seq.append(seq_blur)
            # Random frame masking
            if ds_config['video_transform_do_framemask'] == True:
                if 'video_transform_framemask_mode' in ds_config:
                    if ds_config['video_transform_framemask_mode'] == 'mask_prob':
                        seq_framemask = RandomFrameMasking_singleFrames(mask_probability=ds_config['video_transform_framemask_prob'])
                        transform_seq.append(seq_framemask)
                    elif ds_config['video_transform_framemask_mode'] == 'mask_len':
                        seq_framemask = RandomFrameMasking(mask_len=ds_config['video_transform_framemask_range'], frame_rate=25)
                        transform_seq.append(seq_framemask)
                    else:
                        assert 'No correct video_transform_framemask_mode - ' + ds_config['video_transform_framemask_mode']
                else:
                    seq_framemask = RandomFrameMasking(mask_len=ds_config['video_transform_framemask_range'], frame_rate=25)
                    transform_seq.append(seq_framemask)
            # Random frame area croppinz
            if ds_config['video_transform_do_framecrop'] == True:
                seq_framecrop = RandomFrameCropping(cropping_prob=ds_config['video_transform_framecrop_prob'], cropping_maxAreaShare=ds_config['video_transform_framecrop_maxAreaShare'])
                transform_seq.append(seq_framecrop)
            # Normalize
            if ds_config['video_transform_normalize_mode'] == 'div255':
                seq_normalize = Normalize_DivideBy255()
                transform_seq.append(seq_normalize)
            elif ds_config['video_transform_normalize_mode'] == 'min_max':
                seq_normalize = Normalize_minmax()
                transform_seq.append(seq_normalize)
            elif ds_config['video_transform_normalize_mode'] == 'mean_std':
                seq_normalize = transforms.Normalize(mean=ds_config['video_transform_normalize_mean'], std=ds_config['video_transform_normalize_std'])
                transform_seq.append(seq_normalize)
            elif ds_config['video_transform_normalize_mode'] == 'gray':
                seq_normalize = transforms.Grayscale()
                transform_seq.append(seq_normalize)
            else:
                assert 'No correct video_transform_normalize_mode - ' + ds_config['video_transform_normalize_mode']
        else:
            # Resize inputs
            if ds_config['video_transform_resize_mode'] == 'resize':
                seq_resize = transforms.Resize(ds_config['video_transform_input_shape'])
                transform_seq.append(seq_resize)
            elif ds_config['video_transform_resize_mode'] == 'crop':
                seq_resize = transforms.CenterCrop(ds_config['video_transform_input_shape'])
                transform_seq.append(seq_resize)
            else:
                assert 'No correct video_transform_resize_mode - ' + ds_config['video_transform_resize_mode']
            # Normalize
            if ds_config['video_transform_normalize_mode'] == 'div255':
                seq_normalize = Normalize_DivideBy255()
                transform_seq.append(seq_normalize)
            elif ds_config['video_transform_normalize_mode'] == 'min_max':
                seq_normalize = Normalize_minmax()
                transform_seq.append(seq_normalize)
            elif ds_config['video_transform_normalize_mode'] == 'mean_std':
                seq_normalize = transforms.Normalize(mean=ds_config['video_transform_normalize_mean'], std=ds_config['video_transform_normalize_std'])
                transform_seq.append(seq_normalize)
            elif ds_config['video_transform_normalize_mode'] == 'gray':
                seq_normalize = transforms.Grayscale()
                transform_seq.append(seq_normalize)
            else:
                assert 'No correct video_transform_normalize_mode - ' + ds_config['video_transform_normalize_mode']

        self.transforms_seq = torch.nn.Sequential(*transform_seq)


    def set_seeds(self, seed: int):
        """
        Setzt die Seeds für random, numpy und torch für reproduzierbare Ergebnisse.
        Args:
            seed (int): Der Seed-Wert, der für alle Zufallsquellen verwendet werden soll.
        """
        # Seed für den Python-Zufalls-Generator
        random.seed(seed)
        # Seed für numpy
        np.random.seed(seed)
        # Seed für PyTorch (CPU und CUDA)
        torch.manual_seed(seed)
        # Für CUDA: Sicherstellen, dass reproduzierbare Ergebnisse aktiviert sind
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # Falls mehrere GPUs verwendet werden
        # Zusätzliche Einstellungen für deterministisches Verhalten in PyTorch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def __len__(self):
        return len(self.file_ids)


    # Pepare Video
    def load_video(self, fname):
        video = cv2.VideoCapture(fname)
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(torch.tensor(frame).permute(2,0,1))

        return torch.stack(frames)

    
    # Prepare and augment Audio
    def calc_snr_factor(self, signal, noise, snr):
        p_signal = (torch.sum(signal**2)/signal.shape[1]).item()
        p_noise  = (torch.sum(noise**2)/noise.shape[1]).item()
        targ_noise_p = p_signal/(10**(snr/10))
        noise_factor = (targ_noise_p/p_noise)**0.5
        # # calc SNR 
        # p_signal = np.sum(signal**2)/len(signal)
        # p_noise = np.sum((noise*noise_factor)**2)/len(noise)
        # SNR_val = 10 * (np.log10(p_signal) - np.log10(p_noise))
        return noise_factor
    
    def adjust_noise_length(self, signal, noise, noise_start_offset):
        if signal.shape[1]>noise.shape[1]:
            factor = int(signal.shape[1]/noise.shape[1])+1
            noise = torch.cat([noise for i in range(factor)], dim=1)
        if self.stage=='train':
            if noise.shape[1]-signal.shape[1]==0:
                start_id = 0
            else:
                # start_id = torch.randint(0, noise.shape[1]-signal.shape[1], (1,))
                start_id = int((noise.shape[1]-signal.shape[1]) * noise_start_offset)
        else:
            start_id = 0
        noise = noise[:,start_id:start_id+signal.shape[1]]
        return noise
        
    def add_noise(self, noise_fn, audio, snr, noise_start_offset):
        # noise = self.load_audio(noise_fn)
        noise, sample_rate_noise = torchaudio.load(noise_fn)
        # Resample auf 16kHz falls nötig
        if sample_rate_noise != 16000:
            resample_transform = T.Resample(orig_freq=sample_rate_noise, new_freq=16000)
            noise = resample_transform(noise)
            sample_rate = 16000
        noise_factor = self.calc_snr_factor(audio, noise, snr)
        noise = noise * noise_factor

        # p_signal = torch.mean(audio**2)  # Mittlere Leistung des Signals
        # p_noise = torch.mean(noise**2)  # Mittlere Leistung des Rauschens
        # if p_noise == 0:
        #     raise ValueError("Noise power is zero, cannot compute SNR.")
        # snr_db = 10 * torch.log10(p_signal / p_noise)
        # print(snr_db, flush=True)

        noise = self.adjust_noise_length(audio, noise, noise_start_offset)
        mixed = audio + noise
        return mixed, noise


    # Spec-augmentation
    # roughly based on https://arxiv.org/pdf/1904.08779.pdf
    def augment_spec(self, melspectrum: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment augmentation to the mel-spectrogram.

        Args:
        - melspectrum (torch.Tensor): Mel-spectrogram tensor.

        Returns:
        - torch.Tensor: Augmented mel-spectrogram tensor.
        """
        melspectrum_aug = melspectrum.clone()
        spec_length = melspectrum_aug.shape[2]
        max_specauglength = min([self.specaug_maxSequLen, int(self.specaug_maxlenratio*spec_length)])
        aug_length = torch.randint(0, max_specauglength, (1,)).item()
        if aug_length>0:
            spec_start = torch.randint(0, spec_length-aug_length, (1,)).item()
            melspectrum_aug[:,:,spec_start:spec_start+aug_length] = 0
        aug_channels = torch.randint(0, self.specaug_maxchannels, (1,)).item()
        if aug_channels>0:
            spec_start = torch.randint(0, melspectrum_aug.shape[1]-aug_channels, (1,)).item()
            melspectrum_aug[:, spec_start:spec_start+aug_channels, :] = 0
        return melspectrum_aug


    # Spec-augmentation
    # roughly based on https://arxiv.org/pdf/1904.08779.pdf
    # SpecAugment with spec augmentation for each second of input mel-spectrogram
    def augment_spec_multi(self, melspectrum: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment augmentation to the mel-spectrogram.

        Args:
        - melspectrum (torch.Tensor): Mel-spectrogram tensor.

        Returns:
        - torch.Tensor: Augmented mel-spectrogram tensor.
        """
        B, D, L = melspectrum.shape
        spec_mean = torch.mean(melspectrum).item()
        split_len = self.ds_config['specaug_splitlen']

        i = 0
        for i in range(int(L/split_len)):
            num_aug_channels = torch.randint(0, self.specaug_maxchannels, (1,)).item()
            start_aug_channel = torch.randint(0, D-num_aug_channels, (1,)).item()
            end_aug_channel = start_aug_channel + num_aug_channels

            num_aug_length = torch.randint(0, self.specaug_maxSequLen, (1,)).item()
            start_aug_length = torch.randint(0, split_len-num_aug_length, (1,)).item() + i*split_len
            end_aug_length = start_aug_length + num_aug_length

            melspectrum[:, start_aug_channel:end_aug_channel, i*split_len:(i+1)*split_len] = spec_mean
            melspectrum[:, :, start_aug_length:end_aug_length] = spec_mean

        if (i+1)*split_len < L:
            L_diff = L - (i+1)*split_len

            num_aug_channels = torch.randint(0, self.specaug_maxchannels, (1,)).item()
            start_aug_channel = torch.randint(0, D-num_aug_channels, (1,)).item()
            end_aug_channel = start_aug_channel + num_aug_channels

            num_aug_length = torch.randint(0, int(L_diff*self.specaug_maxlenratio), (1,)).item()
            start_aug_length = torch.randint(0, L_diff-num_aug_length, (1,)).item() + (i+1)*split_len
            end_aug_length = start_aug_length + num_aug_length

            melspectrum[:, start_aug_channel:end_aug_channel, (i+1)*split_len:] = spec_mean
            melspectrum[:, :, start_aug_length:end_aug_length] = spec_mean

        return melspectrum



    def __getitem__(self, idx):
        # Aktuelle Datei und zugehörigen Speaker finden
        file_id = self.file_ids[idx]

        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None

        # Lade das Haupt-Audio
        audio_fn = os.path.join(self.audio_dir, file_id+'.wav')
        if not os.path.exists(audio_fn):
            print(f"File {audio_fn} does not exist.", flush=True)
            print(klajsd)

        waveform, sample_rate = torchaudio.load(audio_fn)
        # Resample auf 16kHz falls nötig
        if sample_rate != 16000:
            resample_transform = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resample_transform(waveform)
            sample_rate = 16000

        if waveform.shape[1] > self.max_duration * sample_rate:
            waveform = waveform[:,:self.max_duration*sample_rate]

        # Prepare noisy audio
        snr_value = self.didicated__snr_list[idx]
        noise_fn = self.didicated__noise_fid_list[idx]

        # if not os.path.exists(noise_fn):
        #     print(f"File {noise_fn} does not exist.", flush=True)
        #     print(klajsd)
        
        if snr_value > 30:
            waveform_noisy = waveform
            noise = torch.zeros_like(waveform)
        else:
            if not os.path.exists(noise_fn):
                print(f"File {noise_fn} does not exist.", flush=True)
                print(klajsd)
            noise_start_offset = self.noise_start_offset[idx]
            waveform_noisy, noise = self.add_noise(noise_fn, waveform, snr_value, noise_start_offset)

        # Video
        video_fn = os.path.join(self.video_dir, file_id+'.mp4')
        if not os.path.exists(video_fn):
            print(f"File {video_fn} does not exist.", flush=True)
            print(klajsd)
        frames = self.load_video(video_fn)
        # frames = frames.unsqueeze(0)
        frames = self.transforms_seq(frames.float())
        frames = frames.unsqueeze(0)

        # video shape B, L, C, H, W [1,L,3,H,W]
        # audio shape B, L      
        min_frames = min([int(30*self.ds_config['video_fps']) , frames.shape[1], int(waveform.shape[1] * self.ds_config['video_fps'] / sample_rate)])
        waveform = waveform[:,:int(min_frames * sample_rate / self.ds_config['video_fps'])]
        waveform_noisy = waveform_noisy[:,:int(min_frames * sample_rate / self.ds_config['video_fps'])]
        waveform_noisy_no_pad = waveform_noisy.clone()
        noise = noise[:,:int(min_frames * sample_rate / self.ds_config['video_fps'])]
        frames = frames[:,:min_frames]

        # Padding
        waveB, waveL = waveform.shape
        waveform_pad = torch.zeros((waveB, 30*sample_rate - waveL))
        waveform = torch.cat([waveform, waveform_pad], dim=1)
        waveform_noisy = torch.cat([waveform_noisy, waveform_pad], dim=1)
        noise = torch.cat([noise, waveform_pad], dim=1)

        framesB, framesL, framesC, framesH, framesW = frames.shape
        frames_mean = torch.mean(frames)
        # frames_pad = torch.ones((framesB, 30*self.ds_config['video_fps']-framesL, framesC, framesH, framesW)) * frames_mean
        frames_pad = torch.zeros((framesB, 30*self.ds_config['video_fps']-framesL, framesC, framesH, framesW))
        frames_padded = torch.cat([frames, frames_pad], dim=1)

        inputs_informations_audio = self.whisper_processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True)
        mel_spec_clean = inputs_informations_audio.input_features

        inputs_informations_audio_noisy = self.whisper_processor(waveform_noisy.squeeze(), sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True)
        mel_spec_noisy = inputs_informations_audio_noisy.input_features


        inputs_informations_noise = self.whisper_processor(noise.squeeze(), sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True)
        mel_spec_noise_only = inputs_informations_noise.input_features

        # Spec augment noisy mel spectrum
        mel_spec_noisy_no_aug = mel_spec_noisy.clone()
        if self.stage == 'train':
            if 'specaug_strategy' in self.ds_config and self.ds_config['specaug_strategy'] == 'modSpecAug':
                mel_spec_noisy[:, :, :int(4*min_frames)] = self.augment_spec_multi(mel_spec_noisy[:, :, :int(4*min_frames)])
            else:
                mel_spec_noisy[:, :, :int(4*min_frames)] = self.augment_spec(mel_spec_noisy[:, :, :int(4*min_frames)])

        attention_mask_clean = inputs_informations_audio.attention_mask

        # Melspectrum zero padding to make training possible
        mel_spec_clean[:, :, int(4*min_frames):] = 0
        mel_spec_noisy[:, :, int(4*min_frames):] = 0
        mel_spec_noisy_no_aug[:, :, int(4*min_frames):] = 0
        mel_spec_noise_only[:, :, int(4*min_frames):] = 0

        return {
            'mel_spec_clean': mel_spec_clean,
            'mel_spec_noisy': mel_spec_noisy,
            'mel_spec_noisy_no_aug': mel_spec_noisy_no_aug,
            'mel_spec_noise_only': mel_spec_noise_only,
            'attention_mask_clean': attention_mask_clean,
            # 'video_frames': frames,
            'video_frames_padded': frames_padded,
            'video_len': min_frames,
            'waveform_noisy': waveform_noisy_no_pad,
            'file_id': file_id,
            'label': label,
            'SNR': snr_value,
            'noise_fid': noise_fn
            }


    def collate_fn(self, batch):
        """
        Collate function to process and pad batches of samples.

        Args:
            batch: List of samples, where each sample is (mel_spec, embedding, speaker_id, file_id).

        Returns:
            Padded batch of mel spectrograms, tensor of embeddings, list of speaker IDs, list of file IDs.
        """
        mel_specs_clean = []
        mel_specs_noisy = []
        mel_specs_noisy_no_aug = []
        mel_specs_noise_only = []
        attentions_mask_clean = []
        video_frames_padded_list = []
        video_lens = []
        waveforms_noisy = []
        speaker_ids = []
        file_ids = []
        labels = []
        SNRs = []
        noise_fids = []

        for b in batch:
            mel_specs_clean.append(b['mel_spec_clean'])  # Transpose to (time, mel_bins)
            mel_specs_noisy.append(b['mel_spec_noisy'])  # Transpose to (time, mel_bins)
            mel_specs_noisy_no_aug.append(b['mel_spec_noisy_no_aug'])  # Transpose to (time, mel_bins)
            mel_specs_noise_only.append(b['mel_spec_noise_only'])  # Transpose to (time, mel_bins)
            attentions_mask_clean.append(b['attention_mask_clean']) 
            video_frames_padded_list.append(b['video_frames_padded'])
            video_lens.append(b['video_len'])
            waveforms_noisy.append(b['waveform_noisy'])
            file_ids.append(b['file_id'])
            labels.append(b['label'])
            SNRs.append(b['SNR'])
            noise_fids.append(b['noise_fid'])

        mel_specs_clean = torch.cat(mel_specs_clean)
        mel_specs_noisy = torch.cat(mel_specs_noisy)
        mel_specs_noisy_no_aug = torch.cat(mel_specs_noisy_no_aug)
        mel_specs_noise_only = torch.cat(mel_specs_noise_only)
        attentions_mask_clean = torch.cat(attentions_mask_clean)
        video_frames_padded_list = torch.cat(video_frames_padded_list)
        video_frames_list = video_frames_padded_list[:,:max(video_lens)]
        video_lens = torch.tensor(video_lens)

        return {
            'mel_specs_clean': mel_specs_clean,
            'mel_specs_noisy': mel_specs_noisy,
            'mel_specs_noisy_no_aug': mel_specs_noisy_no_aug,
            'mel_specs_noise_only': mel_specs_noise_only,
            'attentions_mask_clean': attentions_mask_clean,
            'video_frames_padded': video_frames_padded_list,
            'video_frames': video_frames_list,
            'waveforms_noisy': waveforms_noisy,
            'video_lens': video_lens,
            'file_ids': file_ids,
            'labels': labels,
            'SNRs': SNRs,
            'noise_fids': noise_fids,
        }

