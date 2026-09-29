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



class AVSR_Dataset__noEmb(Dataset):
    def __init__(
                self, 
                ds_config, 
                audio_dir,
                video_dir,
                fid_list_path,
                label_list_path,
            ):
        self.ds_config = ds_config
        self.audio_dir = audio_dir
        self.video_dir = video_dir
        self.fid_list_path = fid_list_path
        self.label_list_path = label_list_path
        
        # Set whisper processor for melspec generation
        self.whisper_processor = AutoProcessor.from_pretrained(ds_config['whisper_modelname'])


        # Lade fids
        with open(self.fid_list_path, 'r') as f:
            self.file_ids = [line.strip() for line in f]

        if self.label_list_path is not None:
            with open(self.label_list_path, 'r') as f:
                self.labels = [line.strip() for line in f]
        else:
            self.labels = None

        # Transformation # input must bis [L, 3, H, W] 
        transform_seq = list()
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


    def __getitem__(self, idx):
        # Aktuelle Datei und zugehörigen Speaker finden
        file_id = self.file_ids[idx]

        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None

        # Lade das Haupt-Audio
        audio_fn = os.path.join(self.audio_dir, file_id+'.wav')
        waveform, sample_rate = torchaudio.load(audio_fn)
        # Resample auf 16kHz falls nötig
        if sample_rate != 16000:
            resample_transform = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resample_transform(waveform)
            sample_rate = 16000

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
        frames = frames[:,:min_frames]

        # Padding
        waveB, waveL = waveform.shape
        waveform_pad = torch.zeros((waveB, 30*sample_rate - waveL))
        waveform = torch.cat([waveform, waveform_pad], dim=1)

        framesB, framesL, framesC, framesH, framesW = frames.shape
        frames_mean = torch.mean(frames)
        # frames_pad = torch.ones((framesB, 30*self.ds_config['video_fps']-framesL, framesC, framesH, framesW)) * frames_mean
        frames_pad = torch.zeros((framesB, 30*self.ds_config['video_fps']-framesL, framesC, framesH, framesW))
        frames_padded = torch.cat([frames, frames_pad], dim=1)

        inputs_informations_audio = self.whisper_processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt", return_attention_mask=True)
        mel_spec = inputs_informations_audio.input_features

        attention_mask_clean = inputs_informations_audio.attention_mask

        # Melspectrum zero padding to make training possible
        mel_spec[:, :, int(4*min_frames):] = 0

        return {
            'mel_spec': mel_spec,
            'attention_mask_clean': attention_mask_clean,
            'video_frames_padded': frames_padded,
            'video_len': min_frames,
            'file_id': file_id,
            'label': label
            }


    def collate_fn(self, batch):
        """
        Collate function to process and pad batches of samples.

        Args:
            batch: List of samples, where each sample is (mel_spec, embedding, speaker_id, file_id).

        Returns:
            Padded batch of mel spectrograms, tensor of embeddings, list of speaker IDs, list of file IDs.
        """
        mel_specs = []
        attentions_mask_clean = []
        video_frames_padded_list = []
        video_lens = []
        speaker_ids = []
        file_ids = []
        labels = []

        for b in batch:
            mel_specs.append(b['mel_spec'])  # Transpose to (time, mel_bins)
            attentions_mask_clean.append(b['attention_mask_clean']) 
            video_frames_padded_list.append(b['video_frames_padded'])
            video_lens.append(b['video_len'])
            file_ids.append(b['file_id'])
            labels.append(b['label'])

        mel_specs = torch.cat(mel_specs)
        attentions_mask_clean = torch.cat(attentions_mask_clean)
        video_frames_padded_list = torch.cat(video_frames_padded_list)
        video_frames_list = video_frames_padded_list[:,:max(video_lens)]
        video_lens = torch.tensor(video_lens)

        return {
            'mel_specs': mel_specs,
            'attentions_mask_clean': attentions_mask_clean,
            'video_frames_padded': video_frames_padded_list,
            'video_frames': video_frames_list,
            'video_lens': video_lens,
            'file_ids': file_ids,
            'labels': labels,
        }
