import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
from typing import Union


from models.utils.utils import get_chunks, recover_chunks, custom_MHA_wChunking, custom_MHA_wChunking_wPrefixAdaptation, custom_MHA_wChunking_wPrefixAdaptation_KV #, custom_MHA_wChunking_wAttnMasking
from models.utils.feature_extraction import LIPNET_CNN, spec_frontend_CNN #, spec_frontend_CNN_woMask


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x.float()).type(x.dtype)


class SAttn_wConcat_AV_Fusion_layer(nn.Module):

    def __init__(self, 
                dim,
                ffn_dim,
                heads,
                dropout=0.1,
                audio_input_dim=None,
                video_input_dim=None,
                chunk_processing=False,
                chunk_sizes_audio_qkv=None,
                step_sizes_audio_qkv=None, 
                chunk_sizes_video_qkv=None,
                step_sizes_video_qkv=None
            ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.heads = heads
        self.dropout = dropout
        self.audio_input_dim = audio_input_dim
        self.video_input_dim = video_input_dim
        self.chunk_processing = chunk_processing
        self.chunk_sizes_audio_qkv = chunk_sizes_audio_qkv
        self.step_sizes_audio_qkv = step_sizes_audio_qkv

        # Self-attention layers for audio and video and layer normalization
        self.sattn = custom_MHA_wChunking(dim=dim, heads=heads, dropout=dropout, chunk_processing=chunk_processing, chunk_sizes_qkv=chunk_sizes_audio_qkv, step_sizes_qkv=step_sizes_audio_qkv) #, MHA_mode=MHA_mode)
        # self.ln1 = nn.LayerNorm(dim)
        self.ln1 = LayerNorm(dim)

        # Feed-forward networks and layer normalization
        self.ffn = nn.Linear(self.dim, self.dim)
        # self.ln2 = nn.LayerNorm(dim)
        self.ln2 = LayerNorm(dim)

        print(f"### SAttn_wConcat_AV_Fusion_layer with FP32 Layernorm ###\n")


    # def forward(self, audio: torch.Tensor, video: torch.Tensor, lens=None):
    def forward(self, 
                audio_video: torch.Tensor,
                **kwargs):
        
        # Apply first self-attention
        audio_video_, _ = self.sattn(audio_video, audio_video, audio_video)
        audio_video = self.ln1(audio_video + audio_video_)

        # Apply feed forward network and layer normalization
        audio_video_ = self.ffn(audio_video)
        audio_video = self.ln2(audio_video + audio_video_)

        return audio_video

class SAttn_wConcat_AV_Fusion(nn.Module):
    """
    My Multi-Layer Single-Side-Cross-Attention Audio-Visual Fusion Module with standard sequence of decoder blocks but with query from video and key and value from audio.
    The sequence is: Self-Attention → Add+Norm → Cross-Attention → Add+Norm → Feed-Forward → Add+Norm.
    Video is used to improve audio (filtering audio features based on video information).
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.config_preprocessor_audio_video = config['config_preprocessor_audio_video']

        # Information for AV_Fusion_Module_preprocessor
        self.video_feature_extractor = self.config_preprocessor_audio_video['video_feature_extractor']
        self.Lipnet_video_layer = self.config_preprocessor_audio_video['Lipnet_video_layer']
        self.Lipnet_emb_size = self.config_preprocessor_audio_video['Lipnet_emb_size']
        self.Lipnet_num_input_channels = self.config_preprocessor_audio_video['Lipnet_num_input_channels']
        self.audio_feature_extractor = self.config_preprocessor_audio_video['audio_feature_extractor']
        self.specfront_procChannels = self.config_preprocessor_audio_video['Specfront_procChannels']
        self.specfront_layerNum = self.config_preprocessor_audio_video['Specfront_layerNum']
        self.specfront_inp_dim = self.config_preprocessor_audio_video['Specfront_inp_dim']
        self.specfront_out_dim = self.config_preprocessor_audio_video['Specfront_out_dim']

        # Fusion module parameters
        self.processing_dim = config['processing_dim']
        self.ffn_dim = config['ffn_dim']
        self.out_dim = config['out_dim']
        self.heads = config['heads']
        self.num_layer = config['num_layer']
        self.dropout = config['dropout']
        self.chunk_processing = config['chunk_processing']
        self.chunk_sizes_audio_qkv = config['chunk_sizes_audio_qkv']
        self.step_sizes_audio_qkv = config['step_sizes_audio_qkv']
        self.chunk_sizes_video_qkv = config['chunk_sizes_video_qkv']
        self.step_sizes_video_qkv = config['step_sizes_video_qkv']
        self.fusion_do_masking = config['fusion_do_masking']

        print(f"### SAttn_wConcat_AV_Fusion ###\n")
        print(f"### Config: {self.config} ###\n")

        self.spec_frontend  = spec_frontend_CNN(processing_channels=self.specfront_procChannels, layerNum=self.specfront_layerNum, inp_dim=self.specfront_inp_dim, out_dim=self.specfront_out_dim)
        self.video_frontend = LIPNET_CNN(layers=self.Lipnet_video_layer, emb_size=self.Lipnet_emb_size, num_input_channels=self.Lipnet_num_input_channels)

        self.features_proj = nn.Linear(int(self.Lipnet_emb_size + self.specfront_out_dim), self.processing_dim)

        # Create a list of SAttn_wConcat_AV_Fusion_layer layers
        self.mlca_layers = nn.ModuleList([
                SAttn_wConcat_AV_Fusion_layer(
                    dim=self.processing_dim,
                    ffn_dim=self.ffn_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    chunk_processing=self.chunk_processing,
                    chunk_sizes_audio_qkv=self.chunk_sizes_audio_qkv,
                    step_sizes_audio_qkv=self.step_sizes_audio_qkv,
                    chunk_sizes_video_qkv=self.chunk_sizes_video_qkv,
                    step_sizes_video_qkv=self.step_sizes_video_qkv,
                ) for _ in range(self.num_layer)
            ])

        self.out_proj = nn.Linear(self.processing_dim, self.out_dim)


    
    def forward(self, audio, video, vid_lens=None, **kwargs):
        """
        Forward pass through the multi-layer cross-attention for audio-visual fusion.
        1. Get lens_ratio from vid_lens in a range from 0 to 1
        2. Preprocess audio and video inputs to get full length audio and video tensors
        3. Prepare masks for audio and video tensors
        4. Reduce audio and video tensors and masks to necessary length based on lens_ratio
        
        Args:
            audio (torch.Tensor): Audio features of shape (B, D, T_audio).
            video (torch.Tensor): Video frames of shape (B, T_video, C, H, W)
        Returns:
            torch.Tensor: Fused audio features of shape (B, T_audio, D).
            torch.Tensor: Fused video features of shape (B, T_video, D).
        """
        
        if vid_lens is None:
            vid_lens = [video.shape[1]] * video.shape[0]

        B_audio, D_audio, T_audio = audio.shape

        inp_video_mod = video[:, :max(vid_lens)]
        inp_spec_mod  = audio[:,:,:int(4*max(vid_lens))]

        spec_features  = self.spec_frontend(inp_spec_mod).transpose(1,2)
        video_features = self.video_frontend(inp_video_mod, vid_lens).transpose(1,2)

        # Padding to satisfy necessary block size
        sequ_len = spec_features.shape[1]
        if sequ_len%self.chunk_sizes_audio_qkv[0]!=0:
            padding_size = 2 * self.chunk_sizes_audio_qkv[0] - sequ_len%self.chunk_sizes_audio_qkv[0]
        else:
            padding_size = self.chunk_sizes_audio_qkv[0]
        spec_features  = torch.nn.functional.pad(spec_features,  (0,0,0,padding_size), "constant", 0)
        video_features = torch.nn.functional.pad(video_features, (0,0,0,padding_size), "constant", 0)

        cat_features = torch.cat((spec_features, video_features), dim=2)
        cat_features = self.features_proj(cat_features)

        # Iterate through each layer of the multi-layer cross-attention
        for layer in self.mlca_layers:
            cat_features = layer(audio_video=cat_features)

        cat_features = self.out_proj(cat_features)

        # Pad to original length
        pad_len = T_audio - cat_features.shape[1]
        if pad_len>0:
            cat_features = torch.nn.functional.pad(cat_features, (0, 0, 0, pad_len), mode='constant', value=0)
        elif pad_len<0:
            cat_features = cat_features[:,:T_audio,:]
        
        for b in range(B_audio):
            cat_features[b, int(4*vid_lens[b]):] = 0.0
        
        cat_features = cat_features.transpose(1, 2)  # Transpose back to

        return cat_features, None, None 








class SAttn_wConcat_AV_Fusion_wNoiseAdaptPrefix_layer(nn.Module):

    def __init__(self, 
                dim,
                ffn_dim,
                heads,
                dropout=0.1,
                audio_input_dim=None,
                video_input_dim=None,
                chunk_processing=False,
                chunk_sizes_audio_qkv=None,
                step_sizes_audio_qkv=None, 
                chunk_sizes_video_qkv=None,
                step_sizes_video_qkv=None
            ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.heads = heads
        self.dropout = dropout
        self.audio_input_dim = audio_input_dim
        self.video_input_dim = video_input_dim
        self.chunk_processing = chunk_processing
        self.chunk_sizes_audio_qkv = chunk_sizes_audio_qkv
        self.step_sizes_audio_qkv = step_sizes_audio_qkv

        # Self-attention layers for audio and video and layer normalization
        self.sattn = custom_MHA_wChunking_wPrefixAdaptation(dim=dim, heads=heads, dropout=dropout, chunk_processing=chunk_processing, chunk_sizes_qkv=chunk_sizes_audio_qkv, step_sizes_qkv=step_sizes_audio_qkv) #, MHA_mode=MHA_mode)
        # self.ln1 = nn.LayerNorm(dim)
        self.ln1 = LayerNorm(dim)

        # Feed-forward networks and layer normalization
        self.ffn = nn.Linear(self.dim, self.dim)
        # self.ln2 = nn.LayerNorm(dim)
        self.ln2 = LayerNorm(dim)

        print(f"### SAttn_wConcat_AV_Fusion_wNoiseAdaptPrefix_layer with FP32 Layernorm ###\n")


    # def forward(self, audio: torch.Tensor, video: torch.Tensor, lens=None):
    def forward(self, 
                audio_video: torch.Tensor,
                adapt_embedding: torch.Tensor,
                **kwargs):
        
        # Apply first self-attention
        audio_video_, _ = self.sattn(query=audio_video, key=audio_video, value=audio_video, adapt_embedding=adapt_embedding)
        audio_video = self.ln1(audio_video + audio_video_)

        # Apply feed forward network and layer normalization
        audio_video_ = self.ffn(audio_video)
        audio_video = self.ln2(audio_video + audio_video_)

        return audio_video

class SAttn_wConcat_AV_Fusion_wNoiseAdaptPrefix(nn.Module):
    """
    My Multi-Layer Single-Side-Cross-Attention Audio-Visual Fusion Module with standard sequence of decoder blocks but with query from video and key and value from audio.
    The sequence is: Self-Attention → Add+Norm → Cross-Attention → Add+Norm → Feed-Forward → Add+Norm.
    Video is used to improve audio (filtering audio features based on video information).
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.config_preprocessor_audio_video = config['config_preprocessor_audio_video']

        # Information for AV_Fusion_Module_preprocessor
        self.video_feature_extractor = self.config_preprocessor_audio_video['video_feature_extractor']
        self.Lipnet_video_layer = self.config_preprocessor_audio_video['Lipnet_video_layer']
        self.Lipnet_emb_size = self.config_preprocessor_audio_video['Lipnet_emb_size']
        self.Lipnet_num_input_channels = self.config_preprocessor_audio_video['Lipnet_num_input_channels']
        self.audio_feature_extractor = self.config_preprocessor_audio_video['audio_feature_extractor']
        self.specfront_procChannels = self.config_preprocessor_audio_video['Specfront_procChannels']
        self.specfront_layerNum = self.config_preprocessor_audio_video['Specfront_layerNum']
        self.specfront_inp_dim = self.config_preprocessor_audio_video['Specfront_inp_dim']
        self.specfront_out_dim = self.config_preprocessor_audio_video['Specfront_out_dim']

        # Fusion module parameters
        self.processing_dim = config['processing_dim']
        self.ffn_dim = config['ffn_dim']
        self.out_dim = config['out_dim']
        self.heads = config['heads']
        self.num_layer = config['num_layer']
        self.dropout = config['dropout']
        self.chunk_processing = config['chunk_processing']
        self.chunk_sizes_audio_qkv = config['chunk_sizes_audio_qkv']
        self.step_sizes_audio_qkv = config['step_sizes_audio_qkv']
        self.chunk_sizes_video_qkv = config['chunk_sizes_video_qkv']
        self.step_sizes_video_qkv = config['step_sizes_video_qkv']
        self.fusion_do_masking = config['fusion_do_masking']

        print(f"### SAttn_wConcat_AV_Fusion_wNoiseAdaptPrefix ###\n")
        print(f"### Config: {self.config} ###\n")

        self.spec_frontend  = spec_frontend_CNN(processing_channels=self.specfront_procChannels, layerNum=self.specfront_layerNum, inp_dim=self.specfront_inp_dim, out_dim=self.specfront_out_dim)
        self.video_frontend = LIPNET_CNN(layers=self.Lipnet_video_layer, emb_size=self.Lipnet_emb_size, num_input_channels=self.Lipnet_num_input_channels)

        self.features_proj = nn.Linear(int(self.Lipnet_emb_size + self.specfront_out_dim), self.processing_dim)

        # Create a list of SAttn_wConcat_AV_Fusion_layer layers
        self.mlca_layers = nn.ModuleList([
                SAttn_wConcat_AV_Fusion_wNoiseAdaptPrefix_layer(
                    dim=self.processing_dim,
                    ffn_dim=self.ffn_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    chunk_processing=self.chunk_processing,
                    chunk_sizes_audio_qkv=self.chunk_sizes_audio_qkv,
                    step_sizes_audio_qkv=self.step_sizes_audio_qkv,
                    chunk_sizes_video_qkv=self.chunk_sizes_video_qkv,
                    step_sizes_video_qkv=self.step_sizes_video_qkv,
                ) for _ in range(self.num_layer)
            ])

        self.out_proj = nn.Linear(self.processing_dim, self.out_dim)


    
    def forward(self, audio, video, adapt_embedding, vid_lens=None, **kwargs):
        """
        Forward pass through the multi-layer cross-attention for audio-visual fusion.
        1. Get lens_ratio from vid_lens in a range from 0 to 1
        2. Preprocess audio and video inputs to get full length audio and video tensors
        3. Prepare masks for audio and video tensors
        4. Reduce audio and video tensors and masks to necessary length based on lens_ratio
        
        Args:
            audio (torch.Tensor): Audio features of shape (B, D, T_audio).
            video (torch.Tensor): Video frames of shape (B, T_video, C, H, W)
        Returns:
            torch.Tensor: Fused audio features of shape (B, T_audio, D).
            torch.Tensor: Fused video features of shape (B, T_video, D).
        """
        
        if vid_lens is None:
            vid_lens = [video.shape[1]] * video.shape[0]

        B_audio, D_audio, T_audio = audio.shape

        inp_video_mod = video[:, :max(vid_lens)]
        inp_spec_mod  = audio[:,:,:int(4*max(vid_lens))]

        spec_features  = self.spec_frontend(inp_spec_mod).transpose(1,2)
        video_features = self.video_frontend(inp_video_mod, vid_lens).transpose(1,2)

        # Padding to satisfy necessary block size
        sequ_len = spec_features.shape[1]
        if sequ_len%self.chunk_sizes_audio_qkv[0]!=0:
            padding_size = 2 * self.chunk_sizes_audio_qkv[0] - sequ_len%self.chunk_sizes_audio_qkv[0]
        else:
            padding_size = self.chunk_sizes_audio_qkv[0]
        spec_features  = torch.nn.functional.pad(spec_features,  (0,0,0,padding_size), "constant", 0)
        video_features = torch.nn.functional.pad(video_features, (0,0,0,padding_size), "constant", 0)

        cat_features = torch.cat((spec_features, video_features), dim=2)
        cat_features = self.features_proj(cat_features)

        # Iterate through each layer of the multi-layer cross-attention
        for layer in self.mlca_layers:
            cat_features = layer(audio_video=cat_features, adapt_embedding=adapt_embedding)

        cat_features = self.out_proj(cat_features)

        # Pad to original length
        pad_len = T_audio - cat_features.shape[1]
        if pad_len>0:
            cat_features = torch.nn.functional.pad(cat_features, (0, 0, 0, pad_len), mode='constant', value=0)
        elif pad_len<0:
            cat_features = cat_features[:,:T_audio,:]
        
        for b in range(B_audio):
            cat_features[b, int(4*vid_lens[b]):] = 0.0
        
        cat_features = cat_features.transpose(1, 2)  # Transpose back to

        return cat_features, None, None 



class SAttn_wConcat_AV_Fusion_wNoiseAdaptPrefix_KV_layer(nn.Module):

    def __init__(self, 
                dim,
                ffn_dim,
                heads,
                dropout=0.1,
                audio_input_dim=None,
                video_input_dim=None,
                chunk_processing=False,
                chunk_sizes_audio_qkv=None,
                step_sizes_audio_qkv=None, 
                chunk_sizes_video_qkv=None,
                step_sizes_video_qkv=None
            ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.heads = heads
        self.dropout = dropout
        self.audio_input_dim = audio_input_dim
        self.video_input_dim = video_input_dim
        self.chunk_processing = chunk_processing
        self.chunk_sizes_audio_qkv = chunk_sizes_audio_qkv
        self.step_sizes_audio_qkv = step_sizes_audio_qkv

        # Self-attention layers for audio and video and layer normalization
        self.sattn = custom_MHA_wChunking_wPrefixAdaptation_KV(dim=dim, heads=heads, dropout=dropout, chunk_processing=chunk_processing, chunk_sizes_qkv=chunk_sizes_audio_qkv, step_sizes_qkv=step_sizes_audio_qkv) #, MHA_mode=MHA_mode)
        # self.ln1 = nn.LayerNorm(dim)
        self.ln1 = LayerNorm(dim)

        # Feed-forward networks and layer normalization
        self.ffn = nn.Linear(self.dim, self.dim)
        # self.ln2 = nn.LayerNorm(dim)
        self.ln2 = LayerNorm(dim)

        print(f"### SAttn_wConcat_AV_Fusion_wNoiseAdaptPrefix_KV_layer with FP32 Layernorm ###\n")


    # def forward(self, audio: torch.Tensor, video: torch.Tensor, lens=None):
    def forward(self, 
                audio_video: torch.Tensor,
                adapt_embedding: torch.Tensor,
                **kwargs):
        
        # Apply first self-attention
        audio_video_, _ = self.sattn(query=audio_video, key=audio_video, value=audio_video, adapt_embedding=adapt_embedding)
        audio_video = self.ln1(audio_video + audio_video_)

        # Apply feed forward network and layer normalization
        audio_video_ = self.ffn(audio_video)
        audio_video = self.ln2(audio_video + audio_video_)

        return audio_video

class SAttn_wConcat_AV_Fusion_wNoiseAdaptPrefix_KV(nn.Module):
    """
    My Multi-Layer Single-Side-Cross-Attention Audio-Visual Fusion Module with standard sequence of decoder blocks but with query from video and key and value from audio.
    The sequence is: Self-Attention → Add+Norm → Cross-Attention → Add+Norm → Feed-Forward → Add+Norm.
    Video is used to improve audio (filtering audio features based on video information).
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.config_preprocessor_audio_video = config['config_preprocessor_audio_video']

        # Information for AV_Fusion_Module_preprocessor
        self.video_feature_extractor = self.config_preprocessor_audio_video['video_feature_extractor']
        self.Lipnet_video_layer = self.config_preprocessor_audio_video['Lipnet_video_layer']
        self.Lipnet_emb_size = self.config_preprocessor_audio_video['Lipnet_emb_size']
        self.Lipnet_num_input_channels = self.config_preprocessor_audio_video['Lipnet_num_input_channels']
        self.audio_feature_extractor = self.config_preprocessor_audio_video['audio_feature_extractor']
        self.specfront_procChannels = self.config_preprocessor_audio_video['Specfront_procChannels']
        self.specfront_layerNum = self.config_preprocessor_audio_video['Specfront_layerNum']
        self.specfront_inp_dim = self.config_preprocessor_audio_video['Specfront_inp_dim']
        self.specfront_out_dim = self.config_preprocessor_audio_video['Specfront_out_dim']

        # Fusion module parameters
        self.processing_dim = config['processing_dim']
        self.ffn_dim = config['ffn_dim']
        self.out_dim = config['out_dim']
        self.heads = config['heads']
        self.num_layer = config['num_layer']
        self.dropout = config['dropout']
        self.chunk_processing = config['chunk_processing']
        self.chunk_sizes_audio_qkv = config['chunk_sizes_audio_qkv']
        self.step_sizes_audio_qkv = config['step_sizes_audio_qkv']
        self.chunk_sizes_video_qkv = config['chunk_sizes_video_qkv']
        self.step_sizes_video_qkv = config['step_sizes_video_qkv']
        self.fusion_do_masking = config['fusion_do_masking']

        print(f"### SAttn_wConcat_AV_Fusion_wNoiseAdaptPrefix_KV ###\n")
        print(f"### Config: {self.config} ###\n")

        self.spec_frontend  = spec_frontend_CNN(processing_channels=self.specfront_procChannels, layerNum=self.specfront_layerNum, inp_dim=self.specfront_inp_dim, out_dim=self.specfront_out_dim)
        self.video_frontend = LIPNET_CNN(layers=self.Lipnet_video_layer, emb_size=self.Lipnet_emb_size, num_input_channels=self.Lipnet_num_input_channels)

        self.features_proj = nn.Linear(int(self.Lipnet_emb_size + self.specfront_out_dim), self.processing_dim)

        # Create a list of SAttn_wConcat_AV_Fusion_layer layers
        self.mlca_layers = nn.ModuleList([
                SAttn_wConcat_AV_Fusion_wNoiseAdaptPrefix_KV_layer(
                    dim=self.processing_dim,
                    ffn_dim=self.ffn_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    chunk_processing=self.chunk_processing,
                    chunk_sizes_audio_qkv=self.chunk_sizes_audio_qkv,
                    step_sizes_audio_qkv=self.step_sizes_audio_qkv,
                    chunk_sizes_video_qkv=self.chunk_sizes_video_qkv,
                    step_sizes_video_qkv=self.step_sizes_video_qkv,
                ) for _ in range(self.num_layer)
            ])

        self.out_proj = nn.Linear(self.processing_dim, self.out_dim)


    
    def forward(self, audio, video, adapt_embedding, vid_lens=None, **kwargs):
        """
        Forward pass through the multi-layer cross-attention for audio-visual fusion.
        1. Get lens_ratio from vid_lens in a range from 0 to 1
        2. Preprocess audio and video inputs to get full length audio and video tensors
        3. Prepare masks for audio and video tensors
        4. Reduce audio and video tensors and masks to necessary length based on lens_ratio
        
        Args:
            audio (torch.Tensor): Audio features of shape (B, D, T_audio).
            video (torch.Tensor): Video frames of shape (B, T_video, C, H, W)
        Returns:
            torch.Tensor: Fused audio features of shape (B, T_audio, D).
            torch.Tensor: Fused video features of shape (B, T_video, D).
        """
        
        if vid_lens is None:
            vid_lens = [video.shape[1]] * video.shape[0]

        B_audio, D_audio, T_audio = audio.shape

        inp_video_mod = video[:, :max(vid_lens)]
        inp_spec_mod  = audio[:,:,:int(4*max(vid_lens))]

        spec_features  = self.spec_frontend(inp_spec_mod).transpose(1,2)
        video_features = self.video_frontend(inp_video_mod, vid_lens).transpose(1,2)

        # Padding to satisfy necessary block size
        sequ_len = spec_features.shape[1]
        if sequ_len%self.chunk_sizes_audio_qkv[0]!=0:
            padding_size = 2 * self.chunk_sizes_audio_qkv[0] - sequ_len%self.chunk_sizes_audio_qkv[0]
        else:
            padding_size = self.chunk_sizes_audio_qkv[0]
        spec_features  = torch.nn.functional.pad(spec_features,  (0,0,0,padding_size), "constant", 0)
        video_features = torch.nn.functional.pad(video_features, (0,0,0,padding_size), "constant", 0)

        cat_features = torch.cat((spec_features, video_features), dim=2)
        cat_features = self.features_proj(cat_features)

        # Iterate through each layer of the multi-layer cross-attention
        for layer in self.mlca_layers:
            cat_features = layer(audio_video=cat_features, adapt_embedding=adapt_embedding)

        cat_features = self.out_proj(cat_features)

        # Pad to original length
        pad_len = T_audio - cat_features.shape[1]
        if pad_len>0:
            cat_features = torch.nn.functional.pad(cat_features, (0, 0, 0, pad_len), mode='constant', value=0)
        elif pad_len<0:
            cat_features = cat_features[:,:T_audio,:]
        
        for b in range(B_audio):
            cat_features[b, int(4*vid_lens[b]):] = 0.0
        
        cat_features = cat_features.transpose(1, 2)  # Transpose back to

        return cat_features, None, None 





class SAttn_wConcat_AV_Fusion_wNoiseAdaptCAttn_layer(nn.Module):

    def __init__(self, 
                dim,
                ffn_dim,
                heads,
                heads_noise_emb,
                dropout=0.1,
                audio_input_dim=None,
                video_input_dim=None,
                chunk_processing=False,
                chunk_sizes_audio_qkv=None,
                step_sizes_audio_qkv=None, 
                chunk_sizes_video_qkv=None,
                step_sizes_video_qkv=None
            ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.heads = heads
        self.heads_noise_emb = heads_noise_emb # For noise embedding adaptation in cross-attention
        self.dropout = dropout
        self.audio_input_dim = audio_input_dim
        self.video_input_dim = video_input_dim
        self.chunk_processing = chunk_processing
        self.chunk_sizes_audio_qkv = chunk_sizes_audio_qkv
        self.step_sizes_audio_qkv = step_sizes_audio_qkv

        # Self-attention layers for audio and video and layer normalization
        self.cattn_NoiseEmb = custom_MHA_wChunking(dim=dim, heads=heads, dropout=dropout, chunk_processing=False, chunk_sizes_qkv=None, step_sizes_qkv=None) #, MHA_mode=MHA_mode)
        # self.ln1 = nn.LayerNorm(dim)
        self.ln1_NoiseEmb = LayerNorm(dim)

        # Self-attention layers for audio and video and layer normalization
        self.sattn = custom_MHA_wChunking(dim=dim, heads=heads, dropout=dropout, chunk_processing=chunk_processing, chunk_sizes_qkv=chunk_sizes_audio_qkv, step_sizes_qkv=step_sizes_audio_qkv) #, MHA_mode=MHA_mode)
        # self.ln1 = nn.LayerNorm(dim)
        self.ln1 = LayerNorm(dim)

        # Feed-forward networks and layer normalization
        self.ffn = nn.Linear(self.dim, self.dim)
        # self.ln2 = nn.LayerNorm(dim)
        self.ln2 = LayerNorm(dim)

        print(f"### SAttn_wConcat_AV_Fusion_wNoiseAdaptCAttn_layer with FP32 Layernorm ###\n")


    # def forward(self, audio: torch.Tensor, video: torch.Tensor, lens=None):
    def forward(self, 
                audio_video: torch.Tensor,
                adapt_embedding: torch.Tensor,
                **kwargs):

        # Apply first cross-attention with noise embedding adaptation
        adapt_embedding_, _ = self.cattn_NoiseEmb(query=audio_video, key=adapt_embedding, value=adapt_embedding)
        audio_video = self.ln1_NoiseEmb(audio_video + adapt_embedding_)

        # Apply first self-attention
        audio_video_, _ = self.sattn(query=audio_video, key=audio_video, value=audio_video, adapt_embedding=adapt_embedding)
        audio_video = self.ln1(audio_video + audio_video_)

        # Apply feed forward network and layer normalization
        audio_video_ = self.ffn(audio_video)
        audio_video = self.ln2(audio_video + audio_video_)

        return audio_video

class SAttn_wConcat_AV_Fusion_wNoiseAdaptCAttn(nn.Module):
    """
    My Multi-Layer Single-Side-Cross-Attention Audio-Visual Fusion Module with standard sequence of decoder blocks but with query from video and key and value from audio.
    The sequence is: Self-Attention → Add+Norm → Cross-Attention → Add+Norm → Feed-Forward → Add+Norm.
    Video is used to improve audio (filtering audio features based on video information).
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.config_preprocessor_audio_video = config['config_preprocessor_audio_video']

        # Information for AV_Fusion_Module_preprocessor
        self.video_feature_extractor = self.config_preprocessor_audio_video['video_feature_extractor']
        self.Lipnet_video_layer = self.config_preprocessor_audio_video['Lipnet_video_layer']
        self.Lipnet_emb_size = self.config_preprocessor_audio_video['Lipnet_emb_size']
        self.Lipnet_num_input_channels = self.config_preprocessor_audio_video['Lipnet_num_input_channels']
        self.audio_feature_extractor = self.config_preprocessor_audio_video['audio_feature_extractor']
        self.specfront_procChannels = self.config_preprocessor_audio_video['Specfront_procChannels']
        self.specfront_layerNum = self.config_preprocessor_audio_video['Specfront_layerNum']
        self.specfront_inp_dim = self.config_preprocessor_audio_video['Specfront_inp_dim']
        self.specfront_out_dim = self.config_preprocessor_audio_video['Specfront_out_dim']

        # Fusion module parameters
        self.processing_dim = config['processing_dim']
        self.ffn_dim = config['ffn_dim']
        self.out_dim = config['out_dim']
        self.heads = config['heads']
        self.heads_noise_emb = config['heads_noise_emb'] # For noise embedding adaptation in cross-attention
        self.num_layer = config['num_layer']
        self.dropout = config['dropout']
        self.chunk_processing = config['chunk_processing']
        self.chunk_sizes_audio_qkv = config['chunk_sizes_audio_qkv']
        self.step_sizes_audio_qkv = config['step_sizes_audio_qkv']
        self.chunk_sizes_video_qkv = config['chunk_sizes_video_qkv']
        self.step_sizes_video_qkv = config['step_sizes_video_qkv']
        self.fusion_do_masking = config['fusion_do_masking']

        print(f"### SAttn_wConcat_AV_Fusion_wNoiseAdaptCAttn ###\n")
        print(f"### Config: {self.config} ###\n")

        self.spec_frontend  = spec_frontend_CNN(processing_channels=self.specfront_procChannels, layerNum=self.specfront_layerNum, inp_dim=self.specfront_inp_dim, out_dim=self.specfront_out_dim)
        self.video_frontend = LIPNET_CNN(layers=self.Lipnet_video_layer, emb_size=self.Lipnet_emb_size, num_input_channels=self.Lipnet_num_input_channels)

        self.features_proj = nn.Linear(int(self.Lipnet_emb_size + self.specfront_out_dim), self.processing_dim)

        # Create a list of SAttn_wConcat_AV_Fusion_layer layers
        self.mlca_layers = nn.ModuleList([
                SAttn_wConcat_AV_Fusion_wNoiseAdaptCAttn_layer(
                    dim=self.processing_dim,
                    ffn_dim=self.ffn_dim,
                    heads=self.heads,
                    heads_noise_emb=self.heads_noise_emb,
                    dropout=self.dropout,
                    chunk_processing=self.chunk_processing,
                    chunk_sizes_audio_qkv=self.chunk_sizes_audio_qkv,
                    step_sizes_audio_qkv=self.step_sizes_audio_qkv,
                    chunk_sizes_video_qkv=self.chunk_sizes_video_qkv,
                    step_sizes_video_qkv=self.step_sizes_video_qkv,
                ) for _ in range(self.num_layer)
            ])

        self.out_proj = nn.Linear(self.processing_dim, self.out_dim)


    
    def forward(self, audio, video, adapt_embedding, vid_lens=None, **kwargs):
        """
        Forward pass through the multi-layer cross-attention for audio-visual fusion.
        1. Get lens_ratio from vid_lens in a range from 0 to 1
        2. Preprocess audio and video inputs to get full length audio and video tensors
        3. Prepare masks for audio and video tensors
        4. Reduce audio and video tensors and masks to necessary length based on lens_ratio
        
        Args:
            audio (torch.Tensor): Audio features of shape (B, D, T_audio).
            video (torch.Tensor): Video frames of shape (B, T_video, C, H, W)
        Returns:
            torch.Tensor: Fused audio features of shape (B, T_audio, D).
            torch.Tensor: Fused video features of shape (B, T_video, D).
        """
        
        if vid_lens is None:
            vid_lens = [video.shape[1]] * video.shape[0]

        B_audio, D_audio, T_audio = audio.shape

        inp_video_mod = video[:, :max(vid_lens)]
        inp_spec_mod  = audio[:,:,:int(4*max(vid_lens))]

        spec_features  = self.spec_frontend(inp_spec_mod).transpose(1,2)
        video_features = self.video_frontend(inp_video_mod, vid_lens).transpose(1,2)

        # Padding to satisfy necessary block size
        sequ_len = spec_features.shape[1]
        if sequ_len%self.chunk_sizes_audio_qkv[0]!=0:
            padding_size = 2 * self.chunk_sizes_audio_qkv[0] - sequ_len%self.chunk_sizes_audio_qkv[0]
        else:
            padding_size = self.chunk_sizes_audio_qkv[0]
        spec_features  = torch.nn.functional.pad(spec_features,  (0,0,0,padding_size), "constant", 0)
        video_features = torch.nn.functional.pad(video_features, (0,0,0,padding_size), "constant", 0)

        cat_features = torch.cat((spec_features, video_features), dim=2)
        cat_features = self.features_proj(cat_features)

        # Iterate through each layer of the multi-layer cross-attention
        for layer in self.mlca_layers:
            cat_features = layer(audio_video=cat_features, adapt_embedding=adapt_embedding)

        cat_features = self.out_proj(cat_features)

        # Pad to original length
        pad_len = T_audio - cat_features.shape[1]
        if pad_len>0:
            cat_features = torch.nn.functional.pad(cat_features, (0, 0, 0, pad_len), mode='constant', value=0)
        elif pad_len<0:
            cat_features = cat_features[:,:T_audio,:]
        
        for b in range(B_audio):
            cat_features[b, int(4*vid_lens[b]):] = 0.0
        
        cat_features = cat_features.transpose(1, 2)  # Transpose back to

        return cat_features, None, None 






class Gating_mechanism(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim, dropout=0.1, act_fun=nn.Tanh):
        super().__init__()
        self.inp_dim = inp_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.gate = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            act_fun(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid()
        )
    
    def get_gate(self, x):
        gate = self.gate(x)
        return gate
    
    def forward(self, x):
        gate = self.gate(x)
        return gate

class SAttn_wConcat_AV_Fusion_wNoiseGatedWeighting(nn.Module):
    """
    My Multi-Layer Single-Side-Cross-Attention Audio-Visual Fusion Module with standard sequence of decoder blocks but with query from video and key and value from audio.
    The sequence is: Self-Attention → Add+Norm → Cross-Attention → Add+Norm → Feed-Forward → Add+Norm.
    Video is used to improve audio (filtering audio features based on video information).
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.config_preprocessor_audio_video = config['config_preprocessor_audio_video']

        # Information for AV_Fusion_Module_preprocessor
        self.video_feature_extractor = self.config_preprocessor_audio_video['video_feature_extractor']
        self.Lipnet_video_layer = self.config_preprocessor_audio_video['Lipnet_video_layer']
        self.Lipnet_emb_size = self.config_preprocessor_audio_video['Lipnet_emb_size']
        self.Lipnet_num_input_channels = self.config_preprocessor_audio_video['Lipnet_num_input_channels']
        self.audio_feature_extractor = self.config_preprocessor_audio_video['audio_feature_extractor']
        self.specfront_procChannels = self.config_preprocessor_audio_video['Specfront_procChannels']
        self.specfront_layerNum = self.config_preprocessor_audio_video['Specfront_layerNum']
        self.specfront_inp_dim = self.config_preprocessor_audio_video['Specfront_inp_dim']
        self.specfront_out_dim = self.config_preprocessor_audio_video['Specfront_out_dim']

        # Fusion module parameters
        self.processing_dim = config['processing_dim']
        self.ffn_dim = config['ffn_dim']
        self.out_dim = config['out_dim']
        self.heads = config['heads']
        self.num_layer = config['num_layer']
        self.dropout = config['dropout']
        self.chunk_processing = config['chunk_processing']
        self.chunk_sizes_audio_qkv = config['chunk_sizes_audio_qkv']
        self.step_sizes_audio_qkv = config['step_sizes_audio_qkv']
        self.chunk_sizes_video_qkv = config['chunk_sizes_video_qkv']
        self.step_sizes_video_qkv = config['step_sizes_video_qkv']
        self.fusion_do_masking = config['fusion_do_masking']

        # # Gating mechanism
        # self.gating_inp_dim = config['gating_inp_dim']
        # self.gating_hidden_dim = config['gating_hidden_dim']
        # self.gating_out_dim = config['gating_out_dim']
        # self.gating_dropout = config['gating_dropout']


        print(f"### SAttn_wConcat_AV_Fusion_wNoiseAdaptCAttn ###\n")
        print(f"### Config: {self.config} ###\n")

        self.spec_frontend  = spec_frontend_CNN(processing_channels=self.specfront_procChannels, layerNum=self.specfront_layerNum, inp_dim=self.specfront_inp_dim, out_dim=self.specfront_out_dim)
        self.video_frontend = LIPNET_CNN(layers=self.Lipnet_video_layer, emb_size=self.Lipnet_emb_size, num_input_channels=self.Lipnet_num_input_channels)

        # self.gater = Gating_mechanism(inp_dim=self.gating_inp_dim, hidden_dim=self.gating_hidden_dim, out_dim=self.gating_out_dim, dropout=self.gating_dropout)

        self.features_proj = nn.Linear(int(self.Lipnet_emb_size + self.specfront_out_dim), self.processing_dim)
        
        # Create a list of SAttn_wConcat_AV_Fusion_layer layers
        self.mlca_layers = nn.ModuleList([
                SAttn_wConcat_AV_Fusion_layer(
                    dim=self.processing_dim,
                    ffn_dim=self.ffn_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    chunk_processing=self.chunk_processing,
                    chunk_sizes_audio_qkv=self.chunk_sizes_audio_qkv,
                    step_sizes_audio_qkv=self.step_sizes_audio_qkv,
                    chunk_sizes_video_qkv=self.chunk_sizes_video_qkv,
                    step_sizes_video_qkv=self.step_sizes_video_qkv,
                ) for _ in range(self.num_layer)
            ])

        self.out_proj = nn.Linear(self.processing_dim, self.out_dim)


    
    def forward(self, audio, video, gate_weights, vid_lens=None, **kwargs):
        """
        Forward pass through the multi-layer cross-attention for audio-visual fusion.
        1. Get lens_ratio from vid_lens in a range from 0 to 1
        2. Preprocess audio and video inputs to get full length audio and video tensors
        3. Prepare masks for audio and video tensors
        4. Reduce audio and video tensors and masks to necessary length based on lens_ratio
        
        Args:
            audio (torch.Tensor): Audio features of shape (B, D, T_audio).
            video (torch.Tensor): Video frames of shape (B, T_video, C, H, W)
        Returns:
            torch.Tensor: Fused audio features of shape (B, T_audio, D).
            torch.Tensor: Fused video features of shape (B, T_video, D).
        """
        
        if vid_lens is None:
            vid_lens = [video.shape[1]] * video.shape[0]

        B_audio, D_audio, T_audio = audio.shape

        inp_video_mod = video[:, :max(vid_lens)]
        inp_spec_mod  = audio[:,:,:int(4*max(vid_lens))]

        spec_features  = self.spec_frontend(inp_spec_mod).transpose(1,2)
        video_features = self.video_frontend(inp_video_mod, vid_lens).transpose(1,2)

        # Padding to satisfy necessary block size
        sequ_len = spec_features.shape[1]
        if sequ_len%self.chunk_sizes_audio_qkv[0]!=0:
            padding_size = 2 * self.chunk_sizes_audio_qkv[0] - sequ_len%self.chunk_sizes_audio_qkv[0]
        else:
            padding_size = self.chunk_sizes_audio_qkv[0]
        spec_features  = torch.nn.functional.pad(spec_features,  (0,0,0,padding_size), "constant", 0)
        video_features = torch.nn.functional.pad(video_features, (0,0,0,padding_size), "constant", 0)

        cat_features = torch.cat((spec_features, video_features), dim=2)
        # gate_weights = self.gater.get_gate(adapt_embedding).unsqueeze(1)
        # Apply gating weights
        cat_features = gate_weights * cat_features
        # Proj to processing dimension
        cat_features = self.features_proj(cat_features)

        # Iterate through each layer of the multi-layer cross-attention
        for layer in self.mlca_layers:
            cat_features = layer(audio_video=cat_features)

        cat_features = self.out_proj(cat_features)

        # Pad to original length
        pad_len = T_audio - cat_features.shape[1]
        if pad_len>0:
            cat_features = torch.nn.functional.pad(cat_features, (0, 0, 0, pad_len), mode='constant', value=0)
        elif pad_len<0:
            cat_features = cat_features[:,:T_audio,:]
        
        for b in range(B_audio):
            cat_features[b, int(4*vid_lens[b]):] = 0.0
        
        cat_features = cat_features.transpose(1, 2)  # Transpose back to

        return cat_features, None, None 







if __name__ == '__main__':

    import time
    
    blocksize_audio = 160
    setpsize_audio = 80
    
    blocksize_video = 160
    setpsize_video = 80
    
    # blocksize_video = 160
    # setpsize_video = 80

    batch_size = 2
    dim = 80
    dim_audio = 80
    dim_video = 80

    # audio = torch.randn(batch_size, 3000, dim_audio) 
    audio = torch.randn(batch_size, dim_audio, 3000) 
    # video = torch.randn(batch_size, 750, dim_video)
    video = torch.randn(2, 750, 1, 88, 88)
    lens = torch.tensor([153, 88]) #, 0.7, 0.9, 1.0, 0.8, 0.7, 0.9]
    lens = [153, 88]  # Example lengths for each batch item

    config1 = {
        'config_preprocessor_audio_video': {
            'video_feature_extractor': 'LIPNET',  # 'mobilenetV2' or 'LIPNET'
            'Lipnet_video_layer': [[2,1],[2,1],[2,1],[3,1]],  # Only used if video_feature_extractor is 'LIPNET'
            'Lipnet_emb_size': 80,  # Only used if video_feature_extractor is 'LIPNET'
            'Lipnet_num_input_channels': 1,  # Only used if video_feature_extractor is 'LIPNET'
            'audio_feature_extractor': 'CNN',  # 'CNN' or None
            'Specfront_procChannels': 128,  # Only used if audio_feature_extractor is 'CNN'
            'Specfront_layerNum': 7,  # Only used if audio_feature_extractor is 'CNN'
            'Specfront_inp_dim': 80,  # Only used if audio_feature_extractor is 'CNN'
            'Specfront_out_dim': 80
        },
        'processing_dim': dim,
        'ffn_dim': 2*dim,
        'out_dim': dim,
        'heads': 6,
        'num_layer': 12,
        'dropout': 0.1,
        'chunk_processing': True,
        'chunk_sizes_audio_qkv': (blocksize_audio, blocksize_audio, blocksize_audio),
        'step_sizes_audio_qkv': (setpsize_audio, setpsize_audio, setpsize_audio),
        'chunk_sizes_video_qkv': (blocksize_video, blocksize_video, blocksize_video),
        'step_sizes_video_qkv': (setpsize_video, setpsize_video, setpsize_video),
        'fusion_do_masking': True,
    }

    config2 = {
        'inp_dim_audio': dim_audio,
        'inp_dim_video': dim_video,
        'dim': 256,
        'processing_dim': 256,
        'out_dim': dim,
        'heads': 8,
        'num_layer': 12,
        'dropout': 0.1,
        'chunk_processing': True,
        'MHA_mode': 'custom_MHA_old_version',  # 'torch_standard' or 'custom_MHA' or 'custom_MHA_old_version'
        'chunk_sizes_audio_qkv': (blocksize_audio, blocksize_audio, blocksize_audio),
        'step_sizes_audio_qkv': (setpsize_audio, setpsize_audio, setpsize_audio),
        'chunk_sizes_video_qkv': (blocksize_video, blocksize_video, blocksize_video),
        'step_sizes_video_qkv': (setpsize_video, setpsize_video, setpsize_video),
        'do_masking': True,
        'do_input_projection': False,
        'do_output_projection': False,
        'video_feature_extractor': 'LIPNET',  # 'mobilenetV2' or 'LIPNET'
        'Lipnet_video_layer': [[2,1],[2,1],[2,1],[3,1]],  # Only used if video_feature_extractor is 'LIPNET'
        'Lipnet_emb_size': 80,  # Only used if video_feature_extractor is 'LIPNET'
        'Lipnet_num_input_channels': 1,  # Only used if video_feature_extractor is 'LIPNET'
        'audio_feature_extractor': 'CNN',  # 'CNN' or None
        'Specfront_masking': True,  # Only used if audio_feature_extractor is 'CNN'
        'Specfront_procChannels': 64,  # Only used if audio_feature_extractor is 'CNN'
        'Specfront_layerNum': 4,  # Only used if audio_feature_extractor is 'CNN'
        'Specfront_inp_dim': 80,  # Only used if audio_feature_extractor is 'CNN'
        'Specfront_out_dim': 80
    }


    # mlca_layer = MLCA_AV_layer_mod(dim=dim, heads=8, chunk_processing=True, chunk_sizes_audio_qkv=(160, 160, 160), step_sizes_audio_qkv=(80, 80, 80), chunk_sizes_video_qkv=(40, 40, 40), step_sizes_video_qkv=(20, 20, 20))
    # mlca = MLCA_AV_Fusion_mod(config=config1)
    # mlca = myMLSCA_AV_Fusion_mod(config=config1)
    # mlca = myMLSCA_AV_Fusion_mod_CAttnFirst(config=config1)
    
    # mlca = SAttn_wConcat_AV_Fusion(config=config1)
    mlca = SAttn_wConcat_AV_Fusion(config=config1)


    # Get number of parameters
    num_params = sum(p.numel() for p in mlca.mlca_layers.parameters() if p.requires_grad)
    print(f"Number of parameters in MLCA_AV_layer_mod: {num_params}")
    t1 = time.time()
    audio_out, _, video_out = mlca(audio, video, vid_lens=lens)
    t2 = time.time()
    print(f"Time taken for forward pass: {t2 - t1:.4f} seconds")

    print(audio)
