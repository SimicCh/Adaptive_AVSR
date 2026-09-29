import torch
from torch import nn
from typing import Optional, Tuple, Union

from models.utils.custom_MHA_old_version import MultiHeadAttention_oldVersion


def get_chunks(x, blocksize, step):
    """
    Get chunks of the input tensor x.
    Args:
        x: Input tensor of shape [B, T, D]
        blocksize: Size of each chunk
        step: Step size for chunking
    Returns:
        Chunks of shape [B, D, Num_patches, Blocksize]
    """
    B, T, D = x.shape
    x_chunked = x.unfold(dimension=1, size=blocksize, step=step).transpose(2,3).reshape(-1, blocksize, D)
    # ######################################################################
    # # Test that chunking and recovering works correctly
    # num_chunks = x_chunked.shape[0] // B
    # max_diff = 0
    # for b in range(B):
    #     for c in range(num_chunks):
    #         x_chunk = x[b, step*c : step*c + blocksize, :].unsqueeze(0)
    #         a_chunk = x_chunked[b*num_chunks + c, :, :].unsqueeze(0)
    #         print(x_chunk)
    #         print(a_chunk)
    #         max_diff = max(max_diff, torch.max(torch.abs(x_chunk.float() - a_chunk.float())).item())
    # print(f'Max difference between original and chunked tensor: {max_diff}')
    return x_chunked

def recover_chunks(x, blocksize, step, batch_size):
    """
    Recover the original tensor from chunks.
    Args:
        x: Chunks of shape [B*Num_patches, Blocksize, D]
        blocksize: Size of each chunk
        step: Step size for chunking
        batch_size: Batch size of the original tensor
    Returns:
        Reconstructed tensor of shape [B, T, D]
    """

    # Get the shape of the input tensor [Bx, Tx, Dx]
    Bx, Tx, Dx = x.shape
    # Reshape the tensor to [Bx, Num_patches, Blocksize, Dx]
    x_resh = x.reshape(batch_size, -1, blocksize, Dx)
    # Reconstruct the tensor to [B, T, D] of original shape
    # First element of first patch, then the middle elements of all patches, and finally the last element of the last patch
    a = x_resh[:,:,int(step/2):Tx-int(step/2),:]
    x_resh = torch.cat([
            x_resh[:,0,:int(step/2),:],
            x_resh[:,:,int(step/2):Tx-int(step/2),:].reshape(batch_size, -1, Dx),
            x_resh[:,-1,-int(step/2):,:]
        ], dim=1)
    ########################################################################
    # # Test that chunking and recovering works correctly
    # num_chunks = Bx // batch_size
    # max_diff = 0
    # for b in range(batch_size):
    #     x_chunk_first = x[b*num_chunks, :step//2, :].unsqueeze(0)
    #     x_resh_chunk_last = x_resh[b, :step//2, :].unsqueeze(0)
    #     max_diff = max(max_diff, torch.max(torch.abs(x_chunk_first - x_resh_chunk_last)).item())
    #     x_chunk_last = x[(b+1)*num_chunks - 1, -step//2:, :].unsqueeze(0)
    #     x_resh_chunk_last = x_resh[b, -step//2:, :].unsqueeze(0)
    #     max_diff = max(max_diff, torch.max(torch.abs(x_chunk_last - x_resh_chunk_last)).item())
    #     for c in range(num_chunks):
    #         start_idx = step//2
    #         end_idx = blocksize - step//2
    #         x_chunk = x[b*num_chunks + c, start_idx:end_idx, :].unsqueeze(0)
    #         x_resh_chunk = x_resh[b, c*step + start_idx:c*step + end_idx, :].unsqueeze(0)
    #         max_diff = max(max_diff, torch.max(torch.abs(x_chunk - x_resh_chunk)).item())
    # print(f'Max difference between original and recovered tensor: {max_diff}')
    return x_resh



class MLP_mapping(nn.Module):
    def __init__(
            self,
            sizes: Tuple[int, ...],
            bias=True,
            act=nn.Tanh,
            dropout=0.0,
    ):
        super(MLP_mapping, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)



class custom_MHA_wChunking(nn.Module):
    """
    Custom Multi-Head Attention module with chunk-wise processing and masking.
    This module allows for processing input tensors in chunks, which can be useful for long sequences, to increase focus on local patterns and reduce memory usage.
    """
    def __init__(self, 
                 dim,
                 heads,
                 dropout=0.1,
                 chunk_processing=False,
                 chunk_sizes_qkv=(160,160,160),
                 step_sizes_qkv=(80,80,80),
                 # MHA_mode='torch_standard'  # 'torch_standard' or 'custom_MHA'
                ):
        """
        Initialize the custom Multi-Head Attention module.
        Args:
            dim (int): Dimension of the input features.
            heads (int): Number of attention heads.
            chunk_processing (bool): Whether to process input in chunks.
            chunk_sizes (tuple): Sizes of chunks for each modality (key, query, value).
            step_sizes (tuple): Step sizes for each modality (key, query, value).
        """
        # Call the parent class constructor
        super().__init__()
        self.chunk_processing = chunk_processing
        self.chunk_sizes_qkv = chunk_sizes_qkv
        self.step_sizes_qkv = step_sizes_qkv
        # self.MHA_mode = MHA_mode

        # Define the Multi-Head Attention mode
        self.attn = MultiHeadAttention_oldVersion(heads=heads, d_model=dim, dropout_prob=dropout)


    def prepare_masks(self, query, key, value, lens=None):
        # Get and store shapes of the input tensors
        Bq, Tq, Dq = query.shape
        Bk, Tk, Dk = key.shape
        Bv, Tv, Dv = value.shape
        # Generate masks for lens if provided
        if lens is not None:
            q_mask = torch.zeros((Bq, Tq), dtype=torch.bool, device=query.device)
            k_mask = torch.zeros((Bk, Tk), dtype=torch.bool, device=key.device)
            v_mask = torch.zeros((Bv, Tv), dtype=torch.bool, device=value.device)
            for i, l in enumerate(lens):
                q_mask[i, :round(l * Tq)] = True
                k_mask[i, :round(l * Tk)] = True
                v_mask[i, :round(l * Tv)] = True
        else:
            q_mask = torch.ones((Bq, Tq), dtype=torch.bool, device=query.device)
            k_mask = torch.ones((Bk, Tk), dtype=torch.bool, device=key.device)
            v_mask = torch.ones((Bv, Tv), dtype=torch.bool, device=value.device)
        return q_mask, k_mask, v_mask
    
    def padding(self, x, chunk_size):
        """
        Pad the input tensor x to ensure its length is divisible by the chunk size.
        Args:
            x: Input tensor of shape [B, T, D]
            chunk_size: Size of the chunks to pad to
        Returns:
            Padded tensor of shape [B, T', D] where T' is the new length after padding
        """
        # Calculate the new length for padding
        # new_len_x = chunk_size * (int(x.shape[1] / chunk_size) + 1)
        new_len_x = x.shape[1] + chunk_size - x.shape[1] % chunk_size if x.shape[1] % chunk_size != 0 else x.shape[1]
        # If the input tensor is shorter than the new length, pad it
        if x.shape[1] < new_len_x:
            x = torch.nn.functional.pad(x, (0, 0, 0, new_len_x - x.shape[1]), mode='constant', value=0)
        return x

    def forward(self, query, key, value, **kwargs): #, query_mask=None, key_mask=None, **kwargs):
        
        # Get and store shapes of the input tensors
        Bq, Tq, Dq = query.shape
        Bk, Tk, Dk = key.shape
        Bv, Tv, Dv = value.shape

        # Padding, if necessary, to ensure the input tensors are divisible by the chunk size
        if self.chunk_processing:
            query = self.padding(query, self.chunk_sizes_qkv[0])
            key = self.padding(key, self.chunk_sizes_qkv[1])
            value = self.padding(value, self.chunk_sizes_qkv[2])

            # if query_mask is not None and key_mask is not None:
            #     query_mask_pad = self.padding(query_mask.unsqueeze(-1), self.chunk_sizes_qkv[0])
            #     key_mask_pad = self.padding(key_mask.unsqueeze(-1), self.chunk_sizes_qkv[1])
            # else:
            #     query_mask_pad = None
            #     key_mask_pad = None

        # If chunk processing is enabled, get chunks of the input tensors
        if self.chunk_processing:
            query_chunks = get_chunks(query, blocksize=self.chunk_sizes_qkv[0], step=self.step_sizes_qkv[0])
            key_chunks = get_chunks(key, blocksize=self.chunk_sizes_qkv[1], step=self.step_sizes_qkv[1])
            value_chunks = get_chunks(value, blocksize=self.chunk_sizes_qkv[2], step=self.step_sizes_qkv[2])

            # if query_mask_pad is not None and key_mask_pad is not None:
            #     query_mask_chunks = get_chunks(query_mask_pad, blocksize=self.chunk_sizes_qkv[0], step=self.step_sizes_qkv[0])
            #     key_mask_chunks = get_chunks(key_mask_pad, blocksize=self.chunk_sizes_qkv[1], step=self.step_sizes_qkv[1])
            #     attn_mask_chunks = torch.matmul(query_mask_chunks.float(), key_mask_chunks.float().transpose(-2, -1)).bool()
            #     # # Check attn_mask_chunks visually
            #     # import matplotlib.pyplot as plt
            #     # for i in range(attn_mask_chunks.shape[0]):
            #     #     print(i)
            #     #     plt.imshow(attn_mask_chunks[i].cpu().numpy())
            #     #     plt.colorbar()
            #     #     plt.show()
            # else:
            #     attn_mask_chunks = None
        else:
            query_chunks = query
            key_chunks = key
            value_chunks = value
            # attn_mask_chunks = None

        # Apply attention on chunks
        attn_out, attn_output_weights = self.attn(query_chunks, key_chunks, value_chunks, **kwargs) #, mask=attn_mask_chunks, **kwargs)

        # Recover the original shape from chunks
        if self.chunk_processing:
            attn_out = recover_chunks(attn_out, blocksize=self.chunk_sizes_qkv[0], step=self.step_sizes_qkv[0], batch_size=Bq)
            attn_output_weights = recover_chunks(attn_output_weights, blocksize=self.chunk_sizes_qkv[0], step=self.step_sizes_qkv[0], batch_size=Bq)
        
        # remove padding if it was applied
        if self.chunk_processing:
            attn_out = attn_out[:, :Tq, :]
            attn_output_weights = attn_output_weights[:, :Tq, :]
        
        return attn_out, attn_output_weights


class custom_MHA_wChunking_wPrefixAdaptation(nn.Module):
    """
    Custom Multi-Head Attention module with chunk-wise processing and masking.
    This module allows for processing input tensors in chunks, which can be useful for long sequences, to increase focus on local patterns and reduce memory usage.
    """
    def __init__(self, 
                 dim,
                 heads,
                 dropout=0.1,
                 chunk_processing=False,
                 chunk_sizes_qkv=(160,160,160),
                 step_sizes_qkv=(80,80,80),
                 # MHA_mode='torch_standard'  # 'torch_standard' or 'custom_MHA'
                ):
        """
        Initialize the custom Multi-Head Attention module.
        Args:
            dim (int): Dimension of the input features.
            heads (int): Number of attention heads.
            chunk_processing (bool): Whether to process input in chunks.
            chunk_sizes (tuple): Sizes of chunks for each modality (key, query, value).
            step_sizes (tuple): Step sizes for each modality (key, query, value).
        """
        # Call the parent class constructor
        super().__init__()
        self.chunk_processing = chunk_processing
        self.chunk_sizes_qkv = chunk_sizes_qkv
        self.step_sizes_qkv = step_sizes_qkv
        # self.MHA_mode = MHA_mode

        # Define the Multi-Head Attention mode
        self.attn = MultiHeadAttention_oldVersion(heads=heads, d_model=dim, dropout_prob=dropout)


    def padding(self, x, chunk_size):
        """
        Pad the input tensor x to ensure its length is divisible by the chunk size.
        Args:
            x: Input tensor of shape [B, T, D]
            chunk_size: Size of the chunks to pad to
        Returns:
            Padded tensor of shape [B, T', D] where T' is the new length after padding
        """
        # Calculate the new length for padding
        # new_len_x = chunk_size * (int(x.shape[1] / chunk_size) + 1)
        new_len_x = x.shape[1] + chunk_size - x.shape[1] % chunk_size if x.shape[1] % chunk_size != 0 else x.shape[1]
        # If the input tensor is shorter than the new length, pad it
        if x.shape[1] < new_len_x:
            x = torch.nn.functional.pad(x, (0, 0, 0, new_len_x - x.shape[1]), mode='constant', value=0)
        return x

    def forward(self, query, key, value, adapt_embedding, **kwargs):
        
        # Get and store shapes of the input tensors
        Bq, Tq, Dq = query.shape
        Bk, Tk, Dk = key.shape
        Bv, Tv, Dv = value.shape
 
        # Padding, if necessary, to ensure the input tensors are divisible by the chunk size
        if self.chunk_processing:
            query = self.padding(query, self.chunk_sizes_qkv[0])
            key = self.padding(key, self.chunk_sizes_qkv[1])
            value = self.padding(value, self.chunk_sizes_qkv[2])

        # If chunk processing is enabled, get chunks of the input tensors
        if self.chunk_processing:
            query_chunks = get_chunks(query, blocksize=self.chunk_sizes_qkv[0], step=self.step_sizes_qkv[0])
            key_chunks = get_chunks(key, blocksize=self.chunk_sizes_qkv[1], step=self.step_sizes_qkv[1])
            value_chunks = get_chunks(value, blocksize=self.chunk_sizes_qkv[2], step=self.step_sizes_qkv[2])

            if adapt_embedding is not None:
                num_repeats = query_chunks.shape[0] // adapt_embedding.shape[0]
                adapt_embedding_chunks = adapt_embedding.unsqueeze(1).repeat(1, num_repeats, 1).reshape(-1, adapt_embedding.shape[1]).unsqueeze(1)

                query_chunks = torch.cat([adapt_embedding_chunks, query_chunks], dim=1)
                key_chunks = torch.cat([adapt_embedding_chunks, key_chunks], dim=1)
                value_chunks = torch.cat([adapt_embedding_chunks, value_chunks], dim=1)

        else:
            query_chunks = query
            key_chunks = key
            value_chunks = value

        # Apply attention on chunks
        attn_out, attn_output_weights = self.attn(query_chunks, key_chunks, value_chunks, **kwargs)

        if adapt_embedding is not None:
            attn_out = attn_out[:, 1:, :]
            attn_output_weights = attn_output_weights[:, 1:, 1:]

        # Recover the original shape from chunks
        if self.chunk_processing:
            attn_out = recover_chunks(attn_out, blocksize=self.chunk_sizes_qkv[0], step=self.step_sizes_qkv[0], batch_size=Bq)
            attn_output_weights = recover_chunks(attn_output_weights, blocksize=self.chunk_sizes_qkv[0], step=self.step_sizes_qkv[0], batch_size=Bq)
        
        # remove padding if it was applied
        if self.chunk_processing:
            attn_out = attn_out[:, :Tq, :]
            attn_output_weights = attn_output_weights[:, :Tq, :]

        return attn_out, attn_output_weights


class custom_MHA_wChunking_wPrefixAdaptation_KV(nn.Module):
    """
    Custom Multi-Head Attention module with chunk-wise processing and masking.
    This module allows for processing input tensors in chunks, which can be useful for long sequences, to increase focus on local patterns and reduce memory usage.
    """
    def __init__(self, 
                 dim,
                 heads,
                 dropout=0.1,
                 chunk_processing=False,
                 chunk_sizes_qkv=(160,160,160),
                 step_sizes_qkv=(80,80,80),
                 # MHA_mode='torch_standard'  # 'torch_standard' or 'custom_MHA'
                ):
        """
        Initialize the custom Multi-Head Attention module.
        Args:
            dim (int): Dimension of the input features.
            heads (int): Number of attention heads.
            chunk_processing (bool): Whether to process input in chunks.
            chunk_sizes (tuple): Sizes of chunks for each modality (key, query, value).
            step_sizes (tuple): Step sizes for each modality (key, query, value).
        """
        # Call the parent class constructor
        super().__init__()
        self.chunk_processing = chunk_processing
        self.chunk_sizes_qkv = chunk_sizes_qkv
        self.step_sizes_qkv = step_sizes_qkv
        # self.MHA_mode = MHA_mode

        # Define the Multi-Head Attention mode
        self.attn = MultiHeadAttention_oldVersion(heads=heads, d_model=dim, dropout_prob=dropout)


    def padding(self, x, chunk_size):
        """
        Pad the input tensor x to ensure its length is divisible by the chunk size.
        Args:
            x: Input tensor of shape [B, T, D]
            chunk_size: Size of the chunks to pad to
        Returns:
            Padded tensor of shape [B, T', D] where T' is the new length after padding
        """
        # Calculate the new length for padding
        # new_len_x = chunk_size * (int(x.shape[1] / chunk_size) + 1)
        new_len_x = x.shape[1] + chunk_size - x.shape[1] % chunk_size if x.shape[1] % chunk_size != 0 else x.shape[1]
        # If the input tensor is shorter than the new length, pad it
        if x.shape[1] < new_len_x:
            x = torch.nn.functional.pad(x, (0, 0, 0, new_len_x - x.shape[1]), mode='constant', value=0)
        return x

    def forward(self, query, key, value, adapt_embedding, **kwargs):
        
        # Get and store shapes of the input tensors
        Bq, Tq, Dq = query.shape
        Bk, Tk, Dk = key.shape
        Bv, Tv, Dv = value.shape
 
        # Padding, if necessary, to ensure the input tensors are divisible by the chunk size
        if self.chunk_processing:
            query = self.padding(query, self.chunk_sizes_qkv[0])
            key = self.padding(key, self.chunk_sizes_qkv[1])
            value = self.padding(value, self.chunk_sizes_qkv[2])

        # If chunk processing is enabled, get chunks of the input tensors
        if self.chunk_processing:
            query_chunks = get_chunks(query, blocksize=self.chunk_sizes_qkv[0], step=self.step_sizes_qkv[0])
            key_chunks = get_chunks(key, blocksize=self.chunk_sizes_qkv[1], step=self.step_sizes_qkv[1])
            value_chunks = get_chunks(value, blocksize=self.chunk_sizes_qkv[2], step=self.step_sizes_qkv[2])

            if adapt_embedding is not None:
                num_repeats = query_chunks.shape[0] // adapt_embedding.shape[0]
                if len(adapt_embedding.shape) == 2:
                    adapt_embedding = adapt_embedding.unsqueeze(1)
                if len(adapt_embedding.shape) == 3:
                    adapt_embedding = adapt_embedding.unsqueeze(1)
                adapt_embedding_chunks = adapt_embedding.repeat(1, num_repeats, 1, 1).reshape(-1, adapt_embedding.shape[2], adapt_embedding.shape[3])

                # query_chunks = torch.cat([adapt_embedding_chunks, query_chunks], dim=1)
                key_chunks = torch.cat([adapt_embedding_chunks, key_chunks], dim=1)
                value_chunks = torch.cat([adapt_embedding_chunks, value_chunks], dim=1)

        else:
            query_chunks = query
            key_chunks = key
            value_chunks = value

        # Apply attention on chunks
        attn_out, attn_output_weights = self.attn(query_chunks, key_chunks, value_chunks, **kwargs)

        # import matplotlib.pyplot as plt
        # num_emb_vectors = adapt_embedding.shape[-2]
        # # attn_output_weights[:,:,:num_emb_vectors] *= 10.0
        # plt.imshow(attn_output_weights[1].cpu().numpy())
        # plt.colorbar()
        # plt.show()
        # plt.savefig('attn_weights.png', dpi=300)

        # num_emb_vectors = adapt_embedding.shape[-2]
        # full_mass = attn_output_weights.sum(dim=-1)
        # attn_prefix_mass = attn_output_weights[..., :num_emb_vectors].sum(dim=-1)  # [B, H, T_q]
        
        # print('adapt_embedding.shape: ', adapt_embedding.shape, flush=True)
        # print('attn_output_weights.shape: ', attn_output_weights.shape, flush=True)
        # print('attn_prefix_mass.shape: ', attn_prefix_mass.shape, flush=True)
        # print('attn_prefix_mass: ', attn_prefix_mass, flush=True)
        # print('full_mass: ', full_mass, flush=True)
        # print('attn_prefix_mass mean: ', attn_prefix_mass.mean().item(), flush=True)

        # if adapt_embedding is not None:
        #     attn_out = attn_out[:, 1:, :]
        #     attn_output_weights = attn_output_weights[:, 1:, 1:]

        # Recover the original shape from chunks
        if self.chunk_processing:
            attn_out = recover_chunks(attn_out, blocksize=self.chunk_sizes_qkv[0], step=self.step_sizes_qkv[0], batch_size=Bq)
            attn_output_weights = recover_chunks(attn_output_weights, blocksize=self.chunk_sizes_qkv[0], step=self.step_sizes_qkv[0], batch_size=Bq)
        
        # remove padding if it was applied
        if self.chunk_processing:
            attn_out = attn_out[:, :Tq, :]
            attn_output_weights = attn_output_weights[:, :Tq, :]

        return attn_out, attn_output_weights



if __name__ == '__main__':
    
    # Test the custom Multi-Head Attention module
    batch_size = 2
    seq_length = 1000
    dim = 80
    heads = 4

    lens = [1.0, 0.8]  # Example lengths for two sequences

    audio = torch.randn(batch_size, seq_length, dim)
    video = torch.randn(batch_size, seq_length // 4, dim)

    mha1 = custom_MHA_wChunking(dim=dim, heads=heads, chunk_processing=True, chunk_sizes_qkv=(40, 160, 160), step_sizes_qkv=(20, 80, 80))
    mha2 = custom_MHA_wChunking(dim=dim, heads=heads, chunk_processing=True, chunk_sizes_qkv=(160, 40, 40), step_sizes_qkv=(80, 20, 20))
    output1, _ = mha1(video, audio, audio, lens=lens)
    output2, _ = mha2(audio, video, output1, lens=lens)

    print(output1.shape)  # [2, 25, 80] for video
    print(output2.shape)  # [2, 160, 80] for audio


    # # Test the chunking and recovering functions
    # blocksize = 160
    # setpsize = 80
    # 
    # a = torch.randn(2, 3000, 80)
    # B, T, D = a.shape
    # new_len = blocksize * (int(T/blocksize)+1)
    # a_padded = torch.nn.functional.pad(a, (0, 0 ,0 , new_len - T), mode='constant', value=0)
    # print(a_padded.shape)  # [B, D, T]
    # a_resh = get_chunks(a_padded, blocksize=blocksize, step=setpsize)
    # print(a_resh.shape)  # [B, D, Num_patches, Blocksize=4]
    # 
    # a_recovered = recover_chunks(a_resh, blocksize=blocksize, step=setpsize, batch_size=B)
    # a_recovered = a_recovered[:, :T, :]  # Trim to original length
    # print(a_recovered.shape)  # [B, T, D]
    # print(torch.max(torch.abs(a[:,:,:] - a_recovered[:,:,:])))

    print('Done')



