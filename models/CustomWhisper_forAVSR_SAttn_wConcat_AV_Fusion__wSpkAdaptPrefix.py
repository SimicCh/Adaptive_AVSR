import os
import math
from typing import Optional, Tuple, Union
import logging

import numpy as np
import torch
import torch.utils.checkpoint
from torchvision import models as torchvision_models
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import (
    PreTrainedModel,
    WhisperForConditionalGeneration,
    WhisperConfig, GenerationConfig,
    AutoProcessor,
)

from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings
)

from transformers.models.whisper.configuration_whisper import WhisperConfig
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
from transformers.cache_utils import EncoderDecoderCache, StaticCache

from transformers.models.whisper.modeling_whisper import(
    WhisperPreTrainedModel,
    WhisperModel,
    WhisperEncoder,
    WhisperDecoder,
    shift_tokens_right,
    _compute_mask_indices,
    WHISPER_START_DOCSTRING,
    WHISPER_INPUTS_DOCSTRING,
    _CONFIG_FOR_DOC
)

from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)

from speechbrain.inference.speaker import EncoderClassifier

from models.utils.AVSR_FusionBlocks import SAttn_wConcat_AV_Fusion
from models.utils.utils import MLP_mapping

logger = logging.getLogger(__name__)



class CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptPrefix(PreTrainedModel):
    _tied_weights_keys = ["proj_out.weight"]

    def __init__(self, model_name_or_path, model_config, embedding_config, adaptation_config):
        super().__init__(model_config)

        print('------> CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptPrefix init', flush=True)

        self.model_config = model_config
        self.afm = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)

        # self.video_embedding = LIPNET_CNN(layers=embedding_config['layers'], emb_size=embedding_config['emb_size'])
        self.fusion = SAttn_wConcat_AV_Fusion(embedding_config)

        # Prepare speaker adaptation model
        self.adaptation_config = adaptation_config
        print('Loading speaker embedding model from: ', adaptation_config['embedding_model_name'], flush=True)
        print('Speaker adaptation model config: ', adaptation_config, flush=True)
        save_dir = os.path.abspath('./'+adaptation_config['embedding_model_name']) 
        # self.spk_embedding_model = EncoderClassifier.from_hparams(source=adaptation_config['embedding_model_name'], savedir=save_dir)
        self.spk_embedding_model = EncoderClassifier.from_hparams(
                    source=self.adaptation_config['embedding_model_name'],
                    savedir=save_dir,
                    run_opts={"device": self.adaptation_config['device']}
                )
        self.spk_embedding_model.requires_grad_(False)
        self.spk_embedding_model.eval()


        if 'mapping_act_fun' in adaptation_config:
            act_fun = adaptation_config['mapping_act_fun']
            if act_fun == 'relu':
                act_fun = nn.ReLU
            elif act_fun == 'gelu':
                act_fun = nn.GELU
            elif act_fun == 'leakyrelu':
                act_fun = nn.LeakyReLU
            elif act_fun == 'tanh':
                act_fun = nn.Tanh
            else:
                raise ValueError('Invalid activation function: '+str(act_fun))
        else:
            act_fun = nn.Tanh
        
        self.embedding_mapping_net_spk = MLP_mapping(
            (
                adaptation_config['out_dim'],
                adaptation_config['mapping_dim'] // 2,
                adaptation_config['mapping_dim'],
            ),
            dropout=adaptation_config['mapping_dropout'],
            bias=True,
            act=act_fun
        )

        # Freeze decoder 
        if model_config.freeze_decoder:
            self.freeze_decoder()
        # Freeze encoder
        if model_config.freeze_encoder:
            self.freeze_encoder()
        # Freeze embedding 
        if model_config.freeze_embedding:
            self.freeze_embedding()
            self.freeze_encoder_convs()
        else:
            self.make_video_embedding_trainable()
            self.make_encoder_convs_trainable()

        print_dict = {
            'model_name_or_path': model_name_or_path,
            'freeze_embedding': model_config.freeze_embedding,
            'freeze_encoder': model_config.freeze_encoder,
            'freeze_decoder': model_config.freeze_decoder,
        }

        print('Using CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptPrefix with config:', print_dict, flush=True)
        self.enc_embed_loss = MSELoss()
        print('CustomWhisper_forAVSR_SAttn_wConcat_AV_Fusion__wSpkAdaptPrefix with MSE Loss for encoder embedding!', flush=True)


    def make_video_embedding_trainable(self):
        for param in self.fusion.parameters():
            param.requires_grad = True

    def make_encoder_convs_trainable(self):
        self.afm.model.encoder.conv1.requires_grad = True
        self.afm.model.encoder.conv2.requires_grad = True
        print('Encoder conv1 and conv2 are trainable!', flush=True)


    def compute_custom_weight_stats(self, return_weights=False):
        weights = []
        for name, param in self.named_parameters():
            # or "layers.0.self_attn.v_proj.weight" in name or "layers.0.self_attn.q_proj.weight" in name or "layers.31.self_attn.v_proj.weight" in name or "layers.31.self_attn.q_proj.weight" in name:
            if 'mapping_net' in name or 'aux_audio_clf' in name:
                min_val = param.min().item()
                max_val = param.max().item()
                mean_val = param.mean().item()
                logger.info(f"Stats for '{name}': min={min_val:.4f}\tmax={max_val:.4f}\tmean={mean_val:.4f} ")
                if return_weights:
                    weights.append((name, param.clone().detach().cpu()))
        logger.info("=" * 50)
        if return_weights:
            return weights

    def get_encoder(self):
        return self.afm.get_encoder()

    def get_decoder(self):
        return self.afm.get_decoder()


    def freeze_embedding(self):
        for param in self.fusion.parameters():
            param.requires_grad = False

    def freeze_encoder_convs(self):
        self.afm.model.encoder.conv1.requires_grad = False
        self.afm.model.encoder.conv2.requires_grad = False

    def freeze_encoder(self):
        """
        Calling this function will disable the gradient computation for the Whisper encoder so that its parameters will
        not be updated during training.
        """
        self.afm.freeze_encoder()

    def freeze_decoder(self):
        for param in self.afm.model.decoder.parameters():
            param.requires_grad = False
        self.afm.proj_out.requires_grad = False

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (WhisperDecoder, WhisperEncoder)):
            logger.info(
                f"Activating gradient checkpointing for {module.__class__.__name__}"
            )
            module.gradient_checkpointing = value

    def _mask_input_features(
            self,
            input_features: torch.FloatTensor,
            attention_mask: Optional[torch.LongTensor] = None,
        ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        if not getattr(self.config, "apply_spec_augment", True):
            return input_features

        # generate indices & apply SpecAugment along time axis
        batch_size, hidden_size, sequence_length = input_features.size()

        if self.config.mask_time_prob > 0 and self.training:
            # generate indices & apply SpecAugment along time axis
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=input_features.device, dtype=torch.bool)
            mask_time_indices = mask_time_indices[:, None].expand(-1, hidden_size, -1)
            input_features[mask_time_indices] = 0

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=input_features.device, dtype=torch.bool)
            input_features[mask_feature_indices] = 0

        return input_features


    def get_encoder_embedding(
        self,
        mel_specs: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        ):
        encoder_outputs = self.afm.model.encoder(
                input_features=mel_specs,
                attention_mask=attention_mask
            )
        return encoder_outputs

    def forward_whisper(
        self,
        input_features: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Union[EncoderDecoderCache, Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[Tuple[torch.FloatTensor]] = None,
        decoder_position_ids: Optional[Tuple[torch.LongTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple[torch.Tensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the language modeling loss. Indices should either be in `[0, ..., config.vocab_size]`
            or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is
            only computed for the tokens with labels in `[0, ..., config.vocab_size]`. `sequence_length` should be smaller than or equal to `config.max_target_positions`.

        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoProcessor, WhisperForConditionalGeneration
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("openai/whisper-tiny.en")
        >>> model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

        >>> inputs = processor(ds[0]["audio"]["array"], return_tensors="pt")
        >>> input_features = inputs.input_features

        >>> generated_ids = model.generate(inputs=input_features)

        >>> transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> transcription
        ' Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if labels.shape[1] > self.afm.max_target_positions:
                raise ValueError(
                    f"Labels' sequence length {labels.shape[1]} cannot exceed the maximum allowed length of {self.max_target_positions} tokens."
                )
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.afm.model(
            input_features,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            decoder_position_ids=decoder_position_ids,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        lm_logits = self.afm.proj_out(outputs[0])

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


    def forward(self,
            mode: str = 'encoder_embedding',        # 'encoder_embedding', 'full'
            calc_loss: bool = False,
            melspec_inputs: Optional[torch.Tensor] = None,
            melspec_inputs_no_aug: Optional[torch.Tensor] = None,
            raw_audio_inputs: Optional[list] = None,
            video_inputs: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            vid_lens: Optional[torch.Tensor] = None,
            melspec_labels: Optional[torch.FloatTensor] = None,
            melspec_loss_weight: float = 0.0,
            enc_embed_labels: Optional[torch.FloatTensor] = None,
            enc_embed_loss_weight: float = 0.0,
            label_tokens: Optional[torch.LongTensor] = None,
            dec_embed_loss_weight: float = 0.0,
            **kwargs,
            ):
        
        if calc_loss:
            loss = torch.tensor(0.0).to(video_inputs.device)
        else:
            loss = None

        # ############ Prepare embeddings ############
        self.spk_embedding_model.eval()
        spk_embedding_model_device = next(self.spk_embedding_model.parameters()).device
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                # self.spk_embedding_model.float()
                speaker_embedding = list()
                for i in range(len(raw_audio_inputs)):
                    wav = raw_audio_inputs[i].float().to(spk_embedding_model_device)
                    se = self.spk_embedding_model.encode_batch(wav)
                    speaker_embedding.append(se)
                speaker_embedding = torch.cat(speaker_embedding, dim=0)
                
        mapped_speaker_embedding = self.embedding_mapping_net_spk(speaker_embedding)

        melspec_pred, _, _ = self.fusion(audio=melspec_inputs, video=video_inputs, vid_lens=vid_lens)

        # melspec_pred, _, _ = self.fusion(melspec_inputs, video_inputs, vid_lens)
        if calc_loss:
            if melspec_loss_weight > 0.:
                melspec_loss = MSELoss()
                loss += melspec_loss(melspec_pred, melspec_labels) * melspec_loss_weight

        encoder_outputs = self.get_encoder_embedding(
                mel_specs=melspec_pred,
                attention_mask=attention_mask
            )
        
        if mode == 'encoder_embedding':
            
            if calc_loss:
                if enc_embed_loss_weight > 0.:
                    loss += self.enc_embed_loss(encoder_outputs['last_hidden_state'], enc_embed_labels) * enc_embed_loss_weight

            return Seq2SeqLMOutput(
                    loss=loss,
                    logits=None,
                    past_key_values=None,
                    decoder_hidden_states=None,
                    decoder_attentions=None,
                    cross_attentions=None,
                    encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                    encoder_hidden_states=encoder_outputs.hidden_states,
                    encoder_attentions=encoder_outputs.attentions
                )

        elif mode == 'full':
            
            # Concatenate mapped speaker embedding as prefix to encoder outputs
            encoder_outputs_with_prefix = torch.cat(
                    (mapped_speaker_embedding, encoder_outputs.last_hidden_state),
                    dim=1
                )
            pre_prefix_encoder_last_hidden_state = encoder_outputs.last_hidden_state
            encoder_outputs.last_hidden_state = encoder_outputs_with_prefix

            outputs = self.forward_whisper(
                    encoder_outputs=encoder_outputs,
                    # input_features=melspec_pred,
                    labels=label_tokens,
                    attention_mask=attention_mask,
                )
            
            if calc_loss:
                if enc_embed_loss_weight > 0.:
                    loss += self.enc_embed_loss(pre_prefix_encoder_last_hidden_state, enc_embed_labels) * enc_embed_loss_weight
                if dec_embed_loss_weight > 0.:
                    loss += outputs.loss * dec_embed_loss_weight

            
            return Seq2SeqLMOutput(
                    loss=loss,
                    logits=outputs.logits,
                    past_key_values=outputs.past_key_values,
                    decoder_hidden_states=outputs.decoder_hidden_states,
                    decoder_attentions=outputs.decoder_attentions,
                    cross_attentions=outputs.cross_attentions,
                    encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                    encoder_hidden_states=outputs.encoder_hidden_states,
                    encoder_attentions=outputs.encoder_attentions
                )


        else:
            raise ValueError('Invalid mode: '+str(mode))



    def generate(
            self,
            melspec_inputs: Optional[torch.Tensor] = None,
            melspec_inputs_no_aug: Optional[torch.Tensor] = None,
            raw_audio_inputs: Optional[list] = None,
            video_inputs: Optional[torch.Tensor] = None,
            vid_lens: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            generation_config=None,
            logits_processor=None,
            stopping_criteria=None,
            prefix_allowed_tokens_fn=None,
            synced_gpus=False,
            return_timestamps=None,
            task=None,
            language=None,
            is_multilingual=None,
            prompt_ids: Optional[torch.Tensor] = None,
            return_token_timestamps=None,
            **kwargs,
        ):

        # inputs = None
        # inputs = self.video_embedding(video_inputs, vid_lens)
        # melspec_pred, _, _ = self.fusion(melspec_inputs, video_inputs, vid_lens)

        # ############ Prepare embeddings ############
        self.spk_embedding_model.eval()
        spk_embedding_model_device = next(self.spk_embedding_model.parameters()).device
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                # self.spk_embedding_model.float()
                speaker_embedding = list()
                for i in range(len(raw_audio_inputs)):
                    wav = raw_audio_inputs[i].float().to(spk_embedding_model_device)
                    se = self.spk_embedding_model.encode_batch(wav)
                    speaker_embedding.append(se)
                speaker_embedding = torch.cat(speaker_embedding, dim=0)
        
        mapped_speaker_embedding = self.embedding_mapping_net_spk(speaker_embedding)

        melspec_pred, _, _ = self.fusion(audio=melspec_inputs, video=video_inputs, vid_lens=vid_lens)

        encoder_outputs = self.get_encoder_embedding(
                mel_specs=melspec_pred,
                attention_mask=attention_mask
            )
        
        last_hidden_state_mod = torch.cat(
                    (mapped_speaker_embedding, encoder_outputs.last_hidden_state),
                    dim=1
                )

        encoder_outputs = BaseModelOutput(
                        last_hidden_state=last_hidden_state_mod,
                        hidden_states=encoder_outputs.hidden_states,
                        attentions=encoder_outputs.attentions,
                    )

        inputs = torch.zeros_like(melspec_pred)
        return self.afm.generate(
            inputs,
            attention_mask=attention_mask,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            return_timestamps=return_timestamps,
            task=task,
            language=language,
            is_multilingual=is_multilingual,
            prompt_ids=prompt_ids,
            return_token_timestamps=return_token_timestamps,
            encoder_outputs=encoder_outputs,
            **kwargs,
        )

        # return self.afm.generate(
        #     input_features=melspec_pred,
        #     task=task,
        #     language=language,
        #     attention_mask=attention_mask,
        # )



if __name__ == '__main__':
    whisper_name = 'openai/whisper-base'

    processor = AutoProcessor.from_pretrained(whisper_name)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language='en', task='transcribe')
    text = "BUT YOU JUST HAVEN'T FOUND IT YET"
    text = text.lower().strip()
    texts = [text, text]
    labels = processor(text=texts, return_tensors="pt", padding=True).input_ids

    whisper_config = WhisperConfig.from_pretrained(whisper_name)
    whisper_config.freeze_embedding = False
    whisper_config.freeze_encoder = False
    whisper_config.freeze_decoder = True


    embedding_config = dict()
    embedding_config['AV_Fusion_attLayer'] = 12
    embedding_config['AV_Fusion_attHeads'] = 12
    embedding_config['AV_Fusion_blocksize'] = 160
    embedding_config['AV_Fusion_inp_dim'] = 80
    embedding_config['AV_Fusion_proc_dim'] = 80
    embedding_config['AV_Fusion_out_dim'] = 80

    embedding_config['Specfront_procChannels'] = 128
    embedding_config['Specfront_layerNum'] = 7
    embedding_config['Specfront_inp_dim'] = 80
    embedding_config['Specfront_out_dim'] = 80

    embedding_config['Lipnet_video_layer'] = [[2,1],[2,1],[2,1],[3,1]]
    embedding_config['Lipnet_emb_size'] = 80
    embedding_config['Lipnet_num_input_channels'] = 3


    model = CustomWhisper_forAVSR_myLIPNET(whisper_name, whisper_config, embedding_config)
    model.make_video_embedding_trainable()
    model.make_encoder_convs_trainable()

    # Count number of total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params}')
    print(f'Total trainable parameters: {total_trainable_params}')
    
    input_melspec = torch.randn(2, 80, 3000)
    input_videos = torch.randn(2, 90, 3, 88, 88)
    vid_lens = torch.tensor([30, 90])

    melspec_labels = torch.randn(2, 80, 3000)
    enc_emb_labels = torch.randn(2, 1500, 512)
    dec_labels = labels

    # out = model.forward(
    #         mode='encoder_embedding',
    #         calc_loss=True,
    #         melspec_inputs=input_melspec,
    #         video_inputs=input_videos,
    #         vid_lens=vid_lens,
    #         melspec_labels=melspec_labels,
    #         melspec_loss_weight=0.0,
    #         enc_embed_labels=enc_emb_labels,
    #         enc_embed_loss_weight=1.0,
    #         label_tokens=dec_labels,
    #         dec_embed_loss_weight=0.0
    #         )

    out = model.forward(
            mode='full',
            calc_loss=True,
            melspec_inputs=input_melspec,
            video_inputs=input_videos,
            vid_lens=vid_lens,
            melspec_labels=melspec_labels,
            melspec_loss_weight=0.0,
            enc_embed_labels=enc_emb_labels,
            enc_embed_loss_weight=0.25,
            label_tokens=dec_labels,
            dec_embed_loss_weight=0.75
            )


    sequ = model.generate(
        melspec_inputs=input_melspec,
        video_inputs=input_videos,
        vid_lens=vid_lens,
        task='transcribe',
        language='en'
        )

    print('done')
