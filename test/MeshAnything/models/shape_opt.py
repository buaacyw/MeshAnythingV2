from transformers import AutoModelForCausalLM, AutoConfig, OPTConfig
from transformers.models.opt.modeling_opt import OPTForCausalLM, OPTModel, OPTDecoder, OPTLearnedPositionalEmbedding, OPTDecoderLayer
from typing import List, Optional, Tuple, Union
from einops import repeat
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.utils import replace_return_docstrings
from transformers.modeling_outputs import BaseModelOutputWithPast

class ShapeOPTConfig(OPTConfig):
    model_type = "shape_opt"

class ShapeOPT(OPTForCausalLM):
    config_class = ShapeOPTConfig
    def __init__(self, config: ShapeOPTConfig):
        super(OPTForCausalLM, self).__init__(config)
        self.model = ShapeOPTModel(config)

        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class="OPTConfig")
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        face_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional
                tensors are only required when the model is used as a decoder in a Sequence to Sequence model.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, OPTForCausalLM

        >>> model = OPTForCausalLM.from_pretrained("facebook/opt-350m")
        >>> tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious. I'm just a little bit of a weirdo."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids = input_ids,
            face_ids = face_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class ShapeOPTModel(OPTModel):
    config_class = ShapeOPTConfig
    def __init__(self, config: ShapeOPTConfig):
        super(OPTModel,self).__init__(config)
        self.decoder = ShapeOPTDecoder(config)
        # Initialize weights and apply final processing
        self.post_init()

class ShapeOPTDecoder(OPTDecoder):
    config_class = ShapeOPTConfig
    def __init__(self, config: ShapeOPTConfig):
        super(OPTDecoder,self).__init__(config)
        self.config = config
        self.dropout = config.dropout
        self.layerdrop = config.layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.word_embed_proj_dim, self.padding_idx)
        self.hidden_size = config.hidden_size
        self.word_embed_proj_dim = config.word_embed_proj_dim
        self.n_discrete_size = config.n_discrete_size

        self.embed_positions = OPTLearnedPositionalEmbedding(config.max_position_embeddings, config.hidden_size)
        self.token_embed_positions = OPTLoopEmbedding(10, config.word_embed_proj_dim, self.n_discrete_size) #padding_idx=self.padding_idx)

        self.face_per_token = config.face_per_token
        self.cond_length = config.cond_length
        self.cond_embed = nn.Embedding(2, config.word_embed_proj_dim)

        # Note that the only purpose of `config._remove_final_layer_norm` is to keep backward compatibility
        # with checkpoints that have been fine-tuned before transformers v4.20.1
        # see https://github.com/facebookresearch/metaseq/pull/164
        if config.do_layer_norm_before and not config._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                config.hidden_size, elementwise_affine=config.layer_norm_elementwise_affine
            )
        else:
            self.final_layer_norm = None

        self.layers = nn.ModuleList([OPTDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        face_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(num_hidden_layers, num_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        # OPT Decoder
        # print("used my Trans")
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # Transformer Decoder
        if input_ids is not None and inputs_embeds is not None: # when  train and first generate
            assert False
        elif input_ids is not None:
            assert not self.training
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            inputs_embeds = self.embed_tokens(input_ids)
            face_embeds = self.token_embed_positions(attention_mask[:, self.cond_length:], face_ids, input_ids,
                                                     self.face_per_token)
            inputs_embeds += face_embeds
            cond_embed_query = torch.ones((inputs_embeds.shape[0], inputs_embeds.shape[1]), device=inputs_embeds.device,
                                            dtype=inputs_embeds.dtype).long()
            inputs_embeds = inputs_embeds + self.cond_embed(cond_embed_query)

        elif inputs_embeds is not None:
            # assert self.cond and not self.training
            assert not self.training
            self.token_embed_positions.init_state(inputs_embeds)
            total_length = inputs_embeds.shape[1] # B x length x embeding
            cond_embed_query = torch.zeros((inputs_embeds.shape[0], total_length), device=inputs_embeds.device,
                                            dtype=inputs_embeds.dtype).long()
            inputs_embeds = inputs_embeds + self.cond_embed(cond_embed_query)
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        batch_size, seq_length = inputs_embeds.shape[:2] # seq_length not used since mask_seq_length is not used
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values_length + seq_length # not used since attention mask is input

        # embed positions
        if self._use_flash_attention_2:
            # 2d mask is passed through the layers
            assert attention_mask is not None
            causal_attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            attention_mask = (
                torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)
                if attention_mask is None
                else attention_mask
            )
        else:
            raise ValueError("Only flash_attention_2 is supported in MeshAnything")

        pos_embeds = self.embed_positions(attention_mask, past_key_values_length)

        hidden_states = inputs_embeds + pos_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask], ["head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    None,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class OPTLoopEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, n_discrete_size: int):
        super().__init__(num_embeddings, embedding_dim)
        self.state = None
        self.loop_state = None
        self.n_discrete_size = n_discrete_size + 3 # for padding

    def forward(self, attention_mask=None, face_ids = None, input_ids = None, face_per_token = None):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        if face_ids is not None:
            return super().forward(face_ids)

        assert input_ids.shape[1] == 1, "Only one token is allowed for loop embedding"
        assert self.state is not None, "State is not initialized"
        # zero as beginning
        batch_size = input_ids.shape[0]
        face_ids = input_ids.clone().detach()

        for cur_batch_index in range(batch_size):
            cur_ids = input_ids[cur_batch_index]

            idx_in_extra = torch.isin(cur_ids, torch.LongTensor([0, 1, 2]).to(input_ids.device))
            if idx_in_extra:
                self.state[cur_batch_index] = 9  # init
                self.loop_state[cur_batch_index] = 0
            else:
                if cur_ids == self.n_discrete_size:
                    face_ids[cur_batch_index] = 3
                    self.state[cur_batch_index] = 9 # init
                    self.loop_state[cur_batch_index] = 0
                else:
                    if self.state[cur_batch_index] == 0:
                        face_ids[cur_batch_index] = 7 + self.loop_state[cur_batch_index] % 3
                    else:
                        self.state[cur_batch_index] -= 1
                        face_ids[cur_batch_index] = 4 + self.loop_state[cur_batch_index] % 3
                    self.loop_state[cur_batch_index] += 1

        return super().forward(face_ids)

    def init_state(self, template_tensor):
        batch_size = template_tensor.shape[0]
        self.state = torch.zeros((batch_size, 1), dtype=torch.long, device=template_tensor.device)
        self.state[...] = 9
        self.loop_state = torch.zeros((batch_size, 1), dtype=torch.long, device=template_tensor.device)

class OPTFacePositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, attention_mask=None, face_ids = None, input_ids = None, face_per_token = None):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        if face_ids is not None:
            return super().forward(face_ids)

        assert input_ids.shape[1] == 1
        idx_in_extra = torch.isin(input_ids, torch.LongTensor([0, 1, 2]).to(input_ids.device))
        cur_ids = input_ids.clone().detach()

        cur_index = (attention_mask.sum(dim=1, keepdim=True) - 2) % face_per_token + 3
        cur_ids[~idx_in_extra]=cur_index[~idx_in_extra]

        return super().forward(cur_ids)


AutoConfig.register("shape_opt", ShapeOPTConfig)
AutoModelForCausalLM.register(ShapeOPTConfig, ShapeOPT)

