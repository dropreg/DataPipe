from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerModel, TransformerDecoder, TransformerEncoder, base_architecture
from ..modules.transformer_sub_layer import TransformerDecoderSublayer, TransformerEncoderSublayer
import numpy as np
import torch.nn as nn
from fairseq.modules import LayerDropModuleList
from typing import Any, Dict, List, Optional
from torch import Tensor
import torch
import logging

logger = logging.getLogger(__name__)

@register_model("x_transformer")
class XTransformerModel(TransformerModel):

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--encoder-sub-layerdrop', type=float, metavar='D', default=0,
                            help='sub-layer Drop probability for encoder')
        parser.add_argument('--decoder-sub-layerdrop', type=float, metavar='D', default=0,
                            help='sub-layer Drop probability for decoder')
        parser.add_argument('--sublayer-reorder', action='store_true')
        parser.add_argument('--sublayer-drop', action='store_true')

    def __init__(self, args, encoder, decoder):
        super().__init__(args, encoder, decoder)
        
        self.encoder_sub_layerdrop = getattr(args, "encoder_sub_layerdrop", 0.0)
        self.decoder_sub_layerdrop = getattr(args, "decoder_sub_layerdrop", 0.0)
        self.sublayer_reorder = getattr(args, "sublayer_reorder", False)
        self.sublayer_drop = getattr(args, "sublayer_drop", False)

    def reset_encoder_decoder_order(self):
        self.encoder.reset_order()
        self.decoder.reset_order()
    
    def random_select_order_layer(self):
        self.encoder.random_select_order_layer(self.sublayer_reorder, self.sublayer_drop, self.decoder_sub_layerdrop)
        self.decoder.random_select_order_layer(self.sublayer_reorder, self.sublayer_drop, self.decoder_sub_layerdrop)

    def dynamic_dropout_rate(self, p, p_act, p_att, inf_flag=False):
        self.encoder.dynamic_dropout_rate(p, p_act, p_att, inf_flag)
        self.decoder.dynamic_dropout_rate(p, p_act, p_att, inf_flag)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderSubLayerVariant(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderSubLayerVariant(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

class TransformerEncoderSubLayerVariant(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)

        self.layer_current_order = None
        self.layers = nn.ModuleList([])
        self.order_layers, self.vanilla_order = self.build_sublayers(args)
        self.layers.extend(self.order_layers)
        self.reset_order()
    
    def reset_order(self):
        self.layer_current_order = " ".join(self.vanilla_order)
    
    def build_sublayers(self, args):
        layer_prefix = ["s", "f"] * self.num_layers
        stand_order = [prefix + '-' + str(l_idx) for l_idx, prefix in enumerate(layer_prefix)]
        layer_list = []
        for layer_arch in stand_order:
            layer_list.append(TransformerEncoderSublayer(args, layer_arch))
        return layer_list, stand_order
    
    def random_select_order_layer(self, reorder, layer_drop, layer_drop_rate):
        if reorder:
            reorderd_order = []
            for layer_idx in range(self.num_layers):
                start = layer_idx * 2
                reorderd_order.extend(np.random.permutation(self.vanilla_order[start: start+2]))
        else:
            reorderd_order = self.vanilla_order
        
        if layer_drop:
            layerdroped_order = []
            dropout_probs = torch.empty(len(self.layers)).uniform_()
            for o_indx, order in enumerate(reorderd_order):
                if dropout_probs[o_indx] > layer_drop_rate:
                    layerdroped_order.append(order)
        else:
            layerdroped_order = self.vanilla_order
        self.layer_current_order = " ".join(layerdroped_order)

    def dynamic_dropout_rate(self, p, p_act, p_att, inf_flag):
        self.dropout_module.p = p
        self.dropout_module.apply_during_inference = inf_flag
        for l in self.layers:
            l.dropout_module.p = p
            l.dropout_module.apply_during_inference = inf_flag
            if getattr(l, "activation_dropout_module", None):
                l.activation_dropout_module.p = p_act
                l.activation_dropout_module.apply_during_inference = inf_flag
            if getattr(l, "self_attn", None):
                l.self_attn.dropout_module.p = p_att
                l.self_attn.dropout_module.apply_during_inference = inf_flag
        
    def forward(
        self,
        src_tokens,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = []

        # encoder layers
        for layer_arch in self.layer_current_order.split(" "):
            
            _, layer_suffix = layer_arch.split("-")
            idx = int(layer_suffix)
            layer = self.layers[idx]

            x = layer(x, encoder_padding_mask)
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

class TransformerDecoderSubLayerVariant(TransformerDecoder):

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__( args, dictionary, embed_tokens, no_encoder_attn)

        self.layer_current_order = None
        self.layers = nn.ModuleList([])
        self.order_layers, self.vanilla_order = self.build_sublayers(args, no_encoder_attn)
        self.layers.extend(self.order_layers)
        self.reset_order()
    
    def reset_order(self):
        self.layer_current_order = " ".join(self.vanilla_order)
    
    def build_sublayers(self, args, no_encoder_attn=False):
        layer_prefix = ["s", "c", "f"] * self.num_layers
        stand_order = [prefix + '-' + str(l_idx) for l_idx, prefix in enumerate(layer_prefix)]
        layer_list = []
        for layer_arch in stand_order:
            layer_list.append(TransformerDecoderSublayer(args, layer_arch, no_encoder_attn))
        return layer_list, stand_order
    
    def build_fixed_reorder_space(self):
        candidate_order = [["s", "c", "f"], ["s", "f", "c"], ["c", "s", "f"],
                           ["c", "f", "s"], ["f", "s", "c"], ["f", "c", "s"]]
        search_space = []
        for order in candidate_order:
            block_order = []
            for l_num in range(self.num_layers):
                offset = l_num * 3
                for predfix in order:
                    if predfix == 's':
                        suffix = offset
                    elif predfix == 'c':
                        suffix = offset + 1
                    else:
                        suffix = offset + 2
                    block_order.append(predfix + "-" + str(suffix))
            search_space.append(block_order)
        return search_space
    
    def random_select_order_layer(self, reorder, layer_drop, layer_drop_rate):
        
        if reorder:
            reorderd_order = []
            for layer_idx in range(self.num_layers):
                start = layer_idx * 3
                reorderd_order.extend(np.random.permutation(self.vanilla_order[start: start+3]))
        else:
            reorderd_order = self.vanilla_order
        
        if layer_drop:
            layerdroped_order = []
            dropout_probs = torch.empty(len(self.layers)).uniform_()
            for o_indx, order in enumerate(reorderd_order):
                if dropout_probs[o_indx] > layer_drop_rate:
                    layerdroped_order.append(order)
        else:
            layerdroped_order = self.vanilla_order
        self.layer_current_order = " ".join(layerdroped_order)

    def dynamic_dropout_rate(self, p, p_act, p_att, inf_flag):
        self.dropout_module.p = p
        self.dropout_module.apply_during_inference = inf_flag
        for l in self.layers:
            l.dropout_module.p = p
            l.dropout_module.apply_during_inference = inf_flag
            if getattr(l, "activation_dropout_module", None):
                l.activation_dropout_module.p = p_act
                l.activation_dropout_module.apply_during_inference = inf_flag
            if getattr(l, "self_attn", None):
                l.self_attn.dropout_module.p = p_att
                l.self_attn.dropout_module.apply_during_inference = inf_flag
            if getattr(l, "encoder_attn", None):
                l.encoder_attn.dropout_module.p = p_att
                l.encoder_attn.dropout_module.apply_during_inference = inf_flag

    def extract_features_scriptable(
            self,
            prev_output_tokens,
            encoder_out: Optional[Dict[str, List[Tensor]]],
            incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
            full_context_alignment: bool = False,
            alignment_layer: Optional[int] = None,
            alignment_heads: Optional[int] = None,
        ):

        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for layer_arch in self.layer_current_order.split(" "):
            
            _, layer_suffix = layer_arch.split("-")
            idx = int(layer_suffix)
            layer = self.layers[idx]

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}


@register_model_architecture("x_transformer", "x_transformer")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)

@register_model_architecture("x_transformer", "x_transformer_big")
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4096)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.dropout = getattr(args, "dropout", 0.3)
    base_architecture(args)