# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import math
import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

@register_criterion('kd_label_smoothed_cross_entropy')
class KDLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if getattr(model, "reset_encoder_decoder_order", None) is not None:
            model.reset_encoder_decoder_order()
            
        net_output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss
    
    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))
        
        if "word_kl_loss" in logging_outputs[0]:
            word_kl_loss_sum = sum(log.get('word_kl_loss', 0) for log in logging_outputs)
            metrics.log_scalar('word_kl', word_kl_loss_sum / ntokens / math.log(2), ntokens, round=3)
    
    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
    
    ## knowledge distillation for training teacher model

    def forward_teacher(self, model, sample, optimizer, reduce=True):
        # select submdoule sampling method:
        model.random_select_order_layer()
        # standard training paradigm:
        output = model(**sample['net_input'])
        loss, nll_loss = self.compute_loss(model, output, sample, reduce=reduce)
        optimizer.backward(loss)

        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output
    
    ## knowledge distillation for training student model
    
    # label_based kd
    def compute_kl_loss(self, tec_model, model, stu_net_output, tec_net_output_list, pad_mask=None):
        stu_p = model.get_normalized_probs(stu_net_output, log_probs=True)
        tec_p = []
        for tec_net_output in tec_net_output_list:
            tec_p.append(tec_model.get_normalized_probs(tec_net_output, log_probs=False).unsqueeze(-1))
        tec_p = torch.cat(tec_p, dim=-1).mean(-1).squeeze(0)
        loss = torch.nn.functional.kl_div(stu_p, tec_p, reduction='none')
        loss.masked_fill_(pad_mask, 0.)
        return loss.sum()
    
    def forward_label_kd(self, model, tec_model, tec_pass, sample, optimizer, reduce=True):
        
        stu_word_output = model(**sample['net_input'])
        word_loss, nll_loss = self.compute_loss(model, stu_word_output, sample, reduce=reduce)
        # cacluate pad mask from stu ouput
        targets = model.get_targets(sample, stu_word_output).unsqueeze(-1)
        pad_mask = targets.eq(self.padding_idx)
        
        # word distillation
        with torch.no_grad():
            tec_model.eval()
            tec_output_list = []
            for _ in range(tec_pass):
                tec_output_list.append(tec_model(**sample['net_input']))
        word_kl_loss = self.compute_kl_loss(tec_model, model, stu_word_output, tec_output_list, pad_mask)
        loss = word_loss + word_kl_loss
        optimizer.backward(loss)
        
        ntokens = sample['ntokens']
        nsentences = sample['target'].size(0)
        sample_size = sample['ntokens']

        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'word_kl_loss': utils.item(word_kl_loss.data) if reduce else word_kl_loss.data,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    # data_based kd
    def forward_data_kd(self, model, sample, weight, optimizer, reduce=True):
        net_output = model(**sample['net_input'])
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output).view(-1, 1)
        bsz = lprobs.size(0)
        seq_len = lprobs.size(1)
        weight = weight * (bsz / sum(weight))
        weight = weight.unsqueeze(-1)
        weight = weight.expand(bsz, seq_len).contiguous().view(-1, 1).to(target.device)
        
        lprobs = lprobs.view(-1, lprobs.size(-1))
        non_pad_mask = target.ne(self.padding_idx)
        nll_loss = -lprobs.gather(dim=-1, index=target)
        nll_loss *= weight
        smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
        if reduce:
            nll_loss = nll_loss[non_pad_mask].sum()
            smooth_loss = smooth_loss[non_pad_mask].sum()
        
        eps_i = self.eps / lprobs.size(-1)
        loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss        
        optimizer.backward(loss)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output