from os import path
import torch
import logging
from fairseq import utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
import pickle as pkl
logger = logging.getLogger(__name__)


@register_task("kd_translation")
class KDTranslation(TranslationTask):

    @staticmethod
    def add_args(parser):
        TranslationTask.add_args(parser)
        
        parser.add_argument("--train-tec", action="store_true", help="training teacher model")
        parser.add_argument("--train-stu", action="store_true", help="training student model")
        parser.add_argument("--load-tec-model", type=str, help='path to pretrained teacher model')
        parser.add_argument("--tec-pass", type=int, help='teacher pass time')
        parser.add_argument("--load-tec-weight", type=str, help='path to teacher data weight')
        parser.add_argument("--generate-random", type=int, help='generate random seed')

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.train_tec = getattr(args, 'train_tec', False)
        self.train_stu = getattr(args, 'train_stu', False)
        self.load_tec_model = getattr(args, 'load_tec_model', "")
        self.load_tec_weight = getattr(args, 'load_tec_weight', "")
        self.tec_pass = getattr(args, 'tec_pass', 0)
        self.generate_random = getattr(args, 'generate_random', 0)
        self.tec_model = None

    def load_teacher_model(self, tec_model):
        self.tec_model = tec_model

    def build_tec_model(self, cfg):
        model = super(TranslationTask, self).build_model(cfg)
        return model

    def train_step(self, sample, model, criterion, optimizer, update_num, ignore_grad=False):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            if self.train_tec:
                loss, sample_size, logging_output = criterion.forward_teacher(model, sample, optimizer)
            elif self.train_stu:
                if self.load_teacher_model:
                    loss, sample_size, logging_output = criterion.forward_label_kd(model, self.tec_model, self.tec_pass, sample, optimizer)
                else:
                    loss, sample_size, logging_output = criterion(model, sample)
            else:
                loss, sample_size, logging_output = criterion(model, sample)
            return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        if getattr(model, "reset_encoder_decoder_order", None) is not None:
            model.reset_encoder_decoder_order()
        
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):  
        if self.generate_random:
            torch.manual_seed(self.generate_random)
            torch.cuda.manual_seed(self.generate_random)
            dropout_p = 0.0
            dropout_p_act = 0.0
            dropout_p_attn = 0.1
            models[0].dynamic_dropout_rate(dropout_p, dropout_p_act, dropout_p_attn, inf_flag=True)

        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
            )

