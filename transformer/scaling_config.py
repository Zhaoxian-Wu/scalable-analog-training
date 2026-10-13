"""Scientific settings for the Adam training comparison in Figure fig:scaling."""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import runpy


MODEL_SCALES = {
    'tiny': dict(n_layer=2, n_head=4, n_embd=128, parameters=6_826_752),
    'small': dict(n_layer=4, n_head=4, n_embd=256, parameters=16_013_824),
    'medium': dict(n_layer=6, n_head=6, n_embd=384, parameters=29_920_512),
    'large': dict(n_layer=8, n_head=8, n_embd=512, parameters=50_906_112),
    'xlarge': dict(n_layer=12, n_head=12, n_embd=768, parameters=123_551_232),
}


@dataclass
class ScalingConfig:
    mode: str = 'digital'
    model_scale: str = 'tiny'
    dataset: str = 'openwebtext'
    dataset_dir: str = ''
    dataset_root: str = ''
    device: str = 'cuda:0'
    batch_size: int = 4
    global_batch_size: int = 480
    block_size: int = 1024
    vocab_size: int = 50_257
    seed: int = 1337
    learning_rate: float = 6e-4
    min_lr: float = 6e-5
    beta1: float = 0.9
    beta2: float = 0.95
    adam_eps: float = 1e-8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    eval_iters: int = 10
    checkpoint_interval: int = 50
    save_checkpoint: bool = True
    max_iters: int = 0  # Optional short prefix; keep the full scaling schedule.
    output_dir: str = ''
    wandb_log: bool = False
    wandb_project: str = 'scalable-analog-training'
    wandb_entity: str = ''
    dac_bits: int = 16
    adc_bits: int = 9
    c_out: float = 6.0
    backward_rail: float = 3.5
    omega: float = 3.0
    conductance_bound: float = 1.0
    embedding_read_chunk_size: int = 128

    @property
    def shape(self):
        return MODEL_SCALES[self.model_scale]

    @property
    def budget_parameters(self):
        # This correction also permits small synthetic fixtures in the tests.
        return self.shape['parameters'] + (self.vocab_size - 50_257) * self.shape['n_embd']

    @property
    def tokens_per_update(self):
        return self.global_batch_size * self.block_size

    @property
    def planned_updates(self):
        return math.ceil(20 * self.budget_parameters / self.tokens_per_update)

    @property
    def warmup_iters(self):
        return min(400, self.planned_updates // 20)

    @property
    def eval_interval(self):
        return max(1, self.planned_updates // 20)

    @property
    def final_validation_step(self):
        return self.planned_updates // self.eval_interval * self.eval_interval

    def validate(self, world_size=1):
        if self.mode not in {'digital', 'analog'}:
            raise ValueError('This release supports only digital AdamW and analog S-OUT AdamW.')
        if self.model_scale not in MODEL_SCALES:
            raise ValueError(f'model_scale must be one of {list(MODEL_SCALES)}')
        if self.weight_decay != 0:
            raise ValueError('The scaling comparison requires weight_decay=0.')
        for name in ('batch_size', 'global_batch_size', 'block_size', 'vocab_size', 'eval_iters',
                     'embedding_read_chunk_size'):
            if getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive.')
        if self.global_batch_size % (self.batch_size * world_size):
            raise ValueError('batch_size * world_size must divide global_batch_size.')
        if not 0 <= self.max_iters <= self.planned_updates:
            raise ValueError('max_iters must be zero (full schedule) or a prefix of planned_updates.')
        if min(self.omega, self.conductance_bound, self.c_out, self.backward_rail) <= 0:
            raise ValueError('Analog scales and rail multipliers must be positive.')
        if min(self.dac_bits, self.adc_bits) < 2:
            raise ValueError('Converter precision must be at least two bits.')
        if self.checkpoint_interval < 0:
            raise ValueError('checkpoint_interval must be nonnegative.')
        return self


def load_config(path):
    namespace = runpy.run_path(str(Path(path)))
    values = {name: value for name, value in namespace.items() if not name.startswith('_')}
    return ScalingConfig(**values).validate()


def learning_rate_at(step, config):
    if step < config.warmup_iters:
        return config.learning_rate * step / config.warmup_iters
    progress = (step - config.warmup_iters) / max(1, config.planned_updates - config.warmup_iters)
    return config.min_lr + 0.5 * (1 + math.cos(math.pi * min(1, progress))) * (
        config.learning_rate - config.min_lr)


def should_evaluate(step, config):
    return step % config.eval_interval == 0 or step in {
        value for value in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
        if value < config.eval_interval
    }


def config_dict(config):
    return {**asdict(config), 'planned_updates': config.planned_updates,
            'tokens_per_update': config.tokens_per_update}
