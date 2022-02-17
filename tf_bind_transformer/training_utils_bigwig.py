import torch
from torch import nn
from tf_bind_transformer.optimizer import get_optimizer
from tf_bind_transformer.data_bigwig import BigWigDataset, get_bigwig_dataloader
from enformer_pytorch.enformer_pytorch import poisson_loss, pearson_corr_coef

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# helpers for logging and accumulating values across gradient steps

def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log

# simple Trainer class

class BigWigTrainer(nn.Module):
    def __init__(
        self,
        model,
        *,
        factor_fasta_folder,
        bigwig_folder_path,
        loci_path,
        annot_file_path,
        fasta_file,
        train_chromosome_ids,
        valid_chromosome_ids,
        batch_size,
        valid_bigwig_folder_path = None,
        valid_loci_path = None,
        valid_annot_file_path = None,
        downsample_factor = 128,
        target_length = 896,
        lr = 3e-4,
        wd = 0.1,
        validate_every = 250,
        grad_clip_norm = None,
        grad_accum_every = 1,
        held_out_targets = [],
        held_out_cell_types = [],
        context_length = 4096,
        shuffle = False,
        shift_aug_range = (-2, 2),
        rc_aug = False,
        only_ref = ['mm10', 'hg38'],
        checkpoint_filename = './checkpoint.pt',
        include_biotypes_metadata_in_context = False,
        biotypes_metadata_path = None,
        include_biotypes_metadata_columns = ['germ_layer', 'cellline_cat'],
        biotypes_metadata_delimiter = ' | '
    ):
        super().__init__()
        self.model = model

        self.ds = BigWigDataset(
            filter_chromosome_ids = train_chromosome_ids,
            factor_fasta_folder = factor_fasta_folder,
            bigwig_folder = bigwig_folder_path,
            enformer_loci_path = loci_path,
            annot_file = annot_file_path,
            fasta_file = fasta_file,
            exclude_targets = held_out_targets,
            exclude_cell_types = held_out_cell_types,
            only_ref = only_ref,
            target_length = target_length,
            downsample_factor = downsample_factor,
            shift_augs = shift_aug_range,
            rc_aug = rc_aug,
        )

        self.valid_ds = BigWigDataset(
            filter_chromosome_ids = valid_chromosome_ids,
            factor_fasta_folder = factor_fasta_folder,
            bigwig_folder = default(valid_bigwig_folder_path, bigwig_folder_path),
            enformer_loci_path = default(valid_loci_path, loci_path),
            annot_file = default(valid_annot_file_path, annot_file_path),
            fasta_file = fasta_file,
            include_targets = held_out_targets,
            include_cell_types = held_out_cell_types,
            only_ref = only_ref,
            target_length = target_length,
            downsample_factor = downsample_factor,
            shift_augs = shift_aug_range,
            rc_aug = rc_aug,
        )

        self.train_dl = get_bigwig_dataloader(self.ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size)
        self.valid_dl = get_bigwig_dataloader(self.valid_ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size)

        self.optim = get_optimizer(model.parameters(), lr = lr, wd = wd)

        self.grad_accum_every = grad_accum_every
        self.grad_clip_norm = grad_clip_norm

        self.validate_every = validate_every
        self.register_buffer('steps', torch.Tensor([0.]))

        self.checkpoint_filename = checkpoint_filename

    def forward(
        self,
        finetune_enformer_ln_only = True,
        **kwargs
    ):
        grad_accum_every = self.grad_accum_every
        curr_step = int(self.steps.item())
        self.model.train()

        log = {}

        for _ in range(self.grad_accum_every):
            seq, tf_aa, contextual_texts, target = next(self.train_dl)
            seq, target = seq.cuda(), target.cuda()

            loss = self.model(
                seq,
                aa = tf_aa,
                contextual_free_text = contextual_texts,
                target = target,
                finetune_enformer_ln_only = finetune_enformer_ln_only,
                **kwargs
            )

            log = accum_log(log, {'loss': loss.item() / grad_accum_every})
            (loss / self.grad_accum_every).backward()

        print(f'{curr_step} loss: {log["loss"]}')

        if exists(self.grad_clip_norm):
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optim.step()
        self.optim.zero_grad()

        if (curr_step % self.validate_every) == 0:
            self.model.eval()

            for _ in range(self.grad_accum_every):
                seq, tf_aa, contextual_texts, target = next(self.valid_dl)
                seq, target = seq.cuda(), target.cuda()

                pred = self.model(
                    seq,
                    aa = tf_aa,
                    contextual_free_text = contextual_texts,
                )

                valid_poisson_loss = poisson_loss(pred, target)
                valid_corr_coef = pearson_corr_coef(pred, target)

                log = accum_log(log, {
                    'valid_loss': valid_poisson_loss.item() / grad_accum_every,
                    'valid_corr_coef': valid_corr_coef.item() / grad_accum_every
                })

            print(f'{curr_step} valid loss: {log["valid_loss"]}')
            print(f'{curr_step} valid accuracy: {log["valid_corr_coef"]}')

            if curr_step > 0:
                torch.save(self.model.state_dict(), self.checkpoint_filename)

        self.steps += 1
        return log
