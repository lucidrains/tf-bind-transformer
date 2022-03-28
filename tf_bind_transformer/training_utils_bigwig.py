import torch
from torch import nn
from tf_bind_transformer.optimizer import get_optimizer
from tf_bind_transformer.data_bigwig import BigWigDataset, BigWigTracksOnlyDataset, get_bigwig_dataloader, get_bigwig_tracks_dataloader
from enformer_pytorch.modeling_enformer import poisson_loss, pearson_corr_coef

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
        human_factor_fasta_folder,
        annot_file_path,
        human_loci_path,
        mouse_loci_path,
        human_fasta_file,
        mouse_fasta_file,
        batch_size,
        bigwig_tracks_only_folder_path = None,
        bigwig_folder_path = None,
        train_chromosome_ids = None,
        valid_chromosome_ids = None,
        mouse_factor_fasta_folder = None,
        downsample_factor = 128,
        target_length = 896,
        lr = 3e-4,
        wd = 0.1,
        validate_every = 250,
        grad_clip_norm = None,
        grad_accum_every = 1,
        held_out_targets_human = [],
        held_out_targets_mouse = [],
        held_out_cell_types_human = [],
        held_out_cell_types_mouse = [],
        context_length = 4096,
        shuffle = False,
        shift_aug_range = (-2, 2),
        rc_aug = False,
        checkpoint_filename = './checkpoint.pt',
        include_biotypes_metadata_in_context = False,
        biotypes_metadata_path = None,
        include_biotypes_metadata_columns = ['germ_layer', 'cellline_cat'],
        biotypes_metadata_delimiter = ' | ',
        bigwig_reduction_type = 'sum',
        enformer_train_valid_split = True
    ):
        super().__init__()
        assert exists(bigwig_folder_path) or exists(bigwig_tracks_only_folder_path)

        self.model = model

        mouse_factor_fasta_folder = default(mouse_factor_fasta_folder, human_factor_fasta_folder)

        self.human_ds = BigWigDataset(
            filter_chromosome_ids = train_chromosome_ids,
            factor_fasta_folder = human_factor_fasta_folder,
            bigwig_folder = bigwig_folder_path,
            enformer_loci_path = human_loci_path,
            annot_file = annot_file_path,
            fasta_file = human_fasta_file,
            exclude_targets = held_out_targets_human,
            exclude_cell_types = held_out_cell_types_human,
            target_length = target_length,
            context_length = context_length,
            downsample_factor = downsample_factor,
            shift_augs = shift_aug_range,
            rc_aug = rc_aug,
            bigwig_reduction_type = bigwig_reduction_type,
            filter_sequences_by = ('column_4', 'train') if enformer_train_valid_split else None,
            only_ref = ['hg38'],
            factor_species_priority = ['human', 'mouse', 'unknown']
        )

        self.valid_human_ds = BigWigDataset(
            filter_chromosome_ids = valid_chromosome_ids,
            factor_fasta_folder = human_factor_fasta_folder,
            bigwig_folder = bigwig_folder_path,
            enformer_loci_path = human_loci_path,
            annot_file = annot_file_path,
            fasta_file = human_fasta_file,
            include_targets = held_out_targets_human,
            include_cell_types = held_out_cell_types_human,
            target_length = target_length,
            context_length = context_length,
            downsample_factor = downsample_factor,
            shift_augs = shift_aug_range,
            rc_aug = rc_aug,
            bigwig_reduction_type = bigwig_reduction_type,
            filter_sequences_by = ('column_4', 'valid') if enformer_train_valid_split else None,
            only_ref = ['hg38'],
            factor_species_priority = ['human', 'mouse', 'unknown']
        )

        self.mouse_ds = BigWigDataset(
            filter_chromosome_ids = train_chromosome_ids,
            factor_fasta_folder = mouse_factor_fasta_folder,
            bigwig_folder = bigwig_folder_path,
            enformer_loci_path = mouse_loci_path,
            annot_file = annot_file_path,
            fasta_file = mouse_fasta_file,
            exclude_targets = held_out_targets_mouse,
            exclude_cell_types = held_out_cell_types_mouse,
            target_length = target_length,
            context_length = context_length,
            downsample_factor = downsample_factor,
            shift_augs = shift_aug_range,
            rc_aug = rc_aug,
            bigwig_reduction_type = bigwig_reduction_type,
            filter_sequences_by = ('column_4', 'train') if enformer_train_valid_split else None,
            only_ref = ['mm10'],
            factor_species_priority = ['mouse', 'human', 'unknown']
        )

        self.valid_mouse_ds = BigWigDataset(
            filter_chromosome_ids = valid_chromosome_ids,
            factor_fasta_folder = mouse_factor_fasta_folder,
            bigwig_folder = bigwig_folder_path,
            enformer_loci_path = mouse_loci_path,
            annot_file = annot_file_path,
            fasta_file = mouse_fasta_file,
            include_targets = held_out_targets_mouse,
            include_cell_types = held_out_cell_types_mouse,
            target_length = target_length,
            context_length = context_length,
            downsample_factor = downsample_factor,
            shift_augs = shift_aug_range,
            rc_aug = rc_aug,
            bigwig_reduction_type = bigwig_reduction_type,
            filter_sequences_by = ('column_4', 'valid') if enformer_train_valid_split else None,
            only_ref = ['mm10'],
            factor_species_priority = ['mouse', 'human', 'unknown']
        )

        self.human_head_ds = BigWigTracksOnlyDataset(
            ref = 'hg38',
            bigwig_folder = bigwig_tracks_only_folder_path,
            enformer_loci_path = human_loci_path,
            fasta_file = human_fasta_file,
            annot_file = annot_file_path,
            downsample_factor = downsample_factor,
            target_length = target_length,
            filter_sequences_by = ('column_4', 'train')
        )

        self.valid_human_head_ds = BigWigTracksOnlyDataset(
            ref = 'hg38',
            bigwig_folder = bigwig_tracks_only_folder_path,
            enformer_loci_path = human_loci_path,
            fasta_file = human_fasta_file,
            annot_file = annot_file_path,
            downsample_factor = downsample_factor,
            target_length = target_length,
            filter_sequences_by = ('column_4', 'valid')
        )

        self.mouse_head_ds = BigWigTracksOnlyDataset(
            ref = 'mm10',
            bigwig_folder = bigwig_tracks_only_folder_path,
            enformer_loci_path = mouse_loci_path,
            fasta_file = mouse_fasta_file,
            annot_file = annot_file_path,
            downsample_factor = downsample_factor,
            target_length = target_length,
            filter_sequences_by = ('column_4', 'train')
        )

        self.valid_mouse_head_ds = BigWigTracksOnlyDataset(
            ref = 'mm10',
            bigwig_folder = bigwig_tracks_only_folder_path,
            enformer_loci_path = mouse_loci_path,
            fasta_file = mouse_fasta_file,
            annot_file = annot_file_path,
            downsample_factor = downsample_factor,
            target_length = target_length,
            filter_sequences_by = ('column_4', 'valid')
        )

        len_train_human = len(self.human_ds)
        len_train_mouse = len(self.mouse_ds)
        len_valid_human = len(self.valid_human_ds)
        len_valid_mouse = len(self.valid_mouse_ds)

        len_train_human_head = len(self.human_head_ds)
        len_valid_human_head = len(self.valid_human_head_ds)
        len_train_mouse_head = len(self.mouse_head_ds)
        len_valid_mouse_head = len(self.valid_mouse_head_ds)

        self.has_train = len_train_human > 0 or len_train_mouse > 0 or len_train_human_head > 0 or len_train_mouse_head > 0
        self.has_valid = len_valid_human > 0 or len_valid_mouse > 0 or len_valid_human_head > 0 or len_valid_mouse_head > 0

        if self.has_train:
            print(f'training with {self.human_ds.ntargets} human targets and {self.mouse_ds.ntargets} mice targets')
            print(f'training independent tracks with {self.human_head_ds.ntargets} human targets and {self.mouse_head_ds.ntargets} mouse targets')

        if self.has_valid:
            print(f'validating with {self.valid_human_ds.ntargets} human targets and {self.valid_mouse_ds.ntargets} mice targets')
            print(f'validating independent tracks with {self.valid_human_head_ds.ntargets} human targets and {self.valid_mouse_head_ds.ntargets} mouse targets')

        assert self.has_train and self.has_valid, 'must have training and validation samples in order to proceed'

        self.train_human_dl = get_bigwig_dataloader(self.human_ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size) if len_train_human > 0 else None
        self.train_mouse_dl = get_bigwig_dataloader(self.mouse_ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size) if len_train_mouse > 0 else None

        self.valid_human_dl = get_bigwig_dataloader(self.valid_human_ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size) if len_valid_human > 0 else None
        self.valid_mouse_dl = get_bigwig_dataloader(self.valid_mouse_ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size) if len_valid_mouse > 0 else None

        # dataloader for independent tracks without

        self.train_human_head_dl = get_bigwig_tracks_dataloader(self.human_head_ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size) if len_train_human_head > 0 else None
        self.train_mouse_head_dl = get_bigwig_tracks_dataloader(self.mouse_head_ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size) if len_train_mouse_head > 0 else None

        self.valid_human_head_dl = get_bigwig_tracks_dataloader(self.valid_human_head_ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size) if len_valid_human_head > 0 else None
        self.valid_mouse_head_dl = get_bigwig_tracks_dataloader(self.valid_mouse_head_ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size) if len_valid_mouse_head > 0 else None

        # optimizer

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
        loss_divisor = int(exists(self.train_human_dl)) + int(exists(self.train_mouse_dl)) + int(exists(self.train_human_head_dl)) + int(exists(self.train_mouse_head_dl))

        if exists(self.train_human_dl):
            for _ in range(grad_accum_every):
                seq, tf_aa, contextual_texts, target = next(self.train_human_dl)
                seq, target = seq.cuda(), target.cuda()

                loss = self.model(
                    seq,
                    aa = tf_aa,
                    contextual_free_text = contextual_texts,
                    target = target,
                    finetune_enformer_ln_only = finetune_enformer_ln_only,
                    **kwargs
                )

                log = accum_log(log, {'human_loss': loss.item() / grad_accum_every})
                (loss / self.grad_accum_every / loss_divisor).backward()

            print(f'{curr_step} human loss: {log["human_loss"]}')

        if exists(self.train_mouse_dl):
            for _ in range(grad_accum_every):
                seq, tf_aa, contextual_texts, target = next(self.train_mouse_dl)
                seq, target = seq.cuda(), target.cuda()

                loss = self.model(
                    seq,
                    aa = tf_aa,
                    contextual_free_text = contextual_texts,
                    target = target,
                    finetune_enformer_ln_only = finetune_enformer_ln_only,
                    **kwargs
                )

                log = accum_log(log, {'mouse_loss': loss.item() / grad_accum_every})
                (loss / self.grad_accum_every / loss_divisor).backward()

            print(f'{curr_step} mouse loss: {log["mouse_loss"]}')

        if exists(self.train_human_head_dl):
            for _ in range(grad_accum_every):
                seq, target = next(self.train_human_head_dl)
                seq, target = seq.cuda(), target.cuda()

                loss = self.model(
                    seq,
                    target = target,
                    head = 'human',
                    finetune_enformer_ln_only = finetune_enformer_ln_only,
                    **kwargs
                )

                log = accum_log(log, {'human_head_loss': loss.item() / grad_accum_every})
                (loss / self.grad_accum_every / loss_divisor).backward()

            print(f'{curr_step} human head loss: {log["human_head_loss"]}')

        if exists(self.train_mouse_head_dl):
            for _ in range(grad_accum_every):
                seq, target = next(self.train_mouse_head_dl)
                seq, target = seq.cuda(), target.cuda()

                loss = self.model(
                    seq,
                    target = target,
                    head = 'mouse',
                    finetune_enformer_ln_only = finetune_enformer_ln_only,
                    **kwargs
                )

                log = accum_log(log, {'mouse_head_loss': loss.item() / grad_accum_every})
                (loss / self.grad_accum_every / loss_divisor).backward()

            print(f'{curr_step} mouse head loss: {log["mouse_head_loss"]}')

        # gradient clipping

        if exists(self.grad_clip_norm):
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        # take a gradient step

        self.optim.step()
        self.optim.zero_grad()

        # validation

        if (curr_step % self.validate_every) == 0:
            self.model.eval()

            if exists(self.valid_human_dl):
                for _ in range(grad_accum_every):
                    seq, tf_aa, contextual_texts, target = next(self.valid_human_dl)
                    seq, target = seq.cuda(), target.cuda()

                    pred = self.model(
                        seq,
                        aa = tf_aa,
                        contextual_free_text = contextual_texts,
                    )

                    valid_loss = self.model.loss_fn(pred, target)
                    valid_corr_coef = pearson_corr_coef(pred, target)

                    log = accum_log(log, {
                        'human_valid_loss': valid_loss.item() / grad_accum_every,
                        'human_valid_corr_coef': valid_corr_coef.item() / grad_accum_every
                    })

                print(f'{curr_step} human valid loss: {log["human_valid_loss"]}')
                print(f'{curr_step} human valid pearson R: {log["human_valid_corr_coef"]}')

            if exists(self.valid_mouse_dl):
                for _ in range(grad_accum_every):
                    seq, tf_aa, contextual_texts, target = next(self.valid_mouse_dl)
                    seq, target = seq.cuda(), target.cuda()

                    pred = self.model(
                        seq,
                        aa = tf_aa,
                        contextual_free_text = contextual_texts,
                    )

                    valid_loss = self.model.loss_fn(pred, target)
                    valid_corr_coef = pearson_corr_coef(pred, target)

                    log = accum_log(log, {
                        'mouse_valid_loss': valid_loss.item() / grad_accum_every,
                        'mouse_valid_corr_coef': valid_corr_coef.item() / grad_accum_every
                    })

                print(f'{curr_step} mouse valid loss: {log["mouse_valid_loss"]}')
                print(f'{curr_step} mouse valid pearson R: {log["mouse_valid_corr_coef"]}')

            if exists(self.valid_human_head_dl):
                for _ in range(grad_accum_every):
                    seq, target = next(self.valid_human_head_dl)
                    seq, target = seq.cuda(), target.cuda()

                    pred = self.model(seq, head = 'human')

                    valid_loss = self.model.loss_fn(pred, target)
                    valid_corr_coef = pearson_corr_coef(pred, target).mean()

                    log = accum_log(log, {
                        'human_head_valid_loss': valid_loss.item() / grad_accum_every,
                        'human_head_valid_corr_coef': valid_corr_coef.item() / grad_accum_every
                    })

                print(f'{curr_step} human head valid loss: {log["human_head_valid_loss"]}')
                print(f'{curr_step} human head valid pearson R: {log["human_head_valid_corr_coef"]}')

            if exists(self.valid_mouse_head_dl):
                for _ in range(grad_accum_every):
                    seq, target = next(self.valid_mouse_head_dl)
                    seq, target = seq.cuda(), target.cuda()

                    pred = self.model(seq, head = 'mouse')

                    valid_loss = self.model.loss_fn(pred, target)
                    valid_corr_coef = pearson_corr_coef(pred, target).mean()

                    log = accum_log(log, {
                        'mouse_head_valid_loss': valid_loss.item() / grad_accum_every,
                        'mouse_head_valid_corr_coef': valid_corr_coef.item() / grad_accum_every
                    })


                print(f'{curr_step} mouse head valid loss: {log["mouse_head_valid_loss"]}')
                print(f'{curr_step} mouse head valid pearson R: {log["mouse_head_valid_corr_coef"]}')

                if curr_step > 0:
                    torch.save(self.model.state_dict(), self.checkpoint_filename)

        self.steps += 1
        return log
