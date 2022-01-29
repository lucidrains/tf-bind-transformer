import torch
from torch import nn
from torch.optim import AdamW
from tf_bind_transformer.data import read_bed, collate_dl_outputs, get_dataloader, RemapAllPeakDataset, NegativePeakDataset

def exists(val):
    return val is not None

def separate_weight_decayable_params(params):
    no_wd_params = set([param for param in params if param.ndim < 2])
    wd_params = set(params) - no_wd_params
    return wd_params, no_wd_params

def get_optimizer(params, lr = 3e-4, wd = 1e-1, filter_by_requires_grad = False):
    if filter_by_requires_grad:
        params = list(filter(lambda t: t.requires_grad, params))

    params = set(params)
    wd_params, no_wd_params = separate_weight_decayable_params(params)

    param_groups = [
        {'params': list(wd_params)},
        {'params': list(no_wd_params), 'weight_decay': 0},
    ]

    return AdamW(param_groups, lr = lr, weight_decay = wd)

# simple Trainer class

class Trainer(nn.Module):
    def __init__(
        self,
        model,
        *,
        remap_bed_file,
        negative_bed_file,
        factor_fasta_folder,
        fasta_file,
        train_chromosome_ids,
        valid_chromosome_ids,
        batch_size,
        validate_every = 250,
        grad_clip_norm = None,
        grad_accum_every = 1,
        held_out_targets = [],
        context_length = 4096,
        shuffle = False,
    ):
        super().__init__()
        self.model = model

        remap_df = read_bed(remap_bed_file)
        neg_df = read_bed(negative_bed_file)

        self.ds = RemapAllPeakDataset(
            remap_df = remap_df,
            fasta_file = fasta_file,
            factor_fasta_folder = factor_fasta_folder,
            filter_chromosome_ids = train_chromosome_ids,
            exclude_targets = held_out_targets,
            context_length = context_length
        )

        self.neg_ds = NegativePeakDataset(
            remap_df = remap_df,
            negative_df = neg_df,
            fasta_file = fasta_file,
            factor_fasta_folder = factor_fasta_folder,
            filter_chromosome_ids = train_chromosome_ids,
            exclude_targets = held_out_targets,
            context_length = context_length
        )

        self.valid_ds = RemapAllPeakDataset(
            remap_df = remap_df,
            fasta_file = fasta_file,
            factor_fasta_folder = factor_fasta_folder,
            include_targets = held_out_targets,
            filter_chromosome_ids = valid_chromosome_ids,
            context_length = context_length
        )

        self.valid_neg_ds = NegativePeakDataset(
            remap_df = remap_df,
            negative_df = neg_df,
            fasta_file = fasta_file,
            factor_fasta_folder = factor_fasta_folder,
            filter_chromosome_ids = valid_chromosome_ids,
            include_targets = held_out_targets,
            context_length = context_length
        )

        self.dl = get_dataloader(self.ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size)
        self.neg_dl = get_dataloader(self.neg_ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size)

        self.valid_dl = get_dataloader(self.valid_ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size)
        self.valid_neg_dl = get_dataloader(self.valid_neg_ds, cycle_iter = True, shuffle = shuffle, batch_size = batch_size)

        self.optim = get_optimizer(model.parameters())

        self.grad_accum_every = grad_accum_every
        self.grad_clip_norm = grad_clip_norm

        self.validate_every = validate_every
        self.register_buffer('steps', torch.Tensor([0.]))

    def forward(self):
        curr_step = int(self.steps.item())
        self.model.train()

        total_train_loss = 0
        for _ in range(self.grad_accum_every):
            seq, tf_aa, contextual_texts, _, binary_target = collate_dl_outputs(next(self.dl), next(self.neg_dl))
            seq, binary_target = seq.cuda(), binary_target.cuda()

            loss = self.model(
                seq,
                target = binary_target,
                aa = tf_aa,
                contextual_free_text = contextual_texts,
                finetune_enformer_ln_only = True
            )

            total_train_loss += loss.item()

            (loss / self.grad_accum_every).backward()

        avg_loss = total_train_loss / self.grad_accum_every
        logs = {'loss': avg_loss}
        print(f'{curr_step} loss: {avg_loss}')

        if exists(self.grad_clip_norm):
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optim.step()
        self.optim.zero_grad()

        if not (curr_step % self.validate_every):
            self.model.eval()

            total_valid_loss = 0
            total_valid_accuracies = 0

            for _ in range(self.grad_accum_every):
                seq, tf_aa, contextual_texts, _, binary_target = collate_dl_outputs(next(self.valid_dl), next(self.valid_neg_dl))
                seq, binary_target = seq.cuda(), binary_target.cuda()

                valid_logits = self.model(
                    seq,
                    aa = tf_aa,
                    contextual_free_text = contextual_texts,
                )

                valid_loss = self.model.loss_fn(valid_logits, binary_target.float())
                valid_accuracy = ((valid_logits.sigmoid() > 0.5).int() == binary_target).sum() / (binary_target.numel())

                total_valid_loss += valid_loss.item()
                total_valid_accuracies = valid_accuracy.item()

            avg_valid_loss = total_valid_loss / self.grad_accum_every
            avg_valid_accuracy = total_valid_accuracies / self.grad_accum_every

            logs = {
                **logs,
                'valid_loss': avg_valid_loss,
                'valid_accuracy': avg_valid_accuracy
            }

            print(f'{curr_step} valid loss: {avg_valid_loss}')
            print(f'{curr_step} valid accuracy: {avg_valid_accuracy}')

            torch.save(self.model.state_dict(), f'./checkpoint.pt')

        self.steps += 1
        return logs
