from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.nn.beam_search import BeamSearch
from torch.distributions import Normal, kl_divergence
import numpy as np
from torch.autograd import Variable

from . import Model
from .util import Losses, VAEOut


def masked_mean(vector: torch.Tensor,
                mask: torch.Tensor,
                dim: int,
                keepdim: bool = False,
                eps: float = 1e-8) -> torch.Tensor:
    one_minus_mask = (1.0 - mask.float()).to(dtype=torch.bool)
    replaced_vector = vector.masked_fill(one_minus_mask, 0.0)

    value_sum = torch.sum(replaced_vector, dim=dim, keepdim=keepdim)
    value_count = torch.sum(mask.float(), dim=dim, keepdim=keepdim)
    return value_sum / value_count.clamp(eps)


class BiMeanVAE(Model):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_size: int,
                 latent_dim: int,
                 pad_id: int,
                 bos_id: int,
                 eos_id: int,
                 num_layers: int = 1,
                 free_bit: float = 0.05):
        super().__init__(hidden_size, latent_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id

        self.SMALL_CONST = 1e-15
        self.one_hot_bows_vectors = []
        self.gm_scale=0.9
        self.stepsize=0.1

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_size // 2, num_layers, batch_first=True, bidirectional=True)

        self.proj_z = nn.Linear(hidden_size, 2 * latent_dim)
        self.proj_dec = nn.Sequential(nn.Linear(latent_dim, 2 * hidden_size),
                                      nn.Tanh())

        self.decoder = nn.LSTMCell(embedding_dim + latent_dim, hidden_size)
        self.output_layer = nn.Sequential(nn.Linear(hidden_size, embedding_dim),
                                          nn.Linear(embedding_dim, vocab_size, bias=True))
        self.beam = BeamSearch(self.eos_id, max_steps=128, beam_size=4, )
        self.bad_words = set()
        # Tying weight
        self.output_layer[-1].weight = self.embed.weight

        self.free_bit = free_bit

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor = None,
                do_generate: torch.Tensor = False,
                num_beams: int = 4,
                **kwargs):
        embed = self.embed(src)
        input_mask = torch.ne(src, self.pad_id)

        # Encoding
        input_length = input_mask.sum(dim=1).cpu().tolist()
        packed_embed = nn.utils.rnn.pack_padded_sequence(embed, input_length, batch_first=True, enforce_sorted=False)
        encoded = self.encoder(packed_embed)[0]
        encoded = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True)[0]
        encoded = masked_mean(encoded, input_mask.unsqueeze(-1), dim=1)
        mu, log_var = torch.chunk(self.proj_z(encoded), chunks=2, dim=-1)
        std = torch.exp(0.5 * log_var)

        q = Normal(mu, std)
        p = Normal(0, 1)
        zkl_real = kl_divergence(q, p)
        kl_mask = torch.gt(zkl_real, self.free_bit)
        bz = embed.size(0)
        zkl = zkl_real[kl_mask].sum() / bz
        zkl_real = zkl_real.sum(dim=-1).mean()
        if self.training:
            z = q.rsample()
            assert tgt is not None
            # Training:
            nll = self._recon_loss(tgt, z)
            return Losses(nll=nll, zkl=zkl, zkl_real=zkl_real)
        else:
            # Inference
            z = mu
            if do_generate:
                generated = self.generate(z, num_beams=num_beams)
            else:
                generated = None
            return VAEOut(q=q, generated=generated)

    def _recon_loss(self,
                    tgt_tensor: torch.Tensor,
                    z: torch.Tensor):
        targets = tgt_tensor[:, 1:]
        embed = self.embed(tgt_tensor[:, :-1])
        bz, max_len, _ = embed.size()
        hx, cx = torch.chunk(self.proj_dec(z), 2, dim=-1)
        recon_losses = []
        for t in range(max_len - 1):
            hx, cx = self.decoder(torch.cat((embed[:, t], z), dim=-1), (hx, cx))
            tgt_mask = targets[:, t] != self.pad_id
            non_masked_targets = targets[:, t].masked_select(tgt_mask)
            non_masked_embeddings = hx.masked_select(tgt_mask.unsqueeze(-1)).view(-1, self.hidden_size)
            non_masked_outs = self.output_layer(non_masked_embeddings)
            recon_losses.append(F.cross_entropy(non_masked_outs, non_masked_targets, reduction="sum"))

        recon_loss = torch.sum(torch.stack(recon_losses)) / bz
        return recon_loss

    def generate(self,
                 z: torch.Tensor,
                 num_beams: int = 4,
                 max_tokens: int = 256,
                 bad_words_ids: List[int] = None):
        self.eval()
        if bad_words_ids:
            self.bad_words.update(bad_words_ids)
        bz, device = len(z), z.device
        start_predictions = torch.full((bz,), fill_value=self.bos_id, dtype=torch.long, device=device)
        hx, cx = torch.chunk(self.proj_dec(z), 2, dim=-1)
        decoder_state = {"z": z, "hx": hx, "cx": cx}
        self.beam.beam_size = num_beams
        self.beam.max_steps = max_tokens
        all_top_k_predictions, _ = self.beam.search(start_predictions,
                                                    decoder_state,
                                                    self.step)
        self.bad_words.clear()
        return all_top_k_predictions[:, 0]

    @torch.no_grad()
    def step(self,
             last_predictions: torch.Tensor,
             state: Dict[str, torch.Tensor]):
        z = state["z"]
        hx, cx = self.decoder(torch.cat((self.embed(last_predictions), z), dim=-1), (state["hx"], state["cx"]))
        new_state = {"z": z, "hx": hx, "cx": cx}
        log_softmax = torch.nn.functional.log_softmax(self.output_layer(hx), dim=-1)
        if self.bad_words:
            log_softmax[:, list(self.bad_words)] = float("-inf")
        return log_softmax, new_state

    @staticmethod
    def klw(step: int,
            interval: int,
            r: float = 0.8,
            t: float = 0.0,
            s: int = 10000):
        if step < s:
            return 0.
        else:
            return min((step - s) / s, 1)


    def generatePerturb(self,
                 z: torch.Tensor,
                 num_beams: int = 4,
                 max_tokens: int = 256,
                 bad_words_ids: List[int] = None,
                 with_grad=False):
        torch.set_grad_enabled(with_grad)
        if bad_words_ids:
            self.bad_words.update(bad_words_ids)
        bz, device = len(z), z.device
        start_predictions = torch.full((bz,), fill_value=self.bos_id, dtype=torch.long, device=device)
        hx, cx = torch.chunk(self.proj_dec(z), 2, dim=-1)
        decoder_state = {"z": z, "hx": hx, "cx": cx}
        self.beam.beam_size = num_beams
        self.beam.max_steps = max_tokens
        all_top_k_predictions, _ = self.beam.search(start_predictions,
                                                    decoder_state,
                                                    self.stepPerturb)
        self.bad_words.clear()
        return all_top_k_predictions[:, 0]

    def to_var(self, x, requires_grad=False, volatile=False, device='cuda'):
        if torch.cuda.is_available() and device == 'cuda':
            x = x.cuda()
        elif device != 'cuda':
            x = x.to(device)
        return Variable(x, requires_grad=requires_grad, volatile=volatile)

    def perturb(self, last_predictions, state, unpert_log, num_iterations=3, device="cuda:0",num_classes=32000,kl_scale=0.0,log_probs_after_end=None,sampler_state = {}):
        torch.set_grad_enabled(True)
        with torch.enable_grad():
            z = state["z"]
            hx = state["hx"]
            cx = state["cx"]
            grad_accumulator = [
                    (np.zeros(p.shape).astype("float32"))
                    for p in hx
                ]
            grad_clean = np.zeros(z.shape).astype("float32")

            accumulated_hidden = 0
            loss_per_iter = []
            new_accumulated_hidden = None

            curr = self.to_var(torch.from_numpy(hx.cpu().detach().numpy()),requires_grad=True, device=device)

            for i in range(num_iterations):
                    #print("Iteration ", i + 1)
                    curr_perturbation = [
                        self.to_var(torch.from_numpy(p_), requires_grad=True, device=device)
                        for p_ in grad_accumulator
                    ]

                    hx, cx = self.decoder(torch.cat((self.embed(last_predictions), z), dim=-1), (curr, cx))
                    log_softmax = torch.nn.functional.log_softmax(self.output_layer(hx), dim=-1)               

                    loss = 0.0
                    loss_list = []
                    if True:#loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
                        for one_hot_bow in self.one_hot_bows_vectors:
                            bow_logits = torch.mm(log_softmax, torch.t(one_hot_bow))
                            bow_loss = -torch.log(-torch.sum(bow_logits))
                            loss += bow_loss
                            loss_list.append(bow_loss)
                            #print(" pplm_bow_loss:", loss.data.cpu().numpy())

                    kl_loss = 0.0
                    if kl_scale > 0.0:
                        unpert_probs = (
                                unpert_log + self.SMALL_CONST *
                                (unpert_log <= self.SMALL_CONST).float().to(device).detach()
                        )
                        correction = self.SMALL_CONST * (log_softmax <= self.SMALL_CONST).float().to(
                            device).detach()
                        corrected_probs = log_softmax + correction.detach()
                        kl_loss = kl_scale * (
                            (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
                        )
                        
                        #print(' kl_loss', kl_loss.data.cpu().numpy())
                        loss += kl_loss

                    loss_per_iter.append(loss.data.cpu().numpy())
                    
                    loss.backward(retain_graph=True)
                    grad = -self.stepsize *curr.grad.data.cpu().detach().numpy()
                    curr = self.to_var(torch.from_numpy(curr.cpu().detach().numpy()+grad),requires_grad=True, device=device)

            new_state = {"z":z, "hx":hx, "cx":cx}
            return log_softmax, new_state

    def stepPerturb(self,
             last_predictions,
             state, timestep=0):

        z = state["z"]
        hx, cx = self.decoder(torch.cat((self.embed(last_predictions), z), dim=-1), (state["hx"], state["cx"]))
        new_state = {"z": z, "hx": hx, "cx": cx}
        log_softmax = torch.nn.functional.log_softmax(self.output_layer(hx), dim=-1)

        if timestep > 1:
          pert_log_softmax, pert_state = self.perturb(last_predictions, state, log_softmax)

          log_softmax = -(((-pert_log_softmax) ** self.gm_scale) * (
                    (-log_softmax) ** (1 - self.gm_scale)))  # + SMALL_CONST
            

        if self.bad_words:
            log_softmax[:, list(self.bad_words)] = float("-inf")

        return log_softmax, new_state
