from pathlib import Path
import click
import pandas as pd
import rouge
import nltk
from collections import Counter
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from typing import List
import torch
from tqdm import tqdm
import argparse
import json
from operator import add
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
from transformers.modeling_gpt2 import GPT2LMHeadModel
from coop import VAE, util
from transformers import GPT2Tokenizer
from typing import Iterable, List, Optional, Tuple
from torch import Tensor
from transformers.file_utils import ModelOutput
from transformers.utils import logging


logger = logging.get_logger(__name__)


class ClassificationHead(torch.nn.Module):
    """Classification Head for  transformer encoders"""

    def __init__(self, class_size, embed_size):
        super(ClassificationHead, self).__init__()
        self.class_size = class_size
        self.embed_size = embed_size
        # self.mlp1 = torch.nn.Linear(embed_size, embed_size)
        # self.mlp2 = (torch.nn.Linear(embed_size, class_size))
        self.mlp = torch.nn.Linear(embed_size, class_size)

    def forward(self, hidden_state):
        # hidden_state = F.relu(self.mlp1(hidden_state))
        # hidden_state = self.mlp2(hidden_state)
        logits = self.mlp(hidden_state)
        return logits

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

BAG_OF_WORDS_ARCHIVE_MAP = {
    'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    'monsters': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    'positive_words': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}


def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


def perturb_past(
        past,
        model,
        last,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        one_hot_bows_vectors=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR,
        output_so_far=None
):
    pastZ=True
    # Generate inital perturbed past
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([window_length])
                + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        if verbosity_level >= VERBOSE:
            print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape

        #if pastZ: 
        #model_inputs = model.prepare_inputs_for_generation(input_ids=output_so_far, past_key_values=(z,)) #output_so_far
        #all_logits, _, all_hidden = model(**model_inputs, return_dict=False, output_hidden_states=True, use_cache=True)
        #else:
        all_logits, _, all_hidden = model(last, past_key_values=perturbed_past, return_dict=False, output_hidden_states=True, use_cache=True, pertPast=True)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            for one_hot_bow in one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                bow_loss = -torch.log(torch.sum(bow_logits))
                bow_idx = torch.where(bow_logits>0)[1]
                #print("-"*10)
                #print([tokenizerGPT.decode(bow_indices[0][idx]) for idx in bow_idx])
                #print("-"*10)
                loss += bow_loss
                loss_list.append(bow_loss)
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_bow_loss:", loss.data.cpu().numpy())

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            if verbosity_level >= VERY_VERBOSE:
                print(' kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward(retain_graph=True)

        # calculate gradient norms
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                for index, p_ in enumerate(curr_perturbation)
            ]

        # normalize gradients
        grad = [
            -stepsize *
            (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def prepare_inputs_for_generation(self, input_ids, **kwargs):
    """
    Implement in subclasses of :class:`~transfomers.PreTrainedModel` for custom behavior to prepare inputs in the
    generate method.
    """
    return {"input_ids": input_ids}

def adjust_logits_during_generation(self, logits, **kwargs):
    """
    Implement in subclasses of :class:`~transfomers.PreTrainedModel` for custom behavior to adjust the logits in
    the generate method.
    """
    return logits

def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
    """
    Enforce the repetition penalty (from the `CTRL paper <https://arxiv.org/abs/1909.05858>`__).
    """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty

def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p
        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


@staticmethod
def _reorder_cache(past: Tuple, beam_idx: Tensor) -> Tuple[Tensor]:
    return tuple(layer_past.index_select(1, beam_idx) for layer_past in past)


def calc_banned_ngram_tokens(prev_input_ids: Tensor, num_hypos: int, no_repeat_ngram_size: int, cur_len: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids: Iterable[int], bad_words_ids: Iterable[int]) -> Iterable[int]:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_tokens):
            # if bad word tokens are longer than prev tokens they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice, banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def set_scores_to_inf_for_banned_tokens(scores: torch.Tensor, banned_tokens: List[List[int]]) -> None:
    """Modifies the scores in place by setting the banned token positions to `-inf`. Banned token is expected to be
    a list of list of banned tokens to ban in the format [[batch index, vocabulary position],...]
        Args:
            scores: logits distribution of shape (batch size, vocabulary size)
            banned_tokens: list of list of tokens to ban of length (batch_size)
    """
    banned_mask_list = []
    for idx, batch_banned_tokens in enumerate(banned_tokens):
        for token in batch_banned_tokens:
            banned_mask_list.append([idx, token])
    if not banned_mask_list:
        return
    banned_mask = torch.LongTensor(banned_mask_list)
    indices = torch.ones(len(banned_mask))
    # A sparse tensor is generated from a list of coordinates: [[0, 1], [0, 2], [2, 0]]. A conversion to dense tensor generates:
    # [ 0  1  1 ]
    # [ 0  0  0 ]
    # [ 1  0  0 ]

    banned_mask = torch.sparse.LongTensor(banned_mask.t(), indices, scores.size()).to(scores.device).to_dense().bool()
    scores.masked_fill_(banned_mask, -float("inf"))



def postprocess_next_token_scores(
    model,
    scores,
    input_ids,
    no_repeat_ngram_size,
    bad_words_ids,
    cur_len,
    min_length,
    max_length,
    eos_token_id,
    repetition_penalty,
    batch_size,
    num_beams,
    good_words_ids=None,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        model.enforce_repetition_penalty_(
            scores,
            batch_size,
            num_beams,
            input_ids,
            repetition_penalty,
        )

    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        scores[:, eos_token_id] = -float("inf")

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(
            input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    if bad_words_ids is not None:
        # Exclude EOS token (already processed)
        bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids.tolist(), bad_words_ids)
        # Modify the scores in place by setting the banned tokens logits to `-inf`
        set_scores_to_inf_for_banned_tokens(scores, banned_tokens)

    if good_words_ids is not None:
        # Exclude EOS token (already processed)         
        bad_words_ids = list(filter(lambda bad_token_seq: bad_token_seq != [eos_token_id], bad_words_ids))
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids.tolist(), bad_words_ids)
        # Modify the scores in place by setting the banned tokens logits to `-inf`
        set_scores_to_inf_for_banned_tokens(scores, banned_tokens)

    return scores

#@torch.no_grad()
def generateScratch(
    model,
    input_ids: Optional[torch.LongTensor] = None,
    decoder_input_ids: Optional[torch.LongTensor] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    early_stopping: Optional[bool] = None,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    bad_words_ids: Optional[Iterable[int]] = None,
    bos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    one_hot_bows_vectors = None,
    length_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    decoder_start_token_id: Optional[int] = None,
    use_cache: Optional[bool] = None,
    **model_kwargs
) -> torch.LongTensor:
    r"""
    Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
    beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.
    Adapted in part from `Facebook's XLM beam search code
    <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.
    Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
    attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
    indicated are the default values of those config.
    Most of these parameters are explained in more detail in `this blog post
    <https://huggingface.co/blog/how-to-generate>`__.
    Parameters:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            The sequence used as a prompt for the generation. If :obj:`None` the method initializes
            it as an empty :obj:`torch.LongTensor` of shape :obj:`(1,)`.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            initial input_ids for the decoder of encoder-decoder type models. If :obj:`None` then only
            decoder_start_token_id is passed as the first token to the decoder.
        max_length (:obj:`int`, `optional`, defaults to 20):
            The maximum length of the sequence to be generated.
        min_length (:obj:`int`, `optional`, defaults to 10):
            The minimum length of the sequence to be generated.
        do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to use sampling ; use greedy decoding otherwise.
        early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        num_beams (:obj:`int`, `optional`, defaults to 1):
            Number of beams for beam search. 1 means no beam search.
        temperature (:obj:`float`, `optional`, defaults tp 1.0):
            The value used to module the next token probabilities.
        top_k (:obj:`int`, `optional`, defaults to 50):
            The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (:obj:`float`, `optional`, defaults to 1.0):
            If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or
            higher are kept for generation.
        repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
            The parameter for repetition penalty. 1.0 means no penalty. See `this paper
            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
        pad_token_id (:obj:`int`, `optional`):
            The id of the `padding` token.
        bos_token_id (:obj:`int`, `optional`):
            The id of the `beginning-of-sequence` token.
        eos_token_id (:obj:`int`, `optional`):
            The id of the `end-of-sequence` token.
        length_penalty (:obj:`float`, `optional`, defaults to 1.0):
            Exponential penalty to the length. 1.0 means no penalty.
            Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
            order to encourage the model to produce longer sequences.
        no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
            If set to int > 0, all ngrams of that size can only occur once.
        bad_words_ids(:obj:`List[int]`, `optional`):
            List of token ids that are not allowed to be generated. In order to get the tokens of the words that
            should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
        num_return_sequences(:obj:`int`, `optional`, defaults to 1):
            The number of independently computed returned sequences for each element in the batch.
        attention_mask (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
            tokens that are not masked, and 0 for masked tokens.
            If not provided, will default to a tensor the same shape as :obj:`input_ids` that masks the pad token.
            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_start_token_id (:obj:`int`, `optional`):
            If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
        use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should use the past last key/values attentions (if applicable to the model) to
            speed up decoding.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.
    Return:
        :obj:`torch.LongTensor` of shape :obj:`(batch_size * num_return_sequences, sequence_length)`:
        The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
        shorter if all batches finished early due to the :obj:`eos_token_id`.
    Examples::
        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
        outputs = model.generate(max_length=40)  # do greedy decoding
        print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
        tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
        input_context = 'The dog'
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
        for i in range(3): #  3 output sequences were generated
            print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))
        tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
        input_context = 'The dog'
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True)  # generate 3 candidates using sampling
        for i in range(3): #  3 output sequences were generated
            print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))
        tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
        input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
        print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))
        tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
        model = AutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
        input_context = 'My cute dog'  # "Legal" is one of the control codes for ctrl
        bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
        input_ids = tokenizer.encode(input_context, return_tensors='pt')  # encode input context
        outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
    """

    # We cannot generate if the model does not have a LM head
    if model.get_output_embeddings() is None:
        raise AttributeError(
            "You tried to generate sequences with a model that does not have a LM Head."
            "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, `CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
        )

    max_length = max_length if max_length is not None else model.config.max_length
    min_length = min_length if min_length is not None else model.config.min_length
    do_sample = do_sample if do_sample is not None else model.config.do_sample
    early_stopping = early_stopping if early_stopping is not None else model.config.early_stopping
    use_cache = use_cache if use_cache is not None else model.config.use_cache
    num_beams = num_beams if num_beams is not None else model.config.num_beams
    temperature = temperature if temperature is not None else model.config.temperature
    top_k = top_k if top_k is not None else model.config.top_k
    top_p = top_p if top_p is not None else model.config.top_p
    repetition_penalty = repetition_penalty if repetition_penalty is not None else model.config.repetition_penalty
    bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
    length_penalty = length_penalty if length_penalty is not None else model.config.length_penalty
    no_repeat_ngram_size = (
        no_repeat_ngram_size if no_repeat_ngram_size is not None else model.config.no_repeat_ngram_size
    )
    bad_words_ids = bad_words_ids if bad_words_ids is not None else model.config.bad_words_ids
    num_return_sequences = (
        num_return_sequences if num_return_sequences is not None else model.config.num_return_sequences
    )
    decoder_start_token_id = (
        decoder_start_token_id if decoder_start_token_id is not None else model.config.decoder_start_token_id
    )

    if input_ids is not None:
        batch_size = input_ids.shape[0]  # overriden by the input batch_size
    else:
        batch_size = 1

    assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
    assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
    assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
    assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
    assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
    assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
    assert temperature > 0, "`temperature` should be strictly positive."
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert input_ids is not None or (
        isinstance(bos_token_id, int) and bos_token_id >= 0
    ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
    assert pad_token_id is None or (
        isinstance(pad_token_id, int) and (pad_token_id >= 0)
    ), "`pad_token_id` should be a positive integer."
    assert (eos_token_id is None) or (
        isinstance(eos_token_id, int) and (eos_token_id >= 0)
    ), "`eos_token_id` should be a positive integer."
    assert length_penalty > 0, "`length_penalty` should be strictly positive."
    assert (
        isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
    ), "`no_repeat_ngram_size` should be a positive integer."
    assert (
        isinstance(num_return_sequences, int) and num_return_sequences > 0
    ), "`num_return_sequences` should be a strictly positive integer."
    assert (
        bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
    ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

    if input_ids is None:
        assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
            "you should either supply a context to complete as `input_ids` input "
            "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
        )
        input_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
    else:
        assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

    # not allow to duplicate outputs when greedy decoding
    if do_sample is False:
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

    # create attention mask if necessary
    # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
    if (attention_mask is None) and (pad_token_id is not None) and (pad_token_id in input_ids):
        attention_mask = input_ids.ne(pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    # set pad_token_id to eos_token_id if not set. Important that this is done after
    # attention_mask is created
    if pad_token_id is None and eos_token_id is not None:
        logger.warning(
            "Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id)
        )
        pad_token_id = eos_token_id

    # vocab size
    if hasattr(model.config, "vocab_size"):
        vocab_size = model.config.vocab_size
    elif (
        model.config.is_encoder_decoder
        and hasattr(model.config, "decoder")
        and hasattr(model.config.decoder, "vocab_size")
    ):
        vocab_size = model.config.decoder.vocab_size
    else:
        raise ValueError("either self.config.vocab_size or self.config.decoder.vocab_size needs to be defined")

    # set effective batch size and effective batch multiplier according to do_sample
    if do_sample:
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
    else:
        effective_batch_size = batch_size
        effective_batch_mult = 1

    if model.config.is_encoder_decoder:
        if decoder_start_token_id is None:
            # see if BOS token can be used for decoder_start_token_id
            if bos_token_id is not None:
                decoder_start_token_id = bos_token_id
            elif (
                hasattr(model.config, "decoder")
                and hasattr(model.config.decoder, "bos_token_id")
                and model.config.decoder.bos_token_id is not None
            ):
                decoder_start_token_id = model.config.decoder.bos_token_id
            else:
                raise ValueError(
                    "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
                )

        assert hasattr(model, "get_encoder"), "{} should have a 'get_encoder' function defined".format(model)
        assert callable(model.get_encoder), "{} should be a method".format(model.get_encoder)

        # get encoder and store encoder outputs
        encoder = model.get_encoder()
        encoder_outputs: ModelOutput = encoder(input_ids, attention_mask=attention_mask, return_dict=True)

    # Expand input ids if num_beams > 1 or num_return_sequences > 1
    if num_return_sequences > 1 or num_beams > 1:
        input_ids_len = input_ids.shape[-1]
        input_ids = input_ids.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)
        attention_mask = attention_mask.unsqueeze(1).expand(
            batch_size, effective_batch_mult * num_beams, input_ids_len
        )

        input_ids = input_ids.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        attention_mask = attention_mask.contiguous().view(
            effective_batch_size * num_beams, input_ids_len
        )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

    if model.config.is_encoder_decoder:
        device = next(model.parameters()).device
        if decoder_input_ids is not None:
            # give initial decoder input ids
            input_ids = decoder_input_ids.repeat(effective_batch_size * num_beams, 1).to(device)
        else:
            # create empty decoder input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=device,
            )
        cur_len = input_ids.shape[-1]

        assert (
            batch_size == encoder_outputs.last_hidden_state.shape[0]
        ), f"expected encoder_outputs.last_hidden_state to have 1st dimension bs={batch_size}, got {encoder_outputs.last_hidden_state.shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, num_beams * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
        )

        # expand encoder_outputs
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_batch_idxs
        )

        # save encoder_outputs in `model_kwargs`
        model_kwargs["encoder_outputs"] = encoder_outputs

    else:
        cur_len = input_ids.shape[-1]

    assert (
        cur_len < max_length
    ), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

    if num_beams > 1:
        output = _generate_beam_search(
            model,
            input_ids,
            cur_len=cur_len,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            early_stopping=early_stopping,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            batch_size=effective_batch_size,
            num_return_sequences=num_return_sequences,
            length_penalty=length_penalty,
            num_beams=num_beams,
            vocab_size=vocab_size,
            attention_mask=attention_mask,
            use_cache=use_cache,
            model_kwargs=model_kwargs,
            one_hot_bows_vectors=one_hot_bows_vectors
        )
    else:
        output = _generate_no_beam_search(
            model,
            input_ids,
            cur_len=cur_len,
            max_length=max_length,
            min_length=min_length,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            batch_size=effective_batch_size,
            attention_mask=attention_mask,
            use_cache=use_cache,
            model_kwargs=model_kwargs,
        )

    return output

def _generate_no_beam_search(
    model,
    input_ids,
    cur_len,
    max_length,
    min_length,
    do_sample,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    bad_words_ids,
    pad_token_id,
    eos_token_id,
    batch_size,
    attention_mask,
    use_cache,
    model_kwargs,
):
    """Generate sequences for each example without beam search (num_beams == 1).
    All returned sequence are generated independantly.
    """
    print("no beam search")
    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    sent_lengths = input_ids.new(batch_size).fill_(max_length)

    past = None
    while cur_len < max_length:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
        )

        outputs = model(**model_inputs, return_dict=True)
        next_token_logits = outputs.logits[:, -1, :]

        scores = model.postprocess_next_token_scores(
            scores=next_token_logits,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
        )

        # if model has past, then set the past variable to speed up decoding
        if "past_key_values" in outputs:
            past = outputs.past_key_values
        elif "mems" in outputs:
            past = outputs.mems

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature
            # Top-p/top-k filtering
            next_token_logscores = top_k_top_p_filtering(scores, top_k=top_k, top_p=top_p)
            # Sample
            probs = F.softmax(next_token_logscores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (1 - unfinished_sents)
        else:
            tokens_to_add = next_token

        # add token and increase length by one
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        cur_len = cur_len + 1

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(eos_in_sents.long()).bool()
            sent_lengths.masked_fill_(is_sents_unfinished_and_token_to_add_is_eos, cur_len)
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # extend attention_mask for new generated input if only decoder
        if model.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    return input_ids

def _generate_beam_search(
    model,
    input_ids,
    cur_len,
    max_length,
    min_length,
    do_sample,
    early_stopping,
    temperature,
    top_k,
    top_p,
    repetition_penalty,
    no_repeat_ngram_size,
    bad_words_ids,
    pad_token_id,
    eos_token_id,
    batch_size,
    num_return_sequences,
    length_penalty,
    num_beams,
    vocab_size,
    attention_mask,
    use_cache,
    model_kwargs,
    perturb=True,
    gamma=1.5,
    gm_scale=0.9,
    kl_scale=0.01,
    current_stepsize=0.01,
    one_hot_bows_vectors=None
):
    progress = tqdm()
    progress.set_description("Generating")
    
    """Generate sequences for each example with beam search."""

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]

    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)

    # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
    if do_sample is False:
        beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # cache compute states
    past = None

    # done sentences
    done = [False for _ in range(batch_size)]

    output_so_far = None
    grad_norms = None
    last = None
    last = torch.LongTensor([50258])
    unpert_discrim_loss = 0
    loss_in_time = []
    first=True
    pert_probs = None
    

    while cur_len < max_length:

        #print(input_ids)
        
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
        )
        
        outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)  # (batch_size * num_beams, cur_len, vocab_size)

        # if model has past, then set the past variable to speed up decoding
        if "past_key_values" in outputs:
            past = outputs.past_key_values
        elif "mems" in outputs:
            past = outputs.mems

        ### PT

        unpert_logits, unpert_past, unpert_all_hidden = (outputs.logits, outputs.past_key_values, outputs.hidden_states) #pt
        unpert_last_hidden = unpert_all_hidden[-1] #pt

        accumulated_hidden = unpert_last_hidden[:, :-1, :]
        accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

        if not first and perturb:
          last = input_ids
          pert_past, _, grad_norms, loss_this_iter = perturb_past(
                      past,
                      model,
                      last,
                      unpert_past=unpert_past,
                      unpert_logits=unpert_logits,
                      accumulated_hidden=accumulated_hidden,
                      grad_norms=grad_norms,
                      stepsize=current_stepsize,
                      one_hot_bows_vectors=one_hot_bows_vectors,
                      classifier=None,
                      class_label=None,
                      loss_type=PPLM_BOW,
                      num_iterations=3,
                      horizon_length=1,
                      window_length=0,
                      decay=False,
                      gamma=1.5,
                      kl_scale=0.01,
                      device=next(model.parameters()).device,
                      verbosity_level=REGULAR,
                      output_so_far=output_so_far ####additional
                  )
          loss_in_time.append(loss_this_iter)

          model_inputs = model.prepare_inputs_for_generation(
            input_ids, past=pert_past, attention_mask=attention_mask, use_cache=use_cache, **model_kwargs
          )
          
          pert_logits, past_p, pert_all_hidden = model(**model_inputs, return_dict=False, output_hidden_states=True)

          #pert_logits, past_p, pert_all_hidden = model(last, past_key_values=pert_past, return_dict=False, output_hidden_states=True, use_cache=True, pertPast=True)

          pert_next_token_logits = pert_logits[:, -1, :]
          pert_scores = F.log_softmax(pert_next_token_logits, dim=-1)
          pert_scores = model.postprocess_next_token_scores(
            scores=pert_scores,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=num_beams,
          )



          pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
          pert_probs = F.softmax(pert_logits, dim=-1)
          #_, last = torch.topk(pert_probs, k=2, dim=-1)
          unpert_discrim_loss = 0
          #print(last)
          #print(tokenizerGPT.decode(last))
        

        ### PT

        next_token_logits = outputs.logits[:, -1, :]  # (batch_size * num_beams, vocab_size)

        
        #print(past[0].shape)
        if model.config.is_encoder_decoder and do_sample is False:
            # TODO (PVP) still a bit hacky here - there might be a better solution
            next_token_logits = model.adjust_logits_during_generation(
                next_token_logits, cur_len=cur_len, max_length=max_length
            )

        scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)

        scores = model.postprocess_next_token_scores(
            scores=scores,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=num_beams,
        )

        assert scores.shape == (batch_size * num_beams, vocab_size), "Shapes of scores: {} != {}".format(
            scores.shape, (batch_size * num_beams, vocab_size)
        )

        if do_sample:
            _scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            # Temperature
            if temperature != 1.0:
                _scores = _scores / temperature
            # Top-p/top-k filtering
            _scores = top_k_top_p_filtering(
                _scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
            )  # (batch_size * num_beams, vocab_size)
            # re-organize to group the beam together to sample from all beam_idxs
            _scores = _scores.contiguous().view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
            probs = F.softmax(_scores, dim=-1)

            if perturb and not first:
              temp_scores = scores + beam_scores[:, None].expand_as(scores)
              _pert_scores = pert_scores + beam_scores[:, None].expand_as(pert_scores)
              #new_scores = ((_pert_scores ** gm_scale) * (temp_scores ** (1 - gm_scale)))  # + SMALL_CONST
              new_scores = -1 * ((torch.sign(_pert_scores) * torch.pow(torch.abs(_pert_scores), gm_scale)) * (torch.sign(temp_scores) * torch.pow(torch.abs(temp_scores), (1-gm_scale))))

              temp_var = temp_scores
              #print(new_scores)
              if temperature != 1.0:
                new_scores = new_scores / temperature
              # Top-p/top-k filtering
              new_scores = top_k_top_p_filtering(
                  new_scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
              )  # (batch_size * num_beams, vocab_size)
              # re-organize to group the beam together to sample from all beam_idxs
              new_scores = new_scores.contiguous().view(
                  batch_size, num_beams * vocab_size
              )  # (batch_size, num_beams * vocab_size)
              #print(new_scores)
              # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
              pert_probs = F.softmax(new_scores, dim=-1)
              
              pert_next_tokens = torch.multinomial(pert_probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
              # Compute next scores
              pert_next_scores = torch.gather(new_scores, -1, pert_next_tokens)  # (batch_size, num_beams * 2)
              # sort the sampled vector to make sure that the first num_beams samples are the best
              pert_next_scores, pert_next_scores_indices = torch.sort(pert_next_scores, descending=True, dim=1)
              pert_next_tokens = torch.gather(pert_next_tokens, -1, pert_next_scores_indices)  # (batch_size, num_beams * 2)

              #print("pert_")
              #print(vae.tgt_tokenizers.decode(pert_next_tokens))


              #pert_probs = ((pert_probs ** gm_scale) * (probs ** (1 - gm_scale)))  # + SMALL_CONST
              #pert_probs = top_k_filter(pert_probs, k=top_k, probs=True)  # + SMALL_CONST

              #if torch.sum(pert_probs) <= 1:
              #  pert_probs = pert_probs / torch.sum(pert_probs)

              #last = torch.multinomial(pert_probs, num_samples=1)
              #print("perturbed add")
              #print(last)
              #print(tokenizerGPT.decode(last))

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
            # Compute next scores
            next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
            # sort the sampled vector to make sure that the first num_beams samples are the best
            next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            #print(vae.tgt_tokenizers.decode(next_tokens))


            ## OVERWRITE
            if not first and perturb:
              next_tokens = pert_next_tokens
        else:
            next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)


        assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)

        # next batch beam content
        next_batch_beam = []

        first=False #pt
        progress.update()
        

        # for each sentence
        for batch_idx in range(batch_size):

            # if we are done with this sentence, add a pad token
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content, this will get added to next_batch_beam
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (token_id.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        input_ids[effective_beam_id].clone(),
                        beam_token_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # once the beam for next step is full, don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break

            # Check if we are done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

            #print(next_batch_beam)

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch and update current length
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
        cur_len = cur_len + 1

        # re-order internal states
        if past is not None:
            past = model._reorder_cache(past, beam_idx)

        # extend attention_mask for new generated input if only decoder
        if model.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # test that beam scores match previously calculated scores if not eos and batch_idx not done
        if eos_token_id is not None and all(
            (token_id % vocab_size).item() != eos_token_id for token_id in next_tokens[batch_idx]
        ):
            assert torch.all(
                next_scores[batch_idx, :num_beams] == beam_scores.view(batch_size, num_beams)[batch_idx]
            ), "If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}".format(
                next_scores[:, :num_beams][batch_idx],
                beam_scores.view(batch_size, num_beams)[batch_idx],
            )

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
    output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences

    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    best = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    # prepare for adding eos
    sent_max_len = min(sent_lengths.max().item() + 1, max_length)
    decoded = input_ids.new(output_batch_size, sent_max_len)
    # shorter batches are padded if needed
    if sent_lengths.min().item() != sent_lengths.max().item():
        assert pad_token_id is not None, "`pad_token_id` has to be defined"
        decoded.fill_(pad_token_id)

    # fill with hypotheses and eos_token_id if the latter fits in
    for i, hypo in enumerate(best):
        decoded[i, : sent_lengths[i]] = hypo
        if sent_lengths[i] < max_length:
            decoded[i, sent_lengths[i]] = eos_token_id

    progress.close()
    return decoded



def build_bows_one_hot_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size+3).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors

def extractVector(rev, tokenizer, most_common_cnt = 150, device="cuda"):
  stop_words = set(stopwords.words('english'))
  allreviews = nltk.word_tokenize(" ".join(rev))
  allreviews = [w for w in allreviews if not w.lower() in stop_words and w.isalpha()]

  cnt = Counter()
  for word in allreviews:
    cnt[word] += 1
  mc = cnt.most_common(most_common_cnt)

  bow_indices = [[tokenizer.encode(word[0].strip(), add_prefix_space=True, add_special_tokens=False) for word in mc]]
  one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer, device)
  return one_hot_bows_vectors


def evaluate(rev, summ, vae):
    reviews: List[str] = rev
    z_raw: torch.Tensor = vae.encode(reviews)
    idxes: List[List[int]] = util.powerset(len(reviews))
    zs: torch.Tensor = torch.stack([z_raw[idx].mean(dim=0) for idx in idxes]) # [2^num_reviews - 1 * latent_size]

    outputs: List[str] = vae.generate(zs, bad_words=util.BAD_WORDS)  # First-person pronoun blocking
    best: str = max(outputs, key=lambda x: util.input_output_overlap(inputs=reviews, output=x))
    reference: List[List[str]] = summ
    evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True,
                            stemming=True, ensure_compatibility=True)
    scores = evaluator.get_scores(best, reference)

    print("Trying Options: " + str(len(zs)))
    outputs_all = []
    outputs = []
    for z in zs:
      text = vae.generate(z, bad_words=util.BAD_WORDS)
      outputs_all.append(text)
      best: str = max(text, key=lambda x: util.input_output_overlap(inputs=reviews, output=x))
      outputs.append(best)
    best: str = max(outputs, key=lambda x: util.input_output_overlap(inputs=reviews, output=x))

    reference: List[List[str]] = summ
    evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True,
                            stemming=True, ensure_compatibility=True)
    scores = evaluator.get_scores(best, reference)

    print(scores)
    return (scores, best, outputs)

def evaluatePPLM(rev, summ, tokenizer, vae):
    one_hot_bows_vectors = extractVector(rev, tokenizer)
    reviews: List[str] = rev
    z_raw: torch.Tensor = vae.encode(reviews)
    idxes: List[List[int]] = util.powerset(len(reviews))
    zs: torch.Tensor = torch.stack([z_raw[idx].mean(dim=0) for idx in idxes]) # [2^num_reviews - 1 * latent_size]
    print("Trying Options: " + str(len(zs)))
    outputs_all = []
    outputs = []
    for z in zs:
      z = z.view((1,512))
      z.requires_grad=True
      bz, _ = z.size()
      input_ids = z.new_full((1, 1), dtype=torch.long, fill_value=vae.model.bos_id)
      generated = generateScratch(
        vae.model.decoder,
        input_ids,
        max_length=256,
        min_length=16,
        num_beams=4,
        num_return_sequences=3,
        bad_words=util.BAD_WORDS,
        bos_token_id=vae.model.bos_id,
        pad_token_id=vae.model.pad_id,
        eos_token_id=vae.model.eos_id,
        one_hot_bows_vectors=one_hot_bows_vectors,
        past_key_values=(z,),
        no_repeat_ngram_size=2,
        latent_as_gpt_memory=True,
        do_sample=True,
        latent_as_gpt_emb=True,
        use_cache=True).tolist()
      text = vae.tgt_tokenizers.decode(generated)
      outputs_all.append(text)
      best: str = max(text, key=lambda x: util.input_output_overlap(inputs=reviews, output=x))
      outputs.append(best)
      
    best: str = max(outputs, key=lambda x: util.input_output_overlap(inputs=reviews, output=x))
    reference: List[List[str]] = summ
    evaluator = rouge.Rouge(metrics=["rouge-n", "rouge-l"], max_n=2, limit_length=False, apply_avg=True,
                            stemming=True, ensure_compatibility=True)
    scores = evaluator.get_scores(best, reference)
    print(scores)
    return (scores, best, outputs)


def main():
    print("Evaluating...")
    
    sp = {'pad_token': '<PAD>', 'bos_token': '<BOS>', 'eos_token': '<EOS>'}
    tokenizerGPT = GPT2Tokenizer.from_pretrained("data/gpt2")
    tokenizerGPT.add_special_tokens(sp)

    model_name: str = "megagonlabs/optimus-amzn" #"./" #"megagonlabs/optimus-amzn"  # or "megagonlabs/bimeanvae-amzn", "megagonlabs/optimus-yelp", "megagonlabs/optimus-amzn"
    vae = VAE(model_name)

    with open('dev.json') as json_file:
        data_arr = json.load(json_file)

    output_eval = []

    for data in data_arr:
        #out = evaluate(data["reviews"], data["summary"], vae)
        out = evaluatePPLM(data["reviews"], data["summary"], tokenizerGPT, vae)
        print(out)
        output_eval.append(out)

    with open('eval_results.json', 'w') as outfile:
        json.dump(output_eval, outfile)


if __name__ == '__main__':
    main()