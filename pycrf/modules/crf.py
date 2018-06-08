# ==============================================================================
#
#  Copyright (c) 2018 AllenAI
#
#  Apache License
#  Version 2.0, January 2004
#  http://www.apache.org/licenses/
#
# ==============================================================================

"""Conditional random field."""

import logging
from typing import List, Tuple, Dict, Optional

import torch

import allennlp.nn.util as util


logger = logging.getLogger(__name__)


def logsumexp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    """
    Compute logsumexp in a numerically stable way.

    This is mathematically equivalent to ``tensor.exp().sum(dim, keep=keepdim).log()``.
    This function is typically used for summing log probabilities.

    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A tensor of arbitrary size.

    dim : int, optional (default = -1)
        The dimension of the tensor to apply the logsumexp to.

    keepdim: bool, optional (default = False)
        Whether to retain a dimension of size one at the dimension we reduce over.

    """
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def viterbi_decode(tag_sequence: torch.Tensor,
                   transition_matrix: torch.Tensor,
                   tag_observations: Optional[List[int]] = None):
    """
    Find most likely sequence of tags.

    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.

    Parameters
    ----------
    tag_sequence : torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.

    transition_matrix : torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.

    tag_observations : Optional[List[int]], optional, (default = None)
        A list of length ``sequence_length`` containing the class ids of observed
        elements in the sequence, with unobserved elements being set to -1. Note that
        it is possible to provide evidence which results in degenerate labellings if
        the sequences of tags you provide as evidence cannot transition between each
        other, or those transitions are extremely unlikely. In this situation we log a
        warning, but the responsibility for providing self-consistent evidence ultimately
        lies with the user.

    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : torch.Tensor
        The score of the viterbi path.

    """
    sequence_length, num_tags = list(tag_sequence.size())
    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise ValueError("Observations were provided, but they were not the same length "
                             "as the sequence. Found sequence of length: {} and evidence: {}"
                             .format(sequence_length, tag_observations))
    else:
        tag_observations = [-1 for _ in range(sequence_length)]

    path_scores = []
    path_indices = []

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.
        path_scores.append(one_hot)
    else:
        path_scores.append(tag_sequence[0, :])

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transition_matrix
        scores, paths = torch.max(summed_potentials, 0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.
        if tag_observations[timestep - 1] != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logger.warning("The pairwise potential between tags you have passed as "
                               "observations is extremely unlikely. Double check your evidence "
                               "or transition potentials!")
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.
            path_scores.append(one_hot)
        else:
            path_scores.append(tag_sequence[timestep, :] + scores.squeeze())
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    viterbi_score, best_path = torch.max(path_scores[-1], 0)
    viterbi_path = [int(best_path.numpy())]
    for backward_timestep in reversed(path_indices):
        viterbi_path.append(int(backward_timestep[viterbi_path[-1]]))
    # Reverse the backward path.
    viterbi_path.reverse()

    return viterbi_path, viterbi_score


def allowed_transitions(constraint_type: str,
                        tokens: Dict[int, str]) -> List[Tuple[int, int]]:
    # pylint: disable=too-many-branches
    """
    Given tokens and a constraint type, returns the allowed transitions.

    It will additionally include transitions for the start and end states,
    which are used by the conditional random field.

    Parameters
    ----------
    constraint_type : str, required
        Indicates which constraint to apply. Current choices are "BIO" and "BIOUL".

    tokens : Dict[int, str], required
        A mapping {token_id -> token}. Most commonly this would be the value from
        Vocabulary.get_index_to_token_vocabulary()

    Returns
    -------
    List[Tuple[int, int]]
        The allowed transitions (from_token_id, to_token_id).

    """
    n_tags = len(tokens)
    start_tag = n_tags
    end_tag = n_tags + 1

    allowed = []
    if constraint_type == "BIOUL":
        for i, (from_bioul, *from_entity) in tokens.items():
            for j, (to_bioul, *to_entity) in tokens.items():
                is_allowed = any([
                    # O can transition to O, B-* or U-*
                    # L-x can transition to O, B-*, or U-*
                    # U-x can transition to O, B-*, or U-*
                    from_bioul in ('O', 'L', 'U') and to_bioul in ('O', 'B', 'U'),
                    # B-x can only transition to I-x or L-x
                    # I-x can only transition to I-x or L-x
                    from_bioul in ('B', 'I') and to_bioul in ('I', 'L') and from_entity == to_entity
                ])

                if is_allowed:
                    allowed.append((i, j))

        # start transitions
        for i, (to_bioul, *to_entity) in tokens.items():
            if to_bioul in ('O', 'B', 'U'):
                allowed.append((start_tag, i))

        # end transitions
        for i, (from_bioul, *from_entity) in tokens.items():
            if from_bioul in ('O', 'L', 'U'):
                allowed.append((i, end_tag))

    elif constraint_type == "BIO":
        for i, (from_bio, *from_entity) in tokens.items():
            for j, (to_bio, *to_entity) in tokens.items():

                is_allowed = any([
                    # Can always transition to O or B-x
                    to_bio in ('O', 'B'),
                    # Can only transition to I-x from B-x or I-x
                    to_bio == 'I' and from_bio in ('B', 'I') and from_entity == to_entity
                ])

                if is_allowed:
                    allowed.append((i, j))

        # start transitions
        for i, (to_bio, *to_entity) in tokens.items():
            if to_bio in ('O', 'B'):
                allowed.append((start_tag, i))

        # end transitions
        for i, (from_bio, *from_entity) in tokens.items():
            if from_bio in ('O', 'B', 'I'):
                allowed.append((i, end_tag))

    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")

    return allowed


class ConditionalRandomField(torch.nn.Module):
    """
    Linear Chain Conditional Random Field.

    This module uses the "forward-backward" algorithm to compute
    the log-likelihood of its inputs assuming a conditional random field model.

    See, e.g. http://www.cs.columbia.edu/~mcollins/fb.pdf

    Parameters
    ----------
    num_tags : int, required
        The number of tags.

    constraints : List[Tuple[int, int]], optional (default: None)
        An optional list of allowed transitions (from_tag_id, to_tag_id).
        These are applied to ``viterbi_tags()`` but do not affect ``forward()``.
        These should be derived from `allowed_transitions` so that the
        start and end transitions are handled correctly for your tag type.

    include_start_end_transitions : bool, optional (default: True)
        Whether to include the start and end transition parameters.

    """

    def __init__(self,
                 num_tags: int,
                 constraints: List[Tuple[int, int]] = None,
                 include_start_end_transitions: bool = True) -> None:
        super().__init__()
        self.num_tags = num_tags

        # transitions[i, j] is the logit for transitioning from state i to state j.
        self.transitions = torch.nn.Parameter(torch.Tensor(num_tags, num_tags))

        # _constraint_mask indicates valid transitions (based on supplied constraints).
        # Include special start of sequence (num_tags + 1) and end of sequence tags (num_tags + 2)
        if constraints is None:
            # All transitions are valid.
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(1.)
        else:
            constraint_mask = torch.Tensor(num_tags + 2, num_tags + 2).fill_(0.)
            for i, j in constraints:
                constraint_mask[i, j] = 1.

        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)

        # Also need logits for transitioning from "start" state and to "end" state.
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.Tensor(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.Tensor(num_tags))

        self.reset_parameters()

    def reset_parameters(self):
        """Randomly reset params."""
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self,
                          logits: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-likelihood.

        Computes the (batch_size,) denominator term for the log-likelihood,
        which is the sum of the likelihoods across all possible state sequences.
        """
        batch_size, sequence_length, num_tags = logits.size()

        # Transpose batch size and sequence dimensions
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()

        # Initial alpha is the (batch_size, num_tags) tensor of likelihoods combining the
        # transitions to the initial states and the logits for the first timestep.
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]

        # For each i we compute logits for the transitions from timestep i-1 to timestep i.
        # We do so in a (batch_size, num_tags, num_tags) tensor where the axes are
        # (instance, current_tag, next_tag)
        for i in range(1, sequence_length):
            # The emit scores are for time i ("next_tag") so we broadcast along
            # the current_tag axis.
            emit_scores = logits[i].view(batch_size, 1, num_tags)
            # Transition scores are (current_tag, next_tag) so we broadcast along
            # the instance axis.
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            # Alpha is for the current_tag, so we broadcast along the next_tag axis.
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)

            # Add all the scores together and logexp over the current_tag axis
            inner = broadcast_alpha + emit_scores + transition_scores

            # In valid positions (mask == 1) we want to take the logsumexp over
            # the current_tag dimension of ``inner``. Otherwise (mask == 0) we
            # want to retain the previous alpha.
            alpha = (logsumexp(inner, 1) * mask[i].view(batch_size, 1) +
                     alpha * (1 - mask[i]).view(batch_size, 1))

        # Every sequence needs to end with a transition to the stop_tag.
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha

        # Finally we log_sum_exp along the num_tags dim, result is (batch_size,)
        return logsumexp(stops)

    def _joint_likelihood(self,
                          logits: torch.Tensor,
                          tags: torch.Tensor,
                          mask: torch.LongTensor) -> torch.Tensor:
        """
        Compute the numerator term for the log-likelihood.

        This is just score(inputs, tags).
        """
        batch_size, sequence_length, num_tags = logits.data.shape

        # Transpose batch size and sequence dimensions:
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        tags = tags.transpose(0, 1).contiguous()

        # Start with the transition scores from start_tag to the first tag in each input
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags[0])
        else:
            score = 0.0

        # Broadcast the transition scores to one per batch element
        broadcast_transitions = self.transitions.view(1, num_tags, num_tags)\
            .expand(batch_size, num_tags, num_tags)

        # Add up the scores for the observed transitions and all the inputs but the last
        for i in range(sequence_length - 1):
            # Each is shape (batch_size,)
            current_tag, next_tag = tags[i], tags[i+1]

            # The scores for transitioning from current_tag to next_tag
            transition_score = (
                broadcast_transitions
                # Choose the current_tag-th row for each input
                .gather(1, current_tag.view(batch_size, 1, 1).expand(batch_size, 1, num_tags))
                # Squeeze down to (batch_size, num_tags)
                .squeeze(1)
                # Then choose the next_tag-th column for each of those
                .gather(1, next_tag.view(batch_size, 1))
                # And squeeze down to (batch_size,)
                .squeeze(1)
            )

            # The score for using current_tag
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)

            # Include transition score if next element is unmasked,
            # input_score if this element is unmasked.
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]

        # Transition from last state to "stop" state. To start with, we need to find the last tag
        # for each instance.
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags.gather(
            0, last_tag_index.view(1, batch_size).expand(sequence_length, batch_size))

        # Is (sequence_length, batch_size), but all the columns are the same, so take the first.
        last_tags = last_tags[0]

        # Compute score of transitioning to `stop_tag` from each "last tag".
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0

        # Add the last input if it's not masked.
        last_inputs = logits[-1]                                         # (batch_size, num_tags)
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1))  # (batch_size, 1)
        last_input_score = last_input_score.squeeze()                    # (batch_size,)

        score = score + last_transition_score + last_input_score * mask[-1]

        return score

    def forward(self,
                inputs: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.ByteTensor = None) -> torch.Tensor:
        # pylint: disable=arguments-differ
        """Compute the log likelihood."""
        if mask is None:
            mask = torch.ones(*tags.size(), dtype=torch.long)

        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)

        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(self,
                     logits: torch.Tensor,
                     mask: torch.Tensor) -> List[Tuple[List[int], float]]:
        """
        Use viterbi algorithm to find most likely tags for the given inputs.

        If constraints are applied, disallows all other transitions.
        """
        _, max_seq_length, num_tags = logits.size()

        # Get the tensors out of the variables
        logits, mask = logits.data, mask.data

        # Augment transitions matrix with start and end transitions
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.Tensor(num_tags + 2, num_tags + 2).fill_(-10000.)

        # Apply transition constraints
        constrained_transitions = (
            self.transitions * self._constraint_mask[:num_tags, :num_tags] +
            -10000.0 * (1 - self._constraint_mask[:num_tags, :num_tags])
        )
        transitions[:num_tags, :num_tags] = constrained_transitions.data

        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = (
                self.start_transitions.detach() * self._constraint_mask[start_tag, :num_tags].data +
                -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            )
            transitions[:num_tags, end_tag] = (
                self.end_transitions.detach() * self._constraint_mask[:num_tags, end_tag].data +
                -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())
            )
        else:
            transitions[start_tag, :num_tags] = \
                -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags].detach())
            transitions[:num_tags, end_tag] = \
                -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag].detach())

        best_paths = []
        # Pad the max sequence length by 2 to account for start_tag + end_tag.
        tag_sequence = torch.Tensor(max_seq_length + 2, num_tags + 2)

        for prediction, prediction_mask in zip(logits, mask):
            sequence_length = torch.sum(prediction_mask)

            # Start with everything totally unlikely
            tag_sequence.fill_(-10000.)
            # At timestep 0 we must have the START_TAG
            tag_sequence[0, start_tag] = 0.
            # At steps 1, ..., sequence_length we just use the incoming prediction
            tag_sequence[1:(sequence_length + 1), :num_tags] = prediction[:sequence_length]
            # And at the last timestep we must have the END_TAG
            tag_sequence[sequence_length + 1, end_tag] = 0.

            # We pass the tags and the transitions to ``viterbi_decode``.
            viterbi_path, viterbi_score = \
                util.viterbi_decode(tag_sequence[:(sequence_length + 2)], transitions)
            # Get rid of START and END sentinels and append.
            viterbi_path = viterbi_path[1:-1]
            best_paths.append((viterbi_path, viterbi_score.item()))

        return best_paths
