"""
Modified from https://github.com/jidasheng/bi-lstm-crf
"""

import torch
import torch.nn as nn


def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()


IMPOSSIBLE = -1e4


class CRF(nn.Module):
    """General CRF module.
    The CRF module contain a inner Linear Layer which transform the input from features space to tag space.
    """

    def __init__(self, in_features, num_tags):
        """
        Initialization

        Parameters
        ----------
        in_features: number of features for the input
        num_tags: number of tags. DO NOT include START, STOP tags, they are included internal.
        """
        super().__init__()

        self.num_tags = num_tags + 2
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(in_features, self.num_tags)

        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(self, features, masks):
        """decode tags
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        features = self.fc(features)
        return self._viterbi_decode(features, masks[:, :features.size(1)].float())

    def loss(self, features, ys, masks):
        """negative log likelihood loss
        B: batch size, length: sequence length, D: dimension
        :param features: [B, length, D]
        :param ys: tags, [B, length]
        :param masks: masks for padding, [B, length]
        :return: loss
        """
        features = self.fc(features)

        length = features.size(1)
        masks_ = masks[:, :length].float()

        forward_score = self.__forward_algorithm(features, masks_)
        gold_score = self._score_sentence(features, ys[:, :length].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def _score_sentence(self, features, tags, masks):
        """Gives the score of a provided tag sequence

        Parameters
        ----------
        features: [s_batch, l_seq, n_tags]
        tags: [s_batch, l_seq]
        masks: [s_batch, l_seq]

        Returns
        -------
        [s_batch] score in the log space
        """
        s_batch, l_seq, n_tags = features.shape

        # emission score
        emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full((s_batch, 1), self.start_idx, dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [s_batch, l_seq+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [s_batch]
        last_score = self.transitions[self.stop_idx, last_tag]

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def _viterbi_decode(self, features, masks):
        """decode to tags using viterbi algorithm
        :param features: [s_batch, l_seq, n_tags], batch of unary scores
        :param masks: [s_batch, l_seq] masks
        :return: (best_score, best_paths)
            best_score: [s_batch]
            best_paths: [s_batch, l_seq]
        """
        s_batch, l_seq, n_tags = features.shape

        bps = torch.zeros(s_batch, l_seq, n_tags, dtype=torch.long, device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((s_batch, n_tags), IMPOSSIBLE, device=features.device)  # [s_batch, n_tags]
        max_score[:, self.start_idx] = 0

        for t in range(l_seq):
            mask_t = masks[:, t].unsqueeze(1)  # [s_batch, 1]
            emit_score_t = features[:, t]  # [s_batch, n_tags]

            # [s_batch, 1, n_tags] + [n_tags, n_tags]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [s_batch, n_tags, n_tags]
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(s_batch):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def __forward_algorithm(self, features, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])
        :param features: features. [s_batch, l_seq, n_tags]
        :param masks: [s_batch, l_seq] masks
        :return:    [s_batch], score in the log space
        """
        s_batch, l_seq, n_tags = features.shape

        scores = torch.full((s_batch, n_tags), IMPOSSIBLE, device=features.device)  # [s_batch, n_tags]
        scores[:, self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, n_tags, n_tags]

        # Iterate through the sentence
        for t in range(l_seq):
            emit_score_t = features[:, t].unsqueeze(2)  # [s_batch, n_tags, 1]
            # [s_batch, 1, n_tags] + [1, n_tags, n_tags] + [s_batch, n_tags, 1] => [s_batch, n_tags, n_tags]
            score_t = scores.unsqueeze(1) + trans + emit_score_t
            score_t = log_sum_exp(score_t)  # [s_batch, n_tags]

            mask_t = masks[:, t].unsqueeze(1)  # [s_batch, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores
