from dataclasses import dataclass

import selfies as sf
import torch


@dataclass
class Selfies:
    selfies: str

    def to_seq(self, vocabulary: "SelfiesVocabulary") -> torch.Tensor:
        return torch.tensor(vocabulary.encode(self.selfies))

    @classmethod
    def from_seq(self, seq: torch.Tensor, vocabulary: "SelfiesVocabulary"):
        return Selfies(vocabulary.decode(seq.tolist()))


class SelfiesVocabulary:
    pad = " "  # Padding token, though SELFIEs might not strictly need it
    sos = "!"  # Start of sequence
    eos = "?"  # End of sequence
    pad_idx = 0
    sos_idx = 1
    eos_idx = 2

    def __init__(self):
        self.alphabet = [self.pad, self.sos, self.eos]
        self.token_to_index = {token: idx for idx, token in enumerate(self.alphabet)}
        self.index_to_token = {idx: token for idx, token in enumerate(self.alphabet)}

    def update(self, selfies_string):
        alphabet = set(sf.split_selfies(selfies_string))
        new_tokens = sorted(list(alphabet - set(self.alphabet)))
        for token in new_tokens:
            self.alphabet.append(token)
            self.token_to_index[token] = len(self.alphabet) - 1
            self.index_to_token[len(self.alphabet) - 1] = token
        return self.selfies2seq(Selfies(selfies_string))

    def encode(self, selfies_string):
        """Encodes a selfies string into a list of indices."""
        seq = (
            [self.sos_idx]
            + [self.token_to_index[token] for token in sf.split_selfies(selfies_string)]
            + [self.eos_idx]
        )
        return seq

    def decode(self, seq):
        """Decodes a sequence of indices into a selfies string."""
        tokens = [
            self.index_to_token[idx] for idx in seq if idx != self.pad_idx
        ]  # Remove padding
        tokens = [
            token for token in tokens if token not in [self.sos, self.eos]
        ]  # Remove sos and eos
        return "".join(tokens)

    def selfies2seq(self, selfies: Selfies) -> torch.Tensor:
        return selfies.to_seq(self)

    def seq2selfies(self, seq: torch.Tensor) -> Selfies:
        return Selfies.from_seq(seq, self)

    def batch_update(self, selfies_list: list[str]):
        seq_list = []
        out_selfies_list = []
        for each_selfies in selfies_list:
            if each_selfies.endswith("\n"):
                each_selfies = each_selfies.strip()
            seq_list.append(self.update(each_selfies))
            out_selfies_list.append(each_selfies)
        right_padded_batch_seq = torch.nn.utils.rnn.pad_sequence(
            seq_list, batch_first=True, padding_value=self.pad_idx
        )
        return right_padded_batch_seq, out_selfies_list

    def batch_update_from_file(self, file_path, with_selfies=False):
        selfies_list = open(file_path).readlines()
        seq_tensor, selfies_list = self.batch_update(selfies_list)
        if with_selfies:
            return seq_tensor, selfies_list
        return seq_tensor
