"""
tokenizer.py

Small educational BPE tokenizer.

IMPORTANT:
This is NOT the real Llama 4 tokenizer.
It is designed so that we can understand the complete
text -> tokens -> model -> text pipeline.
"""

from collections import Counter


class BPETokenizer:
    def __init__(self, text, num_merges=100):
        self.num_merges = num_merges

        self.special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
        ]

        self.merges = []

        self.token_to_id = {}
        self.id_to_token = {}

        self._train(text)

    # ---------------------------------------------------------
    # Properties
    # ---------------------------------------------------------

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    @property
    def pad_id(self):
        return self.token_to_id["<PAD>"]

    @property
    def unk_id(self):
        return self.token_to_id["<UNK>"]

    @property
    def bos_id(self):
        return self.token_to_id["<BOS>"]

    @property
    def eos_id(self):
        return self.token_to_id["<EOS>"]

    # ---------------------------------------------------------
    # BPE training
    # ---------------------------------------------------------

    def _get_pair_statistics(self, splits):

        counts = Counter()

        for pieces in splits.values():

            for i in range(len(pieces) - 1):

                pair = (
                    pieces[i],
                    pieces[i + 1],
                )

                counts[pair] += 1

        return counts

    def _merge_pair(self, pair, splits):

        new_splits = {}

        for word, pieces in splits.items():

            merged = []

            i = 0

            while i < len(pieces):

                if (
                    i < len(pieces) - 1
                    and pieces[i] == pair[0]
                    and pieces[i + 1] == pair[1]
                ):
                    merged.append(
                        pieces[i] + pieces[i + 1]
                    )

                    i += 2

                else:
                    merged.append(pieces[i])
                    i += 1

            new_splits[word] = merged

        return new_splits

    def _train(self, text):

        words = text.split()

        splits = {}

        for word in words:

            # Character-level starting point.
            pieces = list(word)

            # End-of-word marker.
            pieces.append("</w>")

            splits[word] = pieces

        # Learn BPE merges.
        for _ in range(self.num_merges):

            stats = self._get_pair_statistics(
                splits
            )

            if not stats:
                break

            best_pair = max(
                stats,
                key=stats.get
            )

            self.merges.append(best_pair)

            splits = self._merge_pair(
                best_pair,
                splits
            )

        # Build vocabulary from final pieces.
        vocabulary = set(
            self.special_tokens
        )

        for pieces in splits.values():

            vocabulary.update(pieces)

        self.token_to_id = {
            token: i
            for i, token in enumerate(
                sorted(vocabulary)
            )
        }

        self.id_to_token = {
            i: token
            for token, i in self.token_to_id.items()
        }

    # ---------------------------------------------------------
    # Encode one word
    # ---------------------------------------------------------

    def _encode_word(self, word):

        pieces = list(word)

        pieces.append("</w>")

        # Apply learned merges in order.
        for pair in self.merges:

            new_pieces = []

            i = 0

            while i < len(pieces):

                if (
                    i < len(pieces) - 1
                    and pieces[i] == pair[0]
                    and pieces[i + 1] == pair[1]
                ):

                    new_pieces.append(
                        pieces[i] + pieces[i + 1]
                    )

                    i += 2

                else:

                    new_pieces.append(
                        pieces[i]
                    )

                    i += 1

            pieces = new_pieces

        ids = []

        for piece in pieces:

            ids.append(
                self.token_to_id.get(
                    piece,
                    self.unk_id
                )
            )

        return ids

    # ---------------------------------------------------------
    # Encode text
    # ---------------------------------------------------------

    def encode(
        self,
        text,
        add_bos=True,
        add_eos=False
    ):

        ids = []

        if add_bos:
            ids.append(self.bos_id)

        for word in text.split():

            ids.extend(
                self._encode_word(word)
            )

        if add_eos:
            ids.append(self.eos_id)

        return ids

    # ---------------------------------------------------------
    # Decode
    # ---------------------------------------------------------

    def decode(self, ids):

        output = []

        for token_id in ids:

            token = self.id_to_token[
                int(token_id)
            ]

            # Special tokens
            if token == "<PAD>":
                continue

            if token == "<BOS>":
                continue

            if token == "<EOS>":
                break

            # ----------------------------------------------------
            # End-of-word marker
            #
            # It can appear as:
            #
            # </w>
            #
            # or attached to a token:
            #
            # dog</w>
            # ----------------------------------------------------

            if token.endswith("</w>"):

                token = token[:-4]

                output.append(
                    token + " "
                )

            else:

                output.append(token)

        return "".join(output).strip()



# -------------------------------------------------------------
# Simple test
# -------------------------------------------------------------

if __name__ == "__main__":

    corpus = """
    hello world
    hello dog
    hello cat
    the dog is happy
    the cat is happy
    the dog likes food
    the cat likes food
    """

    tokenizer = BPETokenizer(
        corpus,
        num_merges=50
    )

    print(
        "Vocabulary size:",
        tokenizer.vocab_size
    )

    text = "hello dog"

    ids = tokenizer.encode(text)

    print("Text:", text)
    print("IDs:", ids)

    print(
        "Decoded:",
        tokenizer.decode(ids)
    )
