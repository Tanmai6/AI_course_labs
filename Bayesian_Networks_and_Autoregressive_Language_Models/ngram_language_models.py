"""First- and second-order autoregressive language models using only Python."""

from collections import Counter, defaultdict
import random

START = "<START>"
END = "<END>"


def tokenize(sentences):
    """Lower-case sentences and add the special boundary tokens."""
    return [[START] + sentence.lower().split() + [END] for sentence in sentences]


class FirstOrderLanguageModel:
    """Estimate P(next | previous) from observed transition counts."""

    def __init__(self, tokenized_sentences):
        self.counts = defaultdict(Counter)
        for sentence in tokenized_sentences:
            for previous, following in zip(sentence[:-1], sentence[1:]):
                self.counts[previous][following] += 1
        self.probabilities = {
            previous: {
                following: count / sum(outcomes.values())
                for following, count in outcomes.items()
            }
            for previous, outcomes in self.counts.items()
        }

    def distribution(self, previous):
        """Return P(X_t | X_{t-1}=previous), or an empty dict if unseen."""
        return dict(self.probabilities.get(previous, {}))

    def predict(self, previous):
        """Return the most probable next token (ties resolved alphabetically)."""
        distribution = self.distribution(previous)
        if not distribution:
            raise ValueError(f"No transitions observed after {previous!r}")
        return max(sorted(distribution), key=distribution.get)

    def generate(self, mode="sampling", max_length=30, rng=None):
        """Generate from START using greedy or sampling-based decoding."""
        if mode not in {"greedy", "sampling"}:
            raise ValueError("mode must be 'greedy' or 'sampling'")
        rng = rng or random
        previous = START
        result = []
        for _ in range(max_length):
            distribution = self.distribution(previous)
            if not distribution:
                raise ValueError(f"No transitions observed after {previous!r}")
            tokens = sorted(distribution)
            if mode == "greedy":
                following = max(tokens, key=distribution.get)
            else:
                following = rng.choices(
                    tokens, weights=[distribution[t] for t in tokens], k=1
                )[0]
            if following == END:
                return " ".join(result)
            result.append(following)
            previous = following
        # Greedy decoding can enter a deterministic cycle in an n-gram model.
        return " ".join(result) + " <TRUNCATED: NO END TOKEN>"


class SecondOrderLanguageModel:
    """Estimate P(next | two_back, previous) from observed token triples."""

    def __init__(self, tokenized_sentences):
        self.counts = defaultdict(Counter)
        for sentence in tokenized_sentences:
            padded = [START] + sentence
            for two_back, previous, following in zip(
                padded[:-2], padded[1:-1], padded[2:]
            ):
                self.counts[(two_back, previous)][following] += 1
        self.probabilities = {
            context: {
                following: count / sum(outcomes.values())
                for following, count in outcomes.items()
            }
            for context, outcomes in self.counts.items()
        }

    def distribution(self, two_back, previous):
        """Return the conditional distribution, or an empty dict if unseen."""
        return dict(self.probabilities.get((two_back, previous), {}))

    def predict(self, two_back, previous):
        distribution = self.distribution(two_back, previous)
        if not distribution:
            raise ValueError(f"No transitions observed after {(two_back, previous)!r}")
        return max(sorted(distribution), key=distribution.get)

    def generate(self, mode="sampling", max_length=30, rng=None):
        if mode not in {"greedy", "sampling"}:
            raise ValueError("mode must be 'greedy' or 'sampling'")
        rng = rng or random
        two_back, previous = START, START
        result = []
        for _ in range(max_length):
            distribution = self.distribution(two_back, previous)
            if not distribution:
                raise ValueError(
                    f"No transitions observed after {(two_back, previous)!r}"
                )
            tokens = sorted(distribution)
            if mode == "greedy":
                following = max(tokens, key=distribution.get)
            else:
                following = rng.choices(
                    tokens, weights=[distribution[t] for t in tokens], k=1
                )[0]
            if following == END:
                return " ".join(result)
            result.append(following)
            two_back, previous = previous, following
        raise RuntimeError("Generation exceeded max_length before reaching <END>")
