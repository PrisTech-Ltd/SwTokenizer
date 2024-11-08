import torch
from transformers import AutoTokenizer
import pandas as pd
from typing import List, Dict, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from datasets import load_dataset
import random


@dataclass
class TokenizationStats:
    """Statistics for comparing tokenizer performance"""
    avg_sequence_length: float
    max_sequence_length: int
    vocab_size: int
    unique_tokens_used: int
    unknown_token_percentage: float
    compression_ratio: float
    token_reduction_percentage: float
    token_efficiency_score: float
    top_new_tokens: List[Tuple[str, int]]


class TokenizerVocabularyExtender:
    def __init__(
            self,
            model_name: str = "google/gemma-2b",
            device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model_name = model_name
        self.base_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = device

    def load_morpheme_vocab(self, csv_path: str) -> Dict[str, int]:
        """Load morpheme vocabulary from CSV file."""
        df = pd.read_csv(csv_path)
        return dict(zip(df['token'], df['frequency']))

    def calculate_compression_metrics(
            self,
            base_tokenizer: AutoTokenizer,
            extended_tokenizer: AutoTokenizer,
            texts: List[str]
    ) -> Tuple[float, float]:
        """Calculate compression metrics comparing base and extended tokenizers."""
        base_lengths = []
        extended_lengths = []

        for text in tqdm(texts, desc="Calculating compression metrics"):
            base_tokens = base_tokenizer.tokenize(text)
            extended_tokens = extended_tokenizer.tokenize(text)

            base_lengths.append(len(base_tokens))
            extended_lengths.append(len(extended_tokens))

        base_lengths = np.array(base_lengths)
        extended_lengths = np.array(extended_lengths)

        ratios = extended_lengths / base_lengths
        avg_compression_ratio = np.mean(ratios)

        total_base_tokens = np.sum(base_lengths)
        total_extended_tokens = np.sum(extended_lengths)
        token_reduction = ((total_base_tokens - total_extended_tokens) / total_base_tokens) * 100

        return avg_compression_ratio, token_reduction

    def calculate_token_efficiency(
            self,
            extended_tokenizer: AutoTokenizer,
            new_tokens: List[str],
            texts: List[str],
            sample_size: int = 10000
    ) -> Tuple[float, List[Tuple[str, int]]]:
        """Calculate how effectively the new tokens are being utilized."""
        if len(texts) > sample_size:
            texts = random.sample(texts, sample_size)

        total_tokens = 0
        new_token_counts = Counter()

        for text in tqdm(texts, desc="Analyzing token efficiency"):
            tokens = extended_tokenizer.tokenize(text)
            total_tokens += len(tokens)

            for token in tokens:
                if token in new_tokens:
                    new_token_counts[token] += 1

        new_token_usage = sum(new_token_counts.values())
        usage_percentage = new_token_usage / total_tokens if total_tokens > 0 else 0

        if new_token_counts:
            probs = np.array(list(new_token_counts.values())) / new_token_usage
            entropy = -np.sum(probs * np.log2(probs))
            max_entropy = np.log2(len(new_tokens))
            distribution_score = entropy / max_entropy if max_entropy > 0 else 0
        else:
            distribution_score = 0

        efficiency_score = (0.7 * usage_percentage) + (0.3 * distribution_score)
        top_tokens = new_token_counts.most_common(20)

        return efficiency_score, top_tokens

    def analyze_vocabulary_overlap(self, morpheme_vocab: Dict[str, int]) -> Dict:
        """Analyze overlap between base tokenizer and morpheme vocabulary."""
        base_vocab = set(self.base_tokenizer.get_vocab().keys())
        morpheme_tokens = set(morpheme_vocab.keys())

        overlap = base_vocab.intersection(morpheme_tokens)
        new_tokens = morpheme_tokens - base_vocab

        return {
            'total_base_tokens': len(base_vocab),
            'total_morpheme_tokens': len(morpheme_tokens),
            'overlap_tokens': len(overlap),
            'new_tokens': len(new_tokens),
            'overlap_percentage': len(overlap) / len(morpheme_tokens) * 100,
            'new_token_list': sorted(list(new_tokens))
        }

    def extend_tokenizer(
            self,
            morpheme_vocab: Dict[str, int],
            min_frequency: int = 5
    ) -> Tuple[AutoTokenizer, List[str]]:
        """Extend the base tokenizer with new morpheme tokens."""
        filtered_morphemes = {
            token: freq for token, freq in morpheme_vocab.items()
            if freq >= min_frequency
        }

        base_vocab = set(self.base_tokenizer.get_vocab().keys())
        new_tokens = [
            token for token in filtered_morphemes.keys()
            if token not in base_vocab
        ]

        extended_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            use_fast=True
        )

        num_added = extended_tokenizer.add_tokens(new_tokens)
        print(f"Added {num_added} new tokens to the vocabulary")

        return extended_tokenizer, new_tokens

    def get_tokenization_stats(
            self,
            tokenizer: AutoTokenizer,
            texts: List[str],
            base_tokenizer=None,
            new_tokens=None
    ) -> TokenizationStats:
        """Compute detailed tokenization statistics for a set of texts."""
        sequence_lengths = []
        all_tokens = []
        unknown_tokens = 0
        total_tokens = 0

        for text in tqdm(texts, desc="Computing tokenization stats"):
            tokens = tokenizer.tokenize(text)
            sequence_lengths.append(len(tokens))
            all_tokens.extend(tokens)
            unknown_tokens += sum(1 for t in tokens if t == tokenizer.unk_token)
            total_tokens += len(tokens)

        compression_ratio = 1.0
        token_reduction_percentage = 0.0
        if base_tokenizer is not None:
            compression_ratio, token_reduction_percentage = self.calculate_compression_metrics(
                base_tokenizer, tokenizer, texts
            )

        token_efficiency_score = 0.0
        top_new_tokens = []
        if new_tokens:
            token_efficiency_score, top_new_tokens = self.calculate_token_efficiency(
                tokenizer, new_tokens, texts
            )

        return TokenizationStats(
            avg_sequence_length=np.mean(sequence_lengths),
            max_sequence_length=max(sequence_lengths),
            vocab_size=len(tokenizer.get_vocab()),
            unique_tokens_used=len(set(all_tokens)),
            unknown_token_percentage=unknown_tokens / total_tokens * 100 if total_tokens > 0 else 0,
            compression_ratio=compression_ratio,
            token_reduction_percentage=token_reduction_percentage,
            token_efficiency_score=token_efficiency_score,
            top_new_tokens=top_new_tokens
        )

    def visualize_analysis(
            self,
            base_tokenizer: AutoTokenizer,
            extended_tokenizer: AutoTokenizer,
            test_texts: List[str],
            morpheme_vocab: Dict[str, int],
            new_tokens: List[str],
            save_path: str = None
    ):
        """Create visualizations comparing original and extended tokenizer."""
        plt.style.use('seaborn')
        fig = plt.figure(figsize=(15, 15))
        gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)

        base_stats = self.get_tokenization_stats(base_tokenizer, test_texts)
        extended_stats = self.get_tokenization_stats(
            extended_tokenizer,
            test_texts,
            base_tokenizer,
            new_tokens
        )

        # 1. Vocabulary Size and Usage Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['Vocab Size', 'Unique Tokens Used']
        base_values = [base_stats.vocab_size, base_stats.unique_tokens_used]
        extended_values = [extended_stats.vocab_size, extended_stats.unique_tokens_used]

        x = np.arange(len(metrics))
        width = 0.35

        ax1.bar(x - width / 2, base_values, width, label='Base Tokenizer')
        ax1.bar(x + width / 2, extended_values, width, label='Extended Tokenizer')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.set_title('Vocabulary Size and Usage')
        ax1.legend()

        # 2. Token Length Distribution
        ax2 = fig.add_subplot(gs[0, 1])
        base_lengths = []
        extended_lengths = []

        for text in test_texts:
            base_lengths.append(len(base_tokenizer.tokenize(text)))
            extended_lengths.append(len(extended_tokenizer.tokenize(text)))

        sns.kdeplot(data=base_lengths, label='Base Tokenizer', ax=ax2)
        sns.kdeplot(data=extended_lengths, label='Extended Tokenizer', ax=ax2)
        ax2.set_title('Token Length Distribution')
        ax2.set_xlabel('Number of Tokens')
        ax2.set_ylabel('Density')
        ax2.legend()

        # 3. Unknown Token Percentage
        ax3 = fig.add_subplot(gs[1, 0])
        unk_percentages = [base_stats.unknown_token_percentage, extended_stats.unknown_token_percentage]
        ax3.bar(['Base Tokenizer', 'Extended Tokenizer'], unk_percentages)
        ax3.set_title('Unknown Token Percentage')
        ax3.set_ylabel('Percentage')

        # 4. Morpheme Frequency Distribution
        ax4 = fig.add_subplot(gs[1, 1])
        frequencies = list(morpheme_vocab.values())
        sns.histplot(data=frequencies, bins=50, ax=ax4)
        ax4.set_title('Morpheme Frequency Distribution')
        ax4.set_xlabel('Frequency')
        ax4.set_ylabel('Count')
        ax4.set_xscale('log')

        # 5. Compression Metrics
        ax5 = fig.add_subplot(gs[2, :])
        metrics = ['Compression Ratio', 'Token Reduction %']
        values = [extended_stats.compression_ratio, extended_stats.token_reduction_percentage]
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

        bars = ax5.bar(metrics, values, color=colors)
        ax5.set_title('Compression Metrics')
        ax5.set_ylabel('Ratio / Percentage')

        for bar in bars:
            height = bar.get_height()
            ax5.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{height:.2f}',
                ha='center',
                va='bottom'
            )

        # 6. Token Efficiency Score
        ax6 = fig.add_subplot(gs[3, 0])
        efficiency_score = extended_stats.token_efficiency_score
        ax6.bar(['Token Efficiency Score'], [efficiency_score], color='#3498db')
        ax6.set_title('Token Efficiency Score')
        ax6.set_ylim(0, 1)

        interpretation = (
            "Excellent" if efficiency_score > 0.8
            else "Good" if efficiency_score > 0.6
            else "Fair" if efficiency_score > 0.4
            else "Poor"
        )
        ax6.text(
            0, efficiency_score,
            f"{efficiency_score:.2f} ({interpretation})",
            ha='center', va='bottom'
        )

        # 7. Top New Tokens Usage
        ax7 = fig.add_subplot(gs[3, 1])
        if extended_stats.top_new_tokens:
            tokens, counts = zip(*extended_stats.top_new_tokens[:10])
            y_pos = np.arange(len(tokens))
            ax7.barh(y_pos, counts)
            ax7.set_yticks(y_pos)
            ax7.set_yticklabels(tokens)
            ax7.invert_yaxis()
            ax7.set_title('Top 10 Most Used New Tokens')
            ax7.set_xlabel('Frequency')

        fig.suptitle('Tokenizer Extension Analysis', fontsize=16, y=1.02)

        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

        # Print statistics
        print("\nTokenizer Statistics:")
        print(f"{'Metric':<25} {'Base':>12} {'Extended':>12} {'Change':>12}")
        print("-" * 63)
        metrics = [
            ("Vocabulary Size", base_stats.vocab_size, extended_stats.vocab_size),
            ("Avg Sequence Length", base_stats.avg_sequence_length, extended_stats.avg_sequence_length),
            ("Unknown Token %", base_stats.unknown_token_percentage, extended_stats.unknown_token_percentage),
            ("Unique Tokens Used", base_stats.unique_tokens_used, extended_stats.unique_tokens_used),
            ("Compression Ratio", 1.0, extended_stats.compression_ratio),
            ("Token Reduction %", 0.0, extended_stats.token_reduction_percentage),
            ("Token Efficiency Score", 0.0, extended_stats.token_efficiency_score)
        ]

        for name, base, ext in metrics:
            change = ((ext - base) / base * 100) if base != 0 else float('inf')
            print(f"{name:<25} {base:>12.2f} {ext:>12.2f} {change:>11.2f}%")


def main():
    # Create a sample morpheme vocabulary CSV
    sample_morphemes = pd.DataFrame({
        'token': ['pre', 'post', 'un', 'ing', 'ed', 'ly', 'tion', 'able'],
        'frequency': [100, 90, 85, 120, 110, 95, 80, 75]
    })
    sample_morphemes.to_csv('sample_morphemes.csv', index=False)

    # Initialize the tokenizer extender
    extender = TokenizerVocabularyExtender(model_name="google/gemma-2b")

    # Load morpheme vocabulary
    morpheme_vocab = extender.load_morpheme_vocab('sample_morphemes.csv')

    # Load sample texts from a dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    test_texts = dataset['text'][:1000]  # Use first 1000 texts for testing

    # Filter out empty texts
    test_texts = [text for text in test_texts if text.strip()]

    # Analyze vocabulary overlap
    overlap_analysis = extender.analyze_vocabulary_overlap(morpheme_vocab)
    print("\nVocabulary Overlap Analysis:")
    for key, value in overlap_analysis.items():
        if key != 'new_token_list':
            print(f"{key}: {value}")

    # Extend the tokenizer
    extended_tokenizer, new_tokens = extender.extend_tokenizer(
        morpheme_vocab,
        min_frequency=5
    )

    # Visualize the analysis
    extender.visualize_analysis(
        extender.base_tokenizer,
        extended_tokenizer,
        test_texts,
        morpheme_vocab,
        new_tokens,
        save_path="tokenizer_analysis.png"
    )


if __name__ == "__main__":
    main()