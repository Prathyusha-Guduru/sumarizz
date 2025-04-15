import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
import re
import argparse
import json
from typing import List, Dict, Tuple, Optional, Set, Union
import os
import matplotlib.pyplot as plt
import seaborn as sns
import ast


import nltk
nltk.download('punkt_tab')


# Make sure to download required NLTK packages
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TechnicalSummaryEvaluator:
    """A class to evaluate technical summaries based on a rubric."""
    
    def __init__(self, 
                 technical_terms: Optional[List[str]] = None,
                 original_text: Optional[str] = None):
        """
        Initialize the evaluator.
        
        Args:
            technical_terms: List of technical terms for the domain
            original_text: The original document text for reference
        """
        self.technical_terms = set(term.lower() for term in technical_terms) if technical_terms else set()
        self.original_text = original_text
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        
        # Define the weights from the rubric
        self.weights = {
            'content': 0.4,
            'precision': 0.25,
            'readability': 0.2,
            'conciseness': 0.15
        }
        
        # For storing the scores
        self.scores = {}
        self.automated_metrics = {}
        
    def extract_key_phrases(self, text: str, top_n: int = 20) -> List[str]:
        """Extract key phrases from text using TF-IDF."""
        sentences = nltk.sent_tokenize(text)
        
        # Use TF-IDF to find important terms
        vectorizer = TfidfVectorizer(stop_words='english', 
                                     ngram_range=(1, 2),  # Include unigrams and bigrams
                                     max_features=100)
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top terms by summing TF-IDF scores across sentences
            importance = np.sum(tfidf_matrix.toarray(), axis=0)
            indices = np.argsort(importance)[::-1][:top_n]
            
            return [feature_names[i] for i in indices]
        except:
            # Fallback if TF-IDF fails (e.g., with very short texts)
            words = nltk.word_tokenize(text.lower())
            words = [w for w in words if w.isalnum() and w not in self.stopwords]
            return [w[0] for w in Counter(words).most_common(top_n)]
    
    def compute_automated_metrics(self, summaries: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Compute automated metrics for each summary.
        
        Args:
            summaries: Dictionary mapping summary IDs to summary texts
            
        Returns:
            Dictionary of metrics for each summary
        """
        metrics = {}
        
        # Extract key phrases from original if available
        original_key_phrases = set()
        if self.original_text:
            original_key_phrases = set(self.extract_key_phrases(self.original_text, top_n=30))
            
        for summary_id, text in summaries.items():
            # Initialize metrics dictionary for this summary
            metrics[summary_id] = {}
            
            # Basic statistics
            words = nltk.word_tokenize(text)
            sentences = nltk.sent_tokenize(text)
            
            metrics[summary_id]['word_count'] = len(words)
            metrics[summary_id]['sentence_count'] = len(sentences)
            metrics[summary_id]['avg_words_per_sentence'] = len(words) / max(1, len(sentences))
            
            # Readability scores
            metrics[summary_id]['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            metrics[summary_id]['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            metrics[summary_id]['gunning_fog'] = textstat.gunning_fog(text)
            
            # Technical term density
            if self.technical_terms:
                text_lower = text.lower()
                term_count = sum(1 for term in self.technical_terms 
                                if re.search(r'\b' + re.escape(term) + r'\b', text_lower))
                metrics[summary_id]['technical_term_count'] = term_count
                metrics[summary_id]['technical_term_density'] = term_count / max(1, len(words))
            
            # Key phrase overlap with original
            if original_key_phrases:
                summary_key_phrases = set(self.extract_key_phrases(text, top_n=20))
                overlap = len(original_key_phrases.intersection(summary_key_phrases))
                metrics[summary_id]['key_phrase_overlap'] = overlap
                metrics[summary_id]['key_phrase_overlap_percent'] = overlap / len(original_key_phrases) * 100
            
            # Vocabulary diversity (lexical richness)
            unique_words = set(w.lower() for w in words if w.isalnum())
            metrics[summary_id]['lexical_diversity'] = len(unique_words) / max(1, len([w for w in words if w.isalnum()]))
        
        self.automated_metrics = metrics
        return metrics
    
    def suggest_scores(self, summaries: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Suggest scores based on automated metrics.
        
        Args:
            summaries: Dictionary mapping summary IDs to summary texts
            
        Returns:
            Dictionary of suggested scores for each summary and criterion
        """
        # First ensure we have metrics
        if not self.automated_metrics:
            self.compute_automated_metrics(summaries)
        
        suggested_scores = {}
        
        for summary_id, metrics in self.automated_metrics.items():
            suggested_scores[summary_id] = {}
            
            # Content score suggestion (based on key phrase overlap if available)
            if 'key_phrase_overlap_percent' in metrics:
                overlap_pct = metrics['key_phrase_overlap_percent']
                if overlap_pct >= 80:
                    content_score = 5
                elif overlap_pct >= 65:
                    content_score = 4
                elif overlap_pct >= 50:
                    content_score = 3
                elif overlap_pct >= 30:
                    content_score = 2
                else:
                    content_score = 1
                suggested_scores[summary_id]['content'] = content_score
            
            # Technical precision (based on technical term density if available)
            if 'technical_term_density' in metrics:
                density = metrics['technical_term_density']
                # These thresholds would need to be calibrated for your domain
                if density >= 0.08:  # At least 8% of words are technical terms
                    precision_score = 5
                elif density >= 0.06:
                    precision_score = 4
                elif density >= 0.04:
                    precision_score = 3
                elif density >= 0.02:
                    precision_score = 2
                else:
                    precision_score = 1
                suggested_scores[summary_id]['precision'] = precision_score
            
            # Readability score (based on Flesch-Kincaid)
            # For technical papers, higher grade levels might be appropriate
            fk_grade = metrics['flesch_kincaid_grade']
            if 10 <= fk_grade <= 16:  # College-level appropriateness for technical content
                readability_score = 5
            elif 16 < fk_grade <= 18 or 8 <= fk_grade < 10:
                readability_score = 4
            elif 18 < fk_grade <= 20 or 6 <= fk_grade < 8:
                readability_score = 3
            elif 20 < fk_grade or fk_grade < 6:
                readability_score = 2
            else:
                readability_score = 1
            suggested_scores[summary_id]['readability'] = readability_score
            
            # Conciseness (based on word count relative to others)
            # This will be computed after we have all summaries
            
        # Now compute conciseness scores based on relative word counts
        word_counts = {sid: metrics['word_count'] for sid, metrics in self.automated_metrics.items()}
        avg_word_count = sum(word_counts.values()) / len(word_counts)
        
        for summary_id in suggested_scores:
            wc = word_counts[summary_id]
            # For technical summaries, we want concise but complete coverage
            ratio = wc / avg_word_count
            if 0.85 <= ratio <= 1.15:  # Close to average, likely optimal
                conciseness_score = 5
            elif 0.7 <= ratio < 0.85 or 1.15 < ratio <= 1.3:
                conciseness_score = 4
            elif 0.5 <= ratio < 0.7 or 1.3 < ratio <= 1.5:
                conciseness_score = 3
            elif 0.3 <= ratio < 0.5 or 1.5 < ratio <= 2.0:
                conciseness_score = 2
            else:
                conciseness_score = 1
            suggested_scores[summary_id]['conciseness'] = conciseness_score
        
        return suggested_scores
    
    def manually_score(self, 
                        summary_id: str, 
                        content: int, 
                        precision: int, 
                        readability: int, 
                        conciseness: int) -> None:
        """
        Set manual scores for a summary.
        
        Args:
            summary_id: Identifier for the summary
            content: Content accuracy score (1-5)
            precision: Technical precision score (1-5)
            readability: Readability & structure score (1-5)
            conciseness: Conciseness score (1-5)
        """
        self.scores[summary_id] = {
            'content': max(1, min(5, content)),  # Clamp between 1-5
            'precision': max(1, min(5, precision)),
            'readability': max(1, min(5, readability)),
            'conciseness': max(1, min(5, conciseness))
        }
    
    def calculate_weighted_scores(self) -> Dict[str, float]:
        """
        Calculate weighted scores for each summary.
        
        Returns:
            Dictionary mapping summary IDs to their total weighted scores
        """
        weighted_scores = {}
        
        for summary_id, criteria_scores in self.scores.items():
            total = 0
            for criterion, score in criteria_scores.items():
                total += score * self.weights[criterion]
            weighted_scores[summary_id] = total
        
        return weighted_scores
    
    def rank_summaries(self) -> List[Tuple[str, float]]:
        """
        Rank summaries by their weighted scores.
        
        Returns:
            List of (summary_id, score) tuples, sorted by score in descending order
        """
        weighted_scores = self.calculate_weighted_scores()
        return sorted(weighted_scores.items(), key=lambda x: x[1], reverse=True)
    
    def select_top_n(self, n: int = 2) -> List[str]:
        """
        Select the top N summaries.
        
        Args:
            n: Number of summaries to select
            
        Returns:
            List of summary IDs for the top N summaries
        """
        ranked = self.rank_summaries()
        return [summary_id for summary_id, _ in ranked[:n]]
    
    def plot_scores(self, filename: str = 'summary_scores.png') -> None:
        """
        Generate a visualization of the scores.
        
        Args:
            filename: Name of the output file
        """
        weighted_scores = self.calculate_weighted_scores()
        
        # Prepare data for plotting
        summary_ids = list(self.scores.keys())
        criteria = ['content', 'precision', 'readability', 'conciseness']
        
        # Create a DataFrame for easy plotting
        data = []
        for summary_id in summary_ids:
            for criterion in criteria:
                raw_score = self.scores[summary_id][criterion]
                weighted_score = raw_score * self.weights[criterion]
                data.append({
                    'Summary': summary_id,
                    'Criterion': criterion.title(),
                    'Raw Score': raw_score,
                    'Weighted Score': weighted_score
                })
        df = pd.DataFrame(data)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot weighted scores by criterion as stacked bars
        ax = plt.subplot(111)
        pivot_df = df.pivot_table(
            index='Summary', 
            columns='Criterion', 
            values='Weighted Score'
        )
        
        pivot_df.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')
        
        # Add total scores as text on top of bars
        for i, summary_id in enumerate(pivot_df.index):
            score = weighted_scores[summary_id]
            plt.text(
                i, 
                score + 0.1, 
                f'{score:.2f}', 
                ha='center', 
                fontweight='bold'
            )
        
        plt.xlabel('Summary')
        plt.ylabel('Weighted Score')
        plt.title('Summary Evaluation Scores')
        plt.legend(title='Criteria')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 5.5)  # Max score is 5
        
        plt.tight_layout()
        plt.savefig(filename)
        
        return filename

def process_csv(csv_filepath: str, output_filepath: str = None) -> pd.DataFrame:
    """
    Process a CSV file containing articles and their summaries, selecting the best 2 summaries for each article.
    
    Args:
        csv_filepath: Path to the CSV file
        output_filepath: Optional path to save the output DataFrame as CSV
        
    Returns:
        DataFrame with columns: article_index, article, selected_output_a, selected_output_b
    """
    # Read the CSV file
    df = pd.read_csv(csv_filepath)
    
    # Prepare the output DataFrame
    output_data = []
    
    # Process each row (article) in the input DataFrame
    for idx, row in df.iterrows():
        print(f"\nProcessing article {idx}...")
        
        # Get the article text
        article_text = row['article']
        
        # Extract keywords for technical terms
        try:
            # Try to parse the keywords string as a list
            if isinstance(row['keywords'], str):
                keywords = ast.literal_eval(row['keywords'])
            else:
                # If it's already a list or another type
                keywords = row['keywords']
        except (SyntaxError, ValueError):
            # If parsing fails, split by commas (fallback)
            if isinstance(row['keywords'], str):
                keywords = [k.strip() for k in row['keywords'].strip('[]').split(',')]
            else:
                keywords = []
        
        # Get the summaries
        summaries = {}
        for i in range(1, 6):
            col_name = f'output {i}'
            if col_name in row and pd.notna(row[col_name]):
                summaries[str(i)] = row[col_name]
        
        if not summaries:
            print(f"  Warning: No summaries found for article {idx}")
            continue
        
        # Create an evaluator
        evaluator = TechnicalSummaryEvaluator(
            technical_terms=keywords,
            original_text=article_text
        )
        
        # Compute metrics and get suggested scores
        evaluator.compute_automated_metrics(summaries)
        suggested_scores = evaluator.suggest_scores(summaries)
        
        # Use the suggested scores
        for summary_id, scores in suggested_scores.items():
            evaluator.manually_score(
                summary_id,
                scores.get('content', 3),
                scores.get('precision', 3),
                scores.get('readability', 3),
                scores.get('conciseness', 3)
            )
        
        # Select the top 2 summaries
        top_two = evaluator.select_top_n(2)
        
        if len(top_two) < 2:
            print(f"  Warning: Could not select 2 summaries for article {idx}")
            # Fill with empty strings if needed
            while len(top_two) < 2:
                top_two.append("")
        
        # Get the selected summaries
        selected_a = summaries[top_two[0]] if top_two[0] else ""
        selected_b = summaries[top_two[1]] if top_two[1] and len(top_two) > 1 else ""
        
        # Add to output data
        output_data.append({
            'article_index': idx,
            'article': article_text,
            'selected_output_a': selected_a,
            'selected_output_b': selected_b
        })
        
        print(f"  Selected outputs: {top_two[0]} and {top_two[1]}")
    
    # Create the output DataFrame
    output_df = pd.DataFrame(output_data)
    
    # Save to CSV if requested
    if output_filepath:
        output_df.to_csv(output_filepath, index=False)
        print(f"\nOutput saved to {output_filepath}")
    
    return output_df

def main():
    parser = argparse.ArgumentParser(description='Select the best summaries from a CSV file.')
    parser.add_argument(
        '--input', 
        required=True,
        type=str, 
        help='Input CSV file containing articles and summaries'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='selected_summaries.csv',
        help='Output CSV file for the selected summaries'
    )
    args = parser.parse_args()
    
    # Process the CSV file
    process_csv(args.input, args.output)

if __name__ == "__main__":
    main()