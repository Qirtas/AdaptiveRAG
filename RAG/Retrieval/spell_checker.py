"""
spell_checker.py
Enhanced spell checking utility with better domain-specific term handling
"""

import os
import logging
from typing import List, Tuple, Optional, Set
from symspellpy import SymSpell, Verbosity
import requests
import re

logger = logging.getLogger("datamite.spell_checker")


class SpellChecker:
    """Spell checker wrapper for SymSpell with domain-specific enhancements"""

    def __init__(
            self,
            dictionary_path: Optional[str] = None,
            custom_terms_path: Optional[str] = None,
            max_edit_distance: int = 3,
            prefix_length: int = 7
    ):
        """
        Initialize the spell checker

        Args:
            dictionary_path: Path to frequency dictionary file
            custom_terms_path: Path to custom domain-specific terms
            max_edit_distance: Maximum edit distance for suggestions
            prefix_length: Prefix length for symspell
        """
        self.sym_spell = SymSpell(max_edit_distance, prefix_length)
        self.max_edit_distance = max_edit_distance
        self.custom_terms = set()  # Store custom terms for priority checking

        # Common words/abbreviations that should never be "corrected"
        self.protected_words = {
            # Common abbreviations and Latin phrases
            'vs', 'vs.', 'etc', 'etc.', 'e.g.', 'i.e.', 'et al.', 'et al',

            # Question words (CRITICAL - often get incorrectly matched)
            'what', 'where', 'when', 'why', 'which', 'who', 'whom', 'whose', 'how',

            # Demonstrative pronouns
            'this', 'that', 'these', 'those',

            # Personal pronouns
            'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them',
            'my', 'your', 'his', 'its', 'our', 'their',
            'mine', 'yours', 'hers', 'ours', 'theirs',

            # Common verbs
            'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
            'do', 'does', 'did', 'done', 'doing',
            'have', 'has', 'had', 'having',
            'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',
            'go', 'get', 'see', 'make', 'know', 'take', 'give', 'come', 'use', 'find',

            # Common prepositions
            'in', 'on', 'at', 'by', 'for', 'to', 'from', 'of', 'with', 'about',
            'as', 'if', 'or', 'but', 'not', 'up', 'out', 'so',

            # Articles and common adjectives
            'a', 'an', 'the',
            'all', 'any', 'each', 'every', 'some', 'many', 'much', 'more', 'most',
            'no', 'few', 'less', 'both', 'other', 'such',

            # Common adverbs
            'now', 'then', 'here', 'there', 'very', 'too', 'also', 'just', 'only',
            'well', 'even', 'still', 'yet', 'already', 'always', 'never', 'often',

            # Common conjunctions
            'and', 'but', 'or', 'nor', 'so', 'for', 'yet',
            'if', 'than', 'that', 'though', 'although', 'because', 'since', 'unless',

            # Numbers and quantifiers
            'one', 'two', 'three', 'first', 'second', 'third', 'once', 'twice'
        }

        # Load dictionary
        if dictionary_path and os.path.exists(dictionary_path):
            self._load_dictionary(dictionary_path)
        else:
            self._download_default_dictionary()

        # Load custom terms if provided - AFTER main dictionary
        if custom_terms_path and os.path.exists(custom_terms_path):
            self._load_custom_terms(custom_terms_path)

    def _download_default_dictionary(self):
        """Download and load default frequency dictionary"""
        dict_path = "frequency_dictionary_en_82_765.txt"

        if not os.path.exists(dict_path):
            logger.info("Downloading default frequency dictionary...")
            url = "https://raw.githubusercontent.com/mammothb/symspellpy/master/symspellpy/frequency_dictionary_en_82_765.txt"
            try:
                response = requests.get(url)
                with open(dict_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                logger.info(f"Dictionary downloaded to {dict_path}")
            except Exception as e:
                logger.error(f"Failed to download dictionary: {e}")
                self._load_basic_dictionary()
                return

        self._load_dictionary(dict_path)

    def _load_dictionary(self, path: str):
        """Load frequency dictionary from file"""
        if not self.sym_spell.load_dictionary(path, 0, 1):
            logger.error(f"Failed to load dictionary from {path}")
            self._load_basic_dictionary()
        else:
            logger.info(f"Loaded dictionary from {path}")

    def _load_basic_dictionary(self):
        """Load a basic built-in dictionary as fallback"""
        basic_words = [
            "access", "cost", "data", "database", "query", "search",
            "information", "retrieval", "document", "vector", "embedding"
        ]
        for word in basic_words:
            self.sym_spell.create_dictionary_entry(word, 1)
        logger.info("Loaded basic fallback dictionary")

    def _load_custom_terms(self, path: str):
        """Load custom domain-specific terms with very high frequency"""
        try:
            short_terms = []  # Track short terms separately
            long_terms = []  # Track longer terms

            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    term = line.strip()
                    # Skip comments and empty lines
                    if not term or term.startswith('#'):
                        continue

                    # Determine if term is short (3 chars or less)
                    is_short = len(term) <= 3

                    # Store in custom terms set for priority checking
                    self.custom_terms.add(term.upper())
                    self.custom_terms.add(term.lower())
                    self.custom_terms.add(term)

                    # For very short terms (2-3 chars), use LOWER frequency
                    # to avoid over-correcting common words
                    if is_short:
                        # Still higher than default, but not overwhelming
                        frequency = 100000
                        short_terms.append(term)
                    else:
                        # Long terms get very high frequency
                        frequency = 1000000
                        long_terms.append(term)

                    # Add with appropriate frequency
                    self.sym_spell.create_dictionary_entry(term, frequency)
                    self.sym_spell.create_dictionary_entry(term.upper(), frequency)
                    self.sym_spell.create_dictionary_entry(term.lower(), frequency)

            logger.info(f"Loaded {len(long_terms)} long custom terms and {len(short_terms)} short terms from {path}")
            logger.debug(f"Short terms (lower priority): {short_terms[:10]}")  # Show first 10
            logger.debug(f"Long terms (high priority): {long_terms[:10]}")  # Show first 10
        except Exception as e:
            logger.error(f"Failed to load custom terms: {e}")

    def _correct_word(self, word: str) -> Tuple[str, float]:
        """
        Correct a single word, prioritizing custom terms

        Returns:
            Tuple of (corrected_word, confidence)
        """
        # Check if word is protected (common word that shouldn't be corrected)
        # Case-insensitive check for protected words
        if word.lower() in self.protected_words:
            return word, 1.0

        # Check if word is already a custom term (exact match)
        if word in self.custom_terms:
            return word, 1.0

        # Check if the word already exists in the main dictionary (is valid English)
        # If it does, only correct it if there's a very close custom term match
        existing_lookup = self.sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=0)
        is_valid_word = len(existing_lookup) > 0

        # Check for close matches in custom terms first
        word_upper = word.upper()
        best_custom_match = None
        best_custom_distance = float('inf')

        for custom_term in self.custom_terms:
            # Calculate edit distance manually for custom terms
            distance = self._calculate_edit_distance(word_upper, custom_term.upper())

            # Only consider very close matches (distance 1) for custom terms
            # And only if the custom term is long enough (4+ chars) or exact match
            if distance == 0:
                # Exact match (case-insensitive)
                return custom_term, 1.0
            elif distance == 1 and len(custom_term) >= 4:
                # Close match with longer term
                if distance < best_custom_distance:
                    best_custom_distance = distance
                    best_custom_match = custom_term

        # If we found a close custom match and the original word is not valid, use it
        if best_custom_match and not is_valid_word:
            logger.debug(f"Matched '{word}' to custom term '{best_custom_match}'")
            return best_custom_match, 0.9

        # If the word is already valid and we found a custom match,
        # only use custom match if it's much more likely to be intended
        # (e.g., in a domain-specific context)
        if best_custom_match and is_valid_word:
            # Don't correct valid words to custom terms unless they're very similar
            # and the custom term is significantly longer
            if len(best_custom_match) > len(word) + 2:
                logger.debug(f"Potentially matched '{word}' to custom term '{best_custom_match}'")
                return best_custom_match, 0.7
            else:
                # Keep the original valid word
                return word, 1.0

        # Fall back to SymSpell for general corrections
        suggestions = self.sym_spell.lookup(
            word,
            Verbosity.CLOSEST,
            max_edit_distance=self.max_edit_distance
        )

        if suggestions:
            best = suggestions[0]

            # Additional protection: Don't correct to protected words if original word is valid
            if best.term.lower() in self.protected_words and is_valid_word and word.lower() not in self.protected_words:
                # Original word is valid but not protected, and suggestion is a protected word
                # This is likely a false positive correction
                return word, 1.0

            # If the word is already valid and the suggestion is only slightly better, keep original
            if is_valid_word and best.distance > 0 and best.term.lower() != word.lower():
                return word, 1.0

            confidence = 1 - (best.distance / len(word)) if len(word) > 0 else 1.0
            return best.term, confidence

        return word, 1.0

    def _calculate_edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings"""
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer than s2
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def correct_query(
            self,
            query: str,
            return_all_variants: bool = False,
            confidence_threshold: float = 0.8
    ) -> Tuple[str, List[str], float]:
        """
        Correct spelling in a query with enhanced word-level correction

        Args:
            query: Input query string
            return_all_variants: Whether to return all possible variants
            confidence_threshold: Minimum confidence to apply correction

        Returns:
            Tuple of (corrected_query, all_variants, confidence_score)
        """
        if not query or not query.strip():
            return query, [query], 1.0

        # First, try compound correction
        compound_suggestions = self.sym_spell.lookup_compound(
            query,
            max_edit_distance=self.max_edit_distance,
            transfer_casing=True
        )

        # Also do word-by-word correction for better domain term handling
        words = query.split()
        corrected_words = []
        total_confidence = 0
        word_corrections_made = False

        for word in words:
            # Remove punctuation for checking, but preserve it
            punctuation = ""
            cleaned_word = word
            if word and word[-1] in ".,!?;:":
                punctuation = word[-1]
                cleaned_word = word[:-1]

            corrected_word, word_conf = self._correct_word(cleaned_word)

            if corrected_word != cleaned_word:
                word_corrections_made = True
                logger.debug(f"Word correction: '{cleaned_word}' -> '{corrected_word}' (conf: {word_conf:.2f})")

            corrected_words.append(corrected_word + punctuation)
            total_confidence += word_conf

        # Calculate average confidence
        avg_confidence = total_confidence / len(words) if words else 1.0

        # Build the corrected query
        word_corrected_query = " ".join(corrected_words)

        # Choose between compound and word-level correction
        variants = [query]  # Always include original

        # Add word-level corrected version if corrections were made
        if word_corrections_made and word_corrected_query != query:
            variants.append(word_corrected_query)
            best_correction = word_corrected_query
            confidence = avg_confidence
        # Otherwise use compound correction if available
        elif compound_suggestions and compound_suggestions[0].term != query:
            best_correction = compound_suggestions[0].term
            confidence = 1 - (compound_suggestions[0].distance / len(query)) if len(query) > 0 else 1.0
            if confidence >= confidence_threshold:
                variants.append(best_correction)
        else:
            best_correction = query
            confidence = 1.0

        # Log corrections
        if best_correction != query:
            logger.info(f"Spell correction: '{query}' -> '{best_correction}' (confidence: {confidence:.2f})")

        # Return all variants if requested
        if return_all_variants:
            # Add more alternatives if available
            if compound_suggestions:
                for suggestion in compound_suggestions[:3]:
                    if suggestion.term not in variants:
                        variants.append(suggestion.term)

        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique_variants.append(v)

        return best_correction, unique_variants, confidence

    def add_domain_term(self, term: str, frequency: int = 1000000):
        """Add a domain-specific term to the dictionary"""
        self.custom_terms.add(term)
        self.custom_terms.add(term.upper())
        self.custom_terms.add(term.lower())
        self.sym_spell.create_dictionary_entry(term, frequency)
        self.sym_spell.create_dictionary_entry(term.upper(), frequency)
        self.sym_spell.create_dictionary_entry(term.lower(), frequency)
        logger.debug(f"Added domain term: {term}")


# Singleton instance
_spell_checker_instance = None


def get_spell_checker(
        dictionary_path: Optional[str] = None,
        custom_terms_path: Optional[str] = None
) -> SpellChecker:
    """
    Get or create singleton spell checker instance

    Args:
        dictionary_path: Path to frequency dictionary
        custom_terms_path: Path to custom domain terms

    Returns:
        SpellChecker instance
    """
    global _spell_checker_instance

    if _spell_checker_instance is None:
        _spell_checker_instance = SpellChecker(
            dictionary_path=dictionary_path,
            custom_terms_path=custom_terms_path
        )

    return _spell_checker_instance


def process_user_query(
        query: str,
        use_variants: bool = True,
        custom_terms_path: Optional[str] = None,
        confidence_threshold: float = 0.7
) -> dict:
    """
    Main function to process user query with spell checking

    Args:
        query: User's input query
        use_variants: Whether to return multiple variants for RAG
        custom_terms_path: Path to domain-specific terms
        confidence_threshold: Minimum confidence for corrections

    Returns:
        Dictionary with processed query information
    """
    checker = get_spell_checker(custom_terms_path=custom_terms_path)

    corrected, variants, confidence = checker.correct_query(
        query,
        return_all_variants=use_variants,
        confidence_threshold=confidence_threshold
    )

    result = {
        'original': query,
        'corrected': corrected,
        'variants': variants,
        'confidence': confidence,
        'needs_correction': corrected != query
    }

    if result['needs_correction']:
        logger.info(f"Query correction applied: '{query}' -> '{corrected}'")

    return result