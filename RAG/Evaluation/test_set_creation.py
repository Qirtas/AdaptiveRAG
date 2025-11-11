"""
DataMite RAG Test Set Generator
Automatically generates test questions with expected answers from CSV data
"""

import json
import os
import random
import string
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# Direct Domain Relevant Questions

def generate_kpi_questions(df: pd.DataFrame) -> List[Dict]:
    """Generate questions for KPIs using multiple templates."""

    print(f"KPI documents length: {len(df)}")
    questions = []
    templates = [
        "What is {name}?",
        "Define {name}.",
        "Explain {name}.",
        "What does {name} measure?"
    ]

    for _, row in df.iterrows():
        kpi_name = row['name'].strip()

        for template in templates:
            question = {
                'question': template.format(name=kpi_name),
                'entity_type': 'KPI',
                'entity_name': kpi_name,
                'expected_document_type': 'KPI',
                'category': 'direct_domain_relevant'
            }
            questions.append(question)

    return questions


def generate_objective_questions(df: pd.DataFrame) -> List[Dict]:
    """Generate questions for Objectives using multiple templates."""
    questions = []
    templates = [
        "What is the {name} objective?",
        "Explain {name}.",
        "Define the objective {name}."
    ]

    for _, row in df.iterrows():
        objective_name = row['name'].strip()

        for template in templates:
            question = {
                'question': template.format(name=objective_name),
                'entity_type': 'Objective',
                'entity_name': objective_name,
                'expected_document_type': 'Objective',
                'category': 'direct_domain_relevant'
            }
            questions.append(question)

    return questions


def generate_bsc_family_questions(df: pd.DataFrame) -> List[Dict]:
    """Generate questions for BSC Families using various templates."""
    questions = []
    templates = [
        "What is {name}?",
        "Explain the {name} perspective.",
        "Define {name}."
    ]

    for _, row in df.iterrows():
        family_name = row['name'].strip()

        for template in templates:
            question = {
                'question': template.format(name=family_name),
                'entity_type': 'BSC_Family',
                'entity_name': family_name,
                'expected_document_type': 'BSC Family',
                'category': 'direct_domain_relevant'
            }
            questions.append(question)

    return questions


def generate_bsc_subfamily_questions(df: pd.DataFrame) -> List[Dict]:
    """Generate questions for BSC Subfamilies using multiple templates."""
    questions = []
    templates = [
        "What is {name}?",
        "Define {name}.",
        "Explain {name}."
    ]

    for _, row in df.iterrows():
        subfamily_name = row['name'].strip()

        for template in templates:
            question = {
                'question': template.format(name=subfamily_name),
                'entity_type': 'BSC_Subfamily',
                'entity_name': subfamily_name,
                'expected_document_type': 'BSC Subfamily',
                'category': 'direct_domain_relevant'
            }
            questions.append(question)

    return questions


def generate_criteria_questions(df: pd.DataFrame) -> List[Dict]:
    """Generate questions for Criteria using multiple templates."""
    questions = []
    templates = [
        "What is {name}?",
        "Explain the {name} criteria."
    ]

    for _, row in df.iterrows():
        criteria_name = row['name'].strip()

        for template in templates:
            question = {
                'question': template.format(name=criteria_name),
                'entity_type': 'Criteria',
                'entity_name': criteria_name,
                'expected_document_type': 'Criteria',
                'category': 'direct_domain_relevant'
            }
            questions.append(question)

    return questions


# Domain Relevant but with typos questions

# Common typo patterns
TYPO_PATTERNS = {
    'swap_adjacent': lambda s, i: s[:i] + s[i+1] + s[i] + s[i+2:] if i < len(s)-1 else s,
    'delete_char': lambda s, i: s[:i] + s[i+1:] if i < len(s) else s,
    'duplicate_char': lambda s, i: s[:i] + s[i] + s[i] + s[i+1:] if i < len(s) else s,
    'replace_char': lambda s, i: s[:i] + random.choice(string.ascii_lowercase) + s[i+1:] if i < len(s) else s,
}

# Common keyboard proximity mistakes
KEYBOARD_NEIGHBORS = {
    'a': ['s', 'q', 'w', 'z'],
    'b': ['v', 'g', 'h', 'n'],
    'c': ['x', 'd', 'f', 'v'],
    'd': ['s', 'e', 'r', 'f', 'c', 'x'],
    'e': ['w', 'r', 'd', 's'],
    'f': ['d', 'r', 't', 'g', 'v', 'c'],
    'g': ['f', 't', 'y', 'h', 'b', 'v'],
    'h': ['g', 'y', 'u', 'j', 'n', 'b'],
    'i': ['u', 'o', 'k', 'j'],
    'j': ['h', 'u', 'i', 'k', 'm', 'n'],
    'k': ['j', 'i', 'o', 'l', 'm'],
    'l': ['k', 'o', 'p'],
    'm': ['n', 'j', 'k'],
    'n': ['b', 'h', 'j', 'm'],
    'o': ['i', 'p', 'l', 'k'],
    'p': ['o', 'l'],
    'q': ['w', 'a'],
    'r': ['e', 't', 'f', 'd'],
    's': ['a', 'w', 'e', 'd', 'x', 'z'],
    't': ['r', 'y', 'g', 'f'],
    'u': ['y', 'i', 'j', 'h'],
    'v': ['c', 'f', 'g', 'b'],
    'w': ['q', 'e', 's', 'a'],
    'x': ['z', 's', 'd', 'c'],
    'y': ['t', 'u', 'h', 'g'],
    'z': ['a', 's', 'x']
}


def introduce_typo(text: str, typo_type: str = 'random') -> str:
    """
    Introduce a single typo into the text.

    Args:
        text: Original text
        typo_type: Type of typo to introduce

    Returns:
        Text with typo
    """
    if len(text) < 2:
        return text

    # Choose a random position (avoiding very start/end for some typo types)
    if typo_type == 'swap_adjacent':
        pos = random.randint(0, len(text) - 2)
        return TYPO_PATTERNS['swap_adjacent'](text, pos)

    elif typo_type == 'delete_char':
        pos = random.randint(1, len(text) - 1)
        return TYPO_PATTERNS['delete_char'](text, pos)

    elif typo_type == 'duplicate_char':
        pos = random.randint(0, len(text) - 1)
        return TYPO_PATTERNS['duplicate_char'](text, pos)

    elif typo_type == 'keyboard_neighbor':
        # Find positions with alphabetic characters
        alpha_positions = [i for i, c in enumerate(text) if c.lower() in KEYBOARD_NEIGHBORS]
        if alpha_positions:
            pos = random.choice(alpha_positions)
            char = text[pos].lower()
            if char in KEYBOARD_NEIGHBORS:
                replacement = random.choice(KEYBOARD_NEIGHBORS[char])
                # Preserve original case
                if text[pos].isupper():
                    replacement = replacement.upper()
                return text[:pos] + replacement + text[pos + 1:]
        return text

    else:  # random
        typo_types = ['swap_adjacent', 'delete_char', 'duplicate_char', 'keyboard_neighbor']
        return introduce_typo(text, random.choice(typo_types))


def create_entity_name_typo(question: Dict) -> Dict:
    """
    Create a typo in the entity name within the question.

    Args:
        question: Original question dictionary

    Returns:
        New question dictionary with entity name typo
    """
    original_name = question['entity_name']
    typo_name = introduce_typo(original_name)

    # Replace the entity name in the question text
    new_question_text = question['question'].replace(original_name, typo_name)

    return {
        'question': new_question_text,
        'original_question': question['question'],
        'entity_type': question['entity_type'],
        'entity_name': question['entity_name'],  # Keep original for evaluation
        'entity_name_with_typo': typo_name,
        'expected_document_type': question['expected_document_type'],
        'category': 'domain_relevant_with_typos',
        'typo_subcategory': 'entity_name_typo',
        'typo_details': f"'{original_name}' → '{typo_name}'"
    }


def create_question_word_typo(question: Dict) -> Dict:
    """
    Create a typo in the question words (not the entity name).

    Args:
        question: Original question dictionary

    Returns:
        New question dictionary with question word typo
    """
    question_text = question['question']
    entity_name = question['entity_name']

    # Split the question to avoid modifying the entity name
    # Find the entity name position
    entity_start = question_text.find(entity_name)

    if entity_start > 0:
        # Work on the part before entity name
        prefix = question_text[:entity_start]
        words = prefix.split()

        if words:
            # Choose a random word to introduce typo
            word_idx = random.randint(0, len(words) - 1)
            original_word = words[word_idx]
            typo_word = introduce_typo(original_word)
            words[word_idx] = typo_word

            # Remake the question
            new_prefix = ' '.join(words)
            new_question_text = new_prefix + question_text[entity_start:]

            return {
                'question': new_question_text,
                'original_question': question['question'],
                'entity_type': question['entity_type'],
                'entity_name': question['entity_name'],
                'expected_document_type': question['expected_document_type'],
                'category': 'domain_relevant_with_typos',
                'typo_subcategory': 'question_word_typo',
                'typo_details': f"'{original_word}' → '{typo_word}'"
            }

    # Fallback: just introduce a typo somewhere in the question
    typo_question = introduce_typo(question_text)
    return {
        'question': typo_question,
        'original_question': question['question'],
        'entity_type': question['entity_type'],
        'entity_name': question['entity_name'],
        'expected_document_type': question['expected_document_type'],
        'category': 'domain_relevant_with_typos',
        'typo_subcategory': 'question_word_typo',
        'typo_details': 'General typo introduced'
    }


def generate_relationship_questions(entities_by_type: Dict[str, List[Dict]],
                                    target_questions: int = 980) -> List[Dict]:
    """
    Generate relationship questions between entity pairs.

    Args:
        entities_by_type: Dictionary mapping entity types to lists of entities
        target_questions: Target number of questions to generate

    Returns:
        List of question dictionaries
    """
    questions = []

    # Relationship templates
    templates = [
        "What is the relation between {name1} and {name2}?",
        "Explain the relation between {name1} and {name2}.",
        "How are {name1} and {name2} related?"
    ]

    # Calculate how many pairs we need
    questions_per_pair = len(templates)
    target_pairs = target_questions // questions_per_pair

    all_pairs = []

    # Generate pairs within same entity type
    for entity_type, entities in entities_by_type.items():
        if len(entities) >= 2:
            # Generate combinations of entities
            entity_pairs = list(combinations(entities, 2))

            # Add entity type info to each pair
            for pair in entity_pairs:
                all_pairs.append({
                    'entity1': pair[0],
                    'entity2': pair[1],
                    'type1': entity_type,
                    'type2': entity_type
                })

    # Also generate some cross-type pairs
    entity_types = list(entities_by_type.keys())
    for i in range(len(entity_types)):
        for j in range(i + 1, len(entity_types)):
            type1, type2 = entity_types[i], entity_types[j]
            entities1 = entities_by_type[type1]
            entities2 = entities_by_type[type2]

            # Sample some pairs between types
            num_cross_pairs = min(10, len(entities1), len(entities2))
            for _ in range(num_cross_pairs):
                entity1 = random.choice(entities1)
                entity2 = random.choice(entities2)
                all_pairs.append({
                    'entity1': entity1,
                    'entity2': entity2,
                    'type1': type1,
                    'type2': type2
                })

    # Sample pairs to reach target number
    if len(all_pairs) > target_pairs:
        selected_pairs = random.sample(all_pairs, target_pairs)
    else:
        selected_pairs = all_pairs

    # Generate questions for selected pairs
    for pair_info in selected_pairs:
        entity1_name = pair_info['entity1']['name']
        entity2_name = pair_info['entity2']['name']
        type1 = pair_info['type1']
        type2 = pair_info['type2']

        for template in templates:
            question = {
                'question': template.format(name1=entity1_name, name2=entity2_name),
                'entity_type': f'{type1}-{type2}',
                'entity_name': f'{entity1_name}::{entity2_name}',
                'entity1_name': entity1_name,
                'entity2_name': entity2_name,
                'expected_document_type': f'{type1}-{type2}',
                'category': 'direct_domain_relevant_relationship',
                'ground_truth_docs': [
                    f"{type1}::{entity1_name}",
                    f"{type2}::{entity2_name}"
                ]
            }
            questions.append(question)

    return questions