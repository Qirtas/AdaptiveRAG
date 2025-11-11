import html
import json
import time
from typing import Dict, List

import requests


def fetch_trivia_questions(num_questions: int = 500, categories_to_avoid: List[int] = None) -> List[Dict]:
    """
    Fetch trivia questions from Open Trivia Database API
    Categories to avoid business-related topics (like economics): [28]
    Good categories for out-of-domain: 9,10,11,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,29,30,31,32
    """
    if categories_to_avoid is None:
        categories_to_avoid = [28]

    all_questions = []
    batch_size = 50

    category_names = {
        9: "General Knowledge", 10: "Entertainment: Books", 11: "Entertainment: Film",
        12: "Entertainment: Music", 13: "Entertainment: Musicals & Theatres",
        14: "Entertainment: Television", 15: "Entertainment: Video Games",
        16: "Entertainment: Board Games", 17: "Science & Nature", 18: "Science: Computers",
        19: "Science: Mathematics", 20: "Mythology", 21: "Sports", 22: "Geography",
        23: "History", 24: "Politics", 25: "Art", 26: "Celebrities", 27: "Animals",
        29: "Entertainment: Comics", 30: "Science: Gadgets", 31: "Entertainment: Japanese Anime & Manga",
        32: "Entertainment: Cartoon & Animations"
    }

    good_categories = [cat for cat in category_names.keys() if cat not in categories_to_avoid]

    while len(all_questions) < num_questions:
        for category in good_categories:
            if len(all_questions) >= num_questions:
                break

            try:
                url = f"https://opentdb.com/api.php?amount={min(batch_size, num_questions - len(all_questions))}&category={category}&type=multiple"
                response = requests.get(url)
                data = response.json()

                if data['response_code'] == 0:  # Success
                    questions = data['results']
                    for q in questions:
                        q['question'] = html.unescape(q['question'])
                        q['category_name'] = category_names.get(category, f"Category {category}")

                    all_questions.extend(questions)
                    print(
                        f"Fetched {len(questions)} questions from {category_names.get(category, f'Category {category}')}. Total: {len(all_questions)}")

                time.sleep(0.5)

            except Exception as e:
                print(f"Error fetching from category {category}: {e}")
                continue

    return all_questions[:num_questions]


def convert_to_custom_format(trivia_questions: List[Dict], output_file: str):
    """
    Convert trivia questions to your custom format
    """
    converted_questions = []

    for i, q in enumerate(trivia_questions, 1):
        question_data = {
            "question": q['question'],
            "entity_type": "General Knowledge",
            "entity_name": f"{q['category_name']} Question",
            "expected_document_type": "None",
            "category": "out_of_domain",
            "question_id": i,
            "ground_truth_docs": [],
            "trivia_category": q['category']        }
        converted_questions.append(question_data)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_questions, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(converted_questions)} questions to custom format")
    print(f"Saved to: {output_file}")

    print("\nSample questions:")
    for i in range(min(3, len(converted_questions))):
        q = converted_questions[i]
        print(f"\nQ{i + 1}: {q['question']}")
