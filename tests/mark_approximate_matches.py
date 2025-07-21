import json
import argparse
import re
from difflib import SequenceMatcher

def load_results(file_path):
    """Load evaluation results from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def suggest_approximate_match(expected, actual):
    """Suggest if the actual answer is an approximate match to the expected answer."""
    if not expected or not actual:
        return False

    # Convert to lowercase for comparison
    expected_lower = expected.lower()
    actual_lower = actual.lower()

    # Direct match or substring
    if expected_lower in actual_lower or actual_lower in expected_lower:
        return True

    # Check similarity ratio
    similarity = SequenceMatcher(None, expected_lower, actual_lower).ratio()
    if similarity > 0.8:  # High similarity threshold
        return True

    # Check for numbers (dates, quantities, etc.)
    expected_numbers = re.findall(r'\d+', expected)
    actual_numbers = re.findall(r'\d+', actual)
    if expected_numbers and actual_numbers:
        common_numbers = set(expected_numbers).intersection(set(actual_numbers))
        if common_numbers:
            return True

    # Check for key terms (nouns, proper nouns)
    expected_words = set(re.findall(r'\b[A-Za-z]{3,}\b', expected_lower))
    actual_words = set(re.findall(r'\b[A-Za-z]{3,}\b', actual_lower))
    common_words = expected_words.intersection(actual_words)
    if len(common_words) >= min(2, len(expected_words)):
        return True

    return False

def save_results(results, output_file):
    """Save the updated evaluation results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

def calculate_metrics(results):
    """Calculate evaluation metrics."""
    total = len(results)
    approximate_matches = sum(1 for r in results if r['approximate_match'] == True)

    metrics = {
        'total_questions': total,
        'approximate_matches': approximate_matches,
        'approximate_match_percentage': (approximate_matches / total) * 100 if total > 0 else 0
    }

    return metrics

def mark_matches(results):
    """Interactively mark approximate matches."""
    for i, result in enumerate(results):
        print(f"\nQuestion {i+1}: {result['question']}")
        print(f"Expected answer: {result['expected_answer']}")
        print(f"System answer: {result['system_answer']}")

        # Suggest an approximate match
        suggestion = None
        if result['approximate_match'] is None:
            suggestion = suggest_approximate_match(result['expected_answer'], result['system_answer'])
            suggestion_text = "Yes" if suggestion else "No"
            print(f"Suggested match: {suggestion_text}")

        if result['approximate_match'] is not None:
            current = "Yes" if result['approximate_match'] else "No"
            choice = input(f"Is this an approximate match? (Currently: {current}) [y/n/s to skip/a to accept suggestion]: ").lower()
        else:
            choice = input(f"Is this an approximate match? [y/n/s to skip/a to accept suggestion]: ").lower()

        if choice == 'y':
            result['approximate_match'] = True
        elif choice == 'n':
            result['approximate_match'] = False
        elif choice == 'a' and suggestion is not None:
            result['approximate_match'] = suggestion
            print(f"Accepted suggestion: {suggestion_text}")
        elif choice == 's':
            # Skip this question
            pass
        else:
            print("Invalid choice. Skipping.")

    return results

def auto_mark_matches(results):
    """Automatically mark approximate matches using the suggestion algorithm."""
    for result in results:
        if result['approximate_match'] is None:
            result['approximate_match'] = suggest_approximate_match(
                result['expected_answer'],
                result['system_answer']
            )

    return results

def main():
    parser = argparse.ArgumentParser(description="Mark approximate matches in evaluation results")
    parser.add_argument("--results", type=str, required=True, help="Path to the evaluation results JSON file")
    parser.add_argument("--output", type=str, help="Path to save the updated results (defaults to overwriting input file)")
    parser.add_argument("--calculate", action="store_true", help="Calculate metrics without marking")
    parser.add_argument("--auto", action="store_true", help="Automatically mark matches using the suggestion algorithm")

    args = parser.parse_args()

    # Load results
    results = load_results(args.results)

    if args.calculate:
        # Just calculate metrics
        metrics = calculate_metrics(results)
        print("\nEvaluation Metrics:")
        print(f"Total questions: {metrics['total_questions']}")
        print(f"Approximate matches: {metrics['approximate_matches']}")
        print(f"Approximate match percentage: {metrics['approximate_match_percentage']:.2f}%")
    elif args.auto:
        # Automatically mark matches
        updated_results = auto_mark_matches(results)

        # Save updated results
        output_file = args.output if args.output else args.results
        save_results(updated_results, output_file)

        # Calculate and display metrics
        metrics = calculate_metrics(updated_results)
        print("\nAutomatic Evaluation Metrics:")
        print(f"Total questions: {metrics['total_questions']}")
        print(f"Approximate matches: {metrics['approximate_matches']}")
        print(f"Approximate match percentage: {metrics['approximate_match_percentage']:.2f}%")
        print("\nNote: These matches were determined automatically and may need manual verification.")
    else:
        # Mark matches interactively
        updated_results = mark_matches(results)

        # Save updated results
        output_file = args.output if args.output else args.results
        save_results(updated_results, output_file)

        # Calculate and display metrics
        metrics = calculate_metrics(updated_results)
        print("\nEvaluation Metrics:")
        print(f"Total questions: {metrics['total_questions']}")
        print(f"Approximate matches: {metrics['approximate_matches']}")
        print(f"Approximate match percentage: {metrics['approximate_match_percentage']:.2f}%")

if __name__ == "__main__":
    main()
