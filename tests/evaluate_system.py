import json
import query_processor
import argparse
from tqdm import tqdm

def load_test_questions(file_path):
    """Load test questions from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def run_evaluation(questions, max_hops=3):
    """Run the evaluation on the test questions."""
    results = []
    
    for question in tqdm(questions, desc="Processing questions"):
        print(f"\nProcessing question: {question['question']}")
        
        # Process the query
        query_result = query_processor.process_query(question['question'], max_hops=max_hops)
        
        # Record the result
        result = {
            'question': question['question'],
            'expected_answer': question['expected_answer'],
            'system_answer': query_result['answer'],
            'facts_sufficient': query_result['facts_sufficient'],
            'hops_performed': query_result['multi_hop_info']['hops_performed'],
            'sub_queries': query_result['multi_hop_info']['sub_queries'],
            'approximate_match': None  # To be filled in manually
        }
        
        results.append(result)
        
        # Print the result
        print(f"Expected answer: {question['expected_answer']}")
        print(f"System answer: {query_result['answer']}")
        print(f"Facts sufficient: {query_result['facts_sufficient']}")
        print(f"Hops performed: {query_result['multi_hop_info']['hops_performed']}")
        if query_result['multi_hop_info']['sub_queries']:
            print("Sub-queries:")
            for i, sub_query in enumerate(query_result['multi_hop_info']['sub_queries']):
                print(f"  {i+1}. {sub_query}")
        
        print("-" * 80)
    
    return results

def save_results(results, output_file):
    """Save the evaluation results to a JSON file."""
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate the multi-hop fact finding agent")
    parser.add_argument("--questions", type=str, required=True, help="Path to the test questions JSON file")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Path to save the results")
    parser.add_argument("--max-hops", type=int, default=3, help="Maximum number of hops to perform")
    
    args = parser.parse_args()
    
    # Load test questions
    questions = load_test_questions(args.questions)
    
    # Run evaluation
    results = run_evaluation(questions, max_hops=args.max_hops)
    
    # Save results
    save_results(results, args.output)
    
    print("\nEvaluation complete. Please manually review the results and mark approximate matches.")
    print("After marking, run this script with the --calculate-metrics flag to get the final metrics.")

if __name__ == "__main__":
    main()
