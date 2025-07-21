import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from rag_fact_reranker import knowledge_base, query_processor
import argparse

def build_knowledge_base(n=2):
    """Build the knowledge base with the specified subset size."""
    print(f"\nBuilding knowledge base with subset size n={n}")
    subset = knowledge_base.load_musique_subset(n=n)
    knowledge_base.build_knowledge_web(subset, clean_existing=True)

    # Demonstrate Neo4j queries
    knowledge_base.demonstrate_neo4j_queries()

    # Demonstrate Pinecone vector search
    knowledge_base.demonstrate_pinecone_queries()

    print("\nKnowledge base built successfully!")

def query_knowledge_base():
    """Query the knowledge base with user input."""
    print("\n" + "="*50)
    print("KNOWLEDGE BASE QUERY")
    print("="*50)

    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")

        if query.lower() == 'exit':
            break

        print("\nProcessing your query...")
        results = query_processor.process_query(query)

        # Display results
        print("\n" + "="*50)
        print("QUERY RESULTS")
        print("="*50)

        print(f"Query: {results['query']}")

        print("\nExtracted Keywords:")
        for keyword in results['extracted_keywords']:
            print(f"- {keyword}")

        print("\nMatched Keywords:")
        for query_keyword, matches in results['matched_keywords'].items():
            print(f"- {query_keyword}:")
            for db_keyword, score in matches[:3]:  # Show top 3 matches
                print(f"  * {db_keyword} (score: {score:.4f})")

        # Display multi-hop information
        print("\n" + "-"*50)
        print("MULTI-HOP RETRIEVAL INFORMATION")
        print("-"*50)

        hops_performed = results['multi_hop_info']['hops_performed']
        print(f"Hops Performed: {hops_performed}")

        if hops_performed > 1:
            print("\nSub-Queries Generated:")
            for i, sub_query in enumerate(results['multi_hop_info']['sub_queries']):
                print(f"Hop {i+2}: {sub_query}")  # +2 because hop 1 is the initial query

            print("\nFacts Retrieved Per Hop:")
            for i, facts in enumerate(results['multi_hop_info']['facts_per_hop'], 1):
                print(f"Hop {i}: {len(facts)} facts")
                # Show a sample of facts from each hop
                for j, fact in enumerate(facts[:2], 1):
                    print(f"  {j}. {fact}")
                if len(facts) > 2:
                    print(f"  ... and {len(facts) - 2} more facts")
        else:
            print("No multi-hop retrieval was needed or performed.")

        print("\n" + "-"*50)

        print("\nRetrieved Facts (Combined):")
        for i, fact in enumerate(results['retrieved_facts'][:5], 1):  # Show top 5 facts
            print(f"{i}. {fact}")

        if len(results['retrieved_facts']) > 5:
            print(f"... and {len(results['retrieved_facts']) - 5} more facts")

        # Display re-ranked facts
        if 'reranked_facts' in results:
            print("\nRe-ranked Facts (Most Relevant):")
            for i, fact in enumerate(results['reranked_facts'], 1):
                print(f"{i}. {fact}")

        print("\nFacts Sufficient:", "Yes" if results['facts_sufficient'] else "No")

        print("\nAnswer:")
        print(results['answer'])

        print("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description="Knowledge Base Management Tool")
    parser.add_argument("--build-knowledge", action="store_true", help="Build knowledge base")
    parser.add_argument("--n", type=int, default=2, help="Size of dataset subset to use")

    args = parser.parse_args()

    driver = None
    try:
        if args.build_knowledge:
            # Build knowledge base mode
            build_knowledge_base(args.n)
        else:
            # Default query mode
            query_knowledge_base()
    finally:
        # Close the Neo4j driver when done
        driver = knowledge_base.get_neo4j_driver()
        if driver:
            driver.close()
            print("\nNeo4j driver closed.")

        print("\nDone!")

if __name__ == "__main__":
    main()
