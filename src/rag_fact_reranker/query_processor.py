import json
import time
import numpy as np
from Levenshtein import distance as levenshtein_distance
from sklearn.metrics.pairwise import cosine_similarity
from . import knowledge_base
from . import llm

def extract_query_keywords(query, max_retries=3, retry_delay=2):
    """
    Extract important keywords from a user query using OpenAI.

    Args:   
        query (str): The user's query
        max_retries (int): Maximum number of retries if API call fails
        retry_delay (int): Delay between retries in seconds

    Returns:
        list: List of extracted keywords
    """
    prompt = f"""Extract the most important keywords (nouns and noun phrases) from the following query.
    Return ONLY a JSON array of strings, where each string is a single keyword or key phrase.
    Focus on entities, concepts, and important terms that would help retrieve relevant information.

    Query: {query}

    Output (JSON array of keyword strings):"""

    for attempt in range(max_retries):
        try:
            response = llm.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You extract important keywords from queries and return them as a JSON array of strings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )

            keywords_text = response.choices[0].message.content.strip()

            # Try to parse the JSON response
            try:
                # Handle case where the model might include ```json and ``` in the response
                if "```json" in keywords_text:
                    keywords_text = keywords_text.split("```json")[1].split("```")[0].strip()
                elif "```" in keywords_text:
                    keywords_text = keywords_text.split("```")[1].strip()

                keywords = json.loads(keywords_text)
                if isinstance(keywords, list):
                    return keywords
                else:
                    print(f"Unexpected response format: {keywords_text}")
                    continue
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {keywords_text}")
                continue

        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return []

    return []

def calculate_keyword_similarity(query_keyword, db_keyword):
    """
    Calculate similarity between two keywords using a combination of:
    1. Cosine similarity of embeddings
    2. Normalized Levenshtein distance

    Args:
        query_keyword (str): Keyword from the query
        db_keyword (str): Keyword from the database

    Returns:
        float: Similarity score between 0 and 1
    """
    # Get embeddings
    query_embedding = llm.get_embedding(query_keyword)
    db_embedding = llm.get_embedding(db_keyword)

    if not query_embedding or not db_embedding:
        # Fall back to just Levenshtein if embeddings fail
        max_len = max(len(query_keyword), len(db_keyword))
        if max_len == 0:
            return 0
        lev_similarity = 1 - (levenshtein_distance(query_keyword, db_keyword) / max_len)
        return lev_similarity

    # Calculate cosine similarity
    cosine_sim = cosine_similarity([query_embedding], [db_embedding])[0][0]

    # Calculate normalized Levenshtein distance
    max_len = max(len(query_keyword), len(db_keyword))
    if max_len == 0:
        lev_similarity = 0
    else:
        lev_similarity = 1 - (levenshtein_distance(query_keyword, db_keyword) / max_len)

    # Combine similarities (weighted average)
    combined_similarity = (0.7 * cosine_sim) + (0.3 * lev_similarity)

    return combined_similarity

def match_keywords(query_keywords, top_k=5, similarity_threshold=0.6):
    """
    Match query keywords with database keywords using Pinecone vector search.

    Args:
        query_keywords (list): Keywords extracted from the query
        top_k (int): Number of similar keywords to retrieve for each query keyword
        similarity_threshold (float): Minimum similarity score to consider a match

    Returns:
        dict: Dictionary mapping query keywords to matched database keywords with scores
    """
    if not query_keywords:
        print("No query keywords provided.")
        return {}

    matches = {}

    for query_keyword in query_keywords:
        print(f"Finding similar keywords for: {query_keyword}")

        # Use Pinecone to find similar keywords
        similar_keywords = knowledge_base.query_similar_keywords(query_keyword, top_k=top_k)

        # Filter by similarity threshold
        keyword_matches = [(keyword, score) for keyword, score in similar_keywords if score >= similarity_threshold]

        # Store matches for this query keyword
        matches[query_keyword] = keyword_matches

        print(f"Found {len(keyword_matches)} matches for '{query_keyword}'")

    return matches

def rank_and_retrieve_facts(matched_keywords, query, top_k=5):
    """
    Retrieve and rank facts based on keyword matches and connectivity using the formula:
    Score = cosSim(query, fact) + degree(fact)

    Where:
    - cosSim(query, fact) is the cosine similarity between query and fact embeddings
    - degree(fact) is the number of query keywords related to the fact

    This creates a 2D ranking where facts are sorted first by degree and then by cosine similarity.

    Args:
        matched_keywords (dict): Dictionary mapping query keywords to matched database keywords
        query (str): The original query text
        top_k (int, default=5): Number of top facts to retrieve

    Returns:
        list: List of retrieved facts
    """
    if not matched_keywords:
        return []

    # Get query embedding
    query_embedding = llm.get_embedding(query)
    if not query_embedding:
        print("Warning: Could not get embedding for query. Using keyword-only ranking.")

    # Track facts and their associated keywords
    fact_keywords = {}  # fact_text -> set of query keywords
    all_facts = set()  # Set of all unique facts

    # Collect all facts and track which query keywords they match
    for query_keyword, db_matches in matched_keywords.items():
        for db_keyword, similarity in db_matches:
            # Get facts associated with this keyword
            facts = knowledge_base.query_facts_by_keyword(db_keyword)

            for fact in facts:
                all_facts.add(fact)
                if fact not in fact_keywords:
                    fact_keywords[fact] = set()
                fact_keywords[fact].add(query_keyword)

    # Calculate scores using the formula: Score = cosSim(query, fact) + degree(fact)
    fact_scores = {}  # fact_text -> (degree, cosine_similarity)

    for fact in all_facts:
        # Calculate degree(fact): number of query keywords related to this fact
        degree = len(fact_keywords[fact])

        # Calculate cosine similarity between query and fact
        if query_embedding:
            fact_embedding = llm.get_embedding(fact)
            if fact_embedding:
                # Calculate cosine similarity
                cos_sim = llm.calculate_cosine_similarity(query_embedding, fact_embedding)
            else:
                print(f"Warning: Could not get embedding for fact: {fact[:50]}...")
                cos_sim = 0.0
        else:
            cos_sim = 0.0

        # Store both components for sorting
        fact_scores[fact] = (degree, cos_sim)

    # Sort facts by the combined score (degree + cosine similarity)
    # This naturally creates a 2D ranking where degree (integer) has higher priority
    # than cosine similarity (fractional)
    ranked_facts = sorted(fact_scores.items(),
                          key=lambda x: x[1][0] + x[1][1],  # degree + cosine similarity
                          reverse=True)

    # Print some debug info
    print(f"Found {len(ranked_facts)} facts with keyword matches")
    if ranked_facts:
        top_fact, (degree, cos_sim) = ranked_facts[0]
        print(f"Top fact has degree {degree} and cosine similarity {cos_sim:.4f}")
        print(f"Top fact: {top_fact[:100]}...")

    # Return top-k facts
    return [fact for fact, _ in ranked_facts[:top_k]]

def judge_fact_sufficiency(query, facts, max_retries=3, retry_delay=2):
    """
    Use OpenAI to judge if the retrieved facts are sufficient to answer the query.

    Args:
        query (str): The user's query
        facts (list): List of retrieved facts
        max_retries (int): Maximum number of retries if API call fails
        retry_delay (int): Delay between retries in seconds

    Returns:
        bool: True if facts are sufficient, False otherwise
    """
    if not facts:
        return False

    facts_text = "\n".join([f"- {fact}" for fact in facts])

    prompt = f"""Given the following query and retrieved facts, determine if the facts are sufficient to provide a complete and accurate answer to the query.

    Query: {query}

    Retrieved Facts:
    {facts_text}

    Are these facts sufficient to answer the query? Respond with ONLY "Yes" or "No".
    """

    for attempt in range(max_retries):
        try:
            response = llm.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You evaluate whether a set of facts is sufficient to answer a query."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )

            answer = response.choices[0].message.content.strip().lower()

            if "yes" in answer:
                return True
            elif "no" in answer:
                return False
            else:
                print(f"Unexpected response: {answer}")
                continue

        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return False

    return False

def generate_answer(query, facts, max_retries=3, retry_delay=2):
    """
    Generate an answer based on the query and retrieved facts.

    Args:
        query (str): The user's query
        facts (list): List of retrieved facts
        max_retries (int): Maximum number of retries if API call fails
        retry_delay (int): Delay between retries in seconds

    Returns:
        str: Generated answer
    """
    if not facts:
        return "I don't have enough information to answer that question."

    facts_text = "\n".join([f"- {fact}" for fact in facts])

    prompt = f"""Based on the following facts, provide a concise and accurate answer to the query.
    Only use information from the provided facts. If the facts don't contain enough information to fully answer the query, acknowledge the limitations in your answer.

    Query: {query}

    Facts:
    {facts_text}

    Answer:"""

    for attempt in range(max_retries):
        try:
            response = llm.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You provide accurate answers based only on the facts provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )

            answer = response.choices[0].message.content.strip()
            return answer

        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return "I encountered an error while generating an answer. Please try again."

    return "I encountered an error while generating an answer. Please try again."

def rerank_facts(query, facts, max_retries=3, retry_delay=2):
    """
    Enhanced re-ranking of facts using LLM to select only those that are most helpful for answering the query.
    Handles special cases like calculations, inferences, and multi-hop reasoning.

    Args:
        query (str): The user's query
        facts (list): List of retrieved facts
        max_retries (int): Maximum number of retries if API call fails
        retry_delay (int): Delay between retries in seconds

    Returns:
        list: List of re-ranked facts that are helpful for answering the query
    """
    if not facts:
        return []

    # Create a numbered list of facts for the LLM
    numbered_facts = []
    for i, fact in enumerate(facts, 1):
        numbered_facts.append(f"{i}. {fact}")

    facts_text = "\n".join(numbered_facts)

    # Analyze the query to determine if it requires special handling
    requires_calculation = any(keyword in query.lower() for keyword in
                              ["how old", "how many", "age", "years", "century", "centuries",
                               "percentage", "fraction", "ratio", "when", "date"])

    requires_inference = any(keyword in query.lower() for keyword in
                            ["why", "how", "reason", "cause", "effect", "impact", "influence",
                             "relationship", "connection", "link", "association"])

    # Enhanced prompt with specific instructions based on query type
    prompt = f"""Question: {query}

Facts:
{facts_text}

Your task is to identify the most relevant facts that directly help answer the question.

"""

    # Add special instructions based on query type
    if requires_calculation:
        prompt += """This question may require calculating dates, ages, time periods, or numerical values.
Look for facts that provide:
- Dates, years, or time periods
- Birth dates or founding dates when age calculations are needed
- Numerical values needed for calculations
- Both parts of information needed for a calculation (e.g., both a birth date AND a reference year)
"""

    if requires_inference:
        prompt += """This question may require making inferences or connections between facts.
Look for facts that:
- Directly address the causal relationship asked about
- Provide context that explains why something happened
- Connect entities mentioned in the question
- Provide background information essential to understanding the relationship
"""

    prompt += """
Return ONLY the numbers of the most relevant facts (e.g., [1, 3, 6]).
Limit your selection to the 3-5 most important facts that directly support answering the question.
If multiple facts contain similar information, select the most specific and comprehensive one.
"""

    for attempt in range(max_retries):
        try:
            response = llm.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert research analyst specializing in fact verification and relevance assessment. Your job is to identify which facts are most relevant to answering a specific question, including facts that might need to be combined or used for calculations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Lower temperature for more consistent results
                max_tokens=500
            )

            result = response.choices[0].message.content.strip()
            print(f"Re-ranking response: {result}")

            # Extract the fact numbers from the response using improved regex patterns
            relevant_facts = []

            # First, try to find a list of numbers like [1, 3, 6]
            import re

            # Look for lists in various formats: [1,2,3], [1, 2, 3], "1, 2, 3", etc.
            number_list_patterns = [
                r'\[([0-9, ]+)\]',  # [1, 2, 3]
                r'Facts?(?:\s+numbers?)?(?:\s*:)?\s*(?:are|is)?\s*(?:numbers?)?(?:\s*:)?\s*([0-9, ]+)',  # Facts: 1, 2, 3
                r'(?:I select|Selected|Relevant|Most relevant)(?:\s+facts?)?(?:\s*:)?\s*(?:numbers?)?(?:\s*:)?\s*([0-9, ]+)',  # Selected facts: 1, 2, 3
                r'(?:Facts?|Numbers?)(?:\s*:)?\s*([0-9, ]+)'  # Facts: 1, 2, 3
            ]

            for pattern in number_list_patterns:
                number_list_match = re.search(pattern, result)
                if number_list_match:
                    # Extract numbers from the list
                    number_str = number_list_match.group(1)
                    try:
                        # Handle various separators: commas, spaces, and/or
                        number_str = re.sub(r'(?:and|&)', ',', number_str)
                        numbers = [int(n.strip()) for n in re.split(r'[,\s]+', number_str) if n.strip() and n.strip().isdigit()]

                        # Get the corresponding facts (adjusting for 1-based indexing)
                        for num in numbers:
                            if 1 <= num <= len(facts):
                                relevant_facts.append(facts[num-1])

                        if relevant_facts:
                            break  # Stop if we found facts with this pattern
                    except ValueError:
                        pass

            # If we couldn't find a list or it was empty, look for numbered points in the text
            if not relevant_facts:
                # Look for patterns like "1.", "2:", "Fact 3", etc.
                number_matches = re.findall(r'(?:^|\n)(?:Fact\s*)?(\d+)[\.:\)]', result)
                for match in number_matches:
                    try:
                        num = int(match)
                        if 1 <= num <= len(facts):
                            relevant_facts.append(facts[num-1])
                    except ValueError:
                        pass

            # If we still don't have any relevant facts, try one more pattern: "Fact #X"
            if not relevant_facts:
                fact_number_matches = re.findall(r'Fact\s*#?\s*(\d+)', result)
                for match in fact_number_matches:
                    try:
                        num = int(match)
                        if 1 <= num <= len(facts):
                            relevant_facts.append(facts[num-1])
                    except ValueError:
                        pass

            # If we still don't have any relevant facts, return the top 3 original facts
            if not relevant_facts:
                print("Could not extract relevant facts from LLM response. Using top 3 original facts.")
                return facts[:min(3, len(facts))]

            print(f"Re-ranked facts: {len(relevant_facts)} out of {len(facts)} original facts")
            return relevant_facts

        except Exception as e:
            print(f"Error on re-ranking attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("Failed to re-rank facts. Using original facts.")
                return facts[:min(3, len(facts))]  # Return top 3 facts if re-ranking fails

    return facts[:min(3, len(facts))]  # Return top 3 facts as fallback

def process_query(query, top_k=5, max_hops=3):
    """
    Process a user query and generate an answer using multi-hop retrieval if needed.

    Args:
        query (str): The user's query
        top_k (int, default=5): Number of top facts to retrieve per hop
        max_hops (int, default=3): Maximum number of retrieval hops to perform

    Returns:
        dict: Dictionary containing the query process results
    """
    results = {
        "query": query,
        "extracted_keywords": [],
        "matched_keywords": {},
        "retrieved_facts": [],
        "facts_sufficient": False,
        "answer": "",
        "multi_hop_info": {
            "hops_performed": 0,
            "sub_queries": [],
            "facts_per_hop": []
        }
    }

    # Step 1: Extract keywords from query
    print(f"Extracting keywords from query: {query}")
    query_keywords = extract_query_keywords(query)
    results["extracted_keywords"] = query_keywords
    print(f"Extracted keywords: {query_keywords}")

    if not query_keywords:
        results["answer"] = "I couldn't extract keywords from your query. Please try rephrasing it."
        return results

    # Step 2: Match query keywords with database keywords
    print("Matching keywords with database...")
    matched_keywords = match_keywords(query_keywords)
    results["matched_keywords"] = matched_keywords

    if not any(matches for matches in matched_keywords.values()):
        results["answer"] = "I couldn't find any relevant information in my knowledge base. Please try a different query."
        return results

    # Step 3: Retrieve and rank facts
    print("Retrieving and ranking facts...")
    retrieved_facts = rank_and_retrieve_facts(matched_keywords, query, top_k=top_k)

    # Initialize multi-hop tracking
    all_facts = retrieved_facts.copy()
    results["multi_hop_info"]["hops_performed"] = 1  # First hop is the initial query
    results["multi_hop_info"]["facts_per_hop"].append(retrieved_facts)

    if not retrieved_facts:
        results["answer"] = "I couldn't find any relevant facts in my knowledge base. Please try a different query."
        return results

    # Step 4: Judge if facts are sufficient
    print("Judging if facts are sufficient...")
    facts_sufficient = judge_fact_sufficiency(query, all_facts)

    # Multi-hop retrieval logic
    current_query = query
    hop_count = 1

    while not facts_sufficient and hop_count < max_hops:
        print(f"\n=== Starting hop {hop_count + 1} ===")

        # Generate sub-query
        print("Generating sub-query...")
        sub_query = llm.generate_sub_query(query, all_facts)
        results["multi_hop_info"]["sub_queries"].append(sub_query)

        # Check if sub-query is the same as the original or previous query
        if sub_query.lower() == query.lower() or sub_query.lower() == current_query.lower() or not sub_query:
            print("Sub-query is the same as original/previous query or empty. Stopping multi-hop.")
            break

        print(f"Sub-query: {sub_query}")
        current_query = sub_query

        # Extract keywords from sub-query
        sub_query_keywords = extract_query_keywords(sub_query)
        print(f"Extracted keywords from sub-query: {sub_query_keywords}")

        if not sub_query_keywords:
            print("Couldn't extract keywords from sub-query. Stopping multi-hop.")
            break

        # Match sub-query keywords with database
        sub_matched_keywords = match_keywords(sub_query_keywords)

        if not any(matches for matches in sub_matched_keywords.values()):
            print("No matches found for sub-query keywords. Stopping multi-hop.")
            break

        # Retrieve facts for sub-query
        sub_retrieved_facts = rank_and_retrieve_facts(sub_matched_keywords, sub_query, top_k=top_k)

        if not sub_retrieved_facts:
            print("No facts retrieved for sub-query. Stopping multi-hop.")
            break

        # Add new facts to the collection (avoid duplicates)
        new_facts = [fact for fact in sub_retrieved_facts if fact not in all_facts]
        all_facts.extend(new_facts)

        # Update multi-hop info
        hop_count += 1
        results["multi_hop_info"]["hops_performed"] = hop_count
        results["multi_hop_info"]["facts_per_hop"].append(sub_retrieved_facts)

        # Check if facts are now sufficient
        facts_sufficient = judge_fact_sufficiency(query, all_facts)

        print(f"Found {len(new_facts)} new facts. Total facts: {len(all_facts)}")
        print(f"Facts sufficient after hop {hop_count}: {'Yes' if facts_sufficient else 'No'}")

        # If we've reached max hops or facts are sufficient, stop
        if hop_count >= max_hops or facts_sufficient:
            break

    # Update results with all gathered facts
    results["retrieved_facts"] = all_facts
    results["facts_sufficient"] = facts_sufficient

    # Step 5: Re-rank facts using LLM
    print("Re-ranking facts...")
    reranked_facts = rerank_facts(query, all_facts)
    results["reranked_facts"] = reranked_facts

    # Step 6: Generate answer using re-ranked facts
    print("Generating answer...")
    answer = generate_answer(query, reranked_facts)
    results["answer"] = answer

    return results
