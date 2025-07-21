import json
import time
import yaml
from openai import OpenAI

# Load configuration from config.yaml
try:
    with open('config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    OPENAI_API_KEY = config.get('openai_api_key')
    EMBEDDING_MODEL = config.get('embedding_model', 'text-embedding-ada-002')
except Exception as e:
    print(f"Error loading config in llm.py: {e}")
    # Fallback values - user must provide credentials
    print("Warning: Could not load config file. Please ensure config/config.yaml exists with your credentials.")
    OPENAI_API_KEY = None
    EMBEDDING_MODEL = "text-embedding-ada-002"

client = OpenAI(api_key=OPENAI_API_KEY)

def extract_keywords(fact, max_retries=3, retry_delay=2):
    """Extract important keywords (nouns, noun phrases) from a fact."""
    prompt = f"""Extract the most important keywords (nouns and noun phrases) from the following fact.
    Return ONLY a JSON array of strings, where each string is a single keyword or key phrase.
    Focus on entities, concepts, and important terms. Limit to 3-5 keywords.

    Fact: {fact}

    Output (JSON array of keyword strings):"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You extract important keywords from text and return them as a JSON array of strings."},
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


def get_embedding(text, model="text-embedding-ada-002", max_retries=3, retry_delay=2):
    """Get embedding for a text using OpenAI's embedding API."""
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(
                model=model,
                input=text
            )
            return response.data[0].embedding

        # [0.3, 0.21, 0.32, .1, 0.231, ...]
        except Exception as e:
            print(f"Error getting embedding on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return None
    return None

def decompose_into_facts(paragraph_text, max_retries=3, retry_delay=2):
    """Use OpenAI to break a paragraph into independent fact statements."""
    prompt = f"""Extract all factual statements from the following paragraph.
    Return ONLY a JSON array of strings, where each string is a single, atomic fact.
    Do not include opinions, interpretations, or redundant information.

    Paragraph: {paragraph_text}

    Output (JSON array of fact strings):"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You extract factual statements from text and return them as a JSON array of strings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )

            facts_text = response.choices[0].message.content.strip()

            try:
                if "```json" in facts_text:
                    facts_text = facts_text.split("```json")[1].split("```")[0].strip()
                elif "```" in facts_text:
                    facts_text = facts_text.split("```")[1].strip()

                facts = json.loads(facts_text)
                if isinstance(facts, list):
                    return facts
                else:
                    print(f"Unexpected response format: {facts_text}")
                    continue
            except json.JSONDecodeError:
                print(f"Failed to parse JSON: {facts_text}")
                continue
        except Exception as e:
            print(f"Error on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return []

    return []

def calculate_cosine_similarity(embedding1, embedding2):
    """
    Calculate the cosine similarity between two embeddings.

    Args:
        embedding1 (list): First embedding vector
        embedding2 (list): Second embedding vector

    Returns:
        float: Cosine similarity between the two embeddings (between -1 and 1)
    """
    import numpy as np

    # Convert to numpy arrays if they aren't already
    v1 = np.array(embedding1)
    v2 = np.array(embedding2)

    # Calculate dot product
    dot_product = np.dot(v1, v2)

    # Calculate magnitudes
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Calculate cosine similarity
    cosine_similarity = dot_product / (magnitude1 * magnitude2)

    return float(cosine_similarity)

def generate_sub_query(original_query, retrieved_facts, max_retries=3, retry_delay=2):
    """
    Generate a sub-query based on the original query and retrieved facts.

    Args:
        original_query (str): The original user query
        retrieved_facts (list): List of facts retrieved so far
        max_retries (int): Maximum number of retries if API call fails
        retry_delay (int): Delay between retries in seconds

    Returns:
        str: A sub-query to retrieve additional information
    """
    facts_text = "\n".join([f"- {fact}" for fact in retrieved_facts])

    prompt = f"""Given an original query and the facts we've retrieved so far, generate a focused sub-query
    that would help us find additional information needed to fully answer the original query.

    Original Query: {original_query}

    Facts Retrieved So Far:
    {facts_text}

    The sub-query should:
    1. Be specific and focused on missing information
    2. Target a different aspect than what we already know
    3. Help bridge gaps in our current knowledge
    4. Be a complete, standalone question

    Sub-Query:"""

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You generate focused sub-queries to find missing information needed to answer a question."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=150
            )

            sub_query = response.choices[0].message.content.strip()
            return sub_query

        except Exception as e:
            print(f"Error generating sub-query on attempt {attempt+1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                return ""

    return ""