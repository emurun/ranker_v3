from datasets import load_dataset
import json
import time
import uuid
import yaml
from tqdm import tqdm
import llm
from neo4j import GraphDatabase
from pinecone import Pinecone

# Load configuration from config.yaml
try:
    with open('config/config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
    PINECONE_API_KEY = config.get('pinecone_api_key')
    PINECONE_INDEX = config.get('pinecone_index', 'knowledge-web-index')
    EMBEDDING_MODEL = config.get('embedding_model', 'text-embedding-ada-002')
    NEO4J_URI = config.get('neo4j_uri')
    NEO4J_USER = config.get('neo4j_user', "neo4j")
    NEO4J_PASSWORD = config.get('neo4j_password')
except Exception as e:
    print(f"Error loading config in knowledge_base.py: {e}")
    # Fallback values - user must provide credentials
    print("Warning: Could not load config file. Please ensure config/config.yaml exists with your credentials.")
    PINECONE_API_KEY = None
    PINECONE_INDEX = "knowledge-web-index"
    EMBEDDING_MODEL = "text-embedding-ada-002"
    NEO4J_URI = None
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = None

# Initialize Neo4j driver - will be created when needed
neo4j_driver = None

# Initialize Pinecone client and index
pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = None

def cleanup_neo4j_database():
    """Clean up the Neo4j database by removing all nodes and relationships."""
    print("Cleaning up Neo4j database...")
    driver = get_neo4j_driver()
    with driver.session() as session:
        # Delete all relationships first
        session.run("MATCH ()-[r]-() DELETE r")
        # Then delete all nodes
        session.run("MATCH (n) DELETE n")
    print("Neo4j database cleaned up successfully.")

def cleanup_pinecone_index():
    """Clean up the Pinecone index by deleting all vectors."""
    print("Cleaning up Pinecone index...")
    index = get_pinecone_index()
    try:
        # Delete all vectors in the index
        index.delete(delete_all=True)
        print("Pinecone index cleaned up successfully.")
    except Exception as e:
        print(f"Error cleaning up Pinecone index: {e}")

def get_neo4j_driver():
    """Get or create a Neo4j driver instance"""
    global neo4j_driver
    if neo4j_driver is None:
        neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        # Verify connectivity
        neo4j_driver.verify_connectivity()
    return neo4j_driver

def get_pinecone_index():
    """Get or create a Pinecone index"""
    global pinecone_index
    if pinecone_index is None:

        # Connect to the index
        pinecone_index = pc.Index(PINECONE_INDEX)
        print(f"Connected to Pinecone index: {PINECONE_INDEX}")

    return pinecone_index

def store_keywords_in_pinecone(keywords_with_embeddings, batch_size=100):
    """Store keywords with embeddings in Pinecone

    Args:
        keywords_with_embeddings: List of tuples (keyword, embedding)
        batch_size: Number of vectors to upsert in a single batch
    """
    if not keywords_with_embeddings:
        print("No keywords with embeddings to store")
        return

    index = get_pinecone_index()

    # Prepare vectors for upsert
    vectors = []
    for keyword, embedding in keywords_with_embeddings:
        # Generate a unique ID for each keyword
        # In a real application, you might want a more sophisticated ID strategy
        vector_id = str(uuid.uuid4())

        # Create vector record
        vector = {
            "id": vector_id,
            "values": embedding,
            "metadata": {
                "keyword": keyword,
                "timestamp": time.time()
            }
        }
        vectors.append(vector)

    # Upsert vectors in batches
    total_vectors = len(vectors)
    for i in range(0, total_vectors, batch_size):
        batch = vectors[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
            print(f"Upserted batch {i//batch_size + 1}/{(total_vectors-1)//batch_size + 1} ({len(batch)} vectors)")
        except Exception as e:
            print(f"Error upserting vectors to Pinecone: {e}")

    print(f"Successfully stored {total_vectors} keyword vectors in Pinecone")

def query_similar_keywords(query_text, top_k=5):
    """Query Pinecone for keywords similar to the query text

    Args:
        query_text: Text to find similar keywords for
        top_k: Number of results to return

    Returns:
        List of similar keywords with their similarity scores
    """
    # Get embedding for query text
    query_embedding = llm.get_embedding(query_text, model=EMBEDDING_MODEL)
    if not query_embedding:
        print("Failed to get embedding for query text")
        return []

    # Query Pinecone
    index = get_pinecone_index()
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )

        # Format results
        similar_keywords = []
        for match in results.matches:
            keyword = match.metadata.get("keyword", "Unknown")
            score = match.score
            similar_keywords.append((keyword, score))

        return similar_keywords
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return []

# Neo4j database functions
def create_fact_node(tx, fact_text, paragraph_id):
    """Create a fact node in Neo4j"""
    query = (
        "MERGE (f:Fact {text: $fact_text}) "
        "SET f.paragraph_id = $paragraph_id "
        "RETURN f"
    )
    result = tx.run(query, fact_text=fact_text, paragraph_id=paragraph_id)
    return result.single()[0]

def create_keyword_node(tx, keyword):
    """Create a keyword node in Neo4j"""
    query = (
        "MERGE (k:Keyword {text: $keyword}) "
        "RETURN k"
    )
    result = tx.run(query, keyword=keyword)
    return result.single()[0]

def create_fact_keyword_relationship(tx, fact_text, keyword):
    """Create a relationship between a fact and a keyword"""
    query = (
        "MATCH (f:Fact {text: $fact_text}) "
        "MATCH (k:Keyword {text: $keyword}) "
        "MERGE (f)-[r:HAS_KEYWORD]->(k) "
        "RETURN r"
    )
    result = tx.run(query, fact_text=fact_text, keyword=keyword)
    return result.single()[0]

def query_facts_by_keyword(keyword):
    """Query facts associated with a specific keyword"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        query = (
            "MATCH (f:Fact)-[:HAS_KEYWORD]->(k:Keyword {text: $keyword}) "
            "RETURN f.text AS fact"
        )
        result = session.run(query, keyword=keyword)
        return [record["fact"] for record in result]

def query_keywords_by_fact(fact_text):
    """Query keywords associated with a specific fact"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        query = (
            "MATCH (f:Fact {text: $fact_text})-[:HAS_KEYWORD]->(k:Keyword) "
            "RETURN k.text AS keyword"
        )
        result = session.run(query, fact_text=fact_text)
        return [record["keyword"] for record in result]

def query_related_facts(keyword1, keyword2, max_distance=2):
    """Find facts that connect two keywords within a certain distance"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        query = (
            "MATCH path = (k1:Keyword {text: $keyword1})<-[:HAS_KEYWORD]-"
            "(f1:Fact)-[:HAS_KEYWORD*1.." + str(max_distance) + "]->"
            "(k2:Keyword {text: $keyword2}) "
            "RETURN f1.text AS fact"
        )
        result = session.run(query, keyword1=keyword1, keyword2=keyword2)
        return [record["fact"] for record in result]

def get_all_keywords():
    """Get all keywords in the database"""
    driver = get_neo4j_driver()
    with driver.session() as session:
        query = "MATCH (k:Keyword) RETURN k.text AS keyword"
        result = session.run(query)
        return [record["keyword"] for record in result]

def demonstrate_neo4j_queries():
    """Demonstrate various Neo4j query capabilities"""
    # No need to store the driver reference here as each query function gets its own
    try:
        # Get all keywords
        all_keywords = get_all_keywords()
        print(f"\nFound {len(all_keywords)} keywords in the database.")
        print(f"Sample keywords: {', '.join(all_keywords[:5])}...")

        if all_keywords:
            # Query facts by a keyword
            sample_keyword = all_keywords[0]
            print(f"\nFacts associated with keyword '{sample_keyword}':")
            facts = query_facts_by_keyword(sample_keyword)
            for fact in facts:
                print(f"- {fact}")

            # If we have facts, get keywords for the first fact
            if facts:
                sample_fact = facts[0]
                print(f"\nKeywords associated with fact: '{sample_fact[:50]}...'")
                keywords = query_keywords_by_fact(sample_fact)
                print(f"Keywords: {', '.join(keywords)}")

                # If we have multiple keywords, try to find related facts
                if len(keywords) >= 2:
                    keyword1 = keywords[0]
                    keyword2 = keywords[1]
                    print(f"\nFacts connecting keywords '{keyword1}' and '{keyword2}':")
                    related_facts = query_related_facts(keyword1, keyword2)
                    for fact in related_facts:
                        print(f"- {fact}")
        else:
            print("No keywords found in the database.")
    except Exception as e:
        print(f"Error querying Neo4j: {e}")

def demonstrate_pinecone_queries():
    """Demonstrate Pinecone vector search capabilities"""
    try:
        # Get all keywords from Neo4j to use as sample queries
        all_keywords = get_all_keywords()

        if not all_keywords:
            print("\nNo keywords found to use for demonstration.")
            return

        # Select a sample keyword for demonstration
        sample_keyword = all_keywords[0]
        print(f"\nDemonstrating Pinecone vector search with keyword: '{sample_keyword}'")

        # Query similar keywords
        similar_keywords = query_similar_keywords(sample_keyword, top_k=5)

        if similar_keywords:
            print(f"\nKeywords similar to '{sample_keyword}':")
            for keyword, score in similar_keywords:
                print(f"- {keyword} (similarity score: {score:.4f})")
        else:
            print(f"No similar keywords found for '{sample_keyword}'")

        # Try another keyword if available
        if len(all_keywords) > 1:
            sample_keyword2 = all_keywords[1]
            print(f"\nTrying another keyword: '{sample_keyword2}'")
            similar_keywords2 = query_similar_keywords(sample_keyword2, top_k=5)

            if similar_keywords2:
                print(f"\nKeywords similar to '{sample_keyword2}':")
                for keyword, score in similar_keywords2:
                    print(f"- {keyword} (similarity score: {score:.4f})")
            else:
                print(f"No similar keywords found for '{sample_keyword2}'")

    except Exception as e:
        print(f"Error demonstrating Pinecone queries: {e}")


def load_musique_subset(n=35, seed=42):
    dataset = load_dataset("dgslibisey/MuSiQue", split="train")
    subset = dataset.shuffle(seed=seed).select(range(n))
    return subset

def extract_paragraphs(dataset):
    """Extract paragraphs from the dataset."""
    all_paragraphs = []
    for i, item in enumerate(dataset):
        for para in item['paragraphs']:
            if para.get('is_supporting', False):
                all_paragraphs.append({
                    'id': f"{item['id']}-{para['idx']}",
                    'title': para.get('title', ''),
                    'text': para.get('paragraph_text', ''),
                    'source_id': item['id']
                })
    return all_paragraphs


def build_knowledge_web(subset, clean_existing=True):
    print("\n\n\n")
    print("Building Knowledge Web...")

    # Clean up existing data if requested
    if clean_existing:
        cleanup_neo4j_database()
        cleanup_pinecone_index()

    paragraphs = extract_paragraphs(subset)
    print(f"Extracted {len(paragraphs)} paragraphs")

    # Get Neo4j driver and create session
    driver = get_neo4j_driver()
    with driver.session() as session:
        for paragraph in tqdm(paragraphs, desc="Processing paragraphs"):
            facts = llm.decompose_into_facts(paragraph['text'])
            print(f"Extracted {len(facts)} facts from paragraph {paragraph['text'][:25]}")

            # Process each fact
            for fact in facts:
                # Extract keywords from the fact
                keywords = llm.extract_keywords(fact)
                print(f"Extracted {len(keywords)} keywords from fact: {fact[:50]}...")

                # Get embeddings for keywords
                keywords_with_embeddings = []
                for keyword in keywords:
                    embedding = llm.get_embedding(keyword, model=EMBEDDING_MODEL)
                    if embedding:
                        keywords_with_embeddings.append((keyword, embedding))

                # Store keywords with embeddings in Pinecone
                if keywords_with_embeddings:
                    store_keywords_in_pinecone(keywords_with_embeddings)
                    print(f"Stored {len(keywords_with_embeddings)} keywords with embeddings in Pinecone")

                # Store fact in Neo4j
                try:
                    # Create fact node
                    session.execute_write(create_fact_node, fact, paragraph['id'])

                    # Create keyword nodes and relationships
                    for keyword in keywords:
                        session.execute_write(create_keyword_node, keyword)
                        session.execute_write(create_fact_keyword_relationship, fact, keyword)

                    print(f"Stored fact and {len(keywords)} keywords in Neo4j")
                except Exception as e:
                    print(f"Error storing in Neo4j: {e}")
