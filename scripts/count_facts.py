import knowledge_base

# Get Neo4j driver
driver = knowledge_base.get_neo4j_driver()

# Count facts
with driver.session() as session:
    # Count facts
    result = session.run("MATCH (f:Fact) RETURN COUNT(f) as count")
    fact_count = result.single()["count"]
    print(f"Number of facts: {fact_count}")
    
    # Count paragraphs (source paragraphs)
    result = session.run("MATCH (f:Fact) RETURN COUNT(DISTINCT f.paragraph_id) as count")
    paragraph_count = result.single()["count"]
    print(f"Number of paragraphs: {paragraph_count}")

# Close the driver
driver.close()
