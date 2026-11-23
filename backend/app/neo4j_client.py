import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

# Load environment variables
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(backend_dir, ".env")
print(f"Loading .env from: {env_path}")
load_dotenv(env_path)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

driver = None
# Initialize driver only if URI is provided
if NEO4J_URI and NEO4J_PASSWORD:
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=5.0,
            max_connection_lifetime=30
        )
        print("SUCCESS: Neo4j driver initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize Neo4j driver: {e}")
        driver = None
else:
    print("WARNING: Neo4j environment variables not set. Graph features will use local fallback.")

def get_driver():
    """Get the Neo4j driver instance"""
    return driver

def is_driver_healthy() -> bool:
    """Safely check if Neo4j is actively reachable"""
    global driver
    if not driver:
        return False
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 AS num")
            record = result.single()
            return record is not None and record["num"] == 1
    except Exception:
        return False
