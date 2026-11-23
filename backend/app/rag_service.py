import faiss
import json
import os
import glob
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Set, Tuple
import sys
import re
from collections import defaultdict

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

# Import Neo4j client
try:
    from .neo4j_client import get_driver, is_driver_healthy
except ImportError:
    try:
        from neo4j_client import get_driver, is_driver_healthy
    except ImportError:
        get_driver = None
        is_driver_healthy = lambda: False

from gemini.gemini_utils import qa

class RAGService:
    def __init__(self):
        """Initialize the RAG service with FAISS index, embedding model, and entity knowledge graph"""
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Paths for FAISS index and chunk metadata (check data/embeddings first, then data/)
        primary_index = os.path.join(backend_dir, "data", "embeddings", "faiss_index.idx")
        secondary_index = os.path.join(backend_dir, "data", "faiss_index.idx")
        self.index_path = primary_index if os.path.exists(primary_index) else secondary_index

        primary_meta = os.path.join(backend_dir, "data", "embeddings", "chunk_metadata.json")
        secondary_meta = os.path.join(backend_dir, "data", "chunk_metadata.json")
        self.metadata_path = primary_meta if os.path.exists(primary_meta) else secondary_meta
        
        self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
        
        # Load FAISS index
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                print(f"SUCCESS: FAISS index loaded with {self.index.ntotal} vectors from {self.index_path}")
            except Exception as e:
                print(f"ERROR: Failed to load FAISS index: {e}")
                self.index = None
        else:
            print(f"WARNING: FAISS index not found at {self.index_path}")
            self.index = None
        
        # Load chunk metadata
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.chunks = json.load(f)
                print(f"SUCCESS: Loaded metadata for {len(self.chunks)} chunks from {self.metadata_path}")
            except Exception as e:
                print(f"ERROR: Failed to load chunk metadata: {e}")
                self.chunks = []
        else:
            print(f"WARNING: Chunk metadata not found at {self.metadata_path}")
            self.chunks = []
        
        # Load embedding model (offline first from local HuggingFace cache)
        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
            print("SUCCESS: Embedding model loaded from local cache")
        except Exception:
            try:
                self.model = SentenceTransformer(self.model_name)
                print("SUCCESS: Embedding model loaded")
            except Exception as e:
                print(f"ERROR: Failed to load embedding model: {e}")
                self.model = None
        
        # Initialize Neo4j driver
        self.driver = None
        if get_driver:
            try:
                self.driver = get_driver()
            except Exception as e:
                print(f"WARNING: Failed to get Neo4j driver: {e}")
        
        # Build local entity graph index from data/entities/ for graph RAG reasoning
        self.local_entity_to_papers = defaultdict(set)
        self.local_paper_to_entities = defaultdict(set)
        self._build_local_entity_index(backend_dir)
        
        # Create paper ID to chunks mapping for faster lookup
        self.paper_chunks_map = self._build_paper_chunks_map()

    def _build_local_entity_index(self, backend_dir: str):
        """Index local biological entities from data/entities/ and sample_kg.json"""
        entities_dir = os.path.join(backend_dir, "data", "entities")
        if os.path.exists(entities_dir):
            entity_files = glob.glob(os.path.join(entities_dir, "*.jsonl"))
            for file_path in entity_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        for line in f:
                            record = json.loads(line)
                            pid = str(record.get("paper_id", ""))
                            if not pid:
                                continue
                            for cat in ["organism", "assay", "gene", "mission", "experimenttype", "outcome"]:
                                for item in record.get(cat, []):
                                    if item and isinstance(item, str):
                                        clean = item.strip().lower()
                                        if clean:
                                            self.local_entity_to_papers[clean].add(pid)
                                            self.local_paper_to_entities[pid].add(clean)
                except Exception as e:
                    print(f"Notice reading entity file {file_path}: {e}")

        # Also load sample_kg.json
        sample_kg = os.path.join(backend_dir, "sample_kg.json")
        if os.path.exists(sample_kg):
            try:
                with open(sample_kg, "r", encoding="utf-8") as f:
                    kg_data = json.load(f)
                    pid = str(kg_data.get("paper_id", "paper50"))
                    for ent in kg_data.get("entities", []):
                        name = ent.get("name", "").strip().lower()
                        if name:
                            self.local_entity_to_papers[name].add(pid)
                            self.local_paper_to_entities[pid].add(name)
            except Exception:
                pass

        print(f"SUCCESS: Local entity index ready with {len(self.local_entity_to_papers)} unique entities")
    
    def _build_paper_chunks_map(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build a mapping from paper_id to list of chunks for faster lookup"""
        paper_map = defaultdict(list)
        for i, chunk in enumerate(self.chunks):
            paper_id = str(chunk.get("paper_id", f"unknown_{i}"))
            paper_map[paper_id].append({
                "chunk_id": chunk.get("chunk_id", i),
                "text": chunk.get("text", ""),
                "page_num": chunk.get("page_num", 1),
                "index": i
            })
        return dict(paper_map)
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract potential entity names from the query for knowledge graph search"""
        # Look for capitalized words and common scientific terms
        words = re.findall(r'\b[A-Z][a-zA-Z0-9_-]*\b', query)
        scientific_terms = re.findall(r'\b(?:gene|protein|cell|dna|rna|mitochondria|immune|microgravity|space|mission|experiment|assay|organism|outcome|radiation|bone|vertebrae|mus musculus|arabidopsis|iss)\b', query.lower())
        entities = list(set([w.lower() for w in words + scientific_terms if len(w) > 2]))
        return entities
    
    def _search_neo4j_entities(self, query: str, entities: List[str]) -> List[Dict[str, Any]]:
        """Search Neo4j or local entity index for entities and related papers"""
        # Try live Neo4j if healthy
        if self.driver and is_driver_healthy():
            try:
                with self.driver.session() as session:
                    cypher_query = """
                    MATCH (p:Paper)-[r1]-(e1)-[r2]-(e2)-[r3]-(p2:Paper)
                    WHERE ANY(entity IN $entities WHERE 
                        toLower(e1.name) CONTAINS toLower(entity) OR 
                        toLower(e2.name) CONTAINS toLower(entity) OR
                        toLower(p.title) CONTAINS toLower(entity) OR
                        toLower(p2.title) CONTAINS toLower(entity)
                    )
                    WITH DISTINCT p, p2, 
                        collect(DISTINCT e1.name) as entities1,
                        collect(DISTINCT e2.name) as entities2,
                        collect(DISTINCT type(r1)) as rel_types1,
                        collect(DISTINCT type(r2)) as rel_types2,
                        collect(DISTINCT type(r3)) as rel_types3
                    RETURN DISTINCT p.paper_id as paper_id, 
                        p.title as title,
                        entities1 + entities2 as all_entities,
                        rel_types1 + rel_types2 + rel_types3 as all_relationships,
                        size(entities1 + entities2) as entity_count
                    ORDER BY entity_count DESC
                    LIMIT 20
                    """
                    result = session.run(cypher_query, entities=entities)
                    papers = []
                    for record in result:
                        papers.append({
                            "paper_id": str(record["paper_id"]),
                            "title": record.get("title", f"Paper {record['paper_id']}"),
                            "entities": record["all_entities"],
                            "relationships": record["all_relationships"],
                            "entity_count": record["entity_count"],
                            "source": "neo4j"
                        })
                    return papers
            except Exception as e:
                print(f"Neo4j search fallback: {e}")

        # Local entity graph search fallback
        papers = []
        paper_score = defaultdict(int)
        paper_entities = defaultdict(list)
        for ent in entities:
            for stored_ent, pids in self.local_entity_to_papers.items():
                if ent in stored_ent or stored_ent in ent:
                    for pid in pids:
                        paper_score[pid] += 1
                        paper_entities[pid].append(stored_ent)
        
        for pid, score in sorted(paper_score.items(), key=lambda x: x[1], reverse=True)[:20]:
            papers.append({
                "paper_id": pid,
                "title": f"Paper {pid}",
                "entities": list(set(paper_entities[pid])),
                "relationships": ["STUDIES", "AFFECTS"],
                "entity_count": score,
                "source": "entity_graph"
            })
        return papers
    
    def _get_related_papers_from_neo4j(self, paper_ids: List[str]) -> List[Dict[str, Any]]:
        """Get papers related to the given paper IDs through Neo4j or shared entity relationships"""
        if not paper_ids:
            return []

        if self.driver and is_driver_healthy():
            try:
                with self.driver.session() as session:
                    cypher_query = """
                    MATCH (p:Paper)-[r]-(e)-[r2]-(p2:Paper)
                    WHERE p.paper_id IN $paper_ids
                    WITH DISTINCT p2, 
                        collect(DISTINCT e.name) as shared_entities,
                        collect(DISTINCT type(r)) as relationship_types
                    RETURN p2.paper_id as paper_id,
                        p2.title as title,
                        shared_entities,
                        relationship_types,
                        size(shared_entities) as shared_entity_count
                    ORDER BY shared_entity_count DESC
                    LIMIT 15
                    """
                    result = session.run(cypher_query, paper_ids=paper_ids)
                    related_papers = []
                    for record in result:
                        related_papers.append({
                            "paper_id": str(record["paper_id"]),
                            "title": record.get("title", f"Paper {record['paper_id']}"),
                            "shared_entities": record["shared_entities"],
                            "relationship_types": record["relationship_types"],
                            "shared_entity_count": record["shared_entity_count"],
                            "source": "neo4j_related"
                        })
                    return related_papers
            except Exception as e:
                print(f"Neo4j related papers fallback: {e}")

        # Local multi-hop entity co-occurrence
        related = []
        seed_entities = set()
        for pid in paper_ids:
            seed_entities.update(self.local_paper_to_entities.get(str(pid), set()))
        
        candidate_papers = defaultdict(list)
        for ent in seed_entities:
            for other_pid in self.local_entity_to_papers.get(ent, set()):
                if other_pid not in paper_ids:
                    candidate_papers[other_pid].append(ent)
        
        for other_pid, shared in sorted(candidate_papers.items(), key=lambda x: len(x[1]), reverse=True)[:15]:
            related.append({
                "paper_id": other_pid,
                "title": f"Paper {other_pid}",
                "shared_entities": shared,
                "relationship_types": ["SHARED_TARGET", "INTERACTS_WITH"],
                "shared_entity_count": len(shared),
                "source": "entity_graph_related"
            })
        return related
    
    def search_chunks(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks using FAISS"""
        if not self.index or not self.model or not self.chunks:
            return []
        
        try:
            # Encode query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            distances, indices = self.index.search(query_embedding, top_k)
            
            results = []
            for idx, score in zip(indices[0], distances[0]):
                if 0 <= idx < len(self.chunks):
                    chunk_info = self.chunks[idx]
                    results.append({
                        "score": float(score),
                        "paper_id": str(chunk_info.get("paper_id", "unknown")),
                        "chunk_id": str(chunk_info.get("chunk_id", idx)),
                        "text": chunk_info.get("text", ""),
                        "page_num": chunk_info.get("page_num", 1)
                    })
            
            return results
        except Exception as e:
            print(f"Error in search_chunks: {e}")
            return []
    
    def enhanced_search_chunks(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Enhanced search that combines FAISS dense retrieval with Graph RAG entity connections"""
        print(f"Enhanced search for: '{query}'")
        
        # Step 1: Get FAISS semantic results
        faiss_results = self.search_chunks(query, top_k=top_k * 2)
        print(f"FAISS found {len(faiss_results)} chunks")
        
        # Step 2: Extract entities and search knowledge graph
        entities = self._extract_entities_from_query(query)
        print(f"Extracted entities: {entities}")
        
        graph_papers = self._search_neo4j_entities(query, entities)
        print(f"Graph search found {len(graph_papers)} candidate papers")
        
        # Step 3: Multi-hop related papers based on FAISS seed papers
        faiss_paper_ids = [r["paper_id"] for r in faiss_results]
        related_papers = self._get_related_papers_from_neo4j(faiss_paper_ids)
        print(f"Found {len(related_papers)} multi-hop related papers")
        
        # Step 4: Assemble results with Graph-boosted diversity
        enhanced_results = []
        for result in faiss_results:
            enhanced_results.append({
                "score": result["score"],
                "paper_id": result["paper_id"],
                "chunk_id": result["chunk_id"],
                "text": result["text"],
                "page_num": result["page_num"],
                "source": "faiss",
                "neo4j_boost": 0,
                "paper_rank": 1
            })
        
        # Add diverse chunks from graph-connected papers
        graph_paper_ids = set(p["paper_id"] for p in graph_papers + related_papers)
        faiss_paper_ids_set = set(faiss_paper_ids)
        
        for paper_id in graph_paper_ids:
            if paper_id not in faiss_paper_ids_set and paper_id in self.paper_chunks_map:
                chunks = self.paper_chunks_map[paper_id]
                for i, chunk in enumerate(chunks[:2]):
                    if len(enhanced_results) < top_k * 1.5:
                        enhanced_results.append({
                            "score": 0.5,
                            "paper_id": paper_id,
                            "chunk_id": chunk["chunk_id"],
                            "text": chunk["text"],
                            "page_num": chunk["page_num"],
                            "source": "graph_diversity",
                            "neo4j_boost": 1,
                            "paper_rank": i + 1
                        })
        
        enhanced_results.sort(key=lambda x: x["score"], reverse=True)
        final_results = enhanced_results[:top_k]
        return final_results
    
    def generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate cited answer based on retrieved chunks"""
        snippets = []
        citations = []
        
        for chunk in chunks:
            snippet_text = f"[{chunk['paper_id']}:{chunk['page_num']}] {chunk['text']}"
            snippets.append(snippet_text)
            citations.append({
                "paper_id": chunk['paper_id'],
                "page_num": chunk['page_num'],
                "score": chunk['score']
            })
        
        try:
            gemini_response = qa(query, snippets)
            answer = gemini_response.get("answer", "Unable to generate answer")
        except Exception as e:
            print(f"Error generating answer: {e}")
            answer = "Error generating answer. Please try again."
        
        return {
            "answer": answer,
            "citations": citations,
            "chunks_used": len(chunks)
        }
    
    def process_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Complete RAG pipeline: enhanced hybrid search + cited answer generation"""
        if not self.index or not self.model or not self.chunks:
            return {
                "query": query,
                "answer": "RAG system not fully initialized. Please verify FAISS index and embedding model.",
                "citations": [],
                "chunks_used": 0,
                "error": "Missing RAG components"
            }
        
        chunks = self.enhanced_search_chunks(query, top_k)
        if not chunks:
            return {
                "query": query,
                "answer": "No relevant information found for your query in the NASA bioscience dataset.",
                "citations": [],
                "chunks_used": 0
            }
        
        answer_data = self.generate_answer(query, chunks)
        unique_papers = len(set(chunk["paper_id"] for chunk in chunks))
        graph_boost_count = sum(1 for chunk in chunks if chunk.get("neo4j_boost", 0) > 0)
        
        return {
            "query": query,
            "answer": answer_data["answer"],
            "citations": answer_data["citations"],
            "chunks_used": answer_data["chunks_used"],
            "retrieved_chunks": chunks,
            "diversity_metrics": {
                "unique_papers": unique_papers,
                "neo4j_boosted_chunks": graph_boost_count,
                "search_method": "enhanced_faiss_graph_rag"
            }
        }

# Global RAG service singleton
rag_service = None

def get_rag_service() -> RAGService:
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service
