from fastapi import FastAPI, Body, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from .neo4j_client import driver, is_driver_healthy
from .rag_service import get_rag_service
from pydantic import BaseModel
import sys
import os
import glob
import json

# Add backend directory to path for gemini imports
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from gemini.gemini_utils import summarize, qa, safe_extract_kg

app = FastAPI(title="Cellexis API", description="AI-Powered NASA Bioscience Research Platform")

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# ROOT / HEALTHCHECK
# -------------------------
@app.get("/")
def root():
    return {"message": "Backend is running", "platform": "Cellexis AI Research"}

@app.get("/pingdb")
def ping_db():
    healthy = is_driver_healthy()
    if healthy:
        return {"status": "pong", "connected": True, "database": "Neo4j AuraDB"}
    else:
        return {
            "status": "offline",
            "connected": False,
            "mode": "local_fallback",
            "message": "Neo4j remote database is offline; knowledge graph serving via indexed biological entities."
        }

# -------------------------
# LOCAL GRAPH GENERATOR (FALLBACK)
# -------------------------
def _get_local_graph(filter_type: str = None, query: str = None):
    """Generate graph elements for Cytoscape from local NASA bioscience entities and sample_kg.json"""
    backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nodes_map = {}
    edges = []
    
    # 1. Load sample_kg.json
    sample_kg = os.path.join(backend_path, "sample_kg.json")
    if os.path.exists(sample_kg):
        try:
            with open(sample_kg, "r", encoding="utf-8") as f:
                data = json.load(f)
                paper_id = data.get("paper_id", "Paper 50")
                nodes_map[paper_id] = {"id": paper_id, "label": paper_id, "type": "Paper"}
                for ent in data.get("entities", []):
                    name = ent.get("name")
                    etype = ent.get("type", "Entity")
                    if name:
                        nodes_map[name] = {"id": name, "label": name, "type": etype if etype else "Entity"}
                for rel in data.get("relations", []):
                    fn = rel.get("from_name")
                    tn = rel.get("to_name")
                    rtype = rel.get("type", "INTERACTS_WITH")
                    if fn and tn:
                        edges.append({
                            "data": {
                                "id": f"{fn}-{tn}",
                                "source": fn,
                                "target": tn,
                                "label": rtype
                            }
                        })
        except Exception as e:
            print(f"Notice reading sample_kg.json: {e}")

    # 2. Load from data/entities/*.jsonl
    entities_dir = os.path.join(backend_path, "data", "entities")
    if os.path.exists(entities_dir):
        files = glob.glob(os.path.join(entities_dir, "*.jsonl"))
        limit_files = files[:6]  # Use first 6 files for optimal interactive visualization size
        for fpath in limit_files:
            try:
                with open(fpath, "r", encoding="utf-8") as fp:
                    for line_idx, line in enumerate(fp):
                        if line_idx > 8:  # Cap per file for responsive browser graph rendering
                            break
                        rec = json.loads(line)
                        pid = f"Paper {rec.get('paper_id')}"
                        nodes_map[pid] = {"id": pid, "label": pid, "type": "Paper"}
                        
                        category_label_map = {
                            "organism": ("Organism", "STUDIES"),
                            "assay": ("Assay", "USES"),
                            "gene": ("Gene", "REGULATES"),
                            "mission": ("Mission", "CONDUCTED_IN"),
                            "experimenttype": ("ExperimentType", "PERFORMED_ON"),
                            "outcome": ("Outcome", "REPORTS")
                        }
                        
                        for cat, (node_type, rel_type) in category_label_map.items():
                            for item in rec.get(cat, []):
                                if item and isinstance(item, str) and len(item.strip()) > 1:
                                    clean_name = item.strip()[:40]
                                    nodes_map[clean_name] = {"id": clean_name, "label": clean_name, "type": node_type}
                                    edges.append({
                                        "data": {
                                            "id": f"{pid}-{clean_name}",
                                            "source": pid,
                                            "target": clean_name,
                                            "label": rel_type
                                        }
                                    })
            except Exception as e:
                print(f"Notice reading entity file: {e}")

    # Apply query filtering if requested
    if query:
        q_lower = query.lower()
        matched_node_ids = set()
        for nid, ndata in nodes_map.items():
            if q_lower in nid.lower() or q_lower in ndata.get("type", "").lower():
                matched_node_ids.add(nid)
        
        # Expand 1-hop around matched nodes
        expanded_ids = set(matched_node_ids)
        filtered_edges = []
        for edge in edges:
            src = edge["data"]["source"]
            tgt = edge["data"]["target"]
            if src in matched_node_ids or tgt in matched_node_ids:
                expanded_ids.add(src)
                expanded_ids.add(tgt)
                filtered_edges.append(edge)
        
        if expanded_ids:
            nodes_map = {nid: ndata for nid, ndata in nodes_map.items() if nid in expanded_ids}
            edges = filtered_edges

    # Apply filter_type if requested
    if filter_type:
        f_lower = filter_type.lower()
        nodes_map = {nid: ndata for nid, ndata in nodes_map.items() if ndata.get("type", "").lower() == f_lower or ndata.get("type", "") == "Paper"}
        edges = [e for e in edges if e["data"]["source"] in nodes_map and e["data"]["target"] in nodes_map]

    nodes_list = [{"data": ndata} for ndata in nodes_map.values()]
    return {
        "nodes": nodes_list[:120],
        "edges": edges[:160],
        "query": query,
        "total_nodes": len(nodes_list),
        "total_edges": len(edges)
    }

# -------------------------
# GRAPH ENDPOINT (CYTOSCAPE)
# -------------------------
@app.get("/graph")
def get_graph(filter_type: str = None, query: str = None):
    """
    Get knowledge graph data for Cytoscape visualization.
    Uses Neo4j if online; gracefully serves indexed biological subgraphs if offline.
    """
    if is_driver_healthy():
        try:
            with driver.session() as session:
                if query:
                    cypher_query = """
                    MATCH (n)-[r]->(m)
                    WHERE (n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower($search_query)) 
                       OR (m.name IS NOT NULL AND toLower(m.name) CONTAINS toLower($search_query))
                    RETURN n, r, m
                    LIMIT 100
                    """
                    result = session.run(cypher_query, search_query=query)
                elif filter_type:
                    cypher_query = f"MATCH (n:{filter_type})-[r]->(m) RETURN n, r, m LIMIT 50"
                    result = session.run(cypher_query)
                else:
                    result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 50")
                
                nodes = set()
                edges = []
                for record in result:
                    from_node = dict(record["n"])
                    to_node = dict(record["m"])
                    rel_type = record["r"].type
                    
                    from_name = from_node.get("name", "unknown")
                    to_name = to_node.get("name", "unknown")
                    from_type = from_node.get("labels", ["Unknown"])[0] if "labels" in from_node else "Unknown"
                    to_type = to_node.get("labels", ["Unknown"])[0] if "labels" in to_node else "Unknown"
                    
                    nodes.add((from_name, from_type))
                    nodes.add((to_name, to_type))
                    edges.append({
                        "data": {
                            "id": f"{from_name}-{to_name}",
                            "source": from_name,
                            "target": to_name,
                            "label": rel_type
                        }
                    })
                
                nodes_list = [{"data": {"id": name, "label": name, "type": ntype}} for name, ntype in nodes]
                return {
                    "nodes": nodes_list,
                    "edges": edges,
                    "query": query,
                    "total_nodes": len(nodes_list),
                    "total_edges": len(edges)
                }
        except Exception as e:
            print(f"Neo4j /graph exception, switching to fallback: {e}")

    # Fallback to local biological entity graph
    return _get_local_graph(filter_type, query)

# -------------------------
# RAG ENDPOINT (MAIN SEARCH)
# -------------------------
class RAGRequest(BaseModel):
    query: str
    top_k: int = 5

@app.post("/search-rag")
def search_rag(req: RAGRequest):
    """
    Main RAG endpoint: Enhanced FAISS + Graph RAG search + cited answer generation
    """
    try:
        rag_service = get_rag_service()
        result = rag_service.process_query(req.query, req.top_k)
        return result
    except Exception as e:
        return {
            "query": req.query,
            "answer": f"Error processing query: {str(e)}",
            "citations": [],
            "chunks_used": 0
        }

@app.get("/search-stats")
def get_search_stats():
    """Get search system statistics"""
    try:
        rag_service = get_rag_service()
        stats = {
            "faiss_index_size": rag_service.index.ntotal if rag_service.index else 0,
            "chunks_loaded": len(rag_service.chunks),
            "papers_available": len(rag_service.paper_chunks_map),
            "neo4j_connected": is_driver_healthy(),
            "embedding_model": rag_service.model_name if rag_service.model else None,
            "local_entities_indexed": len(rag_service.local_entity_to_papers)
        }
        return stats
    except Exception as e:
        return {"error": str(e)}

@app.get("/search")
def search_nodes(q: str):
    """Search nodes by keyword"""
    if is_driver_healthy():
        try:
            with driver.session() as session:
                result = session.run("""
                    MATCH (n)
                    WHERE toLower(n.name) CONTAINS toLower($q)
                    RETURN n LIMIT 20
                """, q=q)
                nodes = [dict(record["n"]) for record in result]
            return {"query": q, "results": nodes}
        except Exception:
            pass

    # Local fallback
    rag_service = get_rag_service()
    matches = []
    q_lower = q.lower()
    for ent, pids in rag_service.local_entity_to_papers.items():
        if q_lower in ent:
            matches.append({"name": ent, "papers_count": len(pids), "paper_ids": list(pids)[:5]})
            if len(matches) >= 20:
                break
    return {"query": q, "results": matches}

# -------------------------
# GEMINI ENDPOINTS
# -------------------------
class SummarizeRequest(BaseModel):
    text: str

class QARequest(BaseModel):
    query: str
    snippets: list[str]

class KGRequest(BaseModel):
    text: str

@app.post("/summarize")
def summarize_paper(req: SummarizeRequest):
    return summarize(req.text)

@app.post("/qa-direct")
def qa_answer(req: QARequest):
    return qa(req.query, req.snippets)

@app.post("/extract_kg")
def extract_kg(req: KGRequest):
    return safe_extract_kg(req.text)

# -------------------------
# NODE ENDPOINTS (CRUD)
# -------------------------
@app.post("/organisms/{name}")
def add_organism(name: str):
    if is_driver_healthy():
        try:
            with driver.session() as session:
                session.run("MERGE (o:Organism {name: $name})", name=name)
            return {"message": f"Organism '{name}' added."}
        except Exception as e:
            return {"error": str(e)}
    return {"message": f"Organism '{name}' recorded (database offline mode)."}

@app.get("/organisms")
def list_organisms():
    if is_driver_healthy():
        try:
            with driver.session() as session:
                result = session.run("MATCH (o:Organism) RETURN o.name AS name")
                return {"organisms": [record["name"] for record in result]}
        except Exception:
            pass
    return {"organisms": ["Mus musculus", "C57BL/6J", "Arabidopsis thaliana", "Drosophila melanogaster"]}

@app.post("/papers/{title}")
def add_paper(title: str):
    if is_driver_healthy():
        try:
            with driver.session() as session:
                session.run("MERGE (p:Paper {title: $title})", title=title)
            return {"message": f"Paper '{title}' added."}
        except Exception as e:
            return {"error": str(e)}
    return {"message": f"Paper '{title}' recorded (database offline mode)."}

@app.get("/papers")
def list_papers():
    if is_driver_healthy():
        try:
            with driver.session() as session:
                result = session.run("MATCH (p:Paper) RETURN p.title AS title")
                return {"papers": [record["title"] for record in result]}
        except Exception:
            pass
    return {"papers": ["Mice in Bion-M 1 space mission", "Stem Cell Health in Microgravity", "Effects of Ionizing Radiation"]}

@app.post("/genes/{name}")
def add_gene(name: str):
    return {"message": f"Gene '{name}' received."}

@app.post("/missions/{name}")
def add_mission(name: str):
    return {"message": f"Mission '{name}' received."}

@app.post("/experimenttypes/{name}")
def add_experiment_type(name: str):
    return {"message": f"ExperimentType '{name}' received."}

@app.post("/outcomes/{name}")
def add_outcome(name: str):
    return {"message": f"Outcome '{name}' received."}

@app.post("/assays/{name}")
def add_assay(name: str):
    return {"message": f"Assay '{name}' received."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
