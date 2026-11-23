import google.generativeai as genai
import json
import os
import re
from dotenv import load_dotenv

# Load env variables from the correct location
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(backend_dir, ".env")
load_dotenv(env_path)

api_key = os.getenv("GEMINI_API_KEY")

# Configure Gemini with REST transport to avoid gRPC / ALTS issues
model = None
if api_key:
    try:
        genai.configure(api_key=api_key, transport='rest')
        # Preferred models in order
        for model_candidate in ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash"]:
            try:
                model = genai.GenerativeModel(model_candidate)
                break
            except Exception:
                continue
    except Exception as e:
        print(f"Warning: Could not configure Gemini: {e}")
        model = None


# 1️⃣ Summarize function
def summarize(text: str) -> dict:
    """
    Summarize a paper chunk in exactly 1 line + 3 bullets (Experiment → Result → Implication).
    With robust fallback synthesis.
    """
    if model:
        try:
            prompt = f"""
Summarize the following text in exactly 1 line + 3 concise bullets.
Bullets must follow the format: Experiment → Result → Implication.

Text:
{text}
"""
            response = model.generate_content(prompt, request_options={"timeout": 12})
            if response and response.text:
                return {"summary": response.text.strip()}
        except Exception as e:
            print(f"Gemini summarize error (using fallback): {e}")

    # Fallback extractive summary
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    first_sent = sentences[0] if sentences else "NASA bioscience research investigation."
    exp_sent = next((s for s in sentences if any(w in s.lower() for w in ['experiment', 'treated', 'investigated', 'conducted', 'assigned', 'dose', 'exposed'])), first_sent)
    res_sent = next((s for s in sentences if any(w in s.lower() for w in ['found', 'result', 'observed', 'decrease', 'increase', 'showed', 'demonstrated'])), (sentences[1] if len(sentences) > 1 else first_sent))
    imp_sent = next((s for s in sentences if any(w in s.lower() for w in ['suggest', 'conclude', 'implication', 'indicate', 'therefore', 'may'])), (sentences[-1] if len(sentences) > 2 else first_sent))

    fallback_summary = (
        f"{first_sent}\n"
        f"• Experiment: {exp_sent}\n"
        f"• Result: {res_sent}\n"
        f"• Implication: {imp_sent}"
    )
    return {"summary": fallback_summary}


# 2️⃣ Question Answering (QA) function
def qa(query: str, snippets: list) -> dict:
    """
    Answer a question using only the provided snippets.
    Always include citations in [paper_id:page_num] format.
    With robust extractive synthesis fallback.
    """
    snippets_text = "\n".join(snippets)

    if model:
        try:
            prompt = f"""
Answer the question using ONLY the following snippets.
Always cite sources using [paper_id:page_num].

Question: {query}
Snippets:
{snippets_text}
"""
            response = model.generate_content(prompt, request_options={"timeout": 12})
            if response and response.text:
                return {"answer": response.text.strip()}
        except Exception as e:
            print(f"Gemini QA error (using fallback): {e}")

    # Fallback synthesis directly from retrieved snippets
    query_words = set(re.findall(r'\b\w{3,}\b', query.lower())) - {'what', 'when', 'where', 'which', 'how', 'does', 'effect', 'effects', 'affect', 'show', 'paper'}
    
    selected_points = []
    seen_citations = set()

    for snip in snippets:
        match = re.match(r'\[([^\]]+)\]\s*(.*)', snip, re.DOTALL)
        if match:
            citation, text = match.group(1), match.group(2)
        else:
            citation, text = "source", snip

        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 25]
        for sentence in sentences:
            s_lower = sentence.lower()
            matches = sum(1 for w in query_words if w in s_lower)
            if matches > 0 and sentence not in selected_points:
                selected_points.append((matches, f"{sentence} [{citation}]"))
                seen_citations.add(citation)

    if selected_points:
        # Sort by relevance matches
        selected_points.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [p[1] for p in selected_points[:4]]
        synthesized_answer = " ".join(top_sentences)
    else:
        # If no strict keyword matches, extract the top sentences from the first 2 snippets
        fallback_lines = []
        for snip in snippets[:2]:
            match = re.match(r'\[([^\]]+)\]\s*(.*)', snip, re.DOTALL)
            citation = match.group(1) if match else "1"
            text = match.group(2) if match else snip
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 30]
            if sentences:
                fallback_lines.append(f"{sentences[0]} [{citation}]")
        synthesized_answer = " ".join(fallback_lines) if fallback_lines else "NASA bioscience publications document relevant physiological and cellular changes under microgravity and radiation conditions."

    return {"answer": synthesized_answer}


# 3️⃣ Knowledge Graph Extraction
def extract_kg(text: str) -> dict:
    """
    Extract entities (Organism, ExperimentType, Assay, Outcome, Mission)
    and relations (STUDIES, USES, PERFORMED_ON, REPORTS) from text.
    """
    if model:
        try:
            prompt = f"""
Extract entities (Organism, ExperimentType, Assay, Outcome, Mission)
and relations (STUDIES, USES, PERFORMED_ON, REPORTS) from the text below.
Return strictly in JSON with two keys: "entities", "relations".

Text:
{text}
"""
            response = model.generate_content(prompt, request_options={"timeout": 12})
            if response and response.text:
                return {"kg_json": response.text.strip()}
        except Exception as e:
            print(f"Gemini extract_kg error (using fallback): {e}")

    # Fallback rule-based entity extraction
    entities = []
    relations = []
    
    # Common entity patterns in NASA bioscience
    organisms = set(re.findall(r'\b(?:Mus musculus|C57BL/6J|mouse|mice|Arabidopsis thaliana|Drosophila|human|rat)\b', text, re.IGNORECASE))
    missions = set(re.findall(r'\b(?:ISS|International Space Station|Expedition \d+|Bion-M1|STS-\d+|SpaceX-\d+)\b', text, re.IGNORECASE))
    assays = set(re.findall(r'\b(?:micro-CT|RNA-seq|PCR|qPCR|electrophoresis|histology|western blot|flow cytometry)\b', text, re.IGNORECASE))
    experiment_types = set(re.findall(r'\b(?:microgravity|radiation|spaceflight|hindlimb suspension|cosmic rays|hypergravity)\b', text, re.IGNORECASE))

    for o in organisms:
        entities.append({"type": "Organism", "name": o})
    for m in missions:
        entities.append({"type": "Mission", "name": m})
    for a in assays:
        entities.append({"type": "Assay", "name": a})
    for e in experiment_types:
        entities.append({"type": "ExperimentType", "name": e})

    # Basic relationship synthesis
    for o in organisms:
        for e in experiment_types:
            relations.append({"from_name": e, "from_type": "ExperimentType", "to_name": o, "to_type": "Organism", "type": "PERFORMED_ON"})
        for m in missions:
            relations.append({"from_name": m, "from_type": "Mission", "to_name": o, "to_type": "Organism", "type": "STUDIES"})

    return {"kg_json": json.dumps({"entities": entities, "relations": relations})}


# 4️⃣ Safe KG extraction (with validation)
def safe_extract_kg(text: str) -> dict:
    """
    Ensures KG JSON is valid. Returns empty lists or parsed fallback if parsing fails.
    """
    try:
        kg = extract_kg(text)
        clean_json = kg["kg_json"].replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_json)
        return {
            "entities": parsed.get("entities", []),
            "relations": parsed.get("relations", [])
        }
    except Exception:
        return {"entities": [], "relations": []}
