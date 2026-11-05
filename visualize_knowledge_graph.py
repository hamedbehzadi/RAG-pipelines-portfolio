"""Create an interactive PyVis visualization for the extracted knowledge graph."""

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_GRAPH = ROOT / "knowledge_graph_gold.json"
DEFAULT_OUTPUT = ROOT / "knowledge_graph_visualization.html"

TYPE_COLORS = {
    "Disease": "#d94841",
    "Symptom": "#f28e2b",
    "RiskFactor": "#8c6d31",
    "DiagnosticMeasure": "#4e79a7",
    "Treatment": "#59a14f",
    "MedicationClass": "#9c755f",
    "Procedure": "#76b7b2",
    "Population": "#b07aa1",
    "Outcome": "#edc948",
    "Other": "#bab0ac",
}


def normalize(text):
    return " ".join(str(text).casefold().split())


def load_graph(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    triples = payload.get("triples", [])
    if not isinstance(triples, list):
        raise ValueError(f"{path} does not contain a 'triples' list")
    entities = payload.get("entities", [])
    if not isinstance(entities, list):
        entities = []
    return entities, triples


def entity_index(entities, triples):
    index = {}
    for entity in entities:
        if not isinstance(entity, dict) or not entity.get("name"):
            continue
        entity_id = entity.get("id") or normalize(entity["name"])
        index[normalize(entity_id)] = {
            "name": entity["name"],
            "types": list(entity.get("types") or ["Other"]),
        }

    for fact in triples:
        for side in ("subject", "object"):
            name = fact.get(side, "").strip()
            if not name:
                continue
            entity = index.setdefault(normalize(name), {"name": name, "types": []})
            entity_type = fact.get(f"{side}_type") or "Other"
            if entity_type not in entity["types"]:
                entity["types"].append(entity_type)
    return index


def relation_matches(fact, relations):
    return not relations or fact.get("relation") in relations


def entity_matches(fact, terms):
    if not terms:
        return True
    subject = normalize(fact.get("subject", ""))
    object_ = normalize(fact.get("object", ""))
    return any(term in subject or term in object_ for term in terms)


def filtered_triples(triples, relations, entity_terms):
    terms = [normalize(term) for term in entity_terms]
    return [
        fact for fact in triples
        if relation_matches(fact, relations) and entity_matches(fact, terms)
    ]


def endpoints(fact):
    return normalize(fact.get("subject", "")), normalize(fact.get("object", ""))


def choose_focus_node(triples, focus):
    focus_key = normalize(focus)
    degree = Counter()
    names = {}
    for fact in triples:
        for side in ("subject", "object"):
            node_id = normalize(fact.get(side, ""))
            if not node_id:
                continue
            degree[node_id] += 1
            names.setdefault(node_id, fact.get(side, "").strip())

    exact = [node_id for node_id in degree if node_id == focus_key]
    candidates = exact or [node_id for node_id in degree if focus_key in node_id]
    if not candidates:
        raise SystemExit(f"No node matched focus {focus!r}.")

    selected = max(candidates, key=lambda node_id: (degree[node_id], -len(node_id)))
    if len(candidates) > 1:
        print(
            f"Focus {focus!r} matched {len(candidates)} nodes; "
            f"using {names.get(selected, selected)!r}."
        )
    return selected


def focus_triples(triples, focus, depth):
    if not focus:
        return triples
    if depth < 1:
        raise SystemExit("--depth must be at least 1.")

    focus_id = choose_focus_node(triples, focus)
    adjacency = defaultdict(set)
    for fact in triples:
        subject_id, object_id = endpoints(fact)
        if not subject_id or not object_id:
            continue
        adjacency[subject_id].add(object_id)
        adjacency[object_id].add(subject_id)

    distances = {focus_id: 0}
    frontier = {focus_id}
    for distance in range(1, depth + 1):
        next_frontier = set()
        for node_id in frontier:
            next_frontier.update(adjacency[node_id])
        next_frontier -= distances.keys()
        for node_id in next_frontier:
            distances[node_id] = distance
        frontier = next_frontier

    return [
        fact for fact in triples
        if endpoints(fact)[0] in distances and endpoints(fact)[1] in distances
    ]


def rank_triples(triples, max_edges):
    if max_edges is None or max_edges <= 0 or len(triples) <= max_edges:
        return triples

    degree = Counter()
    for fact in triples:
        degree[normalize(fact.get("subject", ""))] += 1
        degree[normalize(fact.get("object", ""))] += 1

    return sorted(
        triples,
        key=lambda fact: (
            degree[normalize(fact.get("subject", ""))]
            + degree[normalize(fact.get("object", ""))],
            fact.get("page", 0),
        ),
        reverse=True,
    )[:max_edges]


def node_title(name, types, degree):
    escaped_types = ", ".join(html.escape(kind) for kind in types) or "Other"
    return (
        f"<b>{html.escape(name)}</b><br>"
        f"Type: {escaped_types}<br>"
        f"Connected relationships: {degree}"
    )


def edge_title(fact):
    bits = [
        f"<b>{html.escape(fact.get('relation', 'related_to'))}</b>",
        f"{html.escape(fact.get('subject', ''))} -> {html.escape(fact.get('object', ''))}",
    ]
    if fact.get("page"):
        bits.append(f"Page: {html.escape(str(fact['page']))}")
    if fact.get("evidence"):
        bits.append(f"Evidence: {html.escape(fact['evidence'])}")
    return "<br>".join(bits)


def build_network(entities, triples, output_path, height, width, show_labels):
    try:
        from pyvis.network import Network
    except ImportError as error:
        raise SystemExit(
            "PyVis is not installed. Run: python -m pip install -r requirements-kg.txt"
        ) from error

    entity_lookup = entity_index(entities, triples)
    degree = Counter()
    edge_counts = defaultdict(int)
    for fact in triples:
        degree[normalize(fact.get("subject", ""))] += 1
        degree[normalize(fact.get("object", ""))] += 1

    network = Network(
        height=height,
        width=width,
        directed=True,
        bgcolor="#ffffff",
        font_color="#1f2933",
        cdn_resources="in_line",
    )
    network.barnes_hut(gravity=-30000, central_gravity=0.2, spring_length=150)

    node_ids = set()
    for fact in triples:
        for side in ("subject", "object"):
            name = fact.get(side, "").strip()
            if not name:
                continue
            node_id = normalize(name)
            if node_id in node_ids:
                continue
            node_ids.add(node_id)
            entity = entity_lookup.get(node_id, {"name": name, "types": ["Other"]})
            types = entity.get("types") or ["Other"]
            primary_type = types[0]
            network.add_node(
                node_id,
                label=entity.get("name", name),
                title=node_title(entity.get("name", name), types, degree[node_id]),
                group=primary_type,
                color=TYPE_COLORS.get(primary_type, TYPE_COLORS["Other"]),
                value=max(1, degree[node_id]),
            )

    for fact in triples:
        subject_id = normalize(fact.get("subject", ""))
        object_id = normalize(fact.get("object", ""))
        if not subject_id or not object_id:
            continue
        edge_key = (subject_id, object_id, fact.get("relation", "related_to"))
        edge_counts[edge_key] += 1
        label = fact.get("relation", "related_to") if show_labels else ""
        network.add_edge(
            subject_id,
            object_id,
            label=label,
            title=edge_title(fact),
            arrows="to",
            smooth={"type": "curvedCW", "roundness": min(0.45, edge_counts[edge_key] * 0.08)},
            color="#667085",
        )

    network.set_options("""
var options = {
  "nodes": {
    "borderWidth": 1,
    "font": {"size": 16, "face": "Inter, Arial"}
  },
  "edges": {
    "font": {"size": 11, "align": "middle"},
    "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
    "color": {"inherit": false}
  },
  "interaction": {
    "hover": true,
    "navigationButtons": true,
    "keyboard": true
  },
  "physics": {
    "stabilization": {"iterations": 120}
  }
}
""")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    network.write_html(str(output_path), notebook=False, open_browser=False)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize extracted knowledge graph triples with PyVis."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_GRAPH,
                        help="Path to knowledge_graph_*.json cache.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="HTML file to write.")
    parser.add_argument("--max-edges", type=int, default=80,
                        help="Maximum relationships to render; use 0 for all.")
    parser.add_argument("--relation", action="append", default=[],
                        help="Only include this relation. Can be repeated.")
    parser.add_argument("--entity", action="append", default=['spirometry'],
                        help="Only include triples where subject/object contains this text.")
    parser.add_argument("--focus",
                        help="Show one connected neighborhood around the best matching node.")
    parser.add_argument("--depth", type=int, default=1,
                        help="Number of graph hops to include with --focus.")
    parser.add_argument("--height", default="850px", help="Canvas height, e.g. 850px or 100vh.")
    parser.add_argument("--width", default="100%", help="Canvas width.")
    parser.add_argument("--hide-edge-labels", action="store_true",
                        help="Hide relationship names on edges for dense graphs.")
    return parser.parse_args()


def main():
    args = parse_args()
    entities, triples = load_graph(args.input)
    triples = filtered_triples(triples, set(args.relation), args.entity)
    triples = focus_triples(triples, args.focus, args.depth)
    selected = rank_triples(triples, args.max_edges)
    if not selected:
        raise SystemExit("No triples matched the requested filters.")

    build_network(
        entities=entities,
        triples=selected,
        output_path=args.output,
        height=args.height,
        width=args.width,
        show_labels=not args.hide_edge_labels,
    )
    print(
        f"Wrote {args.output} with {len(selected)} relationships "
        f"from {len(triples)} matching triples."
    )


if __name__ == "__main__":
    main()
