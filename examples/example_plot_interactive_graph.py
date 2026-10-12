"""Render the neuromaps-prime transformation graph as an interactive HTML page.

Run from the repository root:

    uv run examples/example_plot_interactive_graph.py \
        [-o <output_file>] [--yaml-file <yaml_file>]
"""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     # Uses this repo's checkout (editable) via ${PWD}; swap in a pinned
#     # release (e.g. "neuromaps-prime==0.2.0") if you prefer.
#     "neuromaps-prime @ file:///${PWD}",
#     "networkx>=3.6.1",
#     "pyvis>=0.3.2",
# ]
# ///

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import networkx as nx
from pyvis.network import Network

from neuromaps_prime.graph import NeuromapsGraph

_COLOR_BY_SPECIES = {
    "macaque": "#DEAC52",  # Amber
    "human": "#95B46A",  # Sage green
    "chimpanzee": "#C8D0D5",  # Slate gray
    "marmoset": "#e98c8cff",  # Red
}

_COLOR_BY_EDGE_TYPE = {
    "surface_to_surface": "#80B2D4",  # Blue
    "volume_to_volume": "#E08E9A",  # Pink
    "both": "#A28DC7",  # Purple
}

_NODE_COUNT_FIELDS = (
    "surfaces",
    "volumes",
    "surface_annotations",
    "volume_annotations",
)

_RESOURCE_RE = re.compile(r"(\d+)([a-zA-Z]+)")

_PYVIS_OPTIONS = """
const options = {
  "nodes": {
    "borderWidth": 2,
    "borderWidthSelected": 4,
    "font": {
      "size": 20,
      "face": "Nunito, sans-serif"
    }
  },
  "edges": {
    "color": {
      "inherit": true
    },
    "font": {
      "background": "#ffffff",
      "align": "middle",
      "size": 14,
      "face": "Nunito, sans-serif"
    },
    "selfReference": {
      "angle": 0.7853981633974483
    },
    "smooth": {
      "type": "discrete",
      "forceDirection": "none",
      "roundness": 0.3
    }
  },
  "physics": {
    "barnesHut": {
      "gravitationalConstant": -100000,
      "centralGravity": 0.3,
      "springLength": 450,
      "springConstant": 0.03,
      "damping": 0.09,
      "avoidOverlap": 1.0
    },
    "minVelocity": 0.75
  },
  "layout": {
    "randomSeed": 678
  }
}
"""

_EXPORT_SVG_BUTTON = """                        <div class="col-2 pb-2">
                            <button type="button" class="btn btn-outline-secondary btn-block" onclick="exportSVG();">Export SVG</button>
                        </div>"""

_EXPORT_SVG_JS = """              // ---------------------------------------------------------------
              // Export the rendered graph as a true vector SVG (for Inkscape).
              // vis-network draws to <canvas>, so we reconstruct equivalent SVG
              // from the settled layout: node positions/sizes + edge geometry.
              // ---------------------------------------------------------------
              function _esc(s) {
                return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;")
                                .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
              }
              function _darken(hex, f) {
                var c = (hex || "#888888").replace("#", "");
                if (c.length === 3) c = c.split("").map(function (x) { return x + x; }).join("");
                function h(x) { return ("0" + Math.max(0, Math.min(255, x)).toString(16)).slice(-2); }
                return "#" + h(Math.round(parseInt(c.slice(0, 2), 16) * f))
                           + h(Math.round(parseInt(c.slice(2, 4), 16) * f))
                           + h(Math.round(parseInt(c.slice(4, 6), 16) * f));
              }
              function exportSVG() {
                var ids = nodes.getIds();
                var nodeObjs = nodes.get({ returnType: "Object" });
                var edgeArr = edges.get();
                var pos = network.getPositions(ids);

                // node centers (R) + radii from the actual rendered bounding boxes
                var R = {}, C = {};
                var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
                ids.forEach(function (id) {
                  var bb = network.getBoundingBox(id);
                  R[id] = Math.max((bb.right - bb.left) / 2, (bb.bottom - bb.top) / 2);
                  C[id] = { x: pos[id].x, y: pos[id].y };
                  minX = Math.min(minX, bb.left);  maxX = Math.max(maxX, bb.right);
                  minY = Math.min(minY, bb.top);   maxY = Math.max(maxY, bb.bottom);
                });
                var pad = 100;
                var vbX = minX - pad, vbY = minY - pad;
                var vbW = (maxX - minX) + 2 * pad, vbH = (maxY - minY) + 2 * pad;

                var edgeSvg = [], arrowSvg = [], labelSvg = [];
                edgeArr.forEach(function (e) {
                  var a = C[e.from], b = C[e.to];
                  if (!a || !b) return;
                  var cx, cy, via = null, body = network.body && network.body.edges[e.id];
                  if (body && body.edgeType && typeof body.edgeType.getViaNode === "function") {
                    try { via = body.edgeType.getViaNode(); } catch (_) { via = null; }
                  }
                  if (via && isFinite(via.x) && isFinite(via.y)) {
                    cx = via.x; cy = via.y;
                  } else {
                    var dx = b.x - a.x, dy = b.y - a.y, len = Math.sqrt(dx * dx + dy * dy) || 1;
                    cx = (a.x + b.x) / 2 + (-dy / len) * 0.2 * len;
                    cy = (a.y + b.y) / 2 + (dx / len) * 0.2 * len;
                  }
                  function trim(ctr, rad) {
                    var vx = cx - ctr.x, vy = cy - ctr.y, vl = Math.sqrt(vx * vx + vy * vy) || 1;
                    return { x: ctr.x + vx / vl * rad, y: ctr.y + vy / vl * rad };
                  }
                  var s = trim(a, R[e.from]), t = trim(b, R[e.to]);
                  var col = e.color || "#999999", w = e.width || 1;
                  edgeSvg.push('<path d="M ' + s.x + ' ' + s.y + ' Q ' + cx + ' ' + cy + ' ' + t.x + ' ' + t.y +
                               '" fill="none" stroke="' + _esc(col) + '" stroke-width="' + w + '"/>');

                  var arr = (e.arrows || "").toString(), asz = 8 + w * 2;
                  function tri(tip, dirx, diry) {
                    var nx = -diry, ny = dirx, hw = asz * 0.45;
                    var bx = tip.x - dirx * asz, by = tip.y - diry * asz;
                    return '<polygon points="' + tip.x + ',' + tip.y + ' ' +
                           (bx + nx * hw) + ',' + (by + ny * hw) + ' ' +
                           (bx - nx * hw) + ',' + (by - ny * hw) + '" fill="' + _esc(col) + '"/>';
                  }
                  if (arr.indexOf("to") !== -1) {
                    var d1x = t.x - cx, d1y = t.y - cy, d1l = Math.sqrt(d1x * d1x + d1y * d1y) || 1;
                    arrowSvg.push(tri(t, d1x / d1l, d1y / d1l));
                  }
                  if (arr.indexOf("from") !== -1) {
                    var d2x = s.x - cx, d2y = s.y - cy, d2l = Math.sqrt(d2x * d2x + d2y * d2y) || 1;
                    arrowSvg.push(tri(s, d2x / d2l, d2y / d2l));
                  }
                  if (e.label) {
                    var lp = (body && body.edgeType && typeof body.edgeType.getPoint === "function")
                             ? (function () { try { return body.edgeType.getPoint(0.5); } catch (_) { return null; } })()
                             : null;
                    var lx = lp ? lp.x : (0.25 * a.x + 0.5 * cx + 0.25 * b.x);
                    var ly = lp ? lp.y : (0.25 * a.y + 0.5 * cy + 0.25 * b.y);
                    var ang = Math.atan2(b.y - a.y, b.x - a.x) * 180 / Math.PI;
                    if (ang > 90) ang -= 180; else if (ang < -90) ang += 180;
                    var fs = 14, tw = ("" + e.label).length * fs * 0.55, th = fs * 1.3;
                    labelSvg.push('<g transform="rotate(' + ang.toFixed(3) + ' ' + lx + ' ' + ly + ')">' +
                                  '<rect x="' + (lx - tw / 2) + '" y="' + (ly - th / 2) + '" width="' + tw +
                                  '" height="' + th + '" fill="#ffffff"/><text x="' + lx + '" y="' + (ly + fs * 0.35) +
                                  '" font-family="Nunito,sans-serif" font-size="' + fs +
                                  '" font-weight="bold" text-anchor="middle" fill="#343434">' + _esc(e.label) + '</text></g>');
                  }
                });

                var nodeSvg = [];
                ids.forEach(function (id) {
                  var n = nodeObjs[id], c = C[id], r = R[id], fill = n.color || "#cccccc";
                  var lbl = ((n.label != null ? n.label : n.name) || id).toString().trim();
                  nodeSvg.push('<circle cx="' + c.x + '" cy="' + c.y + '" r="' + r + '" fill="' + _esc(fill) +
                               '" stroke="' + _esc(_darken(fill, 0.7)) + '" stroke-width="2"/>');
                  nodeSvg.push('<text x="' + c.x + '" y="' + (c.y + 7) + '" font-family="Nunito,sans-serif" font-size="20" ' +
                               'font-weight="bold" text-anchor="middle" fill="#343434">' + _esc(lbl) + '</text>');
                });

                var svg = '<?xml version="1.0" encoding="UTF-8"?>\\n' +
                  '<svg xmlns="http://www.w3.org/2000/svg" viewBox="' + vbX + ' ' + vbY + ' ' + vbW + ' ' + vbH +
                  '" width="' + vbW + '" height="' + vbH + '">\\n' +
                  '<rect x="' + vbX + '" y="' + vbY + '" width="' + vbW + '" height="' + vbH + '" fill="#ffffff"/>\\n' +
                  '<g id="edges">\\n' + edgeSvg.join("\\n") + '\\n</g>\\n' +
                  '<g id="arrowheads">\\n' + arrowSvg.join("\\n") + '\\n</g>\\n' +
                  '<g id="edge-labels">\\n' + labelSvg.join("\\n") + '\\n</g>\\n' +
                  '<g id="nodes">\\n' + nodeSvg.join("\\n") + '\\n</g>\\n' +
                  '</svg>\\n';

                var blob = new Blob([svg], { type: "image/svg+xml" });
                var url = URL.createObjectURL(blob);
                var a = document.createElement("a");
                a.href = url; a.download = "neuromaps_interactive_graph.svg";
                document.body.appendChild(a); a.click(); document.body.removeChild(a);
                setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
              }
"""


def _resource_sort_key(token: str) -> tuple[int, int, str]:
    """Sort resource tokens like ``41k`` or ``250um`` by their numeric part."""
    match = _RESOURCE_RE.fullmatch(token)
    if match is None:
        return (2, 0, token)
    unit = 0 if match[2].lower() == "k" else 1
    return (unit, int(match[1]), token)


def _clean_nodes(graph: nx.MultiDiGraph) -> None:
    """Replace each node's data model with flat plotting attributes."""
    max_name_len = max(len(name) for name in graph)

    for name, attrs in graph.nodes(data=True):
        data = attrs.pop("data")

        counts = {
            f"n_{field}": len(getattr(data, field)) for field in _NODE_COUNT_FIELDS
        }
        counts["n_total"] = sum(counts.values())

        # Equal-width labels keep the rendered node boxes aligned.
        pad = " " * (max_name_len - len(name))
        attrs.update(
            name=data.name,
            species=data.species,
            description=data.description,
            **counts,
            color=_COLOR_BY_SPECIES[data.species],
            label=f"{pad}{name}{pad}",
            shape="circle",
        )


def _clean_edges(graph: nx.MultiDiGraph) -> None:
    """Replace each edge's data model with flat plotting attributes."""
    for _u, _v, key, attrs in graph.edges(keys=True, data=True):
        data = attrs.pop("data")

        surface_res = {transform.density for transform in data.surface_transforms}
        volume_res = {transform.resolution for transform in data.volume_transforms}
        res = sorted(surface_res | volume_res, key=_resource_sort_key)

        attrs.update(
            n_surface_transforms=len(data.surface_transforms),
            n_volume_transforms=len(data.volume_transforms),
            n_surface_transforms_res=len(surface_res),
            n_volume_transforms_res=len(volume_res),
            n_xfms=len(res),
            weight=len(res),
            res=res,
            type=key,
            color=_COLOR_BY_EDGE_TYPE[key],
            label=", ".join(res),
        )


def _merge_parallel_edges(graph: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse the surface and volume parallel edges of each pair into one."""
    flat = nx.DiGraph()
    flat.add_nodes_from(graph.nodes(data=True))

    for u, v, _key, attrs in graph.edges(keys=True, data=True):
        if not flat.has_edge(u, v):
            flat.add_edge(u, v, **dict(attrs))
            continue

        merged = flat[u][v]
        merged["n_surface_transforms"] += attrs["n_surface_transforms"]
        merged["n_volume_transforms"] += attrs["n_volume_transforms"]
        merged["n_surface_transforms_res"] += attrs["n_surface_transforms_res"]
        merged["n_volume_transforms_res"] += attrs["n_volume_transforms_res"]

        res = sorted(set(merged["res"]) | set(attrs["res"]), key=_resource_sort_key)
        edge_type = "both" if merged["type"] != attrs["type"] else merged["type"]
        merged.update(
            res=res,
            label=", ".join(res),
            n_xfms=len(res),
            weight=len(res),
            type=edge_type,
            color=_COLOR_BY_EDGE_TYPE[edge_type],
        )

    return flat


def _merge_bidirectional_edges(graph: nx.DiGraph) -> nx.DiGraph:
    """Collapse u->v / v->u pairs with identical labels into one two-headed edge."""
    seen: set[frozenset[str]] = set()
    to_remove: list[tuple[str, str]] = []

    for u, v in graph.edges:
        pair = frozenset((u, v))
        if pair in seen:
            continue
        seen.add(pair)
        if not graph.has_edge(v, u):
            continue
        if graph[u][v].get("label") != graph[v][u].get("label"):
            continue
        graph[u][v]["arrows"] = "to, from"
        to_remove.append((v, u))

    for u, v in to_remove:
        graph.remove_edge(u, v)

    return graph


def clean_graph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    """Reduce the graph to a flat directed graph of plotting attributes."""
    _clean_nodes(graph)
    _clean_edges(graph)
    return _merge_bidirectional_edges(_merge_parallel_edges(graph))


def _replace_once(page: str, old: str, new: str, what: str) -> str:
    """Replace a pyvis template anchor, failing loudly if the template changed."""
    if old not in page:
        raise RuntimeError(f"pyvis template changed: could not find {what}.")
    return page.replace(old, new, 1)


_TOOLBAR_ROW_END = (
    "                        </div>\n                    </div>\n                </div>"
)
_TOOLBAR_ROW_WITH_BUTTON = (
    "                        </div>\n"
    + _EXPORT_SVG_BUTTON
    + "\n                    </div>\n                </div>"
)


def inject_svg_export(page: str) -> str:
    """Add the "Export SVG" toolbar button and its client-side handler to ``page``."""
    page = _replace_once(
        page, 'class="col-10 pb-2"', 'class="col-8 pb-2"', "the select-menu column"
    )
    page = _replace_once(
        page, _TOOLBAR_ROW_END, _TOOLBAR_ROW_WITH_BUTTON, "the toolbar row"
    )
    # _EXPORT_SVG_JS ends with a newline, so no separator is needed before the tag.
    return _replace_once(
        page,
        "              drawGraph();\n        </script>",
        f"              drawGraph();\n{_EXPORT_SVG_JS}        </script>",
        "the closing script tag",
    )


def _set_edge_lengths(net: Network) -> None:
    """Assign per-edge spring lengths based on the species of the endpoints.

    Edges within one species scale with the number of spaces of that
    species (``(n_spaces - 1) * 100``); edges between different species
    all share one common length (250).
    """
    species_of = {node["id"]: node.get("species") for node in net.nodes}
    n_spaces = Counter(species_of.values())

    for edge in net.edges:
        source = species_of.get(edge["from"])
        target = species_of.get(edge["to"])
        if source is not None and source == target:
            edge["length"] = (n_spaces[source] - 1) * 100
        else:
            edge["length"] = 250


def main() -> None:
    """Build the interactive graph page and write it next to this script."""
    parser = argparse.ArgumentParser(
        description="Render interactive visualization of Neuromaps-PRIME graph.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output file path to save HTML file to (default: %(default)s)",
        default=Path(__file__).with_name("neuromaps_interactive_graph.html")
    )
    parser.add_argument(
        "--yaml-file",
        type=Path,
        metavar="FILE",
        help=(
            "YAML file representing nodes & edges for Neuromaps-PRIME graph; if none "
            "provided, uses default graph (default: %(default)s)"
        ),
        default=None,
    )
    args = parser.parse_args()

    output: Path = args.output
    if not output.suffix == ".html":
        print("Expected, but did not receive '.html' extension - saving as HTML")
        output = args.output.with_suffix(".html")

    graph = clean_graph(NeuromapsGraph(yaml_file=args.yaml_file))

    net = Network(
        height="1080px",
        width="1920px",
        notebook=False,
        directed=True,
        select_menu=True,
        bgcolor="#ffffff",
    )
    net.from_nx(graph)
    _set_edge_lengths(net)
    net.set_options(_PYVIS_OPTIONS)

    output.write_text(inject_svg_export(net.generate_html()))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
