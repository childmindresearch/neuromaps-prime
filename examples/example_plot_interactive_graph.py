from neuromaps_prime.graph import NeuromapsGraph

from pyvis.network import Network

import networkx as nx
import numpy as np
import re


color_lut = {
    'macaque': "#DEAC52",   # Amber
    'human': "#95B46A",  # Sage green
    'chimpanzee': "#C8D0D5",   # Slate gray
    'marmoset': "#e98c8cff",   # Red
    'volume_to_volume': '#E08E9A', # Pink
    'surface_to_surface': '#80B2D4', # Blue
    'both': '#A28DC7' # Purple
}


def flatten(multiG):
    flatG = nx.DiGraph()
    flatG.add_nodes_from(multiG.nodes(data=True))

    for u, v, attrs in multiG.edges(data=True):
        # If the edge already exists, merge the attributes
        if flatG.has_edge(u, v):
            existing_attrs = flatG[u][v]

            for k, v in existing_attrs.items():
                if k != 'label':
                    existing_attrs[k] += attrs[k]
                else:
                    existing_attrs[k] = ", ".join([existing_attrs[k], attrs[k]])

            if len(existing_attrs['type']) > 1:
                existing_attrs['type'] = 'both'
                existing_attrs['color'] = color_lut[existing_attrs['type']]

        else:
            flatG.add_edge(u, v, **attrs)

    return flatG

def merge_edges(G):
    processed_pairs = set()

    for u, v, attrs in G.edges(data=True):
        # Create a unique structural identifier for the unordered node pair
        pair_id = tuple(sorted([u, v]))
        if pair_id in processed_pairs:
            continue

        current_label = attrs.get('label')

        # Check if the reverse edge exists in the graph
        if G.has_edge(v, u):
            reverse_attrs = G.edges[v, u]
            reverse_label = reverse_attrs.get('label')

            # If they both have labels and they match perfectly
            if current_label == reverse_label:
                # 1. Update the forward edge to render arrows on BOTH ends
                G.edges[u, v]['arrows'] = "to, from"
                
                # 2. Safely remove the duplicate reverse edge
                G.remove_edge(v, u)
                
                # Mark this node pair as completed
                processed_pairs.add(pair_id)

    return G


def clean_graph(G):
    """Removes Neuromaps specific types/attributes and only
    keeps around elements that can be used to inform plotting
    in some way.
    """
    # Do some tedius housekeeping that helps us with nice visualization later...
    max_len = max([len(n) for n in G.nodes])

    # Start with cleaning up the nodes...
    # For every node, unpack the data values as top-level objects
    # and for the surfaces/volumes/annotations, store the counts
    for node, attrs in G.nodes(data=True):
        data_dict = dict(attrs.get('data', {}))
        keys_of_interest = ['name', 'species', 'description']  # Later I use "kois" for this
        for k in keys_of_interest:
            attrs[k] = data_dict.get(k, None)

        len_kois = ['surfaces', 'volumes', 'surface_annotations', 'volume_annotations']
        n_total = 0
        for k in len_kois:
            tmp = len(data_dict.get(k, []))
            attrs[f'n_{k}'] = tmp
            n_total += tmp
        attrs['n_total'] = n_total

        # Add visualization (eg. colour and size) here to get ahead of it later...
        name_pad = int(max_len - len(attrs['name']))
        attrs['color'] = color_lut[attrs['species']]

        attrs['label'] = (" " * name_pad) + attrs['name'] + (" " * name_pad)
        # attrs['size'] = 5 + 5*np.log2(attrs['n_total'])
        attrs['shape'] = 'circle'

        # Clean up the data field that breaks plotting tools
        del attrs['data']

    # Continue by cleaning up the edges... Actually, remake and merge all edges
    # For every edge, unpack the source, dest, surfaces, and volumes and
    # then derive the type based on which of those lists aren't empty.
    for u, v, k, attrs in G.edges(keys=True, data=True):
        data_dict = dict(attrs.get('data', {}))
        kois = ['surface_transforms', 'volume_transforms']
        pattern = re.compile(r"_([0-9]+[a-zA-Z]+)_")
        n_xfms = 0
        attrs['res'] = []
        for koi in kois:
            value = data_dict.get(koi, [])
            attrs[f'n_{koi}'] = len(value)
            attrs['res'] += list(set(pattern.findall(str(_))[0] for _ in value))
            tmp = len(attrs['res'])
            attrs[f'n_{koi}_res'] = tmp
            n_xfms += tmp
        attrs[f'n_xfms'] = n_xfms
        attrs['type'] = k

        # Similarly, add colour and weight to save another loop
        attrs['color'] = color_lut[attrs['type']]
        attrs['weight'] = attrs['n_xfms']
        attrs['label'] = ", ".join(attrs['res'])

        # Networkx quirk of how these are stored... ty google
        del G.edges[u, v, k]['data']

    G = flatten(G)
    G = merge_edges(G)

    return G


G = NeuromapsGraph()
G = clean_graph(G)

net = Network(notebook=False, directed=True, select_menu=True,
              height="1080px", width="1920px", bgcolor="#ffffff")
net.from_nx(G)

net.set_options("""
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
      "gravitationalConstant": -30000,
      "centralGravity": 0.3,
      "springLength": 300,
      "springConstant": 0.04,
      "damping": 0.09,
      "avoidOverlap": 0.5
    },
    "minVelocity": 0.75
  },
  "layout": {
    "randomSeed": 678
  }
}
""")

net.write_html('./neuromaps_interactive_graph.html')

# Post-process the generated HTML to inject Export SVG button and functions
with open('./neuromaps_interactive_graph.html', 'r') as f:
    html_content = f.read()

# 1. Narrow the select column from col-10 to col-8
html_content = html_content.replace('class="col-10 pb-2"', 'class="col-8 pb-2"')

# 2. Inject the Export SVG button after the Reset Selection button's closing </div>
button_html = '''                        <div class="col-2 pb-2">
                            <button type="button" class="btn btn-outline-secondary btn-block" onclick="exportSVG();">Export SVG</button>
                        </div>'''
html_content = html_content.replace(
    '                        </div>\n                    </div>\n                </div>',
    f'                        </div>\n{button_html}\n                    </div>\n                </div>'
)

# 3. Inject the SVG export JavaScript functions before the final </script>
js_functions = '''              // ---------------------------------------------------------------
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
                                  '" text-anchor="middle" fill="#343434">' + _esc(e.label) + '</text></g>');
                  }
                });

                var nodeSvg = [];
                ids.forEach(function (id) {
                  var n = nodeObjs[id], c = C[id], r = R[id], fill = n.color || "#cccccc";
                  var lbl = ((n.label != null ? n.label : n.name) || id).toString().trim();
                  nodeSvg.push('<circle cx="' + c.x + '" cy="' + c.y + '" r="' + r + '" fill="' + _esc(fill) +
                               '" stroke="' + _esc(_darken(fill, 0.7)) + '" stroke-width="2"/>');
                  nodeSvg.push('<text x="' + c.x + '" y="' + (c.y + 7) + '" font-family="Nunito,sans-serif" font-size="20" ' +
                               'text-anchor="middle" fill="#343434">' + _esc(lbl) + '</text>');
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
'''

html_content = html_content.replace('              drawGraph();\n        </script>',
                                    f'              drawGraph();\n{js_functions}        </script>')

with open('./neuromaps_interactive_graph.html', 'w') as f:
    f.write(html_content)

