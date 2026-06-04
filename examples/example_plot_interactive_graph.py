from neuromaps_prime.graph import NeuromapsGraph

from pyvis.network import Network

import networkx as nx
import numpy as np
import re


color_lut = {
    'macaque': "#DEAC52",   # Amber
    'human': "#95B46A",  # Sage green
    'chimpanzee': "#C8D0D5",   # Slate gray
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

