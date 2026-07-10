---
title: 'Neuromaps-PRIME: a graph-based framework for integrating cross-species neuroimaging spaces'
tags:
  - Python
  - neuroimaging
  - non-human primates
  - brain atlases
  - image registration
  - comparative neuroscience
authors:
  - name: Tamsin Rogers
    affiliation: 1
  - name: Jason Kai
    affiliation: 1
  - name: Biraj Shrestha
    affiliation: 1
  - name: Ting Xu
    affiliation: 1
  - name: Gregory Kiar
    affiliation: 1
  - name: Thomas Funck
    affiliation: 1
    corresponding: true
affiliations:
  - name: Child Mind Institute, New York, NY, United States
    index: 1
  - name: Max Planck Institute for Human Cognitive and Brain Sciences, Leipzig, Germany
    index: 2
  - name: Institute of Neurosciences and Medicine (INM-7), Forschungszentrum Jülich, Jülich, Germany
    index: 3
  - name: Institute of Systems Neuroscience, Heinrich Heine University Düsseldorf, Düsseldorf, Germany
    index: 4
  - name: McConnell Brain Imaging Centre, Montreal Neurological Institute, McGill University, Montréal, Quebec, Canada
    index: 5
  - name: Institute of Neurosciences and Medicine (INM-1), Forschungszentrum Jülich, Jülich, Germany
    index: 6
date: 10 July 2026
bibliography: paper.bib
---

# Summary

Brain imaging data are frequently reported in standardized coordinate spaces so that measurements from different individuals can be aligned and compared. The cost of this convention is fragmentation: many different standard spaces are in routine use, and a brain map defined in one space must be transformed before it can be compared with a map defined in another. The problem multiplies once non-human primate (NHP) species are considered alongside human data because of the many NHP coordinate spaces that are in use. Moreover, even within a single coordinate system, the same data may be stored as a cortical surface, a volume, or both, and at varying resolutions. Neuromaps-PRIME is a Python package that treats this as a graph problem: each template space is a node, each transformation an edge, and moving a map from one space to another becomes a matter of finding and applying a path through the graph. The package extends the neuromaps toolbox [@markello2022] and currently ships twelve template spaces across 3 primate species — human, macaque, and chimpanzee — together with curated transformations between them. New spaces, transformations, and annotations are declared in configuration files rather than code.

# Statement of need

Reorganizing and resampling brain data into a common space is a tedious, error-prone prerequisite for almost any quantitative analysis. The redundant preprocessing it demands slows research and undermines reproducibility. This burden falls hardest on comparative and translational neuroimaging, where interest in NHP models is driven by their close neuroanatomical similarity to humans [@milham2018] and where the number of standard coordinate spaces across the macaque, marmoset, and chimpanzee is large and still growing.

Neuromaps-PRIME is built for researchers who need to move brain maps reliably between coordinate spaces or species. It lets a user request any available brain map in any connected space and returns the resampled map, automatically assembling the intermediate steps through the graph when no direct transformation exists. Because spaces and transformations are described in editable configuration files, a group with a custom atlas or unpublished data can register it against the existing network without modifying the software, lowering the barrier to integrating new resources into a reproducible workflow.

# State of the field

The most closely related tool is neuromaps [@markello2022], which popularized programmatic comparison of brain maps but was designed around the human cortex. Its coordinate spaces are hard-coded; it implements only direct pairwise transformations and cannot compose a chain of transforms to bridge spaces that are not directly connected; and it supports surface data only. TemplateFlow [@ciric2022] addresses an adjacent need — a FAIR archive that makes templates and atlases findable and downloadable across species — but it catalogs resources rather than transforming data between them. Lower-level engines such as the Connectome Workbench [@marcus2011] and ANTs [@avants2011] perform individual resampling and registration operations, and cross-species correspondences exist as standalone datasets [@sirmpilatze2020; @xu2020], but none of these provides an integrated layer that selects, composes, and applies the right sequence of transforms on demand. RheMAP [@sirmpilatze2020] and RheMAP-Surf [@zhou2025] are data repositories that provide standard transformations, but only within macaque spaces.

Rather than re-implement this ecosystem, Neuromaps-PRIME extends the neuromaps model and orchestrates these established tools and datasets. Its distinct scholarly contribution is the graph engine itself together with first-class support for multiple species and for volumetric as well as surface data — capabilities that the hard-coded, pairwise, surface-only design of neuromaps cannot accommodate without a fundamental redesign.

# Software design

The package centers on a NeuromapsGraph, a directed multigraph (built on networkx [@hagberg2008]) whose nodes are template spaces, each carrying its species, surfaces, volumes, and annotations, and whose edges are surface or volume transformations. The graph approach was necessary because enumerating every pairwise transform—as neuromaps does—grows quadratically with the number of spaces and quickly becomes unmaintainable. By contrast, adding a node to a graph requires only a single edge to an existing space, after which any other space becomes reachable by path-finding. Given a source and target, the engine finds the shortest weighted path, composes the intermediate transforms, and can insert the composed result back into the graph as a cached edge, trading a small amount of storage for faster repeated queries. Four transformers cover the common operations — surface-to-surface and volume-to-volume resampling, and surface-to-volume and volume-to-surface projection — for both continuous (metric) and discrete (label) data.

Two further design decisions support reproducibility and community growth. First, the network is described in editable YAML files validated against bundled JSON schemas, separating configuration from implementation so that contributors can add spaces and transforms declaratively; data themselves are hosted remotely and fetched on demand into a local cache, avoiding the need to ship large template files. Second, because the underlying transformations require specific versions of external neuroimaging tools, these are invoked through NiWrap [@rupprecht2025styx], which can dispatch to local, Docker, Podman, or Singularity/Apptainer runtimes; this isolates the library from environment-specific version fragility at the cost of an optional container dependency. The transformations themselves draw on Multimodal Surface Matching [@robinson2014], the CIVET environment [@lepage2017], RheMAP volumetric warps [@sirmpilatze2020], and cross-species functional alignment between the human and macaque cortex [@xu2020].

# Research impact statement

Neuromaps-PRIME (v0.1.0) is released under the MIT license, distributed on PyPI, documented online, and maintained with a continuous-integration suite spanning unit, integration, and regression tests. It builds directly on neuromaps, an established and widely cited tool in the human neuroimaging community, and extends that ecosystem to the NHP imaging community organized around resources such as PRIME-DE [@milham2018]. At release the graph integrates twelve template spaces across three species with curated surface and volume transformations, and its configuration-driven design has been exercised through community development activities aimed at adding further spaces and brain maps. We anticipate the framework will serve as reusable infrastructure for cross-species comparative analyses; concrete adoption metrics and dependent publications will accrue as the software matures.

# AI usage disclosure

Generative AI tools were used to assist in drafting and copy-editing this manuscript. All AI-assisted text was reviewed and verified for accuracy by the authors against the software's source code, documentation, and the cited literature, and the authors take full responsibility for its content.

# Acknowledgements

Research reported in this publication was supported by the National Institute of Mental Health of the National Institutes of Health under award number R01MH139565. The content is solely the responsibility of the authors and does not necessarily represent the official views of the National Institutes of Health.

# References
