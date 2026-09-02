"""Coarse lineage grouping for BMMC cell_type labels (PLAN.md sec 3-2 exp B,
sec 3-3 exp C). Hand-built from the actual cell_type values observed in the
cite/multiome AnnData (docs/HISTORY.md 2026-08-13), grouping the ~45
CITE-seq / ~22 Multiome fine-grained labels into broad hematopoietic
lineages. This is a simplification of the full BMMC ontology — good enough
to define "same lineage, different fine type" vs "different lineage
entirely" conditions, but should be revisited if a condition ends up with
too few cells (PLAN.md's minimum-sample-size requirement for exp B).
"""
from __future__ import annotations

CITE_LINEAGE_MAP = {
    "CD14+ Mono": "Myeloid_Mono", "CD16+ Mono": "Myeloid_Mono",
    "pDC": "Myeloid_DC", "cDC1": "Myeloid_DC", "cDC2": "Myeloid_DC",
    "HSC": "Progenitor", "Lymph prog": "Progenitor", "G/M prog": "Progenitor", "MK/E prog": "Progenitor",
    "T prog cycling": "Progenitor",
    "CD4+ T activated": "T_CD4", "CD4+ T naive": "T_CD4",
    "CD4+ T activated integrinB7+": "T_CD4", "CD4+ T CD314+ CD45RA+": "T_CD4", "T reg": "T_CD4",
    "CD8+ T naive": "T_CD8", "CD8+ T CD57+ CD45RO+": "T_CD8", "CD8+ T CD57+ CD45RA+": "T_CD8",
    "CD8+ T TIGIT+ CD45RO+": "T_CD8", "CD8+ T TIGIT+ CD45RA+": "T_CD8", "CD8+ T CD49f+": "T_CD8",
    "CD8+ T CD69+ CD45RO+": "T_CD8", "CD8+ T CD69+ CD45RA+": "T_CD8",
    "CD8+ T naive CD127+ CD26- CD101-": "T_CD8",
    "MAIT": "T_other", "gdT CD158b+": "T_other", "gdT TCRVD2+": "T_other", "dnT": "T_other",
    "NK": "NK_ILC", "NK CD158e1+": "NK_ILC", "ILC1": "NK_ILC", "ILC": "NK_ILC",
    "Naive CD20+ B IGKC+": "B_cell", "Naive CD20+ B IGKC-": "B_cell", "Transitional B": "B_cell",
    "B1 B IGKC+": "B_cell", "B1 B IGKC-": "B_cell",
    "Plasma cell IGKC+": "Plasma", "Plasma cell IGKC-": "Plasma",
    "Plasmablast IGKC+": "Plasma", "Plasmablast IGKC-": "Plasma",
    "Reticulocyte": "Erythroid", "Erythroblast": "Erythroid",
    "Proerythroblast": "Erythroid", "Normoblast": "Erythroid",
}

MULTIOME_LINEAGE_MAP = {
    # Multiome uses a coarser (22-category) cell_type vocabulary; extend as
    # unseen labels are encountered when exp B/C are actually run on it.
    "CD14+ Mono": "Myeloid_Mono", "CD16+ Mono": "Myeloid_Mono",
    "pDC": "Myeloid_DC", "cDC2": "Myeloid_DC",
    "HSC": "Progenitor", "Lymph prog": "Progenitor", "G/M prog": "Progenitor", "MK/E prog": "Progenitor",
    "Naive CD20+ B": "B_cell", "Transitional B": "B_cell", "Plasma cell": "Plasma",
    "CD4+ T activated": "T_CD4", "CD4+ T naive": "T_CD4", "CD8+ T naive": "T_CD8", "CD8+ T": "T_CD8",
    "NK": "NK_ILC", "ILC": "NK_ILC",
    "Erythroblast": "Erythroid", "Proerythroblast": "Erythroid", "Normoblast": "Erythroid",
}


def to_lineage(cell_types, pair: str):
    mapping = CITE_LINEAGE_MAP if pair == "cite" else MULTIOME_LINEAGE_MAP
    return [mapping.get(ct, "Other") for ct in cell_types]
