"""
Gene tokenizer utilities.

The vocabulary is: [PAD]=0, [CLS]=1, [MASK]=2, gene_0=3, gene_1=4, ...
Gene indices correspond to the column order in the processed AnnData HVG matrix.
"""

from __future__ import annotations
from dataclasses import dataclass

PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 1
MASK_TOKEN_ID = 2
SPECIAL_TOKENS = 3


@dataclass
class GeneVocab:
    gene_names: list[str]  # ordered list of HVG names (length = n_genes)

    @property
    def vocab_size(self) -> int:
        return len(self.gene_names) + SPECIAL_TOKENS

    def gene_to_id(self, gene: str) -> int:
        return self.gene_names.index(gene) + SPECIAL_TOKENS

    def id_to_gene(self, token_id: int) -> str:
        if token_id < SPECIAL_TOKENS:
            return {0: "[PAD]", 1: "[CLS]", 2: "[MASK]"}[token_id]
        return self.gene_names[token_id - SPECIAL_TOKENS]

    @classmethod
    def from_anndata(cls, adata) -> "GeneVocab":
        return cls(gene_names=list(adata.var_names))
