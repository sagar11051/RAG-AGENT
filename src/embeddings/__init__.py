"""Embedding module for OVH BGE-M3 integration."""

from src.embeddings.ovh_embeddings import OVHEmbeddings, get_embeddings_client

__all__ = ["OVHEmbeddings", "get_embeddings_client"]
