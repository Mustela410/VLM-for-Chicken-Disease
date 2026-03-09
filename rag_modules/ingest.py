"""
- Supports JSON Knowledge Base + PDF documents
- Hybrid indexing (ChromaDB + BM25)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_modules.engine_v2 import rag_engine_v2

if __name__ == "__main__":
    rag_engine_v2.ingest_data()
    
    print("\n" + "=" * 60)
    print("Ingestion Complete!")
    
    # Show stats
    stats = rag_engine_v2.get_stats()
    print(f"\n Database Stats:")
    print(f"   Total Documents: {stats['db_size']}")
    print("=" * 60)
