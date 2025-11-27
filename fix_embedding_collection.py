"""
Fix for embedding collection stored in regular Python dict.

Your original code:
    self._embedding_collection: dict[str, nn.Embedding] = {
        feat_name: nn.Embedding(cfg["num_embeddings"], cfg["embedding_dim"]).to("cuda")
        for cfg in emb_cfg for feat_name in cfg["feature_names"]
    }

Problem: Regular Python dict doesn't register modules, so embeddings won't be in state_dict.
Solution: Use nn.ModuleDict instead.
"""

import torch
import torch.nn as nn
from typing import Dict


# ============================================================================
# YOUR ORIGINAL CODE (WON'T SAVE EMBEDDINGS)
# ============================================================================

class ModelWithDictEmbedding(nn.Module):
    """Original code - embeddings NOT saved in state_dict."""
    
    def __init__(self, emb_cfg):
        super().__init__()
        # ❌ This won't be saved in state_dict!
        self._embedding_collection: dict[str, nn.Embedding] = {
            feat_name: nn.Embedding(cfg["num_embeddings"], cfg["embedding_dim"]).to("cuda")
            for cfg in emb_cfg for feat_name in cfg["feature_names"]
        }
    
    def forward(self, x, feat_name):
        return self._embedding_collection[feat_name](x)


# ============================================================================
# SOLUTION 1: Use nn.ModuleDict (RECOMMENDED)
# ============================================================================

class ModelWithModuleDictEmbedding(nn.Module):
    """Fixed code - embeddings WILL be saved in state_dict."""
    
    def __init__(self, emb_cfg):
        super().__init__()
        # ✅ Use nn.ModuleDict - this registers all embeddings!
        self._embedding_collection = nn.ModuleDict({
            feat_name: nn.Embedding(cfg["num_embeddings"], cfg["embedding_dim"]).to("cuda")
            for cfg in emb_cfg for feat_name in cfg["feature_names"]
        })
    
    def forward(self, x, feat_name):
        # Usage is the same - still use dict-like access
        return self._embedding_collection[feat_name](x)


# ============================================================================
# SOLUTION 2: Create dict then convert to ModuleDict
# ============================================================================

class ModelWithConvertedModuleDict(nn.Module):
    """Alternative: Create dict first, then convert to ModuleDict."""
    
    def __init__(self, emb_cfg):
        super().__init__()
        # Create regular dict first
        embedding_dict = {
            feat_name: nn.Embedding(cfg["num_embeddings"], cfg["embedding_dim"]).to("cuda")
            for cfg in emb_cfg for feat_name in cfg["feature_names"]
        }
        # Convert to ModuleDict - this registers all modules
        self._embedding_collection = nn.ModuleDict(embedding_dict)
    
    def forward(self, x, feat_name):
        return self._embedding_collection[feat_name](x)


# ============================================================================
# SOLUTION 3: Manual registration (if you must keep regular dict)
# ============================================================================

class ModelWithManuallyRegisteredEmbeddings(nn.Module):
    """Manually register each embedding (not recommended, but works)."""
    
    def __init__(self, emb_cfg):
        super().__init__()
        # Create regular dict
        self._embedding_collection: dict[str, nn.Embedding] = {}
        
        # Manually register each embedding
        for cfg in emb_cfg:
            for feat_name in cfg["feature_names"]:
                embedding = nn.Embedding(
                    cfg["num_embeddings"], 
                    cfg["embedding_dim"]
                ).to("cuda")
                self._embedding_collection[feat_name] = embedding
                # Manually register with add_module
                self.add_module(f'embedding_{feat_name}', embedding)
    
    def forward(self, x, feat_name):
        return self._embedding_collection[feat_name](x)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def example_usage():
    """Demonstrate the fix."""
    
    # Example embedding config
    emb_cfg = [
        {
            "num_embeddings": 1000,
            "embedding_dim": 128,
            "feature_names": ["user_id", "item_id"]
        },
        {
            "num_embeddings": 500,
            "embedding_dim": 64,
            "feature_names": ["category"]
        }
    ]
    
    print("="*60)
    print("Testing Original Code (Dict)")
    print("="*60)
    model_dict = ModelWithDictEmbedding(emb_cfg)
    state_dict = model_dict.state_dict()
    print(f"Keys in state_dict: {list(state_dict.keys())}")
    print(f"Has 'user_id' embedding: {'_embedding_collection.user_id.weight' in state_dict}")
    print("❌ Embeddings are NOT in state_dict!\n")
    
    print("="*60)
    print("Testing Fixed Code (ModuleDict)")
    print("="*60)
    model_fixed = ModelWithModuleDictEmbedding(emb_cfg)
    state_dict = model_fixed.state_dict()
    print(f"Keys in state_dict: {list(state_dict.keys())}")
    print(f"Has 'user_id' embedding: {'_embedding_collection.user_id.weight' in state_dict}")
    print("✅ Embeddings ARE in state_dict!\n")
    
    # Test save/load
    print("="*60)
    print("Testing Save/Load")
    print("="*60)
    torch.save(model_fixed.state_dict(), 'model_with_embeddings.pth')
    
    # Load into new model
    new_model = ModelWithModuleDictEmbedding(emb_cfg)
    new_model.load_state_dict(torch.load('model_with_embeddings.pth'))
    print("✅ Model with embeddings saved and loaded successfully!")
    
    # Verify weights are the same
    original_weight = model_fixed._embedding_collection['user_id'].weight
    loaded_weight = new_model._embedding_collection['user_id'].weight
    print(f"✅ Weights match: {torch.allclose(original_weight, loaded_weight)}")


# ============================================================================
# MIGRATION GUIDE
# ============================================================================

def migration_guide():
    """How to migrate your existing code."""
    print("\n" + "="*60)
    print("MIGRATION GUIDE")
    print("="*60)
    print("""
STEP 1: Change the type annotation and initialization
    Before:
        self._embedding_collection: dict[str, nn.Embedding] = {...}
    
    After:
        self._embedding_collection = nn.ModuleDict({...})
        # Or keep type hint as: nn.ModuleDict[str, nn.Embedding]

STEP 2: Usage remains the same!
    # All these still work:
    self._embedding_collection[feat_name]  # Access
    self._embedding_collection.keys()      # Iterate
    self._embedding_collection.items()     # Iterate
    len(self._embedding_collection)        # Length
    
STEP 3: Save/load works normally
    torch.save(model.state_dict(), 'model.pth')
    model.load_state_dict(torch.load('model.pth'))

STEP 4: State dict keys will be:
    '_embedding_collection.{feat_name}.weight'
    '_embedding_collection.{feat_name}.bias' (if any)
    """)


if __name__ == '__main__':
    # Run example
    example_usage()
    
    # Show migration guide
    migration_guide()
    
    print("\n" + "="*60)
    print("QUICK FIX FOR YOUR CODE:")
    print("="*60)
    print("""
Just change this line:

    # FROM:
    self._embedding_collection: dict[str, nn.Embedding] = {
        feat_name: nn.Embedding(cfg["num_embeddings"], cfg["embedding_dim"]).to("cuda")
        for cfg in emb_cfg for feat_name in cfg["feature_names"]
    }

    # TO:
    self._embedding_collection = nn.ModuleDict({
        feat_name: nn.Embedding(cfg["num_embeddings"], cfg["embedding_dim"]).to("cuda")
        for cfg in emb_cfg for feat_name in cfg["feature_names"]
    })

That's it! Everything else stays the same.
    """)
