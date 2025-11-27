"""
PyTorch nn.Embedding Save and Load Example

This demonstrates how to properly save and load nn.Embedding layers
in PyTorch models, including common issues and solutions.
"""

import torch
import torch.nn as nn


# ============================================================================
# CORRECT WAY: Embedding as part of the model
# ============================================================================

class ModelWithEmbedding(nn.Module):
    """Model with embedding properly registered as a module."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        # Embedding is registered as a module - will be in state_dict
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return self.output(x)


# ============================================================================
# INCORRECT WAY: Embedding not registered (won't be in state_dict)
# ============================================================================

class ModelWithUnregisteredEmbedding(nn.Module):
    """Model with embedding NOT registered - WON'T be saved in state_dict."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        # This embedding is NOT registered as a module attribute
        embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        # Storing in a list or dict won't register it
        self.layers = [embedding]
    
    def forward(self, x):
        x = self.layers[0](x)  # Using unregistered embedding
        x = self.fc(x)
        return self.output(x)


# ============================================================================
# SOLUTION: Manually save/load embedding separately
# ============================================================================

def save_model_with_embedding(model, embedding, filepath):
    """Save model and embedding separately."""
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_state_dict': embedding.state_dict(),
        'embedding_config': {
            'num_embeddings': embedding.num_embeddings,
            'embedding_dim': embedding.embedding_dim,
            'padding_idx': embedding.padding_idx,
            'max_norm': embedding.max_norm,
            'norm_type': embedding.norm_type,
            'scale_grad_by_freq': embedding.scale_grad_by_freq,
            'sparse': embedding.sparse,
        }
    }, filepath)
    print(f"Model and embedding saved to {filepath}")


def load_model_with_embedding(model, filepath, device='cpu'):
    """Load model and embedding from checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Recreate and load embedding
    embedding_config = checkpoint['embedding_config']
    embedding = nn.Embedding(
        num_embeddings=embedding_config['num_embeddings'],
        embedding_dim=embedding_config['embedding_dim'],
        padding_idx=embedding_config.get('padding_idx'),
        max_norm=embedding_config.get('max_norm'),
        norm_type=embedding_config.get('norm_type'),
        scale_grad_by_freq=embedding_config.get('scale_grad_by_freq'),
        sparse=embedding_config.get('sparse', False),
    )
    embedding.load_state_dict(checkpoint['embedding_state_dict'])
    
    print(f"Model and embedding loaded from {filepath}")
    return model, embedding


# ============================================================================
# SOLUTION: Register embedding after creation
# ============================================================================

class ModelWithDynamicEmbedding(nn.Module):
    """Model that registers embedding dynamically."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Create and register embedding
        embedding = nn.Embedding(vocab_size, embedding_dim)
        self.add_module('embedding', embedding)  # Register it!
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.fc(x)
        return self.output(x)


# ============================================================================
# SOLUTION: Use nn.ModuleDict or nn.ModuleList
# ============================================================================

class ModelWithModuleDict(nn.Module):
    """Model using ModuleDict to store embedding."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        # ModuleDict properly registers modules
        self.embeddings = nn.ModuleDict({
            'token_embedding': nn.Embedding(vocab_size, embedding_dim)
        })
        self.fc = nn.Linear(embedding_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        x = self.embeddings['token_embedding'](x)
        x = self.fc(x)
        return self.output(x)


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_correct_usage():
    """Example of correct embedding save/load."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Correct Usage (Embedding in state_dict)")
    print("="*60)
    
    # Create model
    model = ModelWithEmbedding(vocab_size=1000, embedding_dim=128, hidden_dim=256)
    
    # Check state dict
    state_dict = model.state_dict()
    print(f"\nKeys in state_dict: {list(state_dict.keys())}")
    print(f"'embedding.weight' in state_dict: {'embedding.weight' in state_dict}")
    
    # Save model
    torch.save(model.state_dict(), 'model_correct.pth')
    print("Model saved successfully")
    
    # Load model
    new_model = ModelWithEmbedding(vocab_size=1000, embedding_dim=128, hidden_dim=256)
    new_model.load_state_dict(torch.load('model_correct.pth'))
    print("Model loaded successfully")


def example_incorrect_usage():
    """Example of incorrect embedding (not in state_dict)."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Incorrect Usage (Embedding NOT in state_dict)")
    print("="*60)
    
    # Create model
    model = ModelWithUnregisteredEmbedding(
        vocab_size=1000, embedding_dim=128, hidden_dim=256
    )
    
    # Check state dict
    state_dict = model.state_dict()
    print(f"\nKeys in state_dict: {list(state_dict.keys())}")
    print("Notice: 'embedding.weight' is NOT in state_dict!")
    
    # The embedding won't be saved
    torch.save(model.state_dict(), 'model_incorrect.pth')
    print("Model saved (but embedding is missing)")


def example_manual_save_load():
    """Example of manually saving/loading embedding."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Manual Save/Load (For unregistered embeddings)")
    print("="*60)
    
    # Create model and separate embedding
    model = ModelWithUnregisteredEmbedding(
        vocab_size=1000, embedding_dim=128, hidden_dim=256
    )
    embedding = model.layers[0]  # Get the unregistered embedding
    
    # Save both
    save_model_with_embedding(model, embedding, 'model_with_embedding.pth')
    
    # Load both
    new_model = ModelWithUnregisteredEmbedding(
        vocab_size=1000, embedding_dim=128, hidden_dim=256
    )
    new_model, loaded_embedding = load_model_with_embedding(
        new_model, 'model_with_embedding.pth'
    )
    new_model.layers[0] = loaded_embedding
    print("Model and embedding loaded successfully")


def example_dynamic_registration():
    """Example of dynamically registering embedding."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Dynamic Registration (add_module)")
    print("="*60)
    
    # Create model
    model = ModelWithDynamicEmbedding(
        vocab_size=1000, embedding_dim=128, hidden_dim=256
    )
    
    # Check state dict
    state_dict = model.state_dict()
    print(f"\nKeys in state_dict: {list(state_dict.keys())}")
    print(f"'embedding.weight' in state_dict: {'embedding.weight' in state_dict}")
    
    # Save and load normally
    torch.save(model.state_dict(), 'model_dynamic.pth')
    new_model = ModelWithDynamicEmbedding(
        vocab_size=1000, embedding_dim=128, hidden_dim=256
    )
    new_model.load_state_dict(torch.load('model_dynamic.pth'))
    print("Model with dynamically registered embedding saved and loaded")


def example_module_dict():
    """Example using ModuleDict."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Using ModuleDict")
    print("="*60)
    
    # Create model
    model = ModelWithModuleDict(vocab_size=1000, embedding_dim=128, hidden_dim=256)
    
    # Check state dict
    state_dict = model.state_dict()
    print(f"\nKeys in state_dict: {list(state_dict.keys())}")
    print(f"'embeddings.token_embedding.weight' in state_dict: "
          f"{'embeddings.token_embedding.weight' in state_dict}")
    
    # Save and load normally
    torch.save(model.state_dict(), 'model_moduledict.pth')
    new_model = ModelWithModuleDict(vocab_size=1000, embedding_dim=128, hidden_dim=256)
    new_model.load_state_dict(torch.load('model_moduledict.pth'))
    print("Model with ModuleDict saved and loaded")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("PyTorch nn.Embedding Save and Load Examples")
    print("="*60)
    
    # Run examples
    example_correct_usage()
    example_incorrect_usage()
    example_manual_save_load()
    example_dynamic_registration()
    example_module_dict()
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("1. Always register nn.Embedding as a model attribute (self.embedding)")
    print("2. Use self.add_module() if you need to register dynamically")
    print("3. Use nn.ModuleDict or nn.ModuleList for collections")
    print("4. If embedding is unregistered, save/load it separately")
    print("5. Check state_dict.keys() to verify embedding is included")
    print("="*60)
