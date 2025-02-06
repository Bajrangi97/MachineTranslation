import torch
import whisper

# Load Whisper Medium Model
def load_whisper_model():
    model = whisper.load_model("medium").eval()
    return model

#  Compute Activation Energy (Neuron Activation Magnitude)
def compute_activation_energy(model, sample_input):
    layer_activations = []
    audio = sample_input.clone()
    
    with torch.no_grad():
        for i, layer in enumerate(model.encoder.blocks):
            audio = layer(audio)  # Forward pass
            activation_energy = torch.mean(torch.abs(audio)).item()  # Compute activation energy
            layer_activations.append((i, activation_energy))
    
    return sorted(layer_activations, key=lambda x: x[1])  # Sort (low to high)

# Compute Gradient Norms (Saliency Analysis)
def compute_gradient_norms(model, sample_input):
    model.train()
    sample_input.requires_grad = True  # Enable gradients
    
    output = model.encoder(sample_input)
    loss = output.sum()  # Dummy loss
    loss.backward()  # Backpropagate
    
    layer_gradients = []
    for i, layer in enumerate(model.encoder.blocks):
        grad_norm = torch.norm(layer[0].weight.grad).item()
        layer_gradients.append((i, grad_norm))
    
    return sorted(layer_gradients, key=lambda x: x[1])  # Sort (low to high)

# Compute Attention Scores
def compute_attention_scores(model):
    layer_attention_scores = []
    
    for i, layer in enumerate(model.encoder.blocks):
        attn_weights = layer.self_attn.weight.mean().item()  # Extract attention score
        layer_attention_scores.append((i, attn_weights))
    
    return sorted(layer_attention_scores, key=lambda x: x[1])  # Sort (low to high)

#  Compute L1 Norm (Weight Magnitude)
def compute_l1_norm(model):
    layer_l1_norms = []
    
    for i, layer in enumerate(model.encoder.blocks):
        l1_norm = torch.sum(torch.abs(layer[0].weight)).item()
        layer_l1_norms.append((i, l1_norm))
    
    return sorted(layer_l1_norms, key=lambda x: x[1])  # Sort (low to high)

# Rank layers based on combined importance scores
def rank_layers_for_pruning(model, sample_input, num_layers_to_remove=6):
    activation_ranks = compute_activation_energy(model, sample_input)
    gradient_ranks = compute_gradient_norms(model, sample_input)
    attention_ranks = compute_attention_scores(model)
    l1_ranks = compute_l1_norm(model)
    
    importance_scores = {}
    for rank_list in [activation_ranks, gradient_ranks, attention_ranks, l1_ranks]:
        for idx, score in enumerate(rank_list):
            layer_idx = score[0]
            importance_scores[layer_idx] = importance_scores.get(layer_idx, 0) + idx  # Sum ranks
    
    sorted_layers = sorted(importance_scores.items(), key=lambda x: x[1])  # Sort (low score = redundant)
    return [layer_idx for layer_idx, _ in sorted_layers[:num_layers_to_remove]]

# Prune least important layers
def prune_model(model, redundant_layers):
    model.encoder.blocks = torch.nn.ModuleList(
        [layer for i, layer in enumerate(model.encoder.blocks) if i not in redundant_layers]
    )
    model.decoder.blocks = torch.nn.ModuleList(
        [layer for i, layer in enumerate(model.decoder.blocks) if i not in redundant_layers]
    )
    
    print(f"Pruned Model - Encoder Layers: {len(model.encoder.blocks)}, Decoder Layers: {len(model.decoder.blocks)}")
    return model

# Main Execution
def main():
    model = load_whisper_model()
    sample_input = torch.randn(1, 80, 3000)  # Simulated mel spectrogram input
    
    redundant_layers = rank_layers_for_pruning(model, sample_input, num_layers_to_remove=6)
    pruned_model = prune_model(model, redundant_layers)
    
    torch.save(pruned_model.state_dict(), "whisper_medium_pruned.pth")  # Save model
    print("Pruned model saved successfully!")

if __name__ == "__main__":
    main()
