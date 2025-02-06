import torch
import whisper
from torch.optim import AdamW

# Load Whisper Medium model
model = whisper.load_model("medium").eval()

# Sample input (Random Mel spectrogram for analysis)
audio = torch.randn(1, 80, 3000)  # (Batch, Channels, Time)

# --------------------- Step 1: Activation Energy ---------------------
def compute_activation_energy(model, audio):
    layer_activations = []
    with torch.no_grad():
        for i, layer in enumerate(model.encoder.blocks):
            audio = layer(audio)  # Forward through each layer
            activation_energy = torch.mean(torch.abs(audio)).item()  # Compute activation energy
            layer_activations.append((i, activation_energy))
    
    return sorted(layer_activations, key=lambda x: x[1])  # Sort (low energy = redundant)

# --------------------- Step 2: Gradient Norms ---------------------
def compute_gradient_norms(model, audio):
    model.train()
    audio.requires_grad = True
    
    output = model.encoder(audio)  # Forward pass
    loss = output.sum()  # Dummy loss
    loss.backward()  # Backpropagation
    
    layer_gradients = []
    for i, layer in enumerate(model.encoder.blocks):
        grad_norm = torch.norm(layer[0].weight.grad).item()  # Compute L2 norm of gradients
        layer_gradients.append((i, grad_norm))
    
    return sorted(layer_gradients, key=lambda x: x[1])  # Sort (low gradient norm = redundant)

# --------------------- Step 3: Attention Scores ---------------------
def compute_attention_scores(model):
    layer_attention_scores = []
    for i, layer in enumerate(model.encoder.blocks):
        attn_weights = layer.self_attn.weight.mean().item()  # Compute mean attention score
        layer_attention_scores.append((i, attn_weights))

    return sorted(layer_attention_scores, key=lambda x: x[1])  # Sort (low attention = redundant)

# --------------------- Step 4: L1 Norm (Weight Sparsity) ---------------------
def compute_l1_norm(model):
    layer_l1_norms = []
    for i, layer in enumerate(model.encoder.blocks):
        l1_norm = torch.sum(torch.abs(layer[0].weight)).item()  # Compute L1 norm
        layer_l1_norms.append((i, l1_norm))

    return sorted(layer_l1_norms, key=lambda x: x[1])  # Sort (low L1 norm = redundant)

# --------------------- Step 5: Rank and Prune Layers ---------------------
def prune_layers(model, num_layers_to_remove=6):
    # Compute importance scores
    act_energy = compute_activation_energy(model, audio)
    grad_norms = compute_gradient_norms(model, audio)
    attn_scores = compute_attention_scores(model)
    l1_norms = compute_l1_norm(model)

    # Combine rankings (lower ranks = more redundant)
    avg_ranking = {}
    for i in range(len(act_energy)):
        layer_id = act_energy[i][0]
        avg_ranking[layer_id] = (
            i + grad_norms[i][1] + attn_scores[i][1] + l1_norms[i][1]
        )

    # Select lowest-ranked layers for pruning
    pruned_layers = sorted(avg_ranking.items(), key=lambda x: x[1])[:num_layers_to_remove]
    pruned_layer_ids = [layer[0] for layer in pruned_layers]

    # Remove selected layers
    model.encoder.blocks = torch.nn.ModuleList(
        [layer for i, layer in enumerate(model.encoder.blocks) if i not in pruned_layer_ids]
    )
    model.decoder.blocks = torch.nn.ModuleList(
        [layer for i, layer in enumerate(model.decoder.blocks) if i not in pruned_layer_ids]
    )

    print(f"Pruned {num_layers_to_remove} layers. New Encoder Size: {len(model.encoder.blocks)}")
    return model

# --------------------- Step 6: Fine-Tune Pruned Model ---------------------
def fine_tune_model(model, dataset, epochs=5):
    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        for batch in dataset:
            audio = whisper.pad_or_trim(batch["audio"]["array"])
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            optimizer.zero_grad()
            outputs = model(mel)
            loss = outputs.loss  # Compute loss
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1} | Loss: {loss.item()}")

# --------------------- Step 7: Execute Pruning & Fine-tuning ---------------------
pruned_model = prune_layers(model, num_layers_to_remove=6)
# fine_tune_model(pruned_model, dataset)  # Uncomment to fine-tune

# Save the pruned model
torch.save(pruned_model.state_dict(), "whisper_medium_pruned.pth")

print("Model pruning complete and saved!")
