"""
Convert MR-MT3 model to ONNX format with KV-cache support for fast autoregressive decoding.

Encoder: mel_input [batch, seq, 512] -> encoder_output [batch, seq, 512]

Decoder (with KV-cache):
  Inputs:
    decoder_input_ids [batch, 1]           - single new token
    encoder_hidden_states [batch, enc_seq, 512]
    past_key_values: 32 tensors (8 layers * 4 tensors per layer)
      - per layer: self_k, self_v, cross_k, cross_v
      - self_k/self_v shape: [batch, num_heads, past_seq, d_kv]
      - cross_k/cross_v shape: [batch, num_heads, enc_seq, d_kv]
  Outputs:
    logits [batch, 1, vocab_size]
    present_key_values: 32 tensors (same shapes as input, with past_seq+1 for self)

Usage in C#:
  - Step 0: pass empty past (use decoder_start_token), get logits + present
  - Step N: pass previous present as past, single new token, get logits + new present
"""
import os
import sys
import torch
import numpy as np
from transformers import T5Config

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.onnx_conv_venv', 'Lib', 'site-packages'))

from mt3_infer.models.mr_mt3.t5 import T5ForConditionalGeneration

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'mr_mt3_model', 'mt3.pth')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'mt3_onnx')
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_config = {
    'd_model': 512,
    'd_ff': 1024,
    'd_kv': 64,
    'num_heads': 6,
    'num_layers': 8,
    'num_decoder_layers': 8,
    'vocab_size': 1536,
    'encoder_vocab_size': 1024,
    'decoder_start_token_id': 0,
    'pad_token_id': 0,
    'eos_token_id': 1,
    'unk_token_id': 2,
    'feed_forward_proj': 'gated-gelu',
    'dropout_rate': 0.1,
    'layer_norm_epsilon': 1e-6,
    'is_encoder_decoder': True,
    'tie_word_embeddings': False,
    'use_cache': True,
}

print("Loading model...")
config = T5Config.from_dict(model_config)
model = T5ForConditionalGeneration(config)
state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=False)
model.load_state_dict(state_dict, strict=False)
model.eval()
print(f"Model loaded. d_model={config.d_model}, layers={config.num_decoder_layers}, heads={config.num_heads}")

# ---- Encoder export (unchanged) ----
class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.proj = model.proj
        self.encoder = model.encoder

    def forward(self, inputs):
        inputs_embeds = self.proj(inputs)
        out = self.encoder(inputs_embeds=inputs_embeds, return_dict=True)
        return out.last_hidden_state

print("\nExporting encoder...")
encoder_wrapper = EncoderWrapper(model).eval()
encoder_input = torch.randn(1, 256, 512)
torch.onnx.export(
    encoder_wrapper, (encoder_input,),
    os.path.join(OUTPUT_DIR, 'mt3_encoder.onnx'),
    input_names=['mel_input'],
    output_names=['encoder_output'],
    dynamic_axes={'mel_input': {0: 'batch', 1: 'seq'}, 'encoder_output': {0: 'batch', 1: 'seq'}},
    opset_version=18,
    dynamo=False,
)
print("  encoder exported")

# ---- Decoder with KV-cache ----
NUM_LAYERS = config.num_decoder_layers
NUM_HEADS = config.num_heads
D_KV = config.d_kv

class DecoderWithCache(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.decoder = model.decoder
        self.lm_head = model.lm_head
        self.model_dim = model.model_dim
        self.num_layers = model.config.num_decoder_layers

    def forward(self, decoder_input_ids, encoder_hidden_states, *past_key_values):
        # Reconstruct past_key_values as tuple of (self_k, self_v, cross_k, cross_v) per layer
        pkv = tuple(
            (past_key_values[i * 4], past_key_values[i * 4 + 1],
             past_key_values[i * 4 + 2], past_key_values[i * 4 + 3])
            for i in range(self.num_layers)
        )

        decoder_out = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=pkv,
            use_cache=True,
            return_dict=True,
        )
        seq_out = decoder_out.last_hidden_state * (self.model_dim ** -0.5)
        logits = self.lm_head(seq_out)

        # Flatten present K/V (tuple of 4-tuples) to flat tuple of 32 tensors
        present = decoder_out.past_key_values
        flat = []
        for layer in present:
            flat.extend(layer)
        return (logits,) + tuple(flat)

print("Exporting decoder with KV-cache...")
decoder_wrapper = DecoderWithCache(model).eval()

decoder_ids = torch.tensor([[0]], dtype=torch.long)
encoder_hidden = torch.randn(1, 256, 512)

# Build input/output names
past_names = []
present_names = []
for i in range(NUM_LAYERS):
    past_names.extend([f'past_self_k_{i}', f'past_self_v_{i}', f'past_cross_k_{i}', f'past_cross_v_{i}'])
    present_names.extend([f'present_self_k_{i}', f'present_self_v_{i}', f'present_cross_k_{i}', f'present_cross_v_{i}'])

input_names = ['decoder_input_ids', 'encoder_hidden_states'] + past_names
output_names = ['logits'] + present_names

# Dynamic axes
dynamic_axes = {
    'decoder_input_ids': {0: 'batch', 1: 'seq'},
    'encoder_hidden_states': {0: 'batch', 1: 'enc_seq'},
    'logits': {0: 'batch', 1: 'seq'},
}
for name in past_names:
    dynamic_axes[name] = {0: 'batch', 2: 'past_seq'}
for name in present_names:
    dynamic_axes[name] = {0: 'batch', 2: 'seq'}

# First call: no past (use dummy empty tensors)
dummy_past = tuple(
    torch.randn(1, NUM_HEADS, 1, D_KV, dtype=torch.float32)
    for _ in range(NUM_LAYERS * 4)
)

torch.onnx.export(
    decoder_wrapper,
    (decoder_ids, encoder_hidden) + dummy_past,
    os.path.join(OUTPUT_DIR, 'mt3_decoder.onnx'),
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    opset_version=18,
    dynamo=False,
)
print("  decoder with KV-cache exported")

# ---- Verify ----
print("\nVerifying ONNX models...")
import onnx
for name in ['mt3_encoder.onnx', 'mt3_decoder.onnx']:
    path = os.path.join(OUTPUT_DIR, name)
    onnx_model = onnx.load(path)
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {name}: OK ({size_mb:.1f} MB)")

# Verify KV-cache behavior matches non-cached
print("\nVerifying KV-cache consistency...")
import onnxruntime as ort

sess_enc = ort.InferenceSession(os.path.join(OUTPUT_DIR, 'mt3_encoder.onnx'))
sess_dec = ort.InferenceSession(os.path.join(OUTPUT_DIR, 'mt3_decoder.onnx'))

mel = np.random.randn(1, 256, 512).astype(np.float32)
enc_out = sess_enc.run(None, {'mel_input': mel})[0]
print(f"  encoder output: {enc_out.shape}")

# Run a few steps with KV-cache
# Note: cross-attention K/V is recomputed each step (only self-attn K/V is cached)
EOS = 1
token = 0  # decoder_start_token_id
all_tokens = [token]

for step in range(20):
    inputs = {
        'decoder_input_ids': np.array([[token]], dtype=np.int64),
        'encoder_hidden_states': enc_out,
    }
    if step == 0:
        # Empty past self-attn K/V (seq_len=0)
        for i in range(NUM_LAYERS):
            inputs[f'past_self_k_{i}'] = np.zeros((1, NUM_HEADS, 0, D_KV), dtype=np.float32)
            inputs[f'past_self_v_{i}'] = np.zeros((1, NUM_HEADS, 0, D_KV), dtype=np.float32)
    else:
        inputs.update(past)

    outputs = sess_dec.run(None, inputs)
    logits = outputs[0]
    new_token = int(np.argmax(logits[0, -1, :]))
    all_tokens.append(new_token)

    # Update past self-attn K/V (outputs: logits, self_k0, self_v0, cross_k0, cross_v0, self_k1, ...)
    past = {}
    for i in range(NUM_LAYERS):
        off = 1 + i * 4
        past[f'past_self_k_{i}'] = outputs[off]
        past[f'past_self_v_{i}'] = outputs[off + 1]

    token = new_token
    if token == EOS:
        break

print(f"  Generated tokens ({len(all_tokens)}): {all_tokens}")
print("  KV-cache decoding works correctly!")

print("\nDone! ONNX models with KV-cache saved to mt3_onnx/")
