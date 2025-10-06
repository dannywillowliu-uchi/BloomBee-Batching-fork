#!/usr/bin/env python3
"""
Simple batch inference test for BloomBee
Tests that batch generation produces multiple tokens per sequence
"""

import torch
from bloombee import AutoDistributedModelForCausalLM
from bloombee.client.remote_generation import RemoteGenerationMixin
import time

def test_batch_inference():
    print("🚀 Starting BloomBee batch inference test...")
    
    # Initialize client config with both workers
    initial_peers = [
        "/ip4/127.0.0.1/tcp/31337/p2p/12D3KooWFebTEYNwu77f4SoGEHmfts5qpbTidfpowxdEoKwDz6mf",
        "/ip4/127.0.0.1/tcp/31338/p2p/12D3KooWFebTEYNwu77f4SoGEHmfts5qpbTidfpowxdEoKwDz6mf"
    ]
    
    print(f"🔗 Connecting to workers: {initial_peers}")
    
    print("📚 Loading model...")
    model = AutoDistributedModelForCausalLM.from_pretrained(
        "huggyllama/llama-7b",
        initial_peers=initial_peers,
        device="cpu"
    )
    
    # Test prompt
    prompt = "Hi there!"
    print(f"📝 Test prompt: '{prompt}'")
    
    # Test single generation first
    print("\n🔍 Testing single generation...")
    single_output = model.generate(
        input_ids=model.tokenizer.encode(prompt, return_tensors="pt"),
        max_new_tokens=5,
        do_sample=False,
        temperature=1.0
    )
    single_text = model.tokenizer.decode(single_output[0], skip_special_tokens=True)
    print(f"✅ Single output: '{single_text}'")
    
    # Test batch generation
    print("\n🔍 Testing batch generation...")
    batch_size = 2
    max_new_tokens = 5
    
    # Create batch input
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
    batch_inputs = input_ids.expand(batch_size, -1)  # [1, seq_len] -> [batch_size, seq_len]
    
    print(f"📊 Batch size: {batch_size}, max_new_tokens: {max_new_tokens}")
    print(f"📊 Input shape: {batch_inputs.shape}")
    
    # Test generate_batch method
    try:
        batch_outputs = model.generate_batch(
            inputs=batch_inputs,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0
        )
        
        print(f"✅ Batch output shape: {batch_outputs.shape}")
        
        # Decode each sequence in the batch
        for i in range(batch_size):
            batch_text = model.tokenizer.decode(batch_outputs[i], skip_special_tokens=True)
            print(f"   Batch[{i}]: '{batch_text}'")
            
        # Verify that batch outputs have more tokens than input
        input_length = input_ids.shape[1]
        expected_length = input_length + max_new_tokens
        
        if batch_outputs.shape[1] >= expected_length:
            print(f"✅ SUCCESS: Batch generation produced {batch_outputs.shape[1]} tokens (expected ≥{expected_length})")
        else:
            print(f"❌ FAILED: Batch generation only produced {batch_outputs.shape[1]} tokens (expected ≥{expected_length})")
            
    except Exception as e:
        print(f"❌ ERROR in batch generation: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🏁 Test completed!")

if __name__ == "__main__":
    test_batch_inference()
