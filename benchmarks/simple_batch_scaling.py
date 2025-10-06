#!/usr/bin/env python3
"""
Simple batch scaling benchmark to verify batch inference functionality.
Tests batch sizes 1, 2, 4, 8, 16, 32, 64 with 50 token generation.
"""

import asyncio
import time
import torch
import sys
from bloombee.utils import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer

async def run_batch_scaling_test(peer_address, model_name="huggyllama/llama-7b"):
    """Run simple batch scaling test"""
    
    print("="*60)
    print("SIMPLE BATCH SCALING BENCHMARK")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Server: {peer_address}")
    print(f"Generation length: 50 tokens")
    print("="*60)
    
    # Setup
    print("\nLoading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoDistributedModelForCausalLM.from_pretrained(
        model_name,
        initial_peers=[peer_address],
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    print("Setup complete!")
    
    # Test batch sizes
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    results = {}
    
    prompt = "Hello, how are you today?"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    print("\nRunning tests...")
    print(f"{'Batch':<6} {'Time(s)':<8} {'Tokens/s':<10} {'Status':<10}")
    print("-"*40)
    
    for batch_size in batch_sizes:
        try:
            # Prepare batch
            batch_inputs = input_ids.repeat(batch_size, 1)
            
            # Generate
            start = time.time()
            outputs = model.generate_batch(
                inputs=batch_inputs,
                batch_size=batch_size,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            elapsed = time.time() - start
            
            # Calculate metrics
            total_tokens = outputs.shape[0] * outputs.shape[1]
            tokens_per_sec = total_tokens / elapsed
            
            # Store results
            results[batch_size] = {
                'time': elapsed,
                'tokens_per_sec': tokens_per_sec,
                'success': True
            }
            
            print(f"{batch_size:<6} {elapsed:<8.2f} {tokens_per_sec:<10.1f} {'SUCCESS':<10}")
            
            # Brief pause
            await asyncio.sleep(0.5)
            
        except Exception as e:
            print(f"{batch_size:<6} {'N/A':<8} {'N/A':<10} {'FAILED':<10}")
            print(f"  Error: {str(e)[:50]}...")
            results[batch_size] = {
                'error': str(e),
                'success': False
            }
            # Stop on failure
            break
    
    # Print summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    successful = [b for b, r in results.items() if r.get('success', False)]
    if len(successful) >= 2:
        baseline = successful[0]
        baseline_tps = results[baseline]['tokens_per_sec']
        
        print(f"Baseline (Batch {baseline}): {baseline_tps:.1f} tokens/sec")
        print()
        
        for batch in successful[1:]:
            tps = results[batch]['tokens_per_sec']
            speedup = tps / baseline_tps
            efficiency = speedup / (batch / baseline)
            print(f"Batch {batch:3d}: {speedup:5.2f}x speedup, {efficiency:5.1%} efficiency")
    
    print("\n" + "="*60)
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python simple_batch_scaling.py <peer_address> [model_name]")
        print("Example: python simple_batch_scaling.py /ip4/127.0.0.1/tcp/29501/p2p/12D3Koo...")
        sys.exit(1)
    
    peer_address = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "huggyllama/llama-7b"
    
    asyncio.run(run_batch_scaling_test(peer_address, model_name))

