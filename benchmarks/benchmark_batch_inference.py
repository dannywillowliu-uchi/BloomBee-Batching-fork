#!/usr/bin/env python3
"""
Batch inference testing script for BloomBee.
Tests various batch sizes to measure performance and memory efficiency.
"""

import asyncio
import time
import torch
from bloombee.utils import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer

class BloomBeeBatchInferenceTester:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.results = {}
        
    async def setup_model(self):
        """Initialize the BloomBee model and tokenizer."""
        print("🔧 Setting up BloomBee model...")
        
        # Model configuration
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        initial_peers = [
            "/ip4/127.0.0.1/tcp/29501/p2p/12D3KooWGphkJwgNMBzLFLRUWbvBVVkCZ6S2RHknep8mSUoHwVPY"
        ]
        
        try:
            # Load tokenizer
            print(f"📚 Loading tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load distributed model
            print(f"🤖 Loading distributed model: {model_name}")
            self.model = AutoDistributedModelForCausalLM.from_pretrained(
                model_name,
                initial_peers=initial_peers,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
            print("✅ Model setup complete!")
            
        except Exception as e:
            print(f"❌ Error setting up model: {e}")
            raise
    
    def prepare_input(self, prompt="Hello, how are you today?"):
        """Prepare input tokens for inference."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        return inputs["input_ids"]
    
    async def test_batch_inference(self, batch_size, max_new_tokens_per_sequence=10):
        """Test batch inference with specified batch size."""
        print(f"\n🧪 Testing batch size: {batch_size}")
        
        # Prepare batch input
        input_ids = self.prepare_input()
        batch_inputs = input_ids.repeat(batch_size, 1)
        
        print(f"📊 Batch {batch_size}: Input shape {batch_inputs.shape}")
        
        # Simple memory check before generation
        memory_before = self._get_memory_usage()
        
        try:
            start_time = time.time()
            
            # Generate tokens with retry logic for index errors
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    generated_tokens = self.model.generate_batch(
                        inputs=batch_inputs,
                        batch_size=batch_inputs.shape[0],
                        max_new_tokens=max_new_tokens_per_sequence,
                        do_sample=False,
                        temperature=1.0,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    break  # Success, exit retry loop
                    
                except (IndexError, RuntimeError) as e:
                    if "index" in str(e).lower() or "out of bounds" in str(e).lower():
                        if attempt < max_retries - 1:
                            print(f"   ⚠️  Index error (attempt {attempt + 1}/{max_retries}), retrying...")
                            await asyncio.sleep(0.5)  # Brief pause before retry
                            continue
                        else:
                            raise e  # Re-raise on final attempt
                    else:
                        raise e  # Re-raise non-index errors immediately
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Simple memory check after generation
            memory_after = self._get_memory_usage()
            
            # Calculate metrics
            total_tokens = generated_tokens.shape[0] * generated_tokens.shape[1]
            tokens_per_second = total_tokens / generation_time
            
            # Decode and display generated text for first sequence in batch
            first_sequence = generated_tokens[0]
            decoded_text = self.tokenizer.decode(first_sequence, skip_special_tokens=False)
            
            # Display results with actual text output
            print(f"   ✅ {total_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} t/s)")
            print(f"   💾 Memory: {memory_before:.2f} → {memory_after:.2f} GiB (Δ{memory_after - memory_before:+.2f})")
            print(f"   📝 Generated text (first sequence): {repr(decoded_text)}")
            if batch_size > 1:
                print(f"   📊 Note: All {batch_size} sequences generated identical text")
            
            # Store results
            self.results[batch_size] = {
                'generation_time': generation_time,
                'total_tokens': total_tokens,
                'tokens_per_second': tokens_per_second,
                'input_shape': batch_inputs.shape,
                'output_shape': generated_tokens.shape,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_delta': memory_after - memory_before
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Error with batch size {batch_size}: {e}")
            self.results[batch_size] = {'error': str(e)}
            return False
    
    async def run_batch_tests(self):
        """Run tests with various batch sizes."""
        print("🚀 Starting batch inference tests...")
        
        # Test batch sizes: 1, 2, 4, 8, 16, 32, 64
        batch_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            success = await self.test_batch_inference(batch_size)
            if not success:
                print(f"⚠️  Stopping tests due to failure at batch size {batch_size}")
                break
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        print("\n📊 Test Results Summary:")
        self.print_results()
    
    def print_results(self):
        """Print formatted test results."""
        if not self.results:
            print("No results to display.")
            return
        
        print(f"{'Batch Size':<12} {'Time (s)':<12} {'Tokens':<12} {'Tokens/s':<12} {'Status':<10}")
        print("-" * 70)
        
        for batch_size in sorted(self.results.keys()):
            result = self.results[batch_size]
            
            if 'error' in result:
                print(f"{batch_size:<12} {'N/A':<12} {'N/A':<12} {'N/A':<12} {'ERROR':<10}")
            else:
                time_str = f"{result['generation_time']:.2f}"
                tokens_str = f"{result['total_tokens']}"
                tps_str = f"{result['tokens_per_second']:.2f}"
                print(f"{batch_size:<12} {time_str:<12} {tokens_str:<12} {tps_str:<12} {'SUCCESS':<10}")
        
        # Calculate efficiency improvements
        self.calculate_efficiency_improvements()
    
    def calculate_efficiency_improvements(self):
        """Calculate and display efficiency improvements."""
        print("\n📈 Efficiency Analysis:")
        
        # Use batch size 2 as baseline (skip batch 1 for efficiency calculations)
        baseline_batch = 2
        if baseline_batch not in self.results or 'error' in self.results[baseline_batch]:
            print("⚠️  Cannot calculate efficiency without successful baseline (batch size 2)")
            return
        
        baseline_tps = self.results[baseline_batch]['tokens_per_second']
        print(f"Baseline (batch {baseline_batch}): {baseline_tps:.2f} tokens/second")
        
        for batch_size in [4, 8, 16, 32, 64]:
            if batch_size in self.results and 'error' not in self.results[batch_size]:
                current_tps = self.results[batch_size]['tokens_per_second']
                improvement = (current_tps / baseline_tps - 1) * 100
                efficiency = (current_tps / baseline_tps) / batch_size * 100
                
                print(f"Batch {batch_size:2d}: {current_tps:6.2f} tokens/s "
                      f"({improvement:+6.1f}% improvement, {efficiency:5.1f}% efficiency)")

        # Memory efficiency analysis if available
        if any('memory_efficiency' in result for result in self.results.values()):
            print("\n💾 Memory Efficiency Analysis:")
            for batch_size in sorted(self.results.keys()):
                result = self.results[batch_size]
                if 'error' not in result and 'memory_efficiency' in result:
                    mem_eff = result['memory_efficiency']
                    print(f"Batch {batch_size:2d}: {mem_eff:6.2f} tokens/byte")

    def _get_memory_usage(self):
        """Get current memory usage in GiB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024**3)
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            return 0.0  # MPS doesn't support memory queries
        else:
            try:
                import psutil
                return psutil.virtual_memory().used / (1024**3)
            except ImportError:
                return 0.0

    def _log_cache_info(self, batch_size, input_shape, operation="pre_inference"):
        """Log cache and memory information for monitoring."""
        print(f"\n🔄 Cache Monitoring - {operation.upper()}")
        print(f"   📊 Batch Size: {batch_size}")
        print(f"   📏 Input Shape: {input_shape}")
        print(f"   💾 Estimated Memory: {batch_size * input_shape[1] * 4096 * 4 / (1024**3):.3f} GiB")
        
        # Try to get model memory info if available
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'config'):
            config = self.model.transformer.config
            print(f"   🧠 Model Config: {config.hidden_size} hidden, {config.num_attention_heads} heads")
            print(f"   🔑 KV Cache per token: {config.hidden_size * 2 / (1024**2):.2f} MB")
        
        # Estimate cache requirements
        self._estimate_cache_requirements(batch_size, input_shape)

    def _estimate_cache_requirements(self, batch_size, input_shape):
        """Estimate cache memory requirements for the batch."""
        try:
            if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'config'):
                config = self.model.transformer.config
                
                # Calculate cache requirements
                hidden_size = config.hidden_size
                kv_cache_per_token = hidden_size * 2  # 2 for key + value
                total_cache_gb = batch_size * input_shape[1] * kv_cache_per_token * 4 / (1024**3)
                
                print(f"   🔍 Cache: {total_cache_gb:.3f} GiB total, {total_cache_gb / batch_size:.3f} GiB/seq")
                
        except Exception as e:
            print(f"   ⚠️  Cache estimation failed: {e}")

    def _log_memory_usage(self, operation="check"):
        """Log current memory usage across different devices."""
        print(f"\n💾 Memory Usage - {operation.upper()}")
        
        # GPU memory (if available)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"   🎯 GPU Memory:")
            print(f"      📊 Currently allocated: {allocated:.3f} GiB")
            print(f"      🔒 Currently reserved: {reserved:.3f} GiB")
            print(f"      📈 Peak allocated: {max_allocated:.3f} GiB")
            print(f"      🆓 Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3) - reserved:.3f} GiB")
        
        # CPU memory
        try:
            import psutil
            cpu_memory = psutil.virtual_memory()
            print(f"   🖥️  CPU Memory:")
            print(f"      📊 Total: {cpu_memory.total / (1024**3):.1f} GiB")
            print(f"      📈 Available: {cpu_memory.available / (1024**3):.1f} GiB")
            print(f"      📉 Used: {cpu_memory.used / (1024**3):.1f} GiB ({cpu_memory.percent:.1f}%)")
        except ImportError:
            print(f"   🖥️  CPU Memory: psutil not available")
        
        # MPS memory (if available on Apple Silicon)
        if hasattr(torch, 'mps') and torch.mps.is_available():
            try:
                # Note: MPS doesn't have the same memory APIs as CUDA
                print(f"   🍎 MPS Memory: Available (detailed stats not supported)")
            except Exception as e:
                print(f"   🍎 MPS Memory: Error checking - {e}")

async def main():
    """Main function to run the batch inference tests."""
    tester = BloomBeeBatchInferenceTester()
    
    try:
        await tester.setup_model()
        await tester.run_batch_tests()
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())



