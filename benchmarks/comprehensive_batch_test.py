#!/usr/bin/env python3
"""
Comprehensive batch inference testing script for BloomBee TinyLlama.
Tests various combinations of input sizes, batch sizes up to 1024, and generation lengths up to 1024.
"""

import asyncio
import time
import torch
import psutil
import gc
import json
from datetime import datetime
from bloombee.utils import AutoDistributedModelForCausalLM
from transformers import AutoTokenizer

class ComprehensiveBatchTester:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.results = {}
        self.system_stats = []
        self.start_memory = None
        
    async def setup_model(self):
        """Initialize the BloomBee model and tokenizer."""
        print("🔧 Setting up BloomBee model for comprehensive testing...")
        
        # Model configuration
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        initial_peers = [
            "/ip4/127.0.0.1/tcp/29501/p2p/12D3KooWQmptxm41APkFKXDZ73GEz37Xyvx6jtyAJQKCFWuantyj"
        ]
        
        try:
            # Record initial system state
            self.start_memory = self._get_system_memory_info()
            print(f"📊 Initial system memory: {self.start_memory['available_gb']:.2f} GB available")
            
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
    
    def prepare_inputs(self, input_length=8):
        """Prepare input tokens of specified length."""
        prompts = {
            8: "Hello, how are you today?",
            16: "Hello, how are you today? I hope you're having a wonderful day.",
            32: "Hello, how are you today? I hope you're having a wonderful day. Can you tell me about your favorite hobbies and interests?",
            64: "Hello, how are you today? I hope you're having a wonderful day. Can you tell me about your favorite hobbies and interests? I'm particularly interested in learning about creative activities that people enjoy in their free time.",
            128: "Hello, how are you today? I hope you're having a wonderful day. Can you tell me about your favorite hobbies and interests? I'm particularly interested in learning about creative activities that people enjoy in their free time. Some people love painting, others enjoy music, writing, sports, cooking, gardening, or reading books. What brings you joy?"
        }
        
        # Use appropriate prompt or create one of desired length
        if input_length in prompts:
            prompt = prompts[input_length]
        else:
            # Create a prompt of approximately the desired length
            base_prompt = "Hello, how are you today? "
            words_needed = input_length - len(self.tokenizer.encode(base_prompt))
            filler = "Please tell me more about this topic. " * (words_needed // 8 + 1)
            prompt = base_prompt + filler
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]
        
        # Trim or pad to exact length if needed
        if input_ids.shape[1] > input_length:
            input_ids = input_ids[:, :input_length]
        elif input_ids.shape[1] < input_length:
            # Pad with the tokenizer's pad token
            pad_length = input_length - input_ids.shape[1]
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            padding = torch.full((1, pad_length), pad_token_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, padding], dim=1)
        
        return input_ids
    
    async def test_configuration(self, input_length, batch_size, max_new_tokens):
        """Test a specific configuration of input length, batch size, and generation length."""
        config_name = f"input_{input_length}_batch_{batch_size}_gen_{max_new_tokens}"
        print(f"\n🧪 Testing: {config_name}")
        
        # Memory check before test
        memory_before = self._get_system_memory_info()
        
        try:
            # Prepare batch input
            input_ids = self.prepare_inputs(input_length)
            batch_inputs = input_ids.repeat(batch_size, 1)
            
            print(f"   📊 Input shape: {batch_inputs.shape}")
            print(f"   🎯 Generating {max_new_tokens} tokens per sequence")
            
            # Force garbage collection before test
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            start_time = time.time()
            
            # Generate tokens with retry logic
            max_retries = 3
            generated_tokens = None
            
            for attempt in range(max_retries):
                try:
                    generated_tokens = self.model.generate_batch(
                        inputs=batch_inputs,
                        batch_size=batch_inputs.shape[0],
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    break  # Success
                    
                except (IndexError, RuntimeError, MemoryError) as e:
                    error_msg = str(e).lower()
                    if any(keyword in error_msg for keyword in ['index', 'out of bounds', 'memory', 'cuda']):
                        if attempt < max_retries - 1:
                            print(f"   ⚠️  Error (attempt {attempt + 1}/{max_retries}): {str(e)[:100]}...")
                            await asyncio.sleep(1.0)
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            continue
                        else:
                            raise e
                    else:
                        raise e
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Memory check after test
            memory_after = self._get_system_memory_info()
            
            # Calculate metrics
            if generated_tokens is not None:
                total_tokens = generated_tokens.shape[0] * generated_tokens.shape[1]
                tokens_per_second = total_tokens / generation_time
                
                # Decode sample output
                first_sequence = generated_tokens[0]
                decoded_text = self.tokenizer.decode(first_sequence, skip_special_tokens=True)
                sample_text = decoded_text[:200] + "..." if len(decoded_text) > 200 else decoded_text
                
                # Store results
                result = {
                    'input_length': input_length,
                    'batch_size': batch_size,
                    'max_new_tokens': max_new_tokens,
                    'generation_time': generation_time,
                    'total_tokens': total_tokens,
                    'tokens_per_second': tokens_per_second,
                    'memory_before_gb': memory_before['available_gb'],
                    'memory_after_gb': memory_after['available_gb'],
                    'memory_used_gb': memory_before['available_gb'] - memory_after['available_gb'],
                    'sample_text': sample_text,
                    'success': True
                }
                
                self.results[config_name] = result
                
                # Display results
                print(f"   ✅ {total_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} t/s)")
                print(f"   💾 Memory: {memory_before['available_gb']:.2f} → {memory_after['available_gb']:.2f} GB available")
                print(f"   📝 Sample: {repr(sample_text)}")
                
                return True
            else:
                raise RuntimeError("Generation failed - no tokens produced")
                
        except Exception as e:
            print(f"   ❌ Failed: {str(e)[:100]}...")
            
            # Store failure result
            memory_after = self._get_system_memory_info()
            self.results[config_name] = {
                'input_length': input_length,
                'batch_size': batch_size,
                'max_new_tokens': max_new_tokens,
                'error': str(e),
                'memory_before_gb': memory_before['available_gb'],
                'memory_after_gb': memory_after['available_gb'],
                'success': False
            }
            
            return False
    
    async def run_comprehensive_tests(self):
        """Run comprehensive tests with various configurations."""
        print("🚀 Starting comprehensive batch inference tests...")
        
        # Test configurations: (input_length, batch_sizes, max_new_tokens_list)
        test_configs = [
            # Small tests - basic functionality
            (8, [1, 2, 4], [10, 50]),
            (16, [1, 2, 4, 8], [10, 50, 100]),
            
            # Medium tests - scaling up
            (32, [1, 2, 4, 8, 16], [10, 50, 100, 200]),
            (64, [1, 2, 4, 8, 16, 32], [50, 100, 200]),
            
            # Large input tests
            (128, [1, 2, 4, 8, 16], [100, 200, 500]),
            
            # High batch size tests (smaller generations to manage memory)
            (8, [64, 128, 256, 512], [10, 50]),
            (16, [64, 128, 256], [10, 50, 100]),
            (32, [32, 64, 128], [50, 100]),
            
            # High generation tests (smaller batches to manage memory)
            (8, [1, 2, 4], [500, 1000]),
            (16, [1, 2], [500, 1000]),
            (32, [1], [1000]),
            
            # Extreme tests (if system can handle)
            (8, [1024], [10]),  # Maximum batch size
            (8, [1], [1024]),   # Maximum generation length
        ]
        
        total_tests = sum(len(batch_sizes) * len(gen_lengths) for _, batch_sizes, gen_lengths in test_configs)
        completed_tests = 0
        failed_tests = 0
        
        print(f"📊 Total test configurations: {total_tests}")
        
        for input_length, batch_sizes, gen_lengths in test_configs:
            for batch_size in batch_sizes:
                for max_new_tokens in gen_lengths:
                    completed_tests += 1
                    
                    print(f"\n🔄 Progress: {completed_tests}/{total_tests}")
                    
                    success = await self.test_configuration(input_length, batch_size, max_new_tokens)
                    
                    if not success:
                        failed_tests += 1
                        
                        # Stop if we hit too many failures in a row or memory issues
                        if failed_tests >= 5:
                            print(f"\n⚠️  Stopping tests due to {failed_tests} consecutive failures")
                            break
                    else:
                        failed_tests = 0  # Reset failure counter on success
                    
                    # Brief pause between tests
                    await asyncio.sleep(0.5)
                    
                    # Memory check - stop if system is running low
                    current_memory = self._get_system_memory_info()
                    if current_memory['available_gb'] < 2.0:  # Less than 2GB available
                        print(f"\n⚠️  Stopping tests due to low system memory: {current_memory['available_gb']:.2f} GB available")
                        break
                
                if failed_tests >= 5 or current_memory['available_gb'] < 2.0:
                    break
            
            if failed_tests >= 5 or current_memory['available_gb'] < 2.0:
                break
        
        print(f"\n📊 Completed {completed_tests} test configurations")
        self.print_comprehensive_results()
        self.save_results_to_file()
    
    def print_comprehensive_results(self):
        """Print comprehensive test results."""
        if not self.results:
            print("No results to display.")
            return
        
        print(f"\n{'='*80}")
        print("📊 COMPREHENSIVE TEST RESULTS")
        print(f"{'='*80}")
        
        successful_results = {k: v for k, v in self.results.items() if v.get('success', False)}
        failed_results = {k: v for k, v in self.results.items() if not v.get('success', False)}
        
        print(f"✅ Successful tests: {len(successful_results)}")
        print(f"❌ Failed tests: {len(failed_results)}")
        
        if successful_results:
            print(f"\n{'Config':<30} {'Time(s)':<8} {'Tokens':<8} {'T/s':<8} {'Mem(GB)':<8}")
            print("-" * 70)
            
            for config_name in sorted(successful_results.keys()):
                result = successful_results[config_name]
                config_short = f"{result['input_length']}/{result['batch_size']}/{result['max_new_tokens']}"
                time_str = f"{result['generation_time']:.2f}"
                tokens_str = f"{result['total_tokens']}"
                tps_str = f"{result['tokens_per_second']:.1f}"
                mem_str = f"{result['memory_used_gb']:+.2f}"
                
                print(f"{config_short:<30} {time_str:<8} {tokens_str:<8} {tps_str:<8} {mem_str:<8}")
        
        # Analyze performance patterns
        self._analyze_performance_patterns()
        
        # Final system memory check
        final_memory = self._get_system_memory_info()
        print(f"\n💾 SYSTEM MEMORY ANALYSIS:")
        print(f"   Initial available: {self.start_memory['available_gb']:.2f} GB")
        print(f"   Final available: {final_memory['available_gb']:.2f} GB")
        print(f"   Net change: {final_memory['available_gb'] - self.start_memory['available_gb']:+.2f} GB")
        print(f"   Current usage: {final_memory['used_percent']:.1f}%")
    
    def _analyze_performance_patterns(self):
        """Analyze performance patterns from test results."""
        successful_results = [v for v in self.results.values() if v.get('success', False)]
        
        if len(successful_results) < 2:
            return
        
        print(f"\n📈 PERFORMANCE ANALYSIS:")
        
        # Find best performance by tokens/second
        best_tps = max(successful_results, key=lambda x: x['tokens_per_second'])
        print(f"   🏆 Best throughput: {best_tps['tokens_per_second']:.1f} t/s")
        print(f"      Config: {best_tps['input_length']}/{best_tps['batch_size']}/{best_tps['max_new_tokens']}")
        
        # Find most memory efficient
        memory_efficient = [r for r in successful_results if r['memory_used_gb'] >= 0]
        if memory_efficient:
            best_memory = min(memory_efficient, key=lambda x: x['memory_used_gb'] / x['total_tokens'])
            print(f"   💾 Most memory efficient: {best_memory['memory_used_gb'] / best_memory['total_tokens']:.6f} GB/token")
            print(f"      Config: {best_memory['input_length']}/{best_memory['batch_size']}/{best_memory['max_new_tokens']}")
        
        # Batch size scaling analysis
        batch_sizes = sorted(set(r['batch_size'] for r in successful_results))
        if len(batch_sizes) > 1:
            print(f"   📊 Batch size scaling:")
            for batch_size in batch_sizes:
                batch_results = [r for r in successful_results if r['batch_size'] == batch_size]
                if batch_results:
                    avg_tps = sum(r['tokens_per_second'] for r in batch_results) / len(batch_results)
                    print(f"      Batch {batch_size:3d}: {avg_tps:6.1f} avg t/s ({len(batch_results)} tests)")
    
    def save_results_to_file(self):
        """Save detailed results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"comprehensive_test_results_{timestamp}.json"
        
        # Prepare data for JSON serialization
        json_data = {
            'test_info': {
                'timestamp': timestamp,
                'total_tests': len(self.results),
                'successful_tests': len([r for r in self.results.values() if r.get('success', False)]),
                'failed_tests': len([r for r in self.results.values() if not r.get('success', False)]),
                'start_memory_gb': self.start_memory['available_gb'],
                'final_memory_gb': self._get_system_memory_info()['available_gb']
            },
            'results': self.results
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"\n💾 Results saved to: {filename}")
        except Exception as e:
            print(f"\n⚠️  Failed to save results: {e}")
    
    def _get_system_memory_info(self):
        """Get comprehensive system memory information."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'used_percent': memory.percent,
            'free_gb': memory.free / (1024**3)
        }

async def main():
    """Main function to run comprehensive batch inference tests."""
    tester = ComprehensiveBatchTester()
    
    try:
        await tester.setup_model()
        await tester.run_comprehensive_tests()
        
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        if tester.results:
            print("Saving partial results...")
            tester.print_comprehensive_results()
            tester.save_results_to_file()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        if tester.results:
            print("Saving partial results...")
            tester.save_results_to_file()
        raise

if __name__ == "__main__":
    asyncio.run(main())
