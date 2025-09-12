#!/usr/bin/env python3
"""
Simple benchmark script for standalone_main.py
Tests runtime with and without reference model for different numbers of tests
"""

import os
import sys
import time
import argparse
import subprocess
import tempfile
from datetime import datetime


def run_test(passing_dir, failing_dir, max_files, use_model=False, model_path=None, timeout=600):
    """Run standalone_main.py and measure runtime"""
    
    output_dir = tempfile.mkdtemp(prefix="bench_")
    
    try:
        # Build command
        cmd = ["python", "standalone_main.py"]
        cmd.extend(["--passing-dir", passing_dir])
        cmd.extend(["--failing-dir", failing_dir])
        cmd.extend(["--output-dir", output_dir])
        cmd.extend(["--max-files", str(max_files)])
        
        if use_model and model_path:
            if os.path.exists(model_path):
                cmd.extend(["--reference-model", model_path, "--load-model"])
                mode = "with model"
            else:
                cmd.extend(["--reference-model", model_path, "--save-model"])
                mode = "save model"
        else:
            mode = "no model"
        
        print(f"Running {max_files} files, {mode}")
        
        # Measure runtime
        start_time = time.time_ns()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        end_time = time.time_ns()
        
        runtime_ps = (end_time - start_time) * 1000  # Convert ns to ps
        
        if result.returncode == 0:
            print(f"  Success: {runtime_ps:,} ps ({(runtime_ps/1e12):.2f} seconds)")
            return runtime_ps
        else:
            print(f"  Failed: {result.stderr[-200:]}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"  Timeout after {timeout} seconds")
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None
    finally:
        # Cleanup
        try:
            import shutil
            shutil.rmtree(output_dir)
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description="Simple benchmark for standalone_main.py")
    parser.add_argument("--passing-dir", required=True)
    parser.add_argument("--failing-dir", required=True)
    parser.add_argument("--output-file", default="simple_benchmark.txt")
    parser.add_argument("--timeout", type=int, default=600)
    
    args = parser.parse_args()
    
    print("Simple Benchmark for standalone_main.py")
    print("=" * 50)
    
    # Check if script exists
    if not os.path.exists("standalone_main.py"):
        print("Error: standalone_main.py not found")
        sys.exit(1)
    
    test_counts = [5, 10, 15, 20, 25, 30]
    model_path = "benchmark_model.pkl"
    
    # Initialize output file
    with open(args.output_file, "w") as f:
        f.write(f"# Simple Benchmark Results for standalone_main.py\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Format: mode, num_tests, runtime_ps\n")
    
    results = {}
    
    # Phase 1: Without model (baseline)
    print("\nPhase 1: Without reference model")
    print("-" * 30)
    
    for count in test_counts:
        runtime = run_test(args.passing_dir, args.failing_dir, count, use_model=False)
        if runtime:
            results[f"no_model_{count}"] = runtime
            with open(args.output_file, "a") as f:
                f.write(f"no model, {count}, {runtime}\n")
    
    # Phase 2: With model (first run saves, subsequent runs load)
    print("\nPhase 2: With reference model")
    print("-" * 30)
    
    for count in test_counts:
        runtime = run_test(args.passing_dir, args.failing_dir, count, use_model=True, model_path=model_path)
        if runtime:
            if os.path.exists(model_path):
                mode_name = "with model"
            else:
                mode_name = "save model"
            results[f"with_model_{count}"] = runtime
            with open(args.output_file, "a") as f:
                f.write(f"{mode_name}, {count}, {runtime}\n")
    
    # Summary
    print("\nBenchmark Results:")
    print("=" * 50)
    
    for count in test_counts:
        no_model_key = f"no_model_{count}"
        with_model_key = f"with_model_{count}"
        
        if no_model_key in results and with_model_key in results:
            no_model_time = results[no_model_key]
            with_model_time = results[with_model_key]
            
            speedup = ((no_model_time - with_model_time) / no_model_time) * 100
            
            print(f"{count} files:")
            print(f"  No model:   {no_model_time/1e12:.2f}s")
            print(f"  With model: {with_model_time/1e12:.2f}s")
            print(f"  Speedup:    {speedup:+.1f}%")
            print()
    
    print(f"Results saved to: {args.output_file}")
    
    # Cleanup
    try:
        os.remove(model_path)
    except:
        pass


if __name__ == "__main__":
    main()