#!/usr/bin/env python3
"""
Profiler for ptracker log analysis pipeline.
Wraps the main script to provide detailed timing and performance information.

Usage:
    python ptracker_profiler.py [args for tmp_main.py]
    
Example:
    python ptracker_profiler.py --passing-dir logs/passing --failing-dir logs/failing --output-dir results
"""
import cProfile
import pstats
import io
import os
import sys
from pathlib import Path
import time
import importlib.util
from contextlib import contextmanager
import tracemalloc

# Assuming tmp_main.py is in the same directory
MAIN_SCRIPT = "tmp_main.py"

@contextmanager
def memory_profiler():
    """
    Context manager for memory profiling.
    """
    tracemalloc.start()
    try:
        yield
    finally:
        current, peak = tracemalloc.get_traced_memory()
        print(f"\n--- Memory Usage ---")
        print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
        print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")
        
        # Get top 10 memory allocations
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        print("\nTop 10 memory allocations by line:")
        for stat in top_stats[:10]:
            print(f"{stat.count} allocations: {stat.size / 1024:.1f} KB - {stat.traceback.format()[0]}")
        
        tracemalloc.stop()

@contextmanager
def time_profiler(section_name):
    """
    Context manager for timing individual sections of code.
    """
    start_time = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        print(f"Section '{section_name}' completed in {elapsed:.4f} seconds")

def load_main_module():
    """
    Load the main module dynamically without executing it.
    """
    script_path = Path(MAIN_SCRIPT).resolve()
    if not script_path.exists():
        print(f"Error: Cannot find {MAIN_SCRIPT}")
        sys.exit(1)
    
    # Load the module without executing it
    spec = importlib.util.spec_from_file_location("tmp_main", script_path)
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    
    return main_module

def setup_function_profiling(main_module):
    """
    Instrument key functions in the main module for profiling.
    """
    # List of important functions to profile individually
    functions_to_profile = [
        'tokenize_logs',
        'build_reference_model',
        'analyze_failing_logs',
        'handle_reference_model',
        'prepare_results_for_serialization',
        'save_analysis_results',
        'print_analysis_summary'
    ]
    
    original_functions = {}
    
    for func_name in functions_to_profile:
        if hasattr(main_module, func_name):
            original_func = getattr(main_module, func_name)
            original_functions[func_name] = original_func
            
            # Create a wrapper function
            def make_profiled_func(f, name):
                def profiled_wrapper(*args, **kwargs):
                    with time_profiler(name):
                        return f(*args, **kwargs)
                return profiled_wrapper
            
            # Replace the original function with the profiled version
            setattr(main_module, func_name, make_profiled_func(original_func, func_name))
    
    return original_functions

def restore_original_functions(main_module, original_functions):
    """
    Restore the original functions after profiling.
    """
    for func_name, func in original_functions.items():
        setattr(main_module, func_name, func)

def run_profiler():
    """
    Run the profiler on the main script.
    """
    # Save original sys.argv
    original_argv = sys.argv.copy()
    
    # Replace sys.argv with our args (excluding this script name)
    # This allows tmp_main.py to parse the command line arguments correctly
    sys.argv = [MAIN_SCRIPT] + sys.argv[1:]
    
    # Load the main module
    main_module = load_main_module()
    
    # Setup profiling for specific functions
    original_functions = setup_function_profiling(main_module)
    
    # Run the full profile
    profiler = cProfile.Profile()
    
    print("=" * 50)
    print(f"Starting profiling run of ptracker analysis pipeline")
    print(f"With arguments: {' '.join(sys.argv[1:])}")
    print("=" * 50)
    
    # Run with both cProfile and memory profiling
    with memory_profiler():
        start_time = time.time()
        profiler.runcall(main_module.main)
        total_time = time.time() - start_time
        
    # Restore original sys.argv
    sys.argv = original_argv
    
    # Restore original functions
    restore_original_functions(main_module, original_functions)
    
    # Print profiling results
    print("\n" + "=" * 50)
    print(f"Total execution time: {total_time:.4f} seconds")
    print("=" * 50)
    
    # Sort by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions by cumulative time
    print(s.getvalue())
    
    # Sort by total time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('time')
    ps.print_stats(20)  # Top 20 functions by total time
    print("\nTop 20 functions by total time:")
    print(s.getvalue())
    
    # Create a detailed profiling report file
    output_path = Path('profile_results')
    output_path.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    report_file = output_path / f"profile_report_{timestamp}.txt"
    
    with open(report_file, 'w') as f:
        stats = pstats.Stats(profiler, stream=f)
        stats.sort_stats('cumulative')
        f.write("=" * 50 + "\n")
        f.write("PROFILING REPORT\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total execution time: {total_time:.4f} seconds\n")
        f.write("=" * 50 + "\n\n")
        f.write("TOP FUNCTIONS BY CUMULATIVE TIME:\n")
        stats.print_stats(30)
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("FUNCTIONS GROUPED BY MODULE:\n")
        stats.print_stats('tokenizer|feature_extractor|reference_model')
        
        f.write("\n" + "=" * 50 + "\n")
        f.write("CALLERS OF KEY FUNCTIONS:\n")
        key_funcs = [
            'tokenize_logs',
            'build_reference_model',
            'LogTokenizer.tokenize_logs',
            'PtrackerFeatureExtractor.extract_features'
        ]
        for func in key_funcs:
            f.write(f"\nCallers of {func}:\n")
            stats.print_callers(func)
    
    print(f"\nDetailed profiling report saved to: {report_file}")

if __name__ == "__main__":
    run_profiler()