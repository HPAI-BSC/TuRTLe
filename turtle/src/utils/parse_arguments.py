import argparse


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments.
    Args:
        - None
    Returns:
        - argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run a turtle Benchmarking tool.")
    parser.add_argument("--benchmark", help="Define a Benchmark to run (e.g., rtlrepo, verigen).")
    parser.add_argument("--model", help="Specific model to run (requires --benchmark).")
    parser.add_argument("--run_all", action="store_true", help="Run all benchmarks in the project (cannot be used with --benchmark).")
    parser.add_argument("--generation_only", action="store_true", help="Do only the evaluation phase PPA", default=None)
    parser.add_argument("--evaluate_only", action="store_true", help="Do only the evaluation phase PPA", default=None)
    args = parser.parse_args()

    # Validate argument combinations
    if args.run_all and args.benchmark:
        print("Error: --run_all and --benchmark are mutually exclusive. Use only one.")
        exit(1)
        
    if not args.run_all and not args.benchmark:
        print("Error: Either --run_all or --benchmark must be specified.")
        exit(1)
        
    if args.model and not args.benchmark:
        print("Error: --model requires --benchmark to be specified.")
        exit(1)

    return args
