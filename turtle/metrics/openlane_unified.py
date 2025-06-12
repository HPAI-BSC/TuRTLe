import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def reformat_json(input_file, output_file):
    """Reformat a JSON file with proper indentation."""
    try:
        # Read the input file
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        # Ensure the data is properly formatted
        if not isinstance(data, list):
            data = [data]
            
        # Write the formatted JSON with proper indentation
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Successfully reformatted JSON and saved to {output_file}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return False
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def extract_problem_name(ref_path, mode="auto"):
    """
    Extract problem name from the ref_path based on the specified mode.
    
    Args:
        ref_path: Path to extract problem name from
        mode: Extraction mode - "verilogeval" (Prob### format), "rtllm"/"verigen" (flexible), or "auto" (try all)
    """
    if not ref_path:
        return None
    
    # VerilogEval style - extract just the Prob### part
    if mode in ["verilogeval", "auto"]:
        match = re.search(r'(Prob\d+)', ref_path)
        if match:
            return match.group(1)
    
    # RTLLM/VeriGen style - try to extract filename
    if mode in ["rtllm", "verigen", "auto"]:
        # Handle both Unix and Windows paths
        path_parts = re.split(r'[/\\]', ref_path)
        # Get the last non-empty part (filename with extension)
        filename_with_ext = next((p for p in reversed(path_parts) if p), None)
        if filename_with_ext:
            # Remove the extension
            filename = os.path.splitext(filename_with_ext)[0]
            return filename
    
    return None

def standardize_module_name(verilog_content, mode="auto"):
    """
    Change module name to TopModule in the Verilog content unless in rtllm mode.
    
    Args:
        verilog_content: The Verilog code to process
        mode: Processing mode - if "rtllm", keeps original module name
    """
    if not verilog_content or mode == "rtllm":
        return verilog_content
        
    # Check if there's already a module named TopModule
    if re.search(r'module\s+TopModule\b', verilog_content):
        return verilog_content
        
    # Find the module declaration - match more module patterns
    module_patterns = [
        # Standard module pattern with parameters
        r'module\s+(\w+)\s*(\([\s\S]*?\);)',
        # Module with newlines before parameters
        r'module\s+(\w+)\s*\n\s*(\([\s\S]*?\);)',
        # Module without parameters
        r'module\s+(\w+)\s*;',
        # Simplest case - just find any module declaration
        r'module\s+(\w+)'
    ]
    
    for pattern in module_patterns:
        match = re.search(pattern, verilog_content)
        if match:
            original_name = match.group(1)
            # If original name is already TopModule, no change needed
            if original_name == "TopModule":
                return verilog_content
                
            # Replace the module name
            if len(match.groups()) > 1 and match.group(2):
                # Has parameters
                new_declaration = f"module TopModule {match.group(2)}"
                verilog_content = re.sub(pattern, new_declaration, verilog_content, 1)
            else:
                # No parameters or simpler case
                new_declaration = f"module TopModule"
                verilog_content = re.sub(r'module\s+' + re.escape(original_name), 
                                        new_declaration, verilog_content, 1)
            
            # Also replace any endmodule comments that might reference the old name
            verilog_content = re.sub(r'endmodule\s*//\s*' + re.escape(original_name), 
                                    'endmodule // TopModule', verilog_content)
            
            # If we found and replaced a module, break out of the loop
            break
    
    return verilog_content

def detect_clock_ports(verilog_content):
    """Check if the Verilog file has clock ports and return all found clock port names."""
    # Extended list of clock patterns with capture group for the port name
    clock_patterns = [
        r'\binput\s+(?:wire\s+)?(\bclk\b)',
        r'\binput\s+(?:wire\s+)?(\bclock\b)',
        r'\binput\s+(?:wire\s+)?(\bCLK\b)',
        r'\binput\s+(?:wire\s+)?(\bClk\b)',
        r'\binput\s+(?:wire\s+)?(\bCK\b)',
        r'\binput\s+(?:wire\s+)?(\bwclk\b)',
        r'\binput\s+(?:wire\s+)?(\brclk\b)',
        r'\binput\s+(?:wire\s+)?(\bCLK_in\b)',
        r'\binput\s+(?:wire\s+)?(\bclk_a\b)',
        r'\binput\s+(?:wire\s+)?(\bclk_b\b)'
    ]
    
    # Store all found clock ports
    found_clocks = []
    
    for pattern in clock_patterns:
        # Use findall to get all occurrences, not just the first one
        matches = re.findall(pattern, verilog_content)
        found_clocks.extend(matches)
    
    return found_clocks  # Return empty list if no clock ports found

def create_config_for_generation(gen_dir):
    """Create OpenLane config file for a generation directory."""
    # Find TopModule.sv file
    verilog_file = gen_dir / "TopModule.sv"
    if not verilog_file.exists():
        print(f"No TopModule.sv found in {gen_dir}")
        return False
    
    verilog_content = verilog_file.read_text()
    
    # Get all clock port names if present
    clock_ports = detect_clock_ports(verilog_content)
    
    # Base configuration
    config = {
        "DESIGN_NAME": "TopModule",
        "VERILOG_FILES": "dir::TopModule.sv",
        "CLOCK_PERIOD": 10
    }
    
    # Set the clock port(s) based on what was found
    if clock_ports:
        if len(clock_ports) == 1:
            # Single clock case - use a string
            config["CLOCK_PORT"] = clock_ports[0]
            print(f"Detected single clock port '{clock_ports[0]}' in {gen_dir}")
        else:
            # Multiple clocks case - use a list
            config["CLOCK_PORT"] = clock_ports
            print(f"Detected multiple clock ports {clock_ports} in {gen_dir}")
    else:
        config["CLOCK_PORT"] = ""
        print(f"No clock ports detected in {gen_dir}")
    
    # Write configuration file
    config_path = gen_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return True

def create_config_for_rtllm(gen_dir, module_name):
    """Create OpenLane config file for RTLLM mode with original module name."""
    # Find Verilog file
    verilog_file = gen_dir / f"{module_name}.sv"
    if not verilog_file.exists():
        print(f"No {module_name}.sv found in {gen_dir}")
        return False
    
    verilog_content = verilog_file.read_text()
    
    # Get all clock port names if present
    clock_ports = detect_clock_ports(verilog_content)
    
    # Base configuration using the original module name
    config = {
        "DESIGN_NAME": module_name,
        "VERILOG_FILES": f"dir::{module_name}.sv",
        "CLOCK_PERIOD": 10
    }
    
    # Set the clock port(s) based on what was found
    if clock_ports:
        if len(clock_ports) == 1:
            # Single clock case - use a string
            config["CLOCK_PORT"] = clock_ports[0]
            print(f"Detected single clock port '{clock_ports[0]}' in {gen_dir}")
        else:
            # Multiple clocks case - use a list
            config["CLOCK_PORT"] = clock_ports
            print(f"Detected multiple clock ports {clock_ports} in {gen_dir}")
    else:
        config["CLOCK_PORT"] = ""
        print(f"No clock ports detected in {gen_dir}")
    
    # Write configuration file
    config_path = gen_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    return True

def create_problem_structure(
    output_dir, 
    json_file, 
    model_name=None, 
    mode="auto", 
    task_filter=None
):
    """
    Create directory structure for JSON file generations.
    
    Args:
        output_dir: Directory to store output
        json_file: JSON file with generations
        model_name: Name to use for the model (default: derived from output_dir)
        mode: Processing mode - "verilogeval", "rtllm", "verigen", or "auto"
        task_filter: String pattern to filter tasks by (VeriGen specific)
    
    Returns:
        Dictionary mapping problem names to lists of generation directories
    """
    # Use model_name from parameter or extract from output_dir
    model_name = model_name or output_dir.name
    formatted_json = output_dir / "formatted_output.json"
    
    # Reformat the JSON file
    if not reformat_json(json_file, formatted_json):
        return {}
    
    # Read the formatted JSON
    try:
        with open(formatted_json, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading formatted JSON: {e}")
        return {}
    
    # Create base directory for this model
    base_dir = output_dir / "generated_problems"
    os.makedirs(base_dir, exist_ok=True)
    
    # Track progress
    problems_processed = 0
    problems_skipped = 0
    problems_map = {}
    
    # Process each problem group
    for problem_group in data:
        for gen_idx, generation in enumerate(problem_group, 1):
            # VeriGen specific: Check for task_id filtering
            if task_filter and "task_id" in generation:
                task_id = generation.get("task_id", "")
                if task_filter not in task_id:
                    problems_skipped += 1
                    continue
                print(f"Including task with ID: {task_id}")
            
            # Get problem name using appropriate extraction method
            try:
                problem_name = extract_problem_name(generation.get("ref_path", ""), mode)
                if not problem_name:
                    print(f"Skipping generation with no valid problem name: {generation.get('ref_path', 'Unknown')}")
                    continue
            except KeyError:
                print(f"Missing ref_path in generation")
                continue
            
            # Create problem directory
            problem_dir = base_dir / problem_name
            os.makedirs(problem_dir, exist_ok=True)
            
            # Track this problem
            if problem_name not in problems_map:
                problems_map[problem_name] = []
            
            # Create generation directory
            gen_dir = problem_dir / f"generation_{gen_idx}"
            os.makedirs(gen_dir, exist_ok=True)
            problems_map[problem_name].append(gen_dir)
            
            # Write files
            try:
                # Write prompt.txt
                with open(gen_dir / "prompt.txt", 'w') as f:
                    f.write(generation.get("prompt", "No prompt available"))
                
                # VeriGen specific: Write task_id.txt if available
                if "task_id" in generation:
                    with open(gen_dir / "task_id.txt", 'w') as f:
                        f.write(generation.get("task_id", ""))
                
                # Check if we have any generation content
                if "generation" not in generation or not generation["generation"]:
                    print(f"Skipping empty generation for {problem_name} generation_{gen_idx}")
                    continue
                
                # Get original module name for rtllm mode
                original_module_name = None
                if mode == "rtllm":
                    for pattern in [r'module\s+(\w+)', r'module\s+(\w+)\s*\(']:
                        match = re.search(pattern, generation["generation"])
                        if match:
                            original_module_name = match.group(1)
                            break
                
                # Standardize the module name in the Verilog content (preserving original for rtllm mode)
                verilog_content = standardize_module_name(generation["generation"], mode)
                
                # If the Verilog content doesn't have a module declaration, add a basic one
                if "module" not in verilog_content:
                    if mode == "rtllm":
                        module_name = original_module_name or problem_name
                        verilog_content = f"module {module_name}(\n  // Default inputs/outputs\n);\n\n{verilog_content}\n\nendmodule // {module_name}"
                    else:
                        verilog_content = f"module TopModule(\n  // Default inputs/outputs\n);\n\n{verilog_content}\n\nendmodule // TopModule"
                
                # Write Verilog file with appropriate name
                if mode == "rtllm":
                    module_name = original_module_name or problem_name
                    target_file = gen_dir / f"{module_name}.sv"
                else:
                    target_file = gen_dir / "TopModule.sv"
                    
                with open(target_file, 'w') as f:
                    f.write(verilog_content)
                
                # Create config.json
                if mode == "rtllm":
                    create_config_for_rtllm(gen_dir, original_module_name or problem_name)
                else:
                    create_config_for_generation(gen_dir)
                
                problems_processed += 1
            except KeyError as e:
                print(f"Key error in generation: {e}")
                continue
    
    print(f"Created structure for {problems_processed} problems in {model_name}")
    if problems_skipped > 0:
        print(f"Skipped {problems_skipped} problems that did not match filters")
    
    return problems_map if problems_processed > 0 else {}

def read_metrics(file_path):
    metric = {}
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            metric['power'] = data.get('power__total')
            metric['performance'] = 10 - data.get('timing__setup__ws__corner:nom_tt_025C_1v80')
            metric['area'] = data.get('design__instance__area')
    except:
        print(f'OJO. File {file_path} has no metrics')
    return metric

def find_file(folder_path, target_file_name):
    """Searches for a particular file name inside a directory and its subdirectories."""
    for root, dirs, files in os.walk(folder_path):
        if target_file_name in files:
            return os.path.join(root, target_file_name)
    return None

def find_latest_run_metrics(folder_path):
    """Searches for the latest run inside a directory and its subdirectories."""
    latest_dir = None
    latest_time = None
    for entry in os.listdir(folder_path):
        full_path = os.path.join(folder_path, entry)
        # Skip non-directories and entries that don't start with 'RUN_'
        if not os.path.isdir(full_path) or not entry.startswith('RUN_'):
            continue
            
        # Extract timestamp from directory name
        datetime_str = entry[4:]  # Remove 'RUN_' prefix
        try:
            dt = datetime.strptime(datetime_str, '%Y-%m-%d_%H-%M-%S')
        except ValueError:
            continue  # Skip entries with invalid format
            
        # Update latest if current entry is newer
        if latest_time is None or dt > latest_time:
            latest_time = dt
            latest_dir = full_path

    return find_file(latest_dir, 'metrics.json')

def run_openlane_for_generation(gen_dir, problem_name, model_name, pdk_root=None):
    """Run OpenLane for a single generation and check if it completes successfully."""
    print(f"\nProcessing {model_name} / {problem_name} / {gen_dir.name}...")
    original_dir = os.getcwd()
    os.chdir(gen_dir)
    
    try:
        # Build the command with conditional PDK root
        command = ["openlane"]
        if pdk_root:
            command.extend(["--pdk-root", pdk_root])
        
        command.extend([
            "--to", "OpenRoad.STAPrePNR",
            "--from", "Yosys.JSONHeader",
            "config.json"
        ])
        
        # Run the command
        result = subprocess.run(command, capture_output=True, text=True, timeout=300)
        
        # Check if synthesis was successful
        success = "Flow complete" in result.stdout or "OpenRoad.STAPrePNR complete" in result.stdout
        status = "Success" if success else "Failed"
        
        # print immediate feedback
        print(f"{'✅' if success else '❌'} {model_name} / {problem_name} / {gen_dir.name}: {status}")

        metrics = {}
        if success:
            metrics_path = find_latest_run_metrics(os.path.join(gen_dir,'runs'))
            if metrics_path:
                metrics = read_metrics(metrics_path)
        
        return {
            "generation": gen_dir.name,
            "status": status,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "output": result.stdout if not success else "",
            "metrics": metrics,
        }
    except Exception as e:
        print(f"ERROR: {model_name} / {problem_name} / {gen_dir.name}: Error - {str(e)}")
        return { 
            "generation": gen_dir.name,
            "status": "Error",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "output": str(e)
        }
    finally:
        os.chdir(original_dir)

def process_problems(problems_map, model_name, problem_filter=None, pdk_root=None, mode="auto"):
    """Process all problems for a model, optionally filtering by problem list."""
    # Store results for all problems
    model_results = {}
    
    # Process each problem
    for problem_name, gen_dirs in problems_map.items():
        # Skip if we have a filter and this problem is not in it
        if problem_filter and problem_name not in problem_filter:
            print(f"Skipping {problem_name} (not in filter list)")
            continue
            
        # Process each generation for this problem
        problem_results = []
        for gen_dir in gen_dirs:
            if gen_dir.is_dir():
                # Handle different file names based on mode
                if mode == "rtllm":
                    # For rtllm, check for any .sv file
                    if not any(file.suffix == ".sv" for file in gen_dir.iterdir()):
                        print(f"Skipping {gen_dir} - no Verilog file found")
                        continue
                else:
                    # For other modes, check for TopModule.sv
                    if not (gen_dir / "TopModule.sv").exists():
                        print(f"Skipping {gen_dir} - no TopModule.sv found")
                        continue
                    
                result = run_openlane_for_generation(gen_dir, problem_name, model_name, pdk_root)
                problem_results.append(result)
                
        # Store results for this problem if we have any
        if problem_results:
            model_results[problem_name] = problem_results
    
    return model_results

def generate_model_report(model_name, model_results, output_dir):
    """Generate a report for a single model."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = output_dir / f"{model_name}_openlane_status.md"
    
    with open(report_path, 'w') as f:
        f.write(f"# OpenLane Synthesis Results for {model_name}\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Write overall summary for this model
        total_generations = sum(len(results) for results in model_results.values())
        total_successful = sum(
            sum(1 for gen in results if gen["status"] == "Success")
            for results in model_results.values()
        )
        
        f.write(f"## Model Summary\n")
        f.write(f"- Total Problems: {len(model_results)}\n")
        f.write(f"- Total Generations: {total_generations}\n")
        f.write(f"- Total Successful: {total_successful}\n")
        f.write(f"- Total Failed: {total_generations - total_successful}\n")
        f.write(f"- Success Rate: {(total_successful / total_generations * 100) if total_generations > 0 else 0:.2f}%\n\n")
        
        # Write detailed results per problem
        f.write("## Results by Problem\n\n")
        for problem_name, generations in model_results.items():
            successful_gens = sum(1 for g in generations if g["status"] == "Success")
            success_rate = (successful_gens / len(generations) * 100) if generations else 0
            
            f.write(f"### {problem_name}\n")
            f.write(f"- Total Generations: {len(generations)}\n")
            f.write(f"- Successful: {successful_gens}\n")
            f.write(f"- Failed: {len(generations) - successful_gens}\n")
            f.write(f"- Success Rate: {success_rate:.2f}%\n\n")
            
            # Details for each generation
            for gen in generations:
                status_icon = "✅" if gen["status"] == "Success" else "❌"
                f.write(f"#### {gen['generation']}\n")
                f.write(f"{status_icon} **Status**: {gen['status']}\n")
                f.write(f"**Timestamp**: {gen['timestamp']}\n")
                if gen["status"] != "Success" and gen.get("output"):
                    f.write("\n<details><summary>Error Output</summary>\n\n```\n")
                    f.write(gen["output"])
                    f.write("\n```\n</details>\n")
                f.write("\n")
            f.write("---\n\n")
    
    return report_path

def generate_summary_report(all_model_results, output_dir):
    """Generate a summary report across all models."""
    os.makedirs(output_dir, exist_ok=True)
    report_path = output_dir / "summary_openlane_status.md"
    
    with open(report_path, 'w') as f:
        f.write("# OpenLane Synthesis Summary Across All Models\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Models Comparison\n\n")
        f.write("| Model | Problems | Total Generations | Successful | Success Rate |\n")
        f.write("|-------|----------|-------------------|------------|--------------|\n")
        
        # Calculate statistics for each model
        for model_name, model_results in all_model_results.items():
            total_generations = sum(len(results) for results in model_results.values())
            total_successful = sum(
                sum(1 for gen in results if gen["status"] == "Success")
                for results in model_results.values()
            )
            success_rate = (total_successful / total_generations * 100) if total_generations > 0 else 0
            
            f.write(f"| {model_name} | {len(model_results)} | {total_generations} | {total_successful} | {success_rate:.2f}% |\n")
        
        # Detailed problem-by-problem comparison
        f.write("\n## Problem-by-Problem Comparison\n\n")
        
        # Get all unique problem names
        all_problems = set()
        for model_results in all_model_results.values():
            all_problems.update(model_results.keys())
        
        # For each problem, compare across models
        for problem in sorted(all_problems):
            f.write(f"### {problem}\n\n")
            f.write("| Model | Generations | Successful | Success Rate |\n")
            f.write("|-------|-------------|------------|--------------|\n")
            
            for model_name, model_results in all_model_results.items():
                if problem in model_results:
                    generations = model_results[problem]
                    successful = sum(1 for g in generations if g["status"] == "Success")
                    success_rate = (successful / len(generations) * 100) if generations else 0
                    f.write(f"| {model_name} | {len(generations)} | {successful} | {success_rate:.2f}% |\n")
                else:
                    f.write(f"| {model_name} | 0 | 0 | 0.00% |\n")
            
            f.write("\n")
    
    return report_path

def read_problem_list(problem_list_file):
    """Read a list of problems from a text file."""
    try:
        with open(problem_list_file, 'r') as f:
            # Strip whitespace and filter out empty lines
            problems = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Loaded {len(problems)} problems from {problem_list_file}")
        return problems
    except Exception as e:
        print(f"Error reading problem list file: {e}")
        return None

def process_single_json(
    json_file: Union[str, Path], 
    output_dir: Union[str, Path] = None,
    model_name: str = None,
    generate_report: bool = True,
    problem_filter: List[str] = None,
    pdk_root: str = None,
    mode: str = "auto",
    task_filter: str = None
) -> Dict[str, Any]:
    """
    Process a single JSON file with OpenLane verification.
    
    Args:
        json_file: Path to the JSON file containing generations
        output_dir: Directory to store the generated files and reports (default: beside JSON file)
        model_name: Name to use for the model (default: derived from output_dir)
        generate_report: Whether to generate a report (default: True)
        problem_filter: List of problem names to process (default: None = all problems)
        pdk_root: Path to the OpenLane PDK root (default: system default)
        mode: Processing mode - "verilogeval", "rtllm", "verigen", or "auto" (default)
        task_filter: String pattern to filter tasks by (VeriGen specific)
        
    Returns:
        Dictionary with verification results for each problem
    """
    # Convert paths to Path objects
    json_file = Path(json_file)
    
    # If output_dir not specified, create beside the JSON file
    if output_dir is None:
        output_dir = json_file.parent / f"{json_file.stem}_openlane_output"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Use filename as model_name if not provided
    if model_name is None:
        model_name = json_file.stem
    
    print(f"Processing JSON file: {json_file}")
    print(f"Output directory: {output_dir}")
    print(f"Model name: {model_name}")
    print(f"Processing mode: {mode}")
    if task_filter:
        print(f"Task filter: {task_filter}")
    
    # Setup directory structure
    problems_map = create_problem_structure(output_dir, json_file, model_name, mode, task_filter)
    if not problems_map:
        print("Failed to create problem structure or no valid problems found.")
        return {}
    
    # Process problems
    model_results = process_problems(problems_map, model_name, problem_filter, pdk_root, mode)
    
    # Generate report if requested
    if generate_report and model_results:
        report_path = generate_model_report(model_name, model_results, output_dir)
        print(f"Report generated: {report_path}")
    
    return model_results

def process_multiple_jsons(
    json_files: List[Union[str, Path]],
    base_output_dir: Union[str, Path],
    model_names: List[str] = None,
    generate_summary: bool = True,
    problem_filter: List[str] = None,
    pdk_root: str = None,
    modes: List[str] = None,
    task_filters: List[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Process multiple JSON files and generate a summary report.
    
    Args:
        json_files: List of paths to JSON files
        base_output_dir: Base directory to store all outputs
        model_names: List of model names (default: derived from filenames)
        generate_summary: Whether to generate a summary report (default: True)
        problem_filter: List of problem names to process (default: None = all problems)
        pdk_root: Path to the OpenLane PDK root (default: system default)
        modes: Processing modes for each JSON file (default: "auto" for all)
        task_filters: Task filters for each JSON file (default: None for all)
        
    Returns:
        Dictionary with results for each model
    """
    base_output_dir = Path(base_output_dir)
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Prepare parameters
    if model_names is None:
        model_names = [Path(json_file).stem for json_file in json_files]
    
    if modes is None:
        modes = ["auto"] * len(json_files)
    
    if task_filters is None:
        task_filters = [None] * len(json_files)
    
    # Ensure all parameter lists have the same length
    if len(model_names) != len(json_files):
        print("Warning: Length of model_names doesn't match json_files. Using default names.")
        model_names = [Path(json_file).stem for json_file in json_files]
    
    if len(modes) != len(json_files):
        print("Warning: Length of modes doesn't match json_files. Using 'auto' for all.")
        modes = ["auto"] * len(json_files)
    
    if len(task_filters) != len(json_files):
        print("Warning: Length of task_filters doesn't match json_files. Using None for all.")
        task_filters = [None] * len(json_files)
    
    # Process each JSON file
    all_results = {}
    for i, json_file in enumerate(json_files):
        model_name = model_names[i]
        mode = modes[i]
        task_filter = task_filters[i]
        
        # Create model-specific output directory
        model_output_dir = base_output_dir / model_name
        
        print(f"\nProcessing model {i+1}/{len(json_files)}: {model_name}")
        model_results = process_single_json(
            json_file=json_file,
            output_dir=model_output_dir,
            model_name=model_name,
            generate_report=True,
            problem_filter=problem_filter,
            pdk_root=pdk_root,
            mode=mode,
            task_filter=task_filter
        )
        
        if model_results:
            all_results[model_name] = model_results
    
    # Generate summary report if requested
    if generate_summary and all_results:
        summary_path = generate_summary_report(all_results, base_output_dir)
        print(f"\nSummary report generated: {summary_path}")
    
    return all_results

def main():
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process OpenLane verification for HDL generations')
    
    # Main operation modes
    parser.add_argument('--single', action='store_true', help='Process a single JSON file')
    parser.add_argument('--multiple', action='store_true', help='Process multiple JSON files')
    
    # Single JSON mode arguments
    parser.add_argument('--json-file', type=str, help='Path to JSON file (for --single mode)')
    parser.add_argument('--output-dir', type=str, help='Output directory (for --single mode)')
    parser.add_argument('--model-name', type=str, help='Model name to use (for --single mode)')
    
    # Multiple JSON mode arguments
    parser.add_argument('--json-list', type=str, help='Text file with list of JSON files (for --multiple mode)')
    parser.add_argument('--base-dir', type=str, help='Base directory for outputs (for --multiple mode)')
    
    # Common arguments
    parser.add_argument('--problem-list', type=str, help='Text file with list of problems to process')
    parser.add_argument('--pdk-root', type=str, help='Path to OpenLane PDK root')
    parser.add_argument('--mode', type=str, default='auto', 
                        choices=['auto', 'verilogeval', 'rtllm', 'verigen'],
                        help='Processing mode (default: auto)')
    parser.add_argument('--task-filter', type=str, help='Filter generations by task ID pattern (VeriGen)')
    parser.add_argument('--no-report', action='store_true', help='Skip report generation')
    
    args = parser.parse_args()
    
    # Load problem filter if specified
    problem_filter = None
    if args.problem_list:
        problem_filter = read_problem_list(args.problem_list)
        if not problem_filter:
            print("Failed to load problem list. Exiting.")
            return 1
    
    # Process based on mode
    if args.single:
        if not args.json_file:
            print("Error: --json-file is required with --single mode")
            return 1
        
        # Process single JSON file
        process_single_json(
            json_file=args.json_file,
            output_dir=args.output_dir,
            model_name=args.model_name,
            generate_report=not args.no_report,
            problem_filter=problem_filter,
            pdk_root=args.pdk_root,
            mode=args.mode,
            task_filter=args.task_filter
        )
    
    elif args.multiple:
        if not args.json_list:
            print("Error: --json-list is required with --multiple mode")
            return 1
        
        if not args.base_dir:
            print("Error: --base-dir is required with --multiple mode")
            return 1
        
        # Load list of JSON files
        try:
            with open(args.json_list, 'r') as f:
                json_files = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            print(f"Error reading JSON list file: {e}")
            return 1
        
        if not json_files:
            print("No JSON files found in the list")
            return 1
        
        # Process multiple JSON files
        process_multiple_jsons(
            json_files=json_files,
            base_output_dir=args.base_dir,
            generate_summary=not args.no_report,
            problem_filter=problem_filter,
            pdk_root=args.pdk_root,
            modes=[args.mode] * len(json_files),
            task_filters=[args.task_filter] * len(json_files)
        )
    
    else:
        print("Please specify either --single or --multiple mode")
        return 1
    
    return 0

if __name__ == "__main__":
    # If this script is run directly, use the command-line interface
    import sys
    sys.exit(main())
