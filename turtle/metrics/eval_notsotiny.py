"""
NotSoTiny Evaluation Module

This module contains evaluation functions specifically for the NotSoTiny benchmark.
It evaluates Verilog generations using the original TinyTapeout project test infrastructure
with proper separation between syntax and functionality testing.
"""
from typing import Dict, Any, Optional

import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
import yaml


def eval_notsotiny_generation(
        file_path: Path, ref_path: Path, task_id: str, id: str, debug: bool = False
) -> Dict[str, Any]:
    """
    Evaluate NotSoTiny Verilog generation with syntax/functionality/equivalence separation.

    Args:
        file_path: Path to the generated Verilog file
        ref_path: Path to the reference solution
        id: Task identifier
        debug: Whether to print debug information

    Returns:
        Dictionary containing evaluation results:
        {
            "syntax_passed": bool,
            "equiv_passed": bool,
            "passfail": str,
            "syntax_error": str,
            "func_error": str,
            "equiv_error": str,
            "warnings": List[str],
            "top_module": str,
            "eqy_return_code": int,
            "equiv_method": str
        }
    """

    # Initialize result dictionary
    result = {
        "syntax_passed": False,
        "equiv_passed": False,
        "passfail": "",
        "syntax_error": "",
        "func_error": "",
        "equiv_error": "",
        "warnings": [],
        "top_module": None,
        "eqy_return_code": None,
        "equiv_method": "error",
    }

    # STEP 1: Basic syntax check using Icarus Verilog
    syntax_result = run_syntax_check(file_path, debug)
    result["syntax_passed"] = syntax_result["syntax_valid"]
    result["syntax_error"] = syntax_result["error_message"]
    result["warnings"].extend(syntax_result["warnings"])

    if not syntax_result["syntax_valid"]:
        result["passfail"] = f"Syntax error (id={id}): {syntax_result['error_message']}"
        return result

    # STEP 2: Formal equivalence check

    # Get project directory from test path
    project_dir = ref_path.parent if ref_path.is_dir() else ref_path.parent.parent

    equiv_result = run_equivalence_check(file_path, Path(ref_path), project_dir, task_id, debug)
    result["equiv_passed"] = equiv_result["equiv_passed"]
    result["equiv_error"] = equiv_result["error_message"]
    result["top_module"] = equiv_result["top_module"]
    result["eqy_return_code"] = equiv_result["eqy_return_code"]
    result["equiv_method"] = equiv_result["equiv_method"]

    # Determine overall result
    if result["syntax_passed"] and result["equiv_passed"] and result["func_passed"]:
        result["passfail"] = "Success"
    elif not result["syntax_passed"]:
        result["passfail"] = f"Syntax error (id={id}): {result['syntax_error']}"
    elif not result["equiv_passed"]:
        result["passfail"] = f"Equivalence error (id={id}): {result['equiv_error']}"
    else:
        raise ValueError(f"Invalid state of {result}")

    return result


def run_syntax_check(file_path: Path, debug: bool = False) -> Dict[str, Any]:
    """
    Run pure syntax check using Icarus Verilog without running tests.

    Args:
        file_path: Path to the Verilog file to check
        debug: Whether to print debug information

    Returns:
        Dictionary with syntax check results:
        {
            "syntax_valid": bool,
            "error_message": str,
            "warnings": List[str]
        }
    """
    try:
        # Use iverilog in syntax-check-only mode (-t null means no code generation)
        result = subprocess.run(
            [
                "iverilog",
                "-Wall",
                "-Winfloop",
                "-Wno-timescale",
                "-g2012",
                "-t",
                "null",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=1300,
        )

        syntax_valid = result.returncode == 0
        error_message = ""
        warnings = []

        # Parse iverilog output for errors and warnings
        if result.stderr:
            lines = result.stderr.split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                if "error:" in line.lower():
                    # Extract the error message
                    if not error_message:  # Take the first error
                        error_message = clean_error_message(line)
                elif "warning:" in line.lower():
                    warnings.append(clean_error_message(line))

        # If syntax failed but no specific error message found, use general output
        if not syntax_valid and not error_message:
            if result.stderr:
                error_message = result.stderr.strip()[:200]  # Limit length
            else:
                error_message = f"Compilation failed with return code {result.returncode}"

        return {
            "syntax_valid": syntax_valid,
            "error_message": error_message,
            "warnings": warnings,
        }

    except subprocess.TimeoutExpired:
        error_msg = "Syntax check timed out (30s)"
        return {"syntax_valid": False, "error_message": error_msg, "warnings": []}

    except FileNotFoundError:
        error_msg = "Icarus Verilog (iverilog) not found - please install it"
        return {"syntax_valid": False, "error_message": error_msg, "warnings": []}

    except Exception as e:
        error_msg = f"Syntax check failed: {str(e)}"
        return {"syntax_valid": False, "error_message": error_msg, "warnings": []}


def find_makefile(test_dir: Path) -> Optional[Path]:
    """
    Find a Makefile in the test directory.

    Args:
        test_dir: Directory to search for Makefile

    Returns:
        Path to found Makefile or None if not found
    """
    makefiles = ["Makefile", "makefile", "Makefile.sim", "Makefile.test"]

    for makefile_name in makefiles:
        makefile_path = test_dir / makefile_name
        if makefile_path.exists() and makefile_path.is_file():
            return makefile_path

    return None


def analyze_test_failure(
    result: subprocess.CompletedProcess, debug: bool = False
) -> str:
    """
    Analyze test failure output to provide meaningful error message.

    Args:
        result: Completed subprocess result from make command
        debug: Whether to print debug information

    Returns:
        Human-readable error message
    """
    output = (result.stderr + "\n" + result.stdout).lower()

    # Common error patterns and their meanings
    error_patterns = [
        (r"assertion.*failed", "Test assertion failed - incorrect behavior"),
        (r"test.*failed", "Functional test failed - incorrect behavior"),
        (r"timeout", "Test timed out - possible infinite loop or deadlock"),
        (r"simulation.*error", "Simulation error during testing"),
        (r"syntax error", "Syntax error in test setup"),
        (r"module.*not found", "Required module not found"),
        (r"port.*not found", "Required port not found in module"),
        (r"signal.*not found", "Required signal not found"),
        (r"compilation.*failed", "Test compilation failed"),
        (
            r"elaboration.*failed",
            "Elaboration failed - module instantiation issues",
        ),
        (r"no such file", "Required test file not found"),
        (r"permission denied", "Permission error accessing test files"),
    ]

    # Check for specific error patterns
    for pattern, message in error_patterns:
        if re.search(pattern, output):
            return message

    # If no specific pattern matched, try to extract a meaningful line
    for line in (result.stderr + "\n" + result.stdout).split("\n"):
        line = line.strip()
        if any(keyword in line.lower() for keyword in ["error:", "failed:", "exception:"]):
            # Clean up the error message
            cleaned = clean_error_message(line)
            if len(cleaned) > 10:  # Only use if it's substantial
                return cleaned[:200]  # Limit length

    # Fallback to generic message
    return f"Test failed with return code {result.returncode}"


def clean_error_message(message: str) -> str:
    """
    Clean up error message by removing file paths and irrelevant details.

    Args:
        message: Raw error message

    Returns:
        Cleaned error message
    """
    # Remove absolute paths, keep only filename
    message = re.sub(r"/[^/\s]*/", "", message)

    # Remove common prefixes
    prefixes_to_remove = [
        r"^\s*[^:]*:\s*",  # Remove "filename:" prefix
        r"^\s*error:\s*",  # Remove "error:" prefix
        r"^\s*warning:\s*",  # Remove "warning:" prefix
        r"^\s*make\[\d+\]:\s*",  # Remove make process info
    ]

    for prefix in prefixes_to_remove:
        message = re.sub(prefix, "", message, flags=re.IGNORECASE)

    # Clean up whitespace
    message = " ".join(message.split())

    return message.strip()


def run_equivalence_check(
        generated_file: Path, reference_file: Path, project_dir: Path, task_id: str, debug: bool = False
) -> Dict[str, Any]:
    """
    Run formal equivalence check using EQY/Yosys.
    Both generated_file and reference_file should be complete modules.v files.
    
    IMPORTANT: 
    - Uses 15-minute timeout
    - If timeout occurs WITHOUT failure, treats as PASS (equiv_passed=True)
    - Only returns False if EQY explicitly reports non-equivalence or errors
    - Tracks the method of equivalence determination
    
    Returns:
        {
            "equiv_passed": bool,  # ALWAYS bool, never None
            "error_message": str,
            "top_module": str,
            "eqy_return_code": int or "timeout",
            "equiv_method": str  # "proven", "timeout_pass", "failed", "error"
        }
    """

    result = {
        "equiv_passed": False,  # Default to False
        "error_message": "",
        "top_module": None,
        "eqy_return_code": None,
        "equiv_method": "error",  # Default to error
    }

    try:
        top_module = task_id.replace("task_", "")
        result["top_module"] = top_module

        if not top_module:
            result["error_message"] = "Could not extract top module from project info.yaml"
            result["equiv_passed"] = False
            result["equiv_method"] = "error"
            return result

        # Get absolute paths for both files
        abs_generated_file = os.path.abspath(str(generated_file))
        abs_reference_file = os.path.abspath(str(reference_file))

        # Verify both files exist
        if not os.path.exists(abs_generated_file):
            result["error_message"] = f"Generated file not found: {abs_generated_file}"
            result["equiv_passed"] = False
            result["equiv_method"] = "error"
            return result

        if not os.path.exists(abs_reference_file):
            result["error_message"] = f"Reference file not found: {abs_reference_file}"
            result["equiv_passed"] = False
            result["equiv_method"] = "error"
            return result

        # Check if generated file is empty
        gen_size = os.path.getsize(abs_generated_file)
        if gen_size == 0:
            result["error_message"] = "Generated file is empty"
            result["equiv_passed"] = False
            result["equiv_method"] = "error"
            return result

        # Create EQY script with absolute paths
        script_content = create_equivalence_eqy_script_absolute(
            abs_reference_file, abs_generated_file, top_module
        )

        # Create isolated temporary directory for this EQY run
        temp_dir = tempfile.mkdtemp(prefix=f"eqy_check_{os.getpid()}_")

        try:
            # Write script to temp directory
            script_file_path = os.path.join(temp_dir, "equiv_check.eqy")
            with open(script_file_path, "w") as f:
                f.write(script_content)
                f.flush()

            # Run EQY from the isolated temp directory with 15-minute timeout
            eqy_result = subprocess.run(
                ["eqy", script_file_path],
                capture_output=True,
                text=True,
                timeout=900,  # 15 minutes = 900 seconds
                cwd=temp_dir,
            )

            result["eqy_return_code"] = eqy_result.returncode

            # Parse results
            output_combined = (eqy_result.stdout + " " + (eqy_result.stderr or "")).lower()

            if eqy_result.returncode == 0:
                # Check for explicit success messages
                if (
                    "equiv_simple: success" in output_combined
                    or "equivalent" in output_combined
                    or "proven" in output_combined
                ):
                    result["equiv_passed"] = True
                    result["error_message"] = ""
                    result["equiv_method"] = "proven"
                elif "mismatch" in output_combined or "not equivalent" in output_combined:
                    result["equiv_passed"] = False
                    result["error_message"] = "Designs are not equivalent (counterexample found)"
                    result["equiv_method"] = "failed"
                else:
                    # Return code 0 but no clear success message - assume success
                    result["equiv_passed"] = True
                    result["error_message"] = ""
                    result["equiv_method"] = "proven"

            else:
                # ANY non-zero return code = NOT EQUIVALENT
                result["equiv_passed"] = False
                result["equiv_method"] = "failed"
                
                # Try to extract meaningful error message
                error_detail = extract_eqy_error_detail(eqy_result.stdout, eqy_result.stderr)
                
                if eqy_result.returncode == 1:
                    if "not equivalent" in output_combined or "mismatch" in output_combined:
                        result["error_message"] = "Designs are not equivalent (counterexample found)"
                    elif "module" in output_combined and "not found" in output_combined:
                        result["error_message"] = f"Module '{top_module}' not found in generated design"
                    else:
                        result["error_message"] = f"EQY check failed: {error_detail}"
                    
                elif eqy_result.returncode == 2:
                    result["error_message"] = f"Generated design has errors: {error_detail}"
                    
                else:
                    result["error_message"] = f"EQY failed (RC {eqy_result.returncode}): {error_detail}"


        finally:
            # Cleanup temp directory
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass

    except subprocess.TimeoutExpired:
        # TIMEOUT = PASS (no failure detected within 15 minutes)
        result["error_message"] = ""
        result["equiv_passed"] = True
        result["eqy_return_code"] = "timeout"
        result["equiv_method"] = "timeout_pass"

    except FileNotFoundError:
        result["error_message"] = "EQY not found - please install eqy/yosys"
        result["equiv_passed"] = False
        result["equiv_method"] = "error"

    except Exception as e:
        result["error_message"] = f"Equivalence check error: {str(e)}"
        result["equiv_passed"] = False
        result["equiv_method"] = "error"

    return result


def extract_eqy_error_detail(stdout: str, stderr: str) -> str:
    """Extract specific error details from EQY output."""
    output = stdout + "\n" + (stderr or "")
    
    # Common EQY/Yosys error patterns
    error_patterns = [
        (r"ERROR: (.+?)(?:\n|$)", 1),
        (r"error: (.+?)(?:\n|$)", 1),
        (r"(Module .+ not found)", 0),
        (r"(Identifier .+ is implicitly declared)", 0),
        (r"(syntax error[^\n]*)", 0),
        (r"prep: (.+?)(?:\n|$)", 1),
    ]
    
    for pattern, group in error_patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            error_text = match.group(group).strip()
            # Limit length and clean up
            return error_text[:200]
    
    # Look for lines containing "error" or "failed"
    for line in output.split("\n"):
        line_lower = line.lower()
        if "error" in line_lower or "failed" in line_lower:
            cleaned = line.strip()
            if len(cleaned) > 10:
                return cleaned[:200]
    
    return "Verification failed"


def create_equivalence_eqy_script_absolute(
    reference_file: str, generated_file: str, top_module: str
) -> str:
    """
    Create EQY script content for equivalence checking using ABSOLUTE paths.
    This matches the approach from the working self-equivalence checker.

    Args:
        reference_file: Absolute path to reference modules.v
        generated_file: Absolute path to generated modules.v
        top_module: Name of the top module to check

    Returns:
        EQY script content as string
    """
    script_content = f"""[gold]
read_verilog {reference_file}
prep -top {top_module}
memory_map

[gate]
read_verilog {generated_file}
prep -top {top_module}
memory_map

[strategy basic]
use sat

[script]
equiv_simple
"""
    return script_content


def run_basic_syntax_check(
    verilog_content: str, temp_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run a basic syntax check on Verilog content using Icarus Verilog.

    Args:
        verilog_content: The Verilog code to check
        temp_dir: Temporary directory to use (optional)

    Returns:
        Dictionary with syntax check results
    """
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
        cleanup_temp = True
    else:
        cleanup_temp = False

    try:
        # Write Verilog content to temporary file
        verilog_file = temp_dir / "test_syntax.v"
        with open(verilog_file, "w", encoding="utf-8") as f:
            f.write(verilog_content)

        # Use the main syntax check function
        result = run_syntax_check(verilog_file, debug=False)

        return {
            "syntax_valid": result["syntax_valid"],
            "error_message": result["error_message"],
            "warnings": result["warnings"],
        }

    except Exception as e:
        return {
            "syntax_valid": False,
            "error_message": f"Error during syntax check: {str(e)}",
            "warnings": [],
        }

    finally:
        if cleanup_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
