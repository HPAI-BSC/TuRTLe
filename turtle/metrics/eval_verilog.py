import re
from pathlib import Path

from metrics.generic_eval import main

from .safe_subprocess import run

LANG_NAME = "Verilog"
LANG_EXT = ".sv"

# VerilogEval uses the following error codes
code_to_reason = {
    # No failures:
    '.': 'Pass',
    # Compile-time failures:
    'C': 'Compiler error',
    'S': 'Syntax Error',
    'e': 'Explicit Cast Required',
    '0': 'Sized Numeric Constant Error',
    'n': 'No Sensitivities Warning',
    'w': 'Declared as Wire',
    'm': 'Unknown Module Type',
    'p': 'Unable to Bind Wire/Reg',
    'c': 'Unable to Bind Wire/Reg `clk`',
    # Run-time failures:
    'T': 'Timeout',
    'r': 'Async reset found',
    'R': 'Runtime error'
}

def eval_verilog_eval(path: Path, test: Path, ref: Path, flag: bool = False):
    binary = ".".join(str(path).split(".")[:-1])
    result_dict = {
        'passed': False,
        'syntax_passed': False,
        'func_passed': False,
        'result': '',
        'passfail': '?',
    }

    # syntax check
    compile_cmd = f"iverilog -Wall -Winfloop -Wno-timescale -g2012 -s tb -o {binary} {path} {test} {ref}"
    compile_result = run(compile_cmd, shell=True)
    out = compile_result.stdout
    err = compile_result.stderr.lower()
    if flag:
        print("-"*30 + "COMPILE RESULT" + "-"*30)
        print(compile_result.__dict__)
        print("-"*70)
    if compile_result.exit_code == 0:
        result_dict['syntax_passed'] = True
        result_dict['passfail'] = '.'
    else:
        result_dict['result'] = "failed: compile error."
        result_dict['passfail'] = 'S'
        # process compiler log
        if "this assignment requires an explicit cast" in err:
            result_dict['passfail'] = 'e'
        elif "sized numeric constant must have a size greater than zero" in err:
            result_dict['passfail'] = '0'
        elif "always_comb process has no sensitivities" in err:
            result_dict['passfail'] = 'n'
        elif "found no sensitivities so it will never trigger" in err:
            result_dict['passfail'] = 'n'
        elif "is declared here as wire" in err:
            result_dict['passfail'] = 'w'
        elif "unknown module type" in err:
            result_dict['passfail'] = 'm'
        elif "unable to bind wire/reg/memory `clk'" in err:
            result_dict['passfail'] = 'c'
        elif "unable to bind wire/reg" in err:
            result_dict['passfail'] = 'p'
        elif compile_result.timeout == True:
            result_dict['passfail'] = 'T'
        elif "error" in err: # error not parsed, we assign a compiler error
            result_dict['passfail'] = 'C'
        result_dict['passfail'] = code_to_reason[result_dict['passfail']]
        return result_dict

    sim_cmd = f"vvp -n {binary}"
    execution_result = run(
        sim_cmd,
        shell=True,
    )
    if flag:
        print("-"*30 + "BUILD AND RUN RESULT" + "-"*30)
        print(execution_result.__dict__)
        print("-"*70)

    out = execution_result.stdout
    err = execution_result.stderr.lower()
    match = re.search(r'Mismatches: ([0-9]*) in ([0-9]*) samples', out)
    if "syntax error" in err:
        result_dict['result'] = "failed: syntax error in simulation."
        result_dict['passfail'] = 'R'
    elif len(err) > 0:
        result_dict['result'] = "failed: runtime error during simulation."
        result_dict['passfail'] = 'R'
    elif match:
        cor, tot = [int(i) for i in match.groups()]
        if cor == 0:
            result_dict['func_passed'] = True
            result_dict['passed'] = True
            result_dict['result'] = "passed"
        else:
            result_dict['result'] = f"failed: {cor} out of {tot} mismatches."
            result_dict['passfail'] = 'R'
    else:
        result_dict['result'] = "failed: test did not pass"
        result_dict['passfail'] = 'R'

    if execution_result.timeout == True:
        result_dict['result'] = "failed: timed out."
        result_dict['passfail'] = 'T'

    if result_dict['passfail'] == 'R':
        # process verilog to identify possible runtime issues
        with open(path, 'r') as fd:
            for line in fd:
                if "posedge reset" in line:
                    result_dict['passfail'] = 'r'
                    break

                if "negedge reset" in line:
                    result_dict['passfail'] = 'r'
                    break

                if "posedge r)" in line:
                    result_dict['passfail'] = 'r'
                    break
    result_dict['passfail'] = code_to_reason[result_dict['passfail']]
    return result_dict


def eval_rtllm(path: Path, test: Path, ref: Path, flag: bool = False):
    binary = ".".join(str(path).split(".")[:-1])
    result_dict = {
        'passed': False,
        'syntax_passed': False,
        'func_passed': False,
        'result': ''
    }

    # syntax check
    compile_cmd = f"iverilog -g2012 -o {binary} {path} {test}"
    if flag:
        print("-"*30 + "IVERILOG COMMAND" + "-"*30)
        print(compile_cmd)
        print("-"*70)
    compile_result = run(compile_cmd, shell=True)
    if flag:
        print("-"*30 + "COMPILE RESULT" + "-"*30)
        print(compile_result.__dict__)
        print("-"*70)
    if compile_result.exit_code == 0:
        result_dict['syntax_passed'] = True
    else:
        result_dict['result'] = "failed: compile error."
        return result_dict

    # func check
    sim_cmd = f"vvp -n {binary}"
    sim_result = run(sim_cmd, timeout_seconds=15, shell=True)
    if flag:
        print("-"*30 + "SIM RESULT" + "-"*30)
        print(sim_result.__dict__)
        print("-"*70)
    if "ERROR" not in sim_result.stdout and "Passed" in sim_result.stdout:
        result_dict['func_passed'] = True
        result_dict['passed'] = True
        result_dict['result'] = "passed"
    else:
        result_dict['result'] = "failed: test did not pass"

    return result_dict


def eval_script_verigen(path: Path, test: Path, ref: Path, flag: bool):
    binary = ".".join(str(path).split(".")[:-1])
    result_dict = {
        'passed': False,
        'syntax_passed': False,
        'func_passed': False,
        'result': ''
    }

    # syntax check
    compile_cmd = f"iverilog -o {binary} {path} {test}"
    compile_result = run(compile_cmd, shell=True)
    if flag:
        print("-"*30 + "COMPILE RESULT" + "-"*30)
        print(compile_result.__dict__)
        print("-"*70)
    if compile_result.exit_code == 0:
        result_dict['syntax_passed'] = True
    else:
        result_dict['result'] = "failed: compile error."
        return result_dict

    sim_cmd = f"vvp -n {binary}"

    sim_result = run(sim_cmd, timeout_seconds=15, shell=True)
    if "all tests passed" in sim_result.stdout:
        result_dict['func_passed'] = True
        result_dict['passed'] = True
        result_dict['result'] = "passed"

    if flag:
        print("-"*30 + "BUILD AND RUN RESULT" + "-"*30)
        print(sim_result.__dict__)
        print("-"*70)
    

    if sim_result.timeout == True:
        result_dict['result'] = "failed: timed out."

    return result_dict


if __name__ == "__main__":
    main(eval_script, LANG_NAME, LANG_EXT)

