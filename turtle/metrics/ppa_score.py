
import math
import sys
import os


def stable_divide(A,B):
    # Numerically stable computation of A/B
    # Decompose into mantissas and exponents
    m_A, e_A = math.frexp(A)
    m_B, e_B = math.frexp(B)
    
    # Compute ratio of mantissas and exponent difference
    ratio_m = m_A / m_B
    exponent_diff = e_A - e_B
    
    # Reconstruct the result
    return ratio_m * (2.0 ** exponent_diff)


def compute_ppa_score(C, g, n_problems, metric, results_dir):
    """
    C (dict): A dictionary where keys are elements of mathcal{C}, and values are lists of p_{i,j} values.
    g (dict): A dictionary where keys are elements of mathcal{C}, and values are g_i values.
    """
    num_C = len(C)  # Number of elements in mathcal{C}

    total_sum = 0.0
    
    for i, S_i in C.items():
        print(f"i: {i}, S_i: {S_i}")
        # Filter out negative and too large values   
        print('-------------------------------------------------')
        print(f'Processing {i}. Original S_i: {S_i}')     
        filtered_S_i = []
        for x in S_i:
            # analize PPA
            if x > 2*g[i]: # bad values
                with open(os.path.join(results_dir, "bad_ppa.txt"), "a") as log_file:
                    log_file.write('*********BAD VALUE**********\n')
                    log_file.write(f'S_{i} generations {metric}:\n')
                    log_file.write(f'x: {x} in S_i: {S_i}\n')
                    log_file.write(f'Reference value: {g[i]}\n\n')
            elif x < g[i]: # better than human reference
                with open(os.path.join(results_dir, "better_than_human.txt"), "a") as log_file:
                    log_file.write('*********BETTER THAN HUMAN**********\n')
                    log_file.write(f'S_{i} generations {metric}:\n')
                    log_file.write(f'x: {x} in S_i: {S_i}\n')
                    log_file.write(f'Reference value: {g[i]}\n\n')
            # Compute PPA
            if x < 0:
                with open(os.path.join(results_dir, "error_ppa.txt"), "a") as log_file:
                    log_file.write('*********NEGATIVE VALUE**********\n')
                    log_file.write(f'\tS_{i} before filtering: {S_i}. Metric: {metric}\n')
                    log_file.write(f'Discarding negative value {x}. Could be OpenLane error\n')
                    log_file.write(f'\tReference value: {g[i]}\n\n')
            elif x > 2*g[i]:
                filtered_S_i.append(2*g[i]) # worst result is twice the golden solution
            else:
                filtered_S_i.append(x)         
        S_i = filtered_S_i

        while len(S_i) < 5:
            S_i.append(2*g[i]) # worst result is twice the golden solution

        print(f'Filtered S_i: {S_i}')
        print(f'Golden solution: {g[i]}')
        print(f'Adding term {stable_divide(sum(S_i),(g[i] * 5))} to the total sum')
        print('-------------------------------------------------')
        # Aggregate results
        total_sum += stable_divide(sum(S_i),(g[i] * 5))

    # Penalize missing problems
    if num_C < n_problems:
        print(f'Penalizing {n_problems - num_C} missing problems')
        total_sum += 2*(n_problems - num_C) # worst result is 2 (twice the golden solution)
    
    return total_sum / n_problems
