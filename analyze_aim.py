import json
import matplotlib.pyplot as plt
from collections import defaultdict
import re
import argparse
import matplotlib.cm as cm
import numpy as np 

# Assume K_val matches your training parameter, typically 32
K_VAL = 32

def interpret_aim_as_action_numerical(aim_sequence_str, K):
    """
    Interprets an AIM sequence string into a numerical action.
    'C' (Cooperate) -> 0
    'D' (Defect) -> 1
    """
    try:
        # Convert the string representation of AIM sequence '[x, y]' to a list [x, y]
        aim_list = json.loads(aim_sequence_str)
        
        # Ensure it's a list and not empty
        if not isinstance(aim_list, list) or not aim_list:
            raise ValueError("AIM sequence is not a valid list or is empty.")
        
        # The first element should be an integer for interpretation
        first_id = aim_list[0] 
        if not isinstance(first_id, int):
            raise ValueError(f"First AIM ID is not an integer: {first_id}")

        # Determine action based on the first ID
        if first_id < K // 2:
            return 0  # Cooperate
        else:
            return 1  # Defect
    except (json.JSONDecodeError, ValueError, IndexError, TypeError) as e:
        # print(f"Warning: Could not interpret AIM sequence '{aim_sequence_str}' due to {e}. Skipping.")
        return None

def parse_round_from_context(context_str):
    """Extracts the round number from the context string"""
    match = re.search(r'PD Round (\d+)', context_str)
    if match:
        return int(match.group(1))
    return None

def analyze_aim_dictionary(file_path, top_n=10):
    """
    Analyzes the AIM dictionary file, tallies AIM usage, and plots time series.
    """
    try:
        with open(file_path, "r") as f:
            entries = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please ensure the file is in the correct path or specify the correct filename.")
        return
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not in valid JSON format. Please ensure it's a valid JSON file.")
        return

    # Store frequency and corresponding actions for each AIM sequence
    # Key for defaultdict must be hashable, so convert list AIM_id to tuple
    aim_stats = defaultdict(lambda: {'A_usage': 0, 'B_usage': 0, 'A_actions': defaultdict(int), 'B_actions': defaultdict(int)})
    
    # Store occurrence points for each AIM sequence over time, for plotting
    # Key for defaultdict must be hashable, so convert list AIM_id to tuple
    aim_time_series_A = defaultdict(lambda: {'rounds': [], 'actions_numerical': []})
    aim_time_series_B = defaultdict(lambda: {'rounds': [], 'actions_numerical': []})

    print(f"Processing {len(entries)} dictionary entries from '{file_path}'...")

    for entry in entries:
        aim_id_raw = entry.get("aim_id") # Get raw aim_id (could be string or list)
        human_label = entry.get("human_label")
        context = entry.get("context")
        usage_count = entry.get("usage_count", 1) # Default usage_count to 1

        if not all([aim_id_raw, human_label, context]):
            # print(f"Warning: Missing data in entry: {entry}. Skipping.")
            continue

        # Convert aim_id to a consistent string format for interpret_aim_as_action_numerical
        # And convert to tuple for dictionary keys if it's a list.
        if isinstance(aim_id_raw, list):
            aim_id_str = json.dumps(aim_id_raw) # Convert list to string for interpretation function
            aim_id_key = tuple(aim_id_raw)     # Convert list to tuple for dictionary key (hashable)
        elif isinstance(aim_id_raw, str):
            aim_id_str = aim_id_raw
            try: # Try to convert string like "[x, y]" to tuple (x, y)
                aim_id_key = tuple(json.loads(aim_id_str))
            except json.JSONDecodeError: # If it's just a simple string, use it as is
                aim_id_key = aim_id_str
        else:
            # print(f"Warning: Unexpected aim_id type: {type(aim_id_raw)}. Skipping entry: {entry}")
            continue


        round_num = parse_round_from_context(context)
        if round_num is None:
            # print(f"Warning: Could not parse round from context: {context}. Skipping.")
            continue

        action_numerical = interpret_aim_as_action_numerical(aim_id_str, K_VAL)
        if action_numerical is None:
            continue # Skip if interpretation failed

        # Determine if it's Agent A or Agent B based on context
        if "(Response)" in context:
            # Agent B (Responder)
            aim_stats[aim_id_key]['B_usage'] += usage_count
            aim_stats[aim_id_key]['B_actions'][human_label] += usage_count
            aim_time_series_B[aim_id_key]['rounds'].append(round_num)
            aim_time_series_B[aim_id_key]['actions_numerical'].append(action_numerical)
        else:
            # Agent A (Sender)
            aim_stats[aim_id_key]['A_usage'] += usage_count
            aim_stats[aim_id_key]['A_actions'][human_label] += usage_count
            aim_time_series_A[aim_id_key]['rounds'].append(round_num)
            aim_time_series_A[aim_id_key]['actions_numerical'].append(action_numerical)
    
    print("\n--- AIM ID Usage Statistics ---")

    # Find Top N AIM IDs for Agent A and B respectively
    # Sort by total usage for the respective agent
    top_aim_A = sorted([ (stats['A_usage'], aim_id_key) for aim_id_key, stats in aim_stats.items() if stats['A_usage'] > 0 ], reverse=True)[:top_n]
    top_aim_B = sorted([ (stats['B_usage'], aim_id_key) for aim_id_key, stats in aim_stats.items() if stats['B_usage'] > 0 ], reverse=True)[:top_n]


    print(f"\nAgent A (Sender) Top {top_n} AIM IDs:")
    for usage, aim_id_key in top_aim_A:
        actions = aim_stats[aim_id_key]['A_actions']
        # Convert tuple key back to list for display if it was originally a list
        display_aim_id = list(aim_id_key) if isinstance(aim_id_key, tuple) else aim_id_key
        print(f"- AIM: {display_aim_id}, Total Usage: {usage}, Corresponding Actions: {dict(actions)}")

    print(f"\nAgent B (Responder) Top {top_n} AIM IDs:")
    for usage, aim_id_key in top_aim_B:
        actions = aim_stats[aim_id_key]['B_actions']
        # Convert tuple key back to list for display if it was originally a list
        display_aim_id = list(aim_id_key) if isinstance(aim_id_key, tuple) else aim_id_key
        print(f"- AIM: {display_aim_id}, Total Usage: {usage}, Corresponding Actions: {dict(actions)}")

    # --- Plotting Time Series ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'Top AIM ID Usage Over Time (K_VAL={K_VAL}, Data from: {file_path})')

    # Use a distinct colormap for qualitative data (e.g., 'tab10' or 'Paired')
    # Make sure to get enough colors for all top_n AIMs
    # FIX: Updated get_cmap usage to avoid deprecation warning
    cmap = plt.colormaps.get_cmap('tab10') # Simpler call for default tab10
    colors = [cmap(i) for i in range(top_n)] 
    # If top_n > 10, consider other colormaps like 'Paired' or 'viridis' directly scaled to top_n
    if top_n > 10:
        cmap = plt.colormaps.get_cmap('viridis') # Example for more colors
        colors = [cmap(i/top_n) for i in range(top_n)]


    # Bar thickness (height for C=0, D=1 actions)
    bar_height = 0.8 
    line_width = 3   

    # Agent A's Plot
    ax0 = axes[0]
    ax0.set_title(f'Agent A (Sender) - Top {top_n} AIM IDs')
    ax0.set_ylabel('Interpreted Action')
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels(['Cooperate (C)', 'Defect (D)'])
    ax0.set_ylim([-0.5, 1.5]) 
    ax0.grid(True, axis='x') 
    ax0.axhline(y=0, color='gray', linestyle='--', linewidth=0.8) # Line at C
    ax0.axhline(y=1, color='gray', linestyle='--', linewidth=0.8) # Line at D

    for i, (usage, aim_id_key) in enumerate(top_aim_A):
        display_aim_id = list(aim_id_key) if isinstance(aim_id_key, tuple) else aim_id_key
        current_color = colors[(i % len(colors))]
        
        rounds = np.array(aim_time_series_A[aim_id_key]['rounds'])
        actions = np.array(aim_time_series_A[aim_id_key]['actions_numerical'])
        
        ax0.vlines(x=rounds, ymin=actions - 0.2, ymax=actions + 0.2, # Make a short vertical line
                   colors=current_color, lw=line_width, label=f'{display_aim_id} (Usage: {usage})')

    ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    # Agent B's Plot
    ax1 = axes[1]
    ax1.set_title(f'Agent B (Responder) - Top {top_n} AIM IDs')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Interpreted Action')
    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(['Cooperate (C)', 'Defect (D)'])
    ax1.set_ylim([-0.5, 1.5]) 
    ax1.grid(True, axis='x') 
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8) # Line at C
    ax1.axhline(y=1, color='gray', linestyle='--', linewidth=0.8) # Line at D
    
    for i, (usage, aim_id_key) in enumerate(top_aim_B):
        display_aim_id = list(aim_id_key) if isinstance(aim_id_key, tuple) else aim_id_key
        current_color = colors[(i % len(colors))]

        rounds = np.array(aim_time_series_B[aim_id_key]['rounds'])
        actions = np.array(aim_time_series_B[aim_id_key]['actions_numerical'])

        ax1.vlines(x=rounds, ymin=actions - 0.2, ymax=actions + 0.2, # Make a short vertical line
                   colors=current_color, lw=line_width, label=f'{display_aim_id} (Usage: {usage})')
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    plt.tight_layout(rect=[0, 0, 0.85, 1]) 
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze AIM dictionary file, tally usage, and plot time series.")
    parser.add_argument('--file', type=str, default="aim_dictionary.json",
                        help='Path to the AIM dictionary JSON file. Defaults to aim_dictionary.json.')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Number of top AIM IDs to display. Defaults to 10.')
    
    args = parser.parse_args()

    analyze_aim_dictionary(file_path=args.file, top_n=args.top_n)