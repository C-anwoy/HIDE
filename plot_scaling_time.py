import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

# --- Apply custom styling (Zero bold fonts, LaTeX sub-captions) ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 20,              
    'font.weight': 'normal',
    'axes.labelsize': 24,         
    'axes.titlesize': 22,
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 18,
    'axes.grid': False,           
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.5,        
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

def plot_scalability():
    file_name = "/home/anwoy/HIDE/data/output/ablations/llama3-8b_SQuAD_1/0.pkl"
    print(f"Loading data from: {file_name}")
    
    with open(file_name, "rb") as f:
        resultDict = pkl.load(f)

    output_tokens_topk_list = []
    single_gen_time_list = []
    hide_time_list = []
    
    for item in resultDict:
        try:
            op_tokens_len = len(item['output_tokens_topk'])
            single_gen_time = item['greedy_generation_time']
            hide_time = item['hsic_time']
            
            output_tokens_topk_list.append(op_tokens_len)
            single_gen_time_list.append(single_gen_time)
            hide_time_list.append(hide_time) # Strictly in seconds
        except:
            continue

    # Convert to numpy arrays
    tokens = np.array(output_tokens_topk_list)
    gen_time = np.array(single_gen_time_list)
    hide_time = np.array(hide_time_list)

    # Calculate percentage overhead (HIDE Time / Generation Time * 100)
    overhead_pct = (hide_time / gen_time) * 100 

    # --- PRUNING LOGIC: Remove overhead > 200% ---
    valid_mask = overhead_pct <= 200.0
    pruned_count = len(tokens) - np.sum(valid_mask)
    print(f"Pruned {pruned_count} edge-cases where overhead exceeded 200%.")
    
    tokens = tokens[valid_mask]
    gen_time = gen_time[valid_mask]
    hide_time = hide_time[valid_mask]
    overhead_pct = overhead_pct[valid_mask]

    fig = plt.figure(figsize=(16, 6))
    
    color_hide = '#377e7f'   # Muted Teal
    color_gen = '#9d2933'    # Muted Dark Red

    # --- Subplot (a): HIDE Time vs Token Budget ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.scatter(tokens, hide_time, alpha=0.6, s=80, c=color_hide)
    
    # Curved, translucent trend line (Polynomial degree 2)
    if len(tokens) > 1:
        z = np.polyfit(tokens, hide_time, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(tokens.min(), tokens.max(), 100)
        ax1.plot(x_trend, p(x_trend), color='black', alpha=0.4, linewidth=3, linestyle='--')
    
    ax1.set_ylabel("HIDE Latency (s)", fontsize=22)
    ax1.set_xlabel(r"Token Budget ($n_{eff}$)" + "\n\n(a) Scalability with Sequence Length", fontsize=22)
    
    # --- Subplot (b): Overhead Comparison (Percentage) ---
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.scatter(gen_time, overhead_pct, alpha=0.7, s=80, c=color_gen)
    
    # Crisp, concise label
    ax2.set_ylabel("Overhead (%)", fontsize=22)
    ax2.set_xlabel("Greedy Generation Time (s)\n\n(b) Relative Overhead", fontsize=22)
    
    # Apply border styling
    for ax in [ax1, ax2]:
        ax.set_title("") 
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    plt.tight_layout()
    fig.subplots_adjust(bottom=0.25) # Make room for the LaTeX sub-captions
    
    plot_path = "Scalability_Analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Scalability plot saved to: {plot_path}")

if __name__ == "__main__":
    plot_scalability()