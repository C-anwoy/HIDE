import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer, util

import _settings

# --- Adjusted Scaling for 0.5\textwidth legibility (No Bold) ---
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 24,              # Increased base size
    'font.weight': 'normal',
    'axes.labelsize': 32,         # Large labels for visibility
    'axes.titlesize': 30,
    'axes.titleweight': 'normal',
    'axes.labelweight': 'normal',
    'xtick.labelsize': 28,         
    'ytick.labelsize': 28,
    'legend.fontsize': 24,         
    'axes.grid': False,           
    'axes.edgecolor': 'black',
    'axes.linewidth': 2.0,        # Thicker borders for sharp scaling
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

def identify_qcbw_subset(df, device="cuda:0"):
    print("\nCalculating QCBW subset for NQ...")
    model = SentenceTransformer(os.path.join(_settings.MODEL_PATH, 'nli-roberta-large'), device=device)
    
    hallucinations = df[df['is_correct'] == 0].copy()
    if len(hallucinations) == 0:
        df['is_qcbw'] = False
        return df

    q_embs = model.encode(hallucinations['question'].tolist(), convert_to_tensor=True)
    g_embs = model.encode(hallucinations['generated_text'].tolist(), convert_to_tensor=True)
    
    sims = util.cos_sim(q_embs, g_embs).diagonal().cpu().numpy()
    hallucinations['q_g_sim'] = sims
    
    threshold = np.percentile(sims, 75)
    qcbw_ids = hallucinations[hallucinations['q_g_sim'] >= threshold]['id'].tolist()
    
    df['is_qcbw'] = df['id'].isin(qcbw_ids)
    return df

def plot_for_model(model_name):
    base_dir = os.path.join(_settings.GENERATION_FOLDER, "mechanistic")
    squad_path = os.path.join(base_dir, f"{model_name}_SQuAD_mechanistic.csv")
    nq_path = os.path.join(base_dir, f"{model_name}_nq_open_mechanistic.csv")

    if not os.path.exists(squad_path) or not os.path.exists(nq_path):
        print(f"CSV files not found for {model_name}.")
        return

    df_squad = pd.read_csv(squad_path)
    df_nq = pd.read_csv(nq_path)

    # --- Force numeric types ---
    cols_to_check = ['HIDE_score', 'Omega', 'Delta_in']
    for col in cols_to_check:
        df_squad[col] = pd.to_numeric(df_squad[col], errors='coerce')
        df_nq[col] = pd.to_numeric(df_nq[col], errors='coerce')

    df_squad.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_nq.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_squad.dropna(subset=cols_to_check, inplace=True)
    df_nq.dropna(subset=cols_to_check, inplace=True)
    
    df_squad = df_squad[df_squad['HIDE_score'] > 0.001]
    df_nq = df_nq[df_nq['HIDE_score'] > 0.001]

    df_nq = identify_qcbw_subset(df_nq)
    df_squad['is_qcbw'] = False 

    # Large canvas with extra gutter space for big fonts
    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
    # Concise titles to save space for larger fonts
    datasets = [('SQuAD', df_squad), ('NQ', df_nq)]
    metrics = [('Omega', r'Attention Mass ($\Omega$)'), 
               ('Delta_in', r'Update Norm ($\Delta_{in}$)')]

    color_correct = '#9d2933'  
    color_halluc = '#377e7f'   
    color_qcbw = '#b37b2d'     

    subplot_labels = [['(a)', '(b)'], ['(c)', '(d)']]

    for col, (title, df) in enumerate(datasets):
        print(f"\nProcessing {title}...")
        
        # --- BALANCING LOGIC (60% Hallucination, 40% Correct) ---
        df_correct_all = df[df['is_correct'] == 1]
        df_halluc_all = df[df['is_correct'] == 0]
        
        n_correct = len(df_correct_all)
        n_halluc_avail = len(df_halluc_all)
        
        if n_halluc_avail > 1.5 * n_correct:
            target_halluc = int(1.5 * n_correct)
            target_correct = n_correct
        else:
            target_halluc = n_halluc_avail
            target_correct = int(n_halluc_avail / 1.5)
            
        correct = df_correct_all.sample(n=target_correct, random_state=42)
        
        qcbw_df = df_halluc_all[df_halluc_all['is_qcbw'] == True]
        other_halluc_df = df_halluc_all[df_halluc_all['is_qcbw'] == False]
        
        if len(qcbw_df) > 0:
            target_qcbw = min(len(qcbw_df), int(target_halluc * 0.40)) 
            target_other = target_halluc - target_qcbw
            
            if target_other > len(other_halluc_df):
                target_other = len(other_halluc_df)
                target_qcbw = min(len(qcbw_df), target_halluc - target_other)
                
            qcbw_sampled = qcbw_df.sample(n=target_qcbw, random_state=42)
            other_sampled = other_halluc_df.sample(n=target_other, random_state=42)
            halluc_sampled = pd.concat([qcbw_sampled, other_sampled])
        else:
            halluc_sampled = other_halluc_df.sample(n=target_halluc, random_state=42)
            
        df_balanced = pd.concat([correct, halluc_sampled])
        
        halluc = halluc_sampled[halluc_sampled['is_qcbw'] == False]
        qcbw = halluc_sampled[halluc_sampled['is_qcbw'] == True]

        for row, (metric_key, metric_label) in enumerate(metrics):
            ax = axes[row, col]
            
            # --- DIFFERENTIAL TRANSPARENCY & Z-ORDERING ---
            
            # ZORDER 2: Background Cloud (Small, transparent)
            ax.scatter(halluc['HIDE_score'], halluc[metric_key], 
                       c=color_halluc, alpha=0.3, s=50, edgecolors='none', 
                       label='Hallucination', marker='s', zorder=2)
            
            # ZORDER 3: Correct Points (Large, opaque, on top)
            ax.scatter(correct['HIDE_score'], correct[metric_key], 
                       c=color_correct, alpha=0.9, s=110, edgecolors='white', linewidths=0.8, 
                       label='Correct', marker='o', zorder=3)
            
            # ZORDER 4: QCBW Stars (Highlight layer)
            if len(qcbw) > 0:
                ax.scatter(qcbw['HIDE_score'], qcbw[metric_key], 
                           c=color_qcbw, alpha=0.85, s=400, edgecolors='black', linewidths=1.2, 
                           label='QCBW', marker='*', zorder=4)

            if len(df_balanced) > 1 and df_balanced['HIDE_score'].std() > 0:
                pcc, _ = pearsonr(df_balanced['HIDE_score'], df_balanced[metric_key])
            else:
                pcc = 0.0
            
            ax.set_title("") 
            ax.set_ylabel(metric_label, labelpad=15)
            
            caption_text = f"{subplot_labels[row][col]} {title}, PCC: {pcc:.3f}"
            ax.set_xlabel(r"HIDE Score ($\widehat{HSIC}$)" + f"\n\n{caption_text}", labelpad=15)
            
            for spine in ax.spines.values():
                spine.set_linewidth(2.0)
            
            if row == 0 and col == 1:
                legend = ax.legend(loc='lower right', frameon=True, edgecolor='black', borderpad=0.8)
                legend.get_frame().set_linewidth(1.5)
                legend.set_zorder(10)
                for lh in legend.legend_handles: 
                    lh.set_alpha(1.0) 

    # High hspace/wspace ensures large labels don't collide
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.12, hspace=0.6, wspace=0.35) 
    
    # PDF format for maximum vector quality in conference papers
    plot_path = os.path.join(base_dir, f"{model_name}_Mechanistic_Flow.pdf")
    plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.3)
    print(f"\nPlot saved successfully to: {plot_path}")

if __name__ == "__main__":
    plot_for_model("llama3-8b")

# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import pearsonr
# from sentence_transformers import SentenceTransformer, util

# import _settings

# # --- Adjusted Scaling for 0.5\textwidth ---
# # We bump these significantly so they match paper body text after scaling.
# plt.rcParams.update({
#     'font.family': 'sans-serif',
#     'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
#     'font.size': 24,              # Increased base size
#     'font.weight': 'normal',
#     'axes.labelsize': 32,         # Large labels for visibility
#     'axes.titlesize': 30,
#     'axes.titleweight': 'normal',
#     'axes.labelweight': 'normal',
#     'xtick.labelsize': 28,         # Large ticks
#     'ytick.labelsize': 28,
#     'legend.fontsize': 24,         # Large legend
#     'axes.grid': False,           
#     'axes.edgecolor': 'black',
#     'axes.linewidth': 2.0,        # Slightly thicker borders for clarity
#     'figure.facecolor': 'white',
#     'axes.facecolor': 'white'
# })

# def identify_qcbw_subset(df, device="cuda:0"):
#     print("\nCalculating QCBW subset for NQ...")
#     model = SentenceTransformer(os.path.join(_settings.MODEL_PATH, 'nli-roberta-large'), device=device)
    
#     hallucinations = df[df['is_correct'] == 0].copy()
#     if len(hallucinations) == 0:
#         df['is_qcbw'] = False
#         return df

#     q_embs = model.encode(hallucinations['question'].tolist(), convert_to_tensor=True)
#     g_embs = model.encode(hallucinations['generated_text'].tolist(), convert_to_tensor=True)
    
#     sims = util.cos_sim(q_embs, g_embs).diagonal().cpu().numpy()
#     hallucinations['q_g_sim'] = sims
    
#     threshold = np.percentile(sims, 75)
#     qcbw_ids = hallucinations[hallucinations['q_g_sim'] >= threshold]['id'].tolist()
    
#     df['is_qcbw'] = df['id'].isin(qcbw_ids)
#     return df

# def plot_for_model(model_name):
#     base_dir = os.path.join(_settings.GENERATION_FOLDER, "mechanistic")
#     squad_path = os.path.join(base_dir, f"{model_name}_SQuAD_mechanistic.csv")
#     nq_path = os.path.join(base_dir, f"{model_name}_nq_open_mechanistic.csv")

#     if not os.path.exists(squad_path) or not os.path.exists(nq_path):
#         print(f"CSV files not found for {model_name}.")
#         return

#     df_squad = pd.read_csv(squad_path)
#     df_nq = pd.read_csv(nq_path)

#     cols_to_check = ['HIDE_score', 'Omega', 'Delta_in']
#     for col in cols_to_check:
#         df_squad[col] = pd.to_numeric(df_squad[col], errors='coerce')
#         df_nq[col] = pd.to_numeric(df_nq[col], errors='coerce')

#     df_squad.dropna(subset=cols_to_check, inplace=True)
#     df_nq.dropna(subset=cols_to_check, inplace=True)
    
#     df_squad = df_squad[df_squad['HIDE_score'] > 0.001]
#     df_nq = df_nq[df_nq['HIDE_score'] > 0.001]

#     df_nq = identify_qcbw_subset(df_nq)
#     df_squad['is_qcbw'] = False 

#     # Keep a large canvas but give the subplots more "gutter" space
#     fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    
#     datasets = [('SQuAD', df_squad), ('NQ', df_nq)]
#     metrics = [('Omega', r'Attention Mass ($\Omega$)'), 
#                ('Delta_in', r'Update Norm ($\Delta_{in}$)')]

#     color_correct = '#9d2933'  
#     color_halluc = '#377e7f'   
#     color_qcbw = '#b37b2d'     

#     subplot_labels = [['(a)', '(b)'], ['(c)', '(d)']]

#     for col, (title, df) in enumerate(datasets):
#         df_correct_all = df[df['is_correct'] == 1]
#         df_halluc_all = df[df['is_correct'] == 0]
        
#         target_total = 1000 
#         target_correct = min(len(df_correct_all), int(target_total * 0.40))
#         target_halluc = min(len(df_halluc_all), int(target_total * 0.60))
        
#         # Budget redistribution for sparsity fix
#         if target_correct < int(target_total * 0.40):
#             target_halluc = min(len(df_halluc_all), target_halluc + (int(target_total * 0.40) - target_correct))
            
#         correct = df_correct_all.sample(n=target_correct, random_state=42)
        
#         qcbw_df = df_halluc_all[df_halluc_all['is_qcbw'] == True]
#         other_halluc_df = df_halluc_all[df_halluc_all['is_qcbw'] == False]
        
#         if len(qcbw_df) > 0:
#             target_qcbw = min(len(qcbw_df), int(target_halluc * 0.40)) 
#             target_other = target_halluc - target_qcbw
#             qcbw_sampled = qcbw_df.sample(n=target_qcbw, random_state=42)
#             other_sampled = other_halluc_df.sample(n=min(target_other, len(other_halluc_df)), random_state=42)
#             halluc_sampled = pd.concat([qcbw_sampled, other_sampled])
#         else:
#             halluc_sampled = other_halluc_df.sample(n=target_halluc, random_state=42)
            
#         df_balanced = pd.concat([correct, halluc_sampled])
#         halluc = halluc_sampled[halluc_sampled['is_qcbw'] == False]
#         qcbw = halluc_sampled[halluc_sampled['is_qcbw'] == True]

#         for row, (metric_key, metric_label) in enumerate(metrics):
#             ax = axes[row, col]
            
#             ax.scatter(halluc['HIDE_score'], halluc[metric_key], 
#                        c=color_halluc, alpha=0.3, s=50, edgecolors='none', 
#                        label='Hallucination', marker='s', zorder=2)
            
#             ax.scatter(correct['HIDE_score'], correct[metric_key], 
#                        c=color_correct, alpha=0.9, s=110, edgecolors='white', linewidths=0.8, 
#                        label='Correct', marker='o', zorder=3)
            
#             if len(qcbw) > 0:
#                 ax.scatter(qcbw['HIDE_score'], qcbw[metric_key], 
#                            c=color_qcbw, alpha=0.85, s=400, edgecolors='black', linewidths=1.2, 
#                            label='QCBW', marker='*', zorder=4)

#             pcc, _ = pearsonr(df_balanced['HIDE_score'], df_balanced[metric_key])
            
#             ax.set_title("") 
#             ax.set_ylabel(metric_label, labelpad=15) # Added padding for large labels
            
#             caption_text = f"{subplot_labels[row][col]} {title}, PCC: {pcc:.3f}"
#             ax.set_xlabel(r"HIDE Score ($\widehat{HSIC}$)" + f"\n\n{caption_text}", labelpad=15)
            
#             for spine in ax.spines.values():
#                 spine.set_linewidth(2.0)
            
#             if row == 0 and col == 1:
#                 legend = ax.legend(loc='lower right', frameon=True, edgecolor='black', borderpad=0.8)
#                 legend.get_frame().set_linewidth(1.5)
#                 for lh in legend.legend_handles: 
#                     lh.set_alpha(1.0) 

#     # Increased hspace and wspace to prevent large labels from overlapping neighboring subplots
#     plt.tight_layout()
#     fig.subplots_adjust(bottom=0.12, hspace=0.6, wspace=0.35) 
    
#     plot_path = os.path.join(base_dir, f"{model_name}_Mechanistic_Flow.pdf")
#     plt.savefig(plot_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.3)
#     print(f"\nPlot saved successfully to: {plot_path}")

# if __name__ == "__main__":
#     plot_for_model("gemma-2-9b")