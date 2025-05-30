import jax.numpy as jnp
import rlax
import jax
# plot_utils.py
import matplotlib.pyplot as plt
import numpy as np

@jax.jit
def average_reward_and_return_in_episode(rewards_array,gamma):
    average_reward=jnp.mean(rewards_array)
    returns_mean=rlax.discounted_returns(rewards_array,gamma*jnp.ones_like(rewards_array),
                                    jnp.zeros_like(rewards_array)).mean()
    return average_reward,returns_mean






    


# def plot_policy_diagnostics(
#     x_true: np.ndarray,
#     y_true: np.ndarray,
#     all_samples_x_hist: np.ndarray, # New: All historical samples X
#     all_samples_y_hist: np.ndarray, # New: All historical samples Y
#     current_samples_x: np.ndarray,  # Samples from the current step X
#     current_samples_y: np.ndarray,  # Samples from the current step Y
#     x_policy_mapped: np.ndarray,
#     policy_pdf: np.ndarray,
#     sampler_info: str = "Unknown Function",
#     title_suffix: str = ""
# ):
#     """
#     Plots true function, historical agent samples, current agent samples, and policy distribution.
#     Does NOT call plt.show() and returns the figure object.
#     """
#     fig, ax1 = plt.subplots(figsize=(12, 7))

#     # Plot the true function
#     color_true_func = 'tab:blue'
#     ax1.set_xlabel('x')
#     ax1.set_ylabel('True Function Value', color=color_true_func)
#     ax1.plot(x_true, y_true, label=f'True Function ({sampler_info})', color=color_true_func, linestyle='--')
#     ax1.tick_params(axis='y', labelcolor=color_true_func)

#     # Plot all historical samples (light color, smaller marker)
#     if all_samples_x_hist.size > 0: # Check if there's any historical data
#         ax1.scatter(all_samples_x_hist, all_samples_y_hist, label='Samples', color='grey', marker='o', s=35, alpha=0.5, zorder=3, edgecolors='black')

#     # Plot the samples taken by the agent at the CURRENT step (more prominent)
#     if current_samples_x.size > 0:
#         ax1.scatter(current_samples_x, current_samples_y, label='Last sample', color='red', marker='o', s=70, alpha=0.9, zorder=5, edgecolors='black')

#     # Create a second y-axis for the policy PDF
#     ax2 = ax1.twinx()
#     color_policy_pdf = 'tab:green'
#     ax2.set_ylabel('Policy PDF', color=color_policy_pdf)
#     ax2.plot(x_policy_mapped, policy_pdf, label='Policy PDF', color=color_policy_pdf, linewidth=2)
#     ax2.fill_between(x_policy_mapped, policy_pdf, color=color_policy_pdf, alpha=0.2)
#     ax2.tick_params(axis='y', labelcolor=color_policy_pdf)
#     ax2.set_ylim(bottom=0) # Ensure PDF y-axis starts at 0

#     # Titles and legends
#     plt.title(f'RL Agent Diagnostics: {sampler_info}{title_suffix}')
#     fig.tight_layout()

#     lines, labels = ax1.get_legend_handles_labels()
#     lines2, labels2 = ax2.get_legend_handles_labels()
#     ax2.legend(lines + lines2, labels + labels2, loc='upper right')

#     plt.grid(True, linestyle=':', alpha=0.7)
    
#     return fig # Return the figure object

# import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.cm as cm # For colormaps

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm # For colormaps

def plot_policy_diagnostics(
    x_true: np.ndarray,
    y_true: np.ndarray,
    all_samples_x_hist: np.ndarray, # All historical samples X
    all_samples_y_hist: np.ndarray, # All historical samples Y
    current_samples_x: np.ndarray,  # Samples from the current step X
    current_samples_y: np.ndarray,  # Samples from the current step Y
    x_policy_mapped: np.ndarray,    # X values for the PDF, shape (num_x_points,)
    policy_pdf: np.ndarray,         # PDF values, shape (num_pdfs, num_x_points)
    func_estimate: np.ndarray, # Optional: Function estimate for the agent
    sampler_info: str = "Unknown Function",
    title_suffix: str = ""
):
    """
    Plots true function, historical agent samples, current agent samples, and policy distributions.
    Handles multiple policy PDFs if policy_pdf is 2D.
    Ensures full PDF visibility and color-codes current samples to their respective PDFs if possible.
    Does NOT call plt.show() and returns the figure object.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot the true function
    color_true_func = 'tab:blue'
    ax1.set_xlabel('x')
    ax1.set_ylabel('True Function Value', color=color_true_func)
    ax1.plot(x_true, y_true, label=f'True Function ({sampler_info})', color=color_true_func, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color_true_func)

    # Plot all historical samples
    if all_samples_x_hist.size > 0:
        ax1.scatter(all_samples_x_hist, all_samples_y_hist, label='Historical Samples', color='grey', marker='o', s=35, alpha=0.5, zorder=3, edgecolors='black')

    # --- Prepare for Policy PDF(s) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Policy PDF(s)', color='tab:green') # General label for PDF axis
    ax2.tick_params(axis='y', labelcolor='tab:green')

    max_observed_pdf_y_value = 0.0
    pdf_colors = [] # To store colors for PDFs, also usable for samples
    
    print(f"Policy PDF shape: {policy_pdf.shape}, x_policy_mapped shape: {x_policy_mapped.shape}")

    # Determine number of PDFs and prepare data/colors
    if policy_pdf.ndim == 1:
        num_pdfs = 1
        pdfs_to_plot = policy_pdf.reshape(1, -1)
        pdf_colors = [plt.colormaps.get_cmap('viridis')(0.5)]
    elif policy_pdf.ndim == 2:
        num_pdfs = policy_pdf.shape[0]
        pdfs_to_plot = policy_pdf
        if num_pdfs > 0:
            pdf_colors = plt.colormaps.get_cmap('viridis')(np.linspace(0, 1, num_pdfs))
        else:
             pdfs_to_plot = np.array([[]])
    else:
        print(f"Warning: policy_pdf has unexpected ndim={policy_pdf.ndim}. Expected 1 or 2. No PDFs will be plotted.")
        pdfs_to_plot = np.array([[]])
        num_pdfs = 0

    # Plot each PDF
    for i in range(num_pdfs):
        current_pdf_values = pdfs_to_plot[i, :]
        current_color = pdf_colors[i]
        pdf_label = f'Policy PDF {i+1}' if num_pdfs > 1 else 'Policy PDF'

        ax2.plot(x_policy_mapped, current_pdf_values, label=pdf_label, color=current_color, linewidth=2)
        ax2.fill_between(x_policy_mapped, current_pdf_values, color=current_color, alpha=0.2)

        if current_pdf_values.size > 0:
            max_observed_pdf_y_value = max(max_observed_pdf_y_value, np.max(current_pdf_values))

    # Set PDF y-axis limits for full visibility
    if num_pdfs > 0 and max_observed_pdf_y_value > 0:
        ax2.set_ylim(bottom=0, top=max_observed_pdf_y_value * 1.05) # 5% margin
    elif num_pdfs > 0: # Max PDF value is 0 or PDFs were empty arrays
        ax2.set_ylim(bottom=0, top=0.1) # Small default height if PDF values are all zero
    else: # No PDFs plotted
        ax2.set_ylim(bottom=0, top=1.0) # Default scale for an empty PDF axis
        
    current_samples_x = np.atleast_1d(current_samples_x)
    current_samples_y = np.atleast_1d(current_samples_y)
    
    if func_estimate is not None:
        # Ensure func_estimate is an array, at least 1D
        _func_estimate = np.atleast_1d(func_estimate)

        # Case 1: func_estimate is a single line (1D array matching x_policy_mapped length)
        if _func_estimate.ndim == 1 and _func_estimate.shape[0] == x_policy_mapped.shape[0]:
            ax1.plot(x_policy_mapped, _func_estimate,
                     label='Policy Function Estimate',
                     color='purple', # Distinct color for a single estimate line
                     linestyle='-.', linewidth=1.5, zorder=2.5) # zorder between true_func and hist_samples

        # Case 2: func_estimate is multiple lines (2D array, one line per PDF, matching x_policy_mapped length)
        elif _func_estimate.ndim == 2 and \
             num_pdfs > 0 and \
             _func_estimate.shape[0] == num_pdfs and \
             _func_estimate.shape[1] == x_policy_mapped.shape[0]:
            for i in range(num_pdfs):
                line_label = f'Policy Estimate Line {i+1}' if num_pdfs > 1 else 'Policy Estimate Line'
                ax1.plot(x_policy_mapped, _func_estimate[i, :],
                         label=line_label,
                         color=pdf_colors[i], # Match PDF color
                         linestyle='-.', linewidth=1.5, zorder=2.5) # zorder
        elif _func_estimate.size > 0 : # func_estimate provided but doesn't match expected shapes for line plotting
            print(f"Warning: func_estimate (shape {_func_estimate.shape}) could not be plotted as line(s) against x_policy_mapped (shape {x_policy_mapped.shape}) with {num_pdfs} PDFs. Skipping.")

    # --- Plot Current Samples ---
    # Color-code them if the number of samples matches the number of PDFs
    if current_samples_x.size > 0:
        # Assume one sample per PDF if sizes match.
        can_color_code_samples = (current_samples_x.size == num_pdfs) and (num_pdfs > 0)

        if can_color_code_samples:
            for i in range(num_pdfs): # num_pdfs equals current_samples_x.size here
                # print(f"Color-coding current sample {i} to PDF {i}.", current_samples_x.shape, current_samples_y.shape)
                
                ax1.scatter(current_samples_x[i], current_samples_y[i],
                            color=pdf_colors[i], # Use the corresponding PDF color
                            marker='o', s=70, alpha=0.9, zorder=5, edgecolors='black',
                            label='_nolegend_') # Individual points don't get separate legend entries
            # Add a single, representative legend entry for all current (color-coded) samples.
            # This scatter call plots no visible data but creates the legend item.
            # Using a neutral color like black for the legend swatch itself.
            ax1.scatter([], [], color='black', marker='o', s=70, edgecolors='black', alpha=0.9,
                        label='Current Samples (color-coded to PDF)')
        else:
            # Condition for printing warning: if there are PDFs but sizes don't match for color-coding
            if num_pdfs > 0 and current_samples_x.size != num_pdfs:
                 print(f"Warning: Number of current samples ({current_samples_x.size}) differs from number of PDFs ({num_pdfs}). Current samples will use default color 'red'.")
            ax1.scatter(current_samples_x, current_samples_y,
                        label='Current Samples', color='red', marker='o',
                        s=70, alpha=0.9, zorder=5, edgecolors='black')

    # --- Titles and Legends ---
    plt.title(f'RL Agent Diagnostics: {sampler_info}{title_suffix}')
    # Ensure tight_layout is called BEFORE legend to get correct bounding boxes.
    fig.tight_layout()

    lines_ax1, labels_ax1 = ax1.get_legend_handles_labels()
    lines_ax2, labels_ax2 = ax2.get_legend_handles_labels()

    # Combine legends. Place on ax1 if ax2 has no items, otherwise let ax2 host.
    if labels_ax2: # If ax2 has legend items (i.e., PDFs were plotted)
        ax2.legend(lines_ax1 + lines_ax2, labels_ax1 + labels_ax2, loc='upper right')
    elif labels_ax1: # Only ax1 has legend items
        ax1.legend(loc='upper right')
    # If neither has items, no legend will be drawn.

    plt.grid(True, linestyle=':', alpha=0.7)

    return fig