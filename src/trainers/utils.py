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






def plot_policy_diagnostics(
    x_true: np.ndarray,
    y_true: np.ndarray,
    samples_x: np.ndarray,
    samples_y: np.ndarray,
    x_policy_mapped: np.ndarray,
    policy_pdf: np.ndarray,
    sampler_info: str = "Unknown Function",
    title_suffix: str = ""
):
    """
    Plots the true function, agent samples, and policy distribution.

    Args:
        x_true: X-values for the true function.
        y_true: Y-values for the true function.
        samples_x: X-values of samples taken by the agent.
        samples_y: Y-values of samples taken by the agent.
        x_policy_mapped: X-values for the policy PDF, mapped to the function's domain.
        policy_pdf: PDF values of the policy.
        sampler_info: Name of the true function (e.g., Ackley).
        title_suffix: Optional suffix for the plot title.
    """
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot the true function
    color_true_func = 'tab:blue'
    ax1.set_xlabel('x')
    ax1.set_ylabel('True Function Value', color=color_true_func)
    ax1.plot(x_true, y_true, label=f'True Function ({sampler_info})', color=color_true_func, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color_true_func)

    # Plot the samples taken by the agent
    ax1.scatter(samples_x, samples_y, label='Agent Samples', color='red', marker='o', s=50, alpha=0.7, zorder=5)

    # Create a second y-axis for the policy PDF
    ax2 = ax1.twinx()
    color_policy_pdf = 'tab:green'
    ax2.set_ylabel('Policy PDF', color=color_policy_pdf)
    ax2.plot(x_policy_mapped, policy_pdf, label='Policy PDF', color=color_policy_pdf, linewidth=2)
    ax2.fill_between(x_policy_mapped, policy_pdf, color=color_policy_pdf, alpha=0.2)
    ax2.tick_params(axis='y', labelcolor=color_policy_pdf)

    # Titles and legends
    plt.title(f'RL Agent Diagnostics: {sampler_info}{title_suffix}')
    fig.tight_layout() # Otherwise the right y-label is slightly clipped
    
    # Combine legends from both axes
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.grid(True, linestyle=':', alpha=0.7)
    plt.show()