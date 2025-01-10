import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from matplotlib.offsetbox import AnchoredText
import seaborn as sns

sns.set(font_scale=1.8)
sns.set_style("ticks")

def kaplan_meier_plot(orien_data, survival, gene, time_col='OverallSurvival (Months)', 
                      status_col='status', title=None, custom_split=None):
    """
    Generate Kaplan-Meier survival plots for a specific gene.
    
    Parameters:
        orien_data (pd.DataFrame): DataFrame with gene expression values (genes as columns).
        survival (pd.DataFrame): DataFrame with survival data.
        gene (str): The gene for which to create the KM plot.
        time_col (str): Column name for survival time in the survival DataFrame. Default is 'OverallSurvival (Months)'.
        status_col (str): Column name for event status in the survival DataFrame. Default is 'status'.
        title (str): Title for the plot. Default is None.
        custom_split (tuple): Optional custom split for high and low groups (e.g., (lower_bound, upper_bound)).
        
    Returns:
        None: Displays the Kaplan-Meier plot.
    """
    t = np.linspace(0, 200)  # Define timeline for KM plots
    gene_data = orien_data[[gene]].copy()
    
    # Apply median split or custom split
    if custom_split:
        low_threshold, high_threshold = custom_split
        gene_data['label'] = np.where(gene_data[gene] >= high_threshold, 'High', 
                                      np.where(gene_data[gene] <= low_threshold, 'Low', 'Middle'))
    else:
        gene_data['label'] = np.where(gene_data[gene] >= gene_data[gene].median(), 'High', 'Low')
    
    survival_with_labels = survival.copy()
    survival_with_labels['label'] = gene_data['label']
    
    # Extract survival times and events
    T_high = survival_with_labels.loc[survival_with_labels['label'] == 'High', time_col]
    E_high = survival_with_labels.loc[survival_with_labels['label'] == 'High', status_col]
    T_low = survival_with_labels.loc[survival_with_labels['label'] == 'Low', time_col]
    E_low = survival_with_labels.loc[survival_with_labels['label'] == 'Low', status_col]
    
    # Log-rank test
    results = logrank_test(T_high, T_low, event_observed_A=E_high, event_observed_B=E_low)
    p_value = results.p_value
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(12, 9))
    ax = plt.subplot()
    
    # Fit KM curves
    kmf_high = KaplanMeierFitter()
    ax = kmf_high.fit(durations=T_high, event_observed=E_high, label='High', timeline=t).plot_survival_function(ax=ax, color='tab:orange')
    
    kmf_low = KaplanMeierFitter()
    ax = kmf_low.fit(durations=T_low, event_observed=E_low, label='Low', timeline=t).plot_survival_function(ax=ax, color='tab:purple')
    
    # Add at-risk counts
    add_at_risk_counts(kmf_high, kmf_low, ax=ax)
    
    # Add p-value annotation
    ax.add_artist(AnchoredText("p = %.2E" % p_value, loc=3, frameon=False))
    
    # Add labels and title
    plt.xlabel('Months')
    if title:
        plt.title(title)
    else:
        plt.title(f'OS by {gene} expression')
    
    # Display the plot
    plt.tight_layout()
    plt.show()
