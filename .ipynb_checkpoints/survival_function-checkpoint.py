from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import numpy as np
import pandas as pd
from tqdm import tqdm
import statsmodels.api as sm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def survival_analysis(gene_expression_data, survival, percentiles=(0.33, 0.66), 
                      survival_columns=('OverallSurvival (Months)', 'status'),
                      correction='bh', debug=False):
    """
    Perform survival analysis comparing top and bottom percentile groups based on gene expression.

    Parameters:
        gene_expression_data (pd.DataFrame): DataFrame with gene expression values (genes as columns).
        survival (pd.DataFrame): DataFrame with survival data.
        percentiles (tuple): Percentiles to split the data. Default is (0.33, 0.66) for thirds.
        survival_columns (tuple): Names of the columns for survival time and event status.
            Default is ('OverallSurvival (Months)', 'status').
        correction (str): Multiple testing correction method. Options are 'bh' (default) or 'bonferroni'.
        debug (bool): If True, print intermediate steps for debugging.

    Returns:
        pd.DataFrame: Results with ranks, test statistics, p-values, corrected p-values, and groups with better survival.
    """
    time_col, status_col = survival_columns

    def label_by_percentiles(value, low, high):
        if value >= high:
            return 'Top'
        elif value <= low:
            return 'Bottom'
        else:
            return 'Middle'

    gene_sep_t = []
    gene_sep_p = []
    hi_low_survival = []

    genes = list(gene_expression_data.columns)

    # Wrap the loop with tqdm for progress tracking
    for i in tqdm(genes, desc="Analyzing genes", unit="gene"):
        gene_data = gene_expression_data[[i]].copy()
        summary = gene_data.describe(percentiles=[percentiles[0], percentiles[1]])
        low_percentile = summary.loc[f'{int(percentiles[0] * 100)}%', i]
        high_percentile = summary.loc[f'{int(percentiles[1] * 100)}%', i]
        
        # Apply function to label each row
        gene_data['label'] = gene_data[i].apply(label_by_percentiles, low=low_percentile, high=high_percentile)
        
        # Create a copy of the survival DataFrame to avoid modifying the original
        survival_with_labels = survival.copy()
        survival_with_labels['label'] = gene_data['label']  # Add labels to the copied DataFrame
        
        # Extract survival times and events for comparison
        T_top = survival_with_labels.loc[survival_with_labels['label'] == 'Top', time_col]
        T_bottom = survival_with_labels.loc[survival_with_labels['label'] == 'Bottom', time_col]
        E_top = survival_with_labels.loc[survival_with_labels['label'] == 'Top', status_col]
        E_bottom = survival_with_labels.loc[survival_with_labels['label'] == 'Bottom', status_col]
        
        # Perform log-rank test
        results = logrank_test(T_top, T_bottom, event_observed_A=E_top, event_observed_B=E_bottom)
        gene_sep_t.append(results.test_statistic)
        gene_sep_p.append(results.p_value)
        
        # Fit Kaplan-Meier curves for high and low groups
        kmf_high = KaplanMeierFitter().fit(durations=T_top, event_observed=E_top)
        kmf_low = KaplanMeierFitter().fit(durations=T_bottom, event_observed=E_bottom)
        
        # Calculate the average survival probability for both groups
        avg_surv_high = kmf_high.survival_function_['KM_estimate'].mean()
        avg_surv_low = kmf_low.survival_function_['KM_estimate'].mean()
        
        if debug:
            print(f"Gene: {i}")
            print(f"  Average Survival Probability (High): {avg_surv_high}")
            print(f"  Average Survival Probability (Low): {avg_surv_low}")
        
        # Determine which group has better survival based on average probabilities
        if avg_surv_high > avg_surv_low:
            hi_low_survival.append('High')  # High group has better survival
        elif avg_surv_high < avg_surv_low:
            hi_low_survival.append('Low')  # Low group has better survival
        else:
            hi_low_survival.append('Equal')  # Survival probabilities are the same

    # Compile results into a DataFrame
    gene_sep = pd.DataFrame(
        list(zip(genes, gene_sep_t, gene_sep_p, hi_low_survival)),
        columns=['gene', 't_statistic', 'p_value', 'group with better survival']
    )
    
    # Sort results by t_statistic in descending order
    gene_sep = gene_sep.sort_values(by='t_statistic', ascending=False)
    
    # Add rank column
    gene_sep['rank'] = range(1, len(gene_sep) + 1)
    gene_sep = gene_sep[['rank', 't_statistic', 'p_value', 'corrected_p_value', 'group with better survival']]
    
    # Apply multiple testing correction
    if correction == 'bh':
        gene_sep['corrected_p_value'] = sm.stats.fdrcorrection(
            gene_sep['p_value'].values, alpha=0.05, method='indep', is_sorted=False
        )[1]
    elif correction == 'bonferroni':
        gene_sep['corrected_p_value'] = sm.stats.multipletests(
            gene_sep['p_value'].values, alpha=0.05, method='bonferroni'
        )[1]
    else:
        raise ValueError("Invalid correction method. Choose 'bh' or 'bonferroni'.")

    return gene_sep.set_index('gene')
