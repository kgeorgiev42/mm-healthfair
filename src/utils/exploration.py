import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.patches import Patch
import seaborn as sns
from tableone import TableOne
import json
from statannotations.Annotator import Annotator

def get_table_one(ed_pts: pl.DataFrame | pl.LazyFrame,  
                  outcome: str, 
                  output_path: str = '../outputs/reference',
                  disp_dict_path: str = '../outputs/reference/feat_name_map.json',
                sensitive_attr_list: list = ['Sex', 'Ethnicity', 'Marital Status', 'Insurance'],
                nn_attr: list = ['Age', '# historical discharge notes', '# Raw Input Tokens', '# Processed Input Tokens'],
                adjust_method = 'bonferroni',
                cat_cols: list = None,
                verbose: bool = False) -> TableOne:
    """
    Generate baseline patient summary with adjusted p-values grouped by outcome.
    """
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect().to_pandas()
    else:
        ed_pts = ed_pts.to_pandas()
    ### Load dictionary containing display names for features
    with open(disp_dict_path, 'r') as f:
        disp_dict = json.load(f)
    ### Infer categorical data if not specified
    if cat_cols is None:
        cat_cols = [el for el in disp_dict.values() if el not in nn_attr]
    ed_disp = ed_pts.rename(columns=disp_dict)
    ed_disp = ed_disp[disp_dict.values()]
    ### Code categorical columns
    for col in cat_cols:
        if col not in sensitive_attr_list:
            ed_disp[col] = np.where(ed_disp[col]==1, 'Yes', 'No')
    ### Generate Table 1 summary with p-values for target outcome
    if verbose:
        print(f'Generating table summary by {outcome} with prevalence {(ed_disp[outcome].value_counts(normalize=True).iloc[1]*100).round(2)}%')
    tb_one_hd = TableOne(ed_disp,
                     categorical=[col for col in cat_cols if outcome!=col], 
                     nonnormal=nn_attr,
                     groupby=outcome, overall=True, pval=True, htest_name=True, tukey_test=True,
                     decimals=0, pval_adjust=adjust_method)
    tb_one_hd.to_html(output_path + f'/table_one_{outcome}.html')
    print(f'Saved table summary grouped by {outcome} to {output_path + f'/table_one_{outcome}.html'}.')
    return tb_one_hd

def assign_age_groups(ed_pts: pl.DataFrame | pl.LazyFrame, 
                      age_col: str = 'anchor_age', 
                      bins: list = [18, 49, 59, 69, 79, 91], 
                      labels: list = ['<50', '50-59', '60-69', '70-79', '80+'],
                      use_lazy: bool = False) -> pl.DataFrame:
    """
    Assign age groups to patients based on age column.
    """
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect()
    ed_pts = ed_pts.with_column(
        pl.cut(
            ed_pts[age_col], 
            bins=bins, 
            labels=labels
        ).alias('age_group')
    )
    return ed_pts.lazy() if use_lazy else ed_pts

def get_age_table_by_sensitive_attr(ed_pts: pl.DataFrame | pl.LazyFrame, 
                                    attr_name: str, 
                                    outcome: str,
                                    value_name_col: str,
                                    use_lazy: bool = False) -> pl.DataFrame:
    """
    Modifies the dataset into long format to group samples with and without the outcome by age.
    """
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect()
    
    ed_inh_long = ed_pts.filter(pl.col(outcome) == 1).melt(id_vars=[attr_name], value_vars=['age_group'], variable_name='variable', value_name=value_name_col)
    ed_inh_long = ed_inh_long.groupby([attr_name, value_name_col]).agg(pl.count().alias('# Patients'))
    ed_y_cts = ed_inh_long.groupby(attr_name).agg(pl.sum('# Patients').alias('Total'))
    ed_inh_long = ed_inh_long.join(ed_y_cts, on=attr_name)
    ed_inh_long = ed_inh_long.with_column((pl.col('# Patients') / pl.col('Total')).round(4).alias('Percentage'))
    
    return ed_inh_long.lazy() if use_lazy else ed_inh_long

def plot_outcome_dist_by_sensitive_attr(ed_pts: pl.DataFrame | pl.LazyFrame, 
                                        attr_col: str,
                                        attr_xlabel: str, 
                                        output_path: str = '../outputs/reference', 
                                        outcome_list: list = ['in_hosp_death', 'ext_stay_7', 'non_home_discharge','icu_admission'],
                                        outcome_title: list = ['In-hospital death', 'Extended stay (>7 days)', 'Non-home discharge', 'ICU admission'],
                                        outcome_legend: dict = {'In-hospital death': ['No', 'Yes'], 'Extended stay (>7 days)': ['No', 'Yes'],
                                                    'Non-home discharge': ['No', 'Yes'], 'ICU admission': ['No', 'Yes']},
                                        maxi: int=2, maxj: int=2,
                                        rot: int=0, 
                                        figsize: tuple = (8,6), 
                                        palette: list = ['#1f77b4', '#ff7f0e']):
    """
    Plots distribution of health outcomes by specified sensitive attribute.
    """
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect()
    ### Set plot config
    plt.rcParams.update({'font.size': 12, 'font.weight': 'normal', 'font.family': 'serif'})
    fig, ax = plt.subplots(2, 2, figsize=figsize, sharey=True)
    fig.suptitle(f'Health outcome distribution by {attr_xlabel}', fontsize=16)
    fig.supxlabel(attr_xlabel, fontsize=14)
    fig.supylabel('Percentage of patients with ED attendance', fontsize=14)
    outcome_idx = 0
    for i in range(maxi):
        for j in range(maxj):
            ed_gr = ed_pts.groupby([attr_col, outcome_list[outcome_idx]]).agg(pl.count().alias('count')).to_pandas()
            total_counts = ed_gr.groupby(attr_col)['count'].transform('sum')
            ed_gr['percentage'] = ed_gr['count'] / total_counts * 100
            
            # Plot relative percentages
            sns.barplot(data=ed_gr, x=attr_col, y='percentage', hue=outcome_list[outcome_idx], ax=ax[i][j],
                        palette=palette)
            handles = [Patch(color=palette[k], label=outcome_legend[outcome_title[outcome_idx]][k]) for k in range(len(outcome_legend[outcome_title[outcome_idx]]))]
            ax[i][j].legend(handles=handles, title=outcome_title[outcome_idx])
            ax[i][j].set_xlabel('')
            ax[i][j].set_ylabel('')
            ax[i][j].yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x:.0f}%'))
            if j==1:
                ax[i][j].tick_params(left=False)
            plt.sca(ax[i][j])
            plt.xticks(rotation=rot, ha='center')
            outcome_idx+=1

    plt.tight_layout()
    plt.savefig(output_path + f'/outcome_dist_by_{attr_col}.png')
    print(f'Saved plot to {output_path + f'/outcome_dist_by_{attr_col}.png'}.')
    plt.show()

def plot_age_dist_by_sensitive_attr(ed_pts: pl.DataFrame | pl.LazyFrame, 
                                    attr_col: str,
                                    attr_xlabel: str, 
                                    output_path: str = '../outputs/reference', 
                                    outcome_list: list = ['in_hosp_death', 'ext_stay_7', 'non_home_discharge','icu_admission'],
                                    outcome_title: list = ['In-hospital death', 'Extended stay (>7 days)', 'Non-home discharge', 'ICU admission'],
                                    maxi: int=2, maxj: int=2,
                                    colors: list = ['#fef0d9', '#fdcc8a', '#fc8d59', '#e34a33', '#b30000'],
                                    labels: list = ['<50', '50-59', '60-69', '70-79', '80+'],
                                    figsize: tuple = (12,8), 
                                    rot: int=0):
    """
    Plots distribution of age by specified sensitive attribute.
    """
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect()
    ### Set plot config
    plt.rcParams.update({'font.size': 12, 'font.weight': 'normal', 'font.family': 'serif'})
    fig, ax = plt.subplots(2, 2, figsize=figsize, sharey=True)
    fig.suptitle('Age distribution by sensitive variable in patients with adverse event.', fontsize=16)
    fig.supxlabel(attr_xlabel, fontsize=14)
    fig.supylabel('Percentage of patients with ED attendance', fontsize=14)
    outcome_idx = 0
    lg_flag = False
    for i in range(maxi):
        for j in range(maxj):
            lg_flag = True if outcome_idx == 1 else False
            ed_long = get_age_table_by_sensitive_attr(ed_pts, attr_col, outcome_list[outcome_idx], 'age_'+attr_xlabel)
            ed_long_pivot = ed_long.pivot(index=attr_col, columns='age_'+attr_xlabel, values='Percentage')
            ed_long_pivot = ed_long_pivot.to_pandas()
            ed_long_pivot.plot(kind='bar', stacked=True, ax=ax[i][j], legend=lg_flag, color=colors)
            ax[i][j].set_title(outcome_title[outcome_idx])
            if lg_flag:
                ax[i][j].legend(title='Age Group', labels=labels, bbox_to_anchor=(1,1))
            if j==1:
                ax[i][j].tick_params(left=False)
            ax[i][j].set_xlabel('')
            ax[i][j].set_ylabel('')
            ax[0][0].yaxis.set_major_formatter(mtick.PercentFormatter(1))
            plt.sca(ax[i][j])
            plt.xticks(rotation=rot, ha='center')
            outcome_idx+=1
            
    plt.tight_layout()
    plt.savefig(output_path + f'/age_dist_by_{attr_col}.png')
    print(f'Saved plot to {output_path + f'/age_dist_by_{attr_col}.png'}.')
    plt.show()

def plot_token_length_by_attribute(ed_pts: pl.DataFrame | pl.LazyFrame, 
                                  output_path: str = '../outputs/reference',
                                    attr_list: list = ['gender', 'insurance', 'race_group', 'marital_status'],
                                    attr_title: list = ['Gender', 'Insurance type', 'Ethnicity', 'Marital status'],
                                    maxi: int=2, maxj:int=2,
                                    figsize: tuple=(8,6), rot: int=0, ylim: tuple=(0,12),
                                    gr_pairs: dict = {'gender': [('M', 'F')], 
                                              'insurance': [('Medicare', 'Medicaid'), ('Medicare', 'Private'), ('Medicaid', 'Private')],
                                              'race_group': [('White', 'Black'), ('White', 'Hispanic/Latino'), ('White', 'Asian'),
                                                             ('Black', 'Hispanic/Latino'), ('Black', 'Asian'), ('Hispanic/Latino', 'Asian')],
                                              'marital_status': [('Divorced', 'Single'), ('Divorced', 'Married'), ('Divorced', 'Widowed'),
                                                                 ('Single', 'Married'), ('Single', 'Widowed'), ('Married', 'Widowed')],
                                              },
                                    suptitle: str = 'BHC token length by sensitive variable in patients with ED attendance.',
                                    outcome_mode: bool = False,
                                    adjust_method: str = 'bonferroni',
                                    test_type: str = 't-test_welch'):
    """"
    Displays violin plots of aggregated BHC token length by sensitive attribute.
    """
    if isinstance(ed_pts, pl.LazyFrame):
        ed_pts = ed_pts.collect()

    ### log transform target tokens as they are right-skewed
    ed_pts = ed_pts.with_column((pl.col('num_target_tokens').log1p()).alias('num_target_tokens_lg'))
    ## Exclude other categories
    ed_pts = ed_pts.filter(pl.col('race_group') != 'Other')
    ed_pts = ed_pts.filter(pl.col('insurance') != 'Other')
    ### Set plot config
    plt.rcParams.update({'font.size': 12, 'font.weight': 'normal', 'font.family': 'serif'})
    fig, ax = plt.subplots(2, 2, figsize=figsize, sharey=True)
    fig.suptitle(suptitle, fontsize=16)
    fig.supylabel('Log-transformed token length', fontsize=14)
    attr_idx = 0
    for i in range(maxi):
        for j in range(maxj):
            ### If exploring health outcomes, relabel the categories
            if outcome_mode:
                ed_pts = ed_pts.with_column(pl.when(pl.col(attr_list[attr_idx]) == 1).then('Y').otherwise('N').alias(attr_list[attr_idx]))
            ax[i][j] = sns.violinplot(data=ed_pts.to_pandas(), x=attr_list[attr_idx], y='num_target_tokens_lg', ax=ax[i][j])
            ax[i][j].set_xlabel(attr_title[attr_idx])
            ax[i][j].set_ylabel('')
            if j == 1:
                ax[i][j].tick_params(left=False)
            plt.sca(ax[i][j])
            plt.xticks(rotation=rot, ha='center')
            plt.ylim(ylim)
            ### Annotate significant differences
            print(f'Annotating significant differences for {attr_list[attr_idx]}')
            annot = Annotator(ax[i][j], data=ed_pts.to_pandas(), x=attr_list[attr_idx], y='num_target_tokens_lg', pairs=gr_pairs[attr_list[attr_idx]])
            annot.configure(test=test_type, text_format='star', loc='outside', verbose=0, comparisons_correction=adjust_method)
            annot._pvalue_format.pvalue_thresholds = [[0.001, '***'], [0.01, '**'], [0.1, '*'], [1, 'ns']]
            annot.apply_and_annotate()
            attr_idx += 1

    plt.tight_layout(pad=0.5)
    plt.savefig(output_path)
    print(f'Saved plot to {output_path}.')
    plt.show()


