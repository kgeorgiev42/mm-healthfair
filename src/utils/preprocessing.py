import numpy as np
import polars as pl
import pandas as pd
import spacy
import json
import os
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from utils.functions import read_icd_mapping, contains_both_ltc_types, rename_fields, get_train_split_summary

###############################
# EHR data preprocessing
###############################

def preproc_icd_module(diagnoses: pl.DataFrame | pl.LazyFrame, 
                       icd_map_path: str = "../config/icd9to10.txt", 
                       map_code_colname: str = "diagnosis_code", only_icd10: bool = True, 
                       ltc_dict_path: str = "../outputs/icd10_codes.json",
                       verbose=True, use_lazy: bool = False) -> pl.DataFrame:
    """Takes an module dataset with ICD codes and puts it in long_format, optionally mapping ICD-codes by a mapping table path.
    Uses custom ICD-10 mapping to generate fields for physical and mental long-term conditions."""

    if isinstance(diagnoses, pl.LazyFrame):
        diagnoses = diagnoses.collect()

    def standardize_icd(mapping, df, root=False):
        """Takes an ICD9 -> ICD10 mapping table and a module dataframe; 
        adds column with converted ICD10 column"""
        
        def icd_9to10(icd):
            # If root is true, only map an ICD 9 -> 10 according to the ICD9's root (first 3 digits)
            if root:
                icd = icd[:3]
            try:
                # Many ICD-9's do not have a 1-to-1 mapping; get first index of mapped codes
                return mapping.filter(pl.col(map_code_colname) == icd).select('icd10cm').to_series()[0]
            except:
                #print("Error on code", icd)
                return np.nan

        # Create new column with original codes as default
        col_name = 'icd10_convert'
        if root: col_name = 'root_' + col_name
        df = df.with_columns(pl.col('icd_code').alias(col_name).cast(pl.Utf8))

        # Convert ICD9 codes to ICD10 in a vectorized manner
        icd9_codes = df.filter(pl.col('icd_version') == 9).select('icd_code').unique().to_series().to_list()
        icd9_to_icd10_map = {code: icd_9to10(code) for code in icd9_codes}
        
        df = df.with_columns(
            pl.when(pl.col('icd_version') == 9)
            .then(pl.col('icd_code').apply(lambda x: icd9_to_icd10_map.get(x, np.nan), return_dtype=pl.Utf8))
            .otherwise(pl.col(col_name))
            .alias(col_name)
        )

        if only_icd10:
            # Column for just the roots of the converted ICD10 column
            df = df.with_columns(pl.col(col_name).apply(lambda x: x[:3] if isinstance(x, str) else np.nan, return_dtype=pl.Utf8).alias('root'))

        return df

    # Optional ICD mapping if argument passed
    if icd_map_path:
        icd_map = read_icd_mapping(icd_map_path)
        diagnoses = standardize_icd(icd_map, diagnoses, root=True)
        diagnoses = diagnoses.filter(pl.col('root_icd10_convert').is_not_null())
        if verbose:
            print("# unique ICD-9 codes", diagnoses.filter(pl.col('icd_version') == 9).select('icd_code').n_unique())
            print("# unique ICD-10 codes", diagnoses.filter(pl.col('icd_version') == 10).select('icd_code').n_unique())
            print("# unique ICD-10 codes (After converting ICD-9 to ICD-10)", diagnoses.select('root_icd10_convert').n_unique())
            print("# unique ICD-10 codes (After clinical grouping ICD-10 codes)", diagnoses.select('root').n_unique())
            print("# Unique patients:  ", diagnoses.select('hadm_id').n_unique())

    diagnoses = diagnoses.select(['subject_id', 'hadm_id', 'seq_num', 'long_title', 'root_icd10_convert'])
    #### Create features for long-term chronic conditions
    if ltc_dict_path:
        with open(ltc_dict_path, 'r') as json_dict:
            ltc_dict = json.load(json_dict)
        ### Initialise long-term condition column
        diagnoses = diagnoses.with_columns(pl.lit('Undefined').alias('ltc_code').cast(pl.Utf8))
        print('Applying LTC coding to diagnoses...')
        for ltc_group, codelist in tqdm(ltc_dict.items()):
            #print("Group:", ltc_group, "Codes:", codelist)
            for code in codelist:
                diagnoses = diagnoses.with_columns(
                    pl.when(pl.col('root_icd10_convert').str.starts_with(code))
                    .then(pl.lit(ltc_group))
                    .otherwise(pl.col('ltc_code'))
                    .alias('ltc_code')
                    .cast(pl.Utf8)
                )

    return diagnoses.lazy() if use_lazy else diagnoses

def get_ltc_features(admits_last: pl.DataFrame | pl.LazyFrame,
                     diagnoses: pl.DataFrame | pl.LazyFrame,
                     ltc_dict_path: str = "../outputs/icd10_codes.json",
                     verbose=True, use_lazy: bool = False) -> pl.DataFrame:
    """Generates features for long-term conditions from a diagnoses table and a dictionary of ICD-10 codes."""
    
    if isinstance(diagnoses, pl.LazyFrame):
        diagnoses = diagnoses.collect()
    if isinstance(diagnoses, pl.LazyFrame):
        admits_last = admits_last.collect()

    ### Comorbidity history
    diag_flat = diagnoses.filter(pl.col('ltc_code') != 'Undefined')
    if verbose:
        print("Number of previous diagnoses recorded in historical ED metadata:", diagnoses.shape[0], diagnoses['subject_id'].n_unique())
    
    ### Create list for each row in ltc_code column
    diag_flat = diag_flat.groupby('subject_id').agg(pl.col('ltc_code').apply(set).alias('ltc_code'))
    
    ### If dict is populated generate categorical columns for each long-term condition
    if ltc_dict_path:
        with open(ltc_dict_path, 'r') as json_dict:
            ltc_dict = json.load(json_dict)
        for ltc_code, _ in ltc_dict.items():
            diag_flat = diag_flat.with_columns(
                pl.col('ltc_code').apply(lambda x: 1 if ltc_code in x else 0).alias(ltc_code)
            )

    ### Create features for multimorbidity
    diag_flat = diag_flat.with_columns([
        pl.col('ltc_code').apply(contains_both_ltc_types, return_dtype=pl.Int8).alias('phys_men_multimorbidity'),
        pl.col('ltc_code').apply(len, return_dtype=pl.Int8).alias('n_unique_conditions'),
        pl.when(pl.col('ltc_code').apply(len, return_dtype=pl.Int8) > 1).then(1).otherwise(0).alias('is_multimorbid'),
        pl.when(pl.col('ltc_code').apply(len, return_dtype=pl.Int8) > 3).then(1).otherwise(0).alias('is_complex_multimorbid')
    ])
    
    ### Merge with base patient data
    admits_last = admits_last.join(diag_flat, on='subject_id', how='left')
    admits_last = admits_last.with_columns([
        pl.col(col).cast(pl.Int8).fill_null(0) for col in diag_flat.drop(['subject_id', 'ltc_code']).columns
    ])
    
    return admits_last.lazy() if use_lazy else admits_last

def transform_sensitive_attributes(ed_pts: pl.DataFrame) -> pl.DataFrame:
    """Maps any sensitive attributes to predefined categories and data types.

    Args:
        ed_pts (pl.DataFrame): ED attendance patients.

    Returns:
        pl.DataFrame: Updated data.
    """

    ed_pts = ed_pts.with_columns([
        pl.col('anchor_age').cast(pl.Int16),
        pl.when(pl.col('race').str.to_lowercase().str.contains('white|middle eastern|portuguese'))
          .then(pl.lit('White'))
          .when(pl.col('race').str.to_lowercase().str.contains('black|caribbean island'))
          .then(pl.lit('Black'))
          .when(pl.col('race').str.to_lowercase().str.contains('hispanic|south american'))
          .then(pl.lit('Hispanic/Latino'))
          .when(pl.col('race').str.to_lowercase().str.contains('asian'))
          .then(pl.lit('Asian'))
          .otherwise(pl.lit('Other'))
          .alias('race_group'),
        pl.col('marital_status').str.to_lowercase().str.to_titlecase()
    ])

    return ed_pts

def prepare_medication_features(medications: pl.DataFrame | pl.LazyFrame,
                                admits_last: pl.DataFrame | pl.LazyFrame,
                                top_n: int = 50,
                                use_lazy: bool = False) -> pl.DataFrame:
    """Generates count features for drug-level medication history."""
    if isinstance(medications, pl.LazyFrame):
        medications = medications.collect()
    if isinstance(admits_last, pl.LazyFrame):
        admits_last = admits_last.collect()

    ### Convert to pandas for easier manipulation
    medications = medications.to_pandas()
    admits_last = admits_last.to_pandas()
    
    medications['charttime'] = pd.to_datetime(medications['charttime'])
    medications['edregtime'] = pd.to_datetime(medications['edregtime'])
    medications = medications[(medications['charttime']<medications['edregtime'])]

    ### Clean and prepare medication text
    medications['medication'] = medications['medication'].str.lower().str.strip().str.replace(' ', '_').str.replace('-', '_')
    
    ### Get top_n (most commonly found) medications
    top_meds = medications['medication'].value_counts().head(top_n).index.tolist()
    
    #### Filter most common medications
    medications = medications[medications['medication'].isin(top_meds)]
    
    ### Clean some of the top medication fields
    medications['medication'] = np.where(medications['medication'].str.contains('vancomycin'), 'vancomycin', medications['medication'])
    medications['medication'] = np.where(medications['medication'].str.contains('acetaminophen'), 'acetaminophen', medications['medication'])
    medications['medication'] = np.where(medications['medication'].str.contains('albuterol_0.083%_neb_soln'), 'albuterol_neb_soln', medications['medication'])
    medications['medication'] = np.where(medications['medication'].str.contains('oxycodone_(immediate_release)', regex=False), 'oxycodone', medications['medication'])
    ### Get days since first and last medication
    medications = medications.sort_values(['subject_id', 'medication', 'charttime'])
    meds_min = medications.drop_duplicates(subset=['subject_id', 'medication'], keep='first')
    meds_max = medications.drop_duplicates(subset=['subject_id', 'medication'], keep='last')
    meds_min = meds_min.rename(columns={'charttime': 'first_date'})
    meds_max = meds_max.rename(columns={'charttime': 'last_date'})
    
    meds_min['dsf'] = (medications['edregtime'] - meds_min['first_date']).dt.days
    meds_max['dsl'] = (medications['edregtime'] - meds_max['last_date']).dt.days
    
    ### Get number of prescriptions
    meds_ids = medications.groupby(['subject_id', 'medication', 'edregtime']).size().reset_index(name='n_presc')
    
    meds_ids = meds_ids.merge(meds_min[['subject_id', 'medication', 'dsf']], on=['subject_id', 'medication'], how='left')
    meds_ids = meds_ids.merge(meds_max[['subject_id', 'medication', 'dsl']], on=['subject_id', 'medication'], how='left')

    #### Pivot table and create drug-specific features
    meds_piv = meds_ids.pivot_table(index='subject_id', columns='medication', values=['n_presc', 'dsf', 'dsl'], fill_value=0)
    meds_piv.columns = [rename_fields('_'.join(col).strip()) for col in meds_piv.columns.values]
    
    meds_piv_total = meds_ids.groupby('subject_id')['medication'].nunique().reset_index(name='total_n_presc')
    
    admits_last = admits_last.merge(meds_piv_total, on='subject_id', how='left')
    admits_last = admits_last.merge(meds_piv, on='subject_id', how='left')
    
    ### Fill missing values
    days_cols = [col for col in admits_last.columns if 'dsf' in col or 'dsl' in col]
    admits_last[days_cols] = admits_last[days_cols].fillna(9999).astype(np.int32)
    
    nums_cols = [col for col in admits_last.columns if 'n_presc' in col]
    admits_last[nums_cols] = admits_last[nums_cols].fillna(0).astype(np.int16)
    
    admits_last['total_n_presc'] = admits_last['total_n_presc'].fillna(0).astype(np.int8)
    admits_last.columns = admits_last.columns.str.replace('(', '').str.replace(')', '')

    admits_last = pl.DataFrame(admits_last)

    return admits_last.lazy() if use_lazy else admits_last
    
def encode_categorical_features(ehr_data: pl.DataFrame) -> pl.DataFrame:
    """Applies one-hot encoding to categorical features.

    Args:
        ehr_data (pl.DataFrame): Static EHR dataset.

    Returns:
        pl.DataFrame: Transformed EHR data.
    """

    # prepare attribute features for one-hot-encoding
    ehr_data = ehr_data.with_columns([
        pl.when(pl.col('race_group') == 'Hispanic/Latino').then(pl.lit('Hispanic_Latino')).otherwise(pl.col('race_group')).alias('race_group'),
        pl.when(pl.col('gender') == 'F').then(1).otherwise(0).cast(pl.Int8).alias('gender_F')
    ])
    ehr_data = ehr_data.to_dummies(columns=["race_group", "marital_status", "insurance"])
    ehr_data = ehr_data.drop(['race', 'gender'])
    return ehr_data

def extract_lookup_fields(ehr_data: pl.DataFrame,
                          lookup_list: list = ['hadm_id', 'yob', 'dod', 'admittime',
                                               'dischtime', 'deathtime', 'edregtime',
                                               'intime', 'outtime',
                                               'edouttime', 'admission_location',
                                               'discharge_location', 'los_days',
                                               'icu_los_days', 'ltc_code',
                                               'num_summaries', 'num_input_tokens',
                                               'num_target_tokens', 'num_measures',
                                               'total_proc_count'],
                          lookup_output_path: str = '../outputs/reference') -> pl.DataFrame:
    """Extract dates and summary fields not suitable for training in a separate dataframe.

    Args:
        ehr_data (pl.DataFrame): Static EHR dataset.

    Returns:
        pl.DataFrame: Transformed EHR data.
    """
    ehr_lookup = ehr_data.select(['subject_id'] + lookup_list)
    ehr_data = ehr_data.drop(lookup_list)
    print(f'Saving lookup fields in EHR data to {lookup_output_path}')
    ehr_lookup.write_csv(os.path.join(lookup_output_path, 'ehr_lookup.csv'))
    return ehr_data

def remove_correlated_features(ehr_data: pl.DataFrame,
                          feats_to_save: list = ['anchor_age', 'gender_F', 
                                                 'race_group_Hispanic_Latino', 'race_group_Black', 'race_group_White',
                                                 'race_group_Asian', 'race_group_Other',
                                                 'marital_status_Married', 'marital_status_Single', 'marital_status_Widowed', 'marital_status_Divorced',
                                                 'insurance_Medicare', 'insurance_Medicaid', 'insurance_Private', 'insurance_Other',
                                                 'in_hosp_death', 'ext_stay_7', 'icu_admission', 'non_home_discharge'],
                          threshold: float = 0.9,
                          method: str = 'pearson',
                          verbose: bool = True) -> pl.DataFrame:
    """Drop highly correlated features from static EHR dataset, while specifying features to explicitly save for training.

    Args:
        ehr_data (pl.DataFrame): Static EHR dataset.

    Returns:
        pl.DataFrame: Transformed EHR data.
    """
    ### Specify features to save
    ehr_save = ehr_data.select(['subject_id'] + feats_to_save)
    ehr_data = ehr_data.drop(['subject_id'] + feats_to_save)
    ### Generate a linear correlation matrix
    corr_matrix = ehr_data.to_pandas().corr(method=method)
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []
    
    for i in tqdm(iters, desc='Dropping highly correlated features...'):
        for j in range(i + 1):
            item = corr_matrix.iloc[j:(j + 1), (i + 1):(i + 2)]
            colname = item.columns
            row = item.index
            val = abs(item.values)
            if val >= threshold:
                #if verbose:
                    #print(f'Detected correlation: {colname.values[0]} || {row.values[0]} || {round(val[0][0], 2)}')
                drop_cols.append(colname.values[0])
    
    to_drop = list(set(drop_cols))
    ehr_data = ehr_data.drop(to_drop)
    ehr_data = ehr_save.select(['subject_id']).hstack(ehr_data)
    ehr_data = ehr_data.join(ehr_save, on='subject_id', how='left')
    
    if verbose:
        print(f'Dropped {len(to_drop)} highly correlated features.')
        print('-------------------------------------')
        print('Full list of dropped features:', to_drop)
        print('-------------------------------------')
        print(f'Final number of EHR features: {ehr_data.shape[1]}/{len(to_drop)+ehr_data.shape[1]}')
    
    return ehr_data

def generate_train_val_test_set(ehr_data: pl.DataFrame,
                                output_path: str = '../outputs/processed_data',
                                outcome_col: str = 'in_hosp_death',
                                output_summary_path: str = '../outputs/exp_data',
                                seed: int = 0,
                                train_ratio: float = 0.8,
                                val_ratio: float = 0.1,
                                test_ratio: float = 0.1,
                                cont_cols: list=['Age'],
                                nn_cols: list=['Age'],
                                disp_dict: dict={
                                    'anchor_age': 'Age',
                                    'gender': 'Gender',
                                    'race_group': 'Ethnicity',
                                    'insurance': 'Insurance',
                                    'marital_status': 'Marital status',
                                    'in_hosp_death': 'In-hospital death',
                                    'ext_stay_7': 'Extended stay',
                                    'non_home_discharge': 'Non-home discharge',
                                    'icu_admission': 'ICU admission',
                                    'is_multimorbid': 'Multimorbidity',
                                    'is_complex_multimorbid': 'Complex multimorbidity'
                                },
                                stratify: bool = True,
                                verbose: bool = True) -> dict:
    """Create train/val/test split from static EHR dataset and save the patient IDs in separate files.

    Args:
        ehr_data (pl.DataFrame): Static EHR dataset.

    Returns:
        pl.DataFrame: Transformed EHR data.
    """
    ### Set stratification columns to include sensitive attributes + target outcome
    ehr_data = ehr_data.to_pandas()
    if stratify:
        strat_target = pd.concat([ehr_data[outcome_col], 
                                  ehr_data['gender'], 
                                  ehr_data['race_group']], axis=1)
        split_target = ehr_data.drop([outcome_col, 'gender', 'race_group'], axis=1)
        ### Generate split dataframes
        train_x, test_x, train_y, test_y = train_test_split(split_target, strat_target, 
                                                            test_size=(1 - train_ratio), 
                                                            random_state=seed, 
                                                            stratify=strat_target)
        val_x, test_x, val_y, test_y = train_test_split(test_x, test_y,
                                                        test_size=test_ratio/(test_ratio + val_ratio),
                                                        random_state=seed,
                                                        stratify=test_y)
    else:
        train_x, test_x, train_y, test_y = train_test_split(ehr_data.drop([outcome_col], axis=1), ehr_data[outcome_col], 
                                                            test_size=(1 - train_ratio), 
                                                            random_state=seed)
        val_x, test_x, val_y, test_y = train_test_split(test_x.drop([outcome_col], axis=1), 
                                                        test_x[outcome_col],
                                                        test_size=test_ratio/(test_ratio + val_ratio),
                                                        random_state=seed)
    train_x = pd.concat([train_x, train_y], axis=1)
    val_x = pd.concat([val_x, val_y], axis=1)
    test_x = pd.concat([test_x, test_y], axis=1)
    train_x['set'] = 'train'
    val_x['set'] = 'val'
    test_x['set'] = 'test'
    ### Print summary statistics
    if verbose:
        print(f'Created split with {train_x.shape[0]}({round(train_x.shape[0]/len(ehr_data), 2)*100}%) samples in train, {val_x.shape[0]}({round(val_x.shape[0]/len(ehr_data), 2)*100}%) samples in validation, and {test_x.shape[0]}({round(test_x.shape[0]/len(ehr_data), 2)*100}%) samples in test.')
        print('Getting summary statistics for split...')
        get_train_split_summary(train_x, val_x, test_x, outcome_col, output_summary_path, 
                                cont_cols, nn_cols, disp_dict, verbose=verbose)
        print(f'Saving train/val/test split IDs to {output_path}')
        
    ### Save patient IDs
    train_x[['subject_id']].to_csv(os.path.join(output_path, 'training_ids.csv'), index=False)
    val_x[['subject_id']].to_csv(os.path.join(output_path, 'validation_ids.csv'), index=False)
    test_x[['subject_id']].to_csv(os.path.join(output_path, 'testing_ids.csv'), index=False)

    return {'train': train_x, 'val': val_x, 'test': test_x}
    
###############################
# Notes preprocessing
###############################


def clean_notes(notes: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    """Cleans notes data by removing any relevant special characters and extra whitespaces."""
    # Remove __
    notes = notes.with_columns(
        target=pl.col("target").str.replace_all(r"\s___\s", " ")
    )
    # Remove any extra whitespaces
    notes = notes.with_columns(
        target=pl.col("target").str.replace_all(r"\s+", " ")
    )
    return notes


def process_text_to_embeddings(notes: pl.DataFrame) -> dict:
    """Generates dictionary containing embeddings from Bio+Discharge ClinicalBERT (mean vector).
    https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT

    Args:
        notes (pl.DataFrame): Dataframe containing notes data.
        use_gpu (bool): Whether to use GPU for inference. Defaults to False.

    Returns:
        dict: Dictionary containing subject_id as keys and average word embeddings as values.
    """
    embeddings_dict = {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    nlp = spacy.load("en_core_sci_md", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")
    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_Discharge_Summary_BERT"
    )
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT").to(device)

    for row in tqdm(
        notes.iter_rows(named=True),
        desc="Generating notes embeddings with ClinicalBERT...",
        total=notes.height,
    ):
        subj_id = row["subject_id"]
        text = row["target"]

        # Turn text into sentences
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        # Tokenize all sentences at once
        inputs = tokenizer(
            sentences,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
            return_attention_mask=False,
        ).to(device)

        # Generate embeddings for all sentences in a single forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            sentence_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

        if sentence_embeddings.size > 0:
            embeddings = np.mean(sentence_embeddings, axis=0)
        else:
            embeddings = np.zeros((768,))  # Handle case with no sentences

        embeddings_dict[subj_id] = embeddings

    return embeddings_dict


###############################
# Time-series preprocessing
###############################


def clean_labevents(labs_data: pl.LazyFrame) -> pl.LazyFrame:
    """Maps non-integer values to None and removes outliers.

    Args:
        events (pl.DataFrame): Events table.

    Returns:
        pl.DataFrame: Cleaned events table.
    """
    labs_data = labs_data.with_columns(
        pl.col("label").str.to_lowercase().str.replace(" ", "_").str.replace(",", "").str.replace('"', "").str.replace(" ", "_"),
        pl.col("charttime").cast(pl.Utf8).str.replace("T", " ").str.strip_chars()
    )
    lab_events = labs_data.with_columns(
            value=pl.when(pl.col("value") == ".").then(None).otherwise(pl.col("value"))
    )
    lab_events = lab_events.with_columns(
        value=pl.when(pl.col("value").str.contains("_|<|ERROR"))
        .then(None)
        .otherwise(pl.col("value"))
        .cast(pl.Float64, strict=False)  # Attempt to cast to Float64, set invalid values to None
    )
    labs_data = labs_data.drop_nulls()

    # Remove outliers using 2 std from mean
    lab_events = lab_events.with_columns(mean=pl.col("value").mean().over(pl.count("label")))
    lab_events = lab_events.with_columns(std=pl.col("value").std().over(pl.count("label")))
    lab_events = lab_events.filter(
        (pl.col("value") < pl.col("mean") + pl.col("std") * 2)
        & (pl.col("value") > pl.col("mean") - pl.col("std") * 2)
    ).drop(["mean", "std"])

    return lab_events


def add_time_elapsed_to_events(
    events: pl.DataFrame, starttime: pl.Datetime, remove_charttime: bool = False
) -> pl.DataFrame:
    """Adds column 'elapsed' which considers time elapsed since starttime.

    Args:
        events (pl.DataFrame): Events table.
        starttime (pl.Datetime): Reference start time.
        remove_charttime (bool, optional): Whether to remove charttime column. Defaults to False.

    Returns:
        pl.DataFrame: Updated events table.
    """
    events = events.with_columns(
        elapsed=((pl.col("charttime") - starttime) / pl.duration(hours=1)).round(1)
    )

    # reorder columns
    if remove_charttime:
        events = events.drop("charttime")

    return events


def convert_events_to_timeseries(events: pl.DataFrame) -> pl.DataFrame:
    """Converts long-form events to wide-form time-series.

    Args:
        events (pl.DataFrame): Long-form events.

    Returns:
        pl.DataFrame: Wide-form time-series of shape (timestamp, features)
    """

    metadata = (
        events.select(["charttime", "label", "value", "linksto"])
        .sort(by=["charttime", "label", "value"])
        .unique(subset=["charttime"], keep="last")
        .sort(by="charttime")
    )

    # get unique label, values and charttimes
    timeseries = (
        events.select(["charttime", "label", "value"])
        .sort(by=["charttime", "label", "value"])
        .unique(subset=["charttime", "label"], keep="last")
    )

    # pivot into wide-form format
    timeseries = timeseries.pivot(
        index="charttime", columns="label", values="value"
    ).sort(by="charttime")

    # join any metadata remaining
    timeseries = timeseries.join(
        metadata.select(["charttime", "linksto"]), on="charttime", how="inner"
    )
    return timeseries

def generate_interval_dataset(ehr_static: pl.DataFrame, ts_data: pl.DataFrame,
                              ehr_regtime: pl.DataFrame,
                              vitals_freq: str = "5h", lab_freq: str = "1h",
                              min_events: int = None, 
                              max_events: int = None,
                              impute: str='value', 
                              include_dyn_mean: bool = False, 
                              no_resample: bool = False,
                              max_elapsed: int = None,
                              verbose: bool = True) -> dict:
    """Generates a multimodal dataset with set intervals for each event source."""
    data_dict = {}
    n = 0
    filter_by_nb_events = 0
    missing_event_src = 0
    filter_by_elapsed_time = 0
    n_src = ts_data.n_unique("linksto")
    
    feature_map = {}
    freq = {}
    print("Getting lookup intervals for each event source..")
    for src in tqdm(ts_data.unique("linksto").get_column("linksto").to_list()):
        feature_map[src] = sorted(
            ts_data.filter(pl.col("linksto") == src)
            .unique("label")
            .get_column("label")
            .to_list()
        )
        freq[src] = vitals_freq if src == "vitalsign" else lab_freq

    min_events = 1 if min_events is None else int(min_events)
    max_events = 1e6 if max_events is None else int(max_events)
    print("Imputing event intervals per patient..")
    ts_data = ts_data.sort(by=["subject_id", "charttime"])
    ehr_regtime = ehr_regtime.sort(by=["subject_id", "edregtime"])
    for id_val in tqdm(ts_data.unique("subject_id").get_column("subject_id").to_list(),
            desc="Generating patient-level data...",
    ):
        pt_events = ts_data.filter(pl.col("subject_id") == id_val)
        edregtime = ehr_regtime.filter(pl.col("subject_id") == id_val).select("edregtime").head(1).item()
        if pt_events.n_unique("linksto") < n_src:
            missing_event_src += 1
            continue

        write_data = True
        ts_data_list = []
        for events_by_src in pt_events.partition_by("linksto"):
            src = events_by_src.select(pl.first("linksto")).item()
            timeseries = convert_events_to_timeseries(events_by_src)
            if (timeseries.shape[0] < min_events) | (timeseries.shape[0] > max_events):
                filter_by_nb_events += 1
                write_data = False
                break
            features = feature_map[src]
            missing_cols = [x for x in features if x not in timeseries.columns]
            timeseries = timeseries.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(c) for c in missing_cols]
            )

            if impute is not None:
                if impute == "mask":
                    timeseries = timeseries.with_columns(
                        [pl.col(f).is_null().alias(f + "_isna") for f in features]
                    )
                    ehr_static = ehr_static.with_columns(
                        [
                            pl.col(f).is_null().alias(f + "_isna")
                            for f in ehr_static.columns
                        ]
                    )
                elif impute in ["forward", "backward"]:
                    timeseries = timeseries.fill_null(strategy=impute)
                    timeseries = timeseries.fill_null(value=-1)
                    ehr_static = ehr_static.fill_null(value=-1)
                elif impute == "value":
                    timeseries = timeseries.fill_null(value=-1)
                    ehr_static = ehr_static.fill_null(value=-1)
                else:
                    raise ValueError(
                        "impute_strategy must be one of [None, mask, value, forward, backward]"
                    )

            if include_dyn_mean:
                timeseries_mean = timeseries.drop(["charttime", "linksto"]).mean()
                timeseries_mean = timeseries_mean.with_columns(pl.all().round(3))
                ehr_static = ehr_static.hstack(timeseries_mean)

            if not no_resample:
                timeseries = timeseries.upsample(time_column="charttime", every="1m")
                timeseries = timeseries.group_by_dynamic(
                    "charttime",
                    every=freq[src],
                ).agg(pl.col(pl.Float64).mean())
                timeseries = timeseries.fill_null(strategy="forward")

            if max_elapsed is not None:
                timeseries = add_time_elapsed_to_events(timeseries, edregtime)
                timeseries = timeseries.filter(pl.col("elapsed") <= max_elapsed)
            else:
                timeseries = add_time_elapsed_to_events(timeseries, edregtime)

            if timeseries.shape[0] == 0:
                filter_by_elapsed_time += 1
                write_data = False
                break
            timeseries = timeseries.select(features)
            ts_data_list.append(timeseries)

        if write_data:
            data_dict[id_val] = {"static": ehr_static}
            for idx, ts in enumerate(ts_data_list):
                data_dict[id_val][f"dynamic_{idx}"] = ts
            n += 1

    if verbose:
        print(f"Successfully processed time-series intervals for {n} patients.")
        print(f"Skipping {filter_by_nb_events} patients with less or greater number of events than specified.")
        print(f"Skipping {missing_event_src} patients due to at least one missing time-series source.")
        print(f"Skipping {filter_by_elapsed_time} patients due to no measures within elapsed time.")
        
    return data_dict
