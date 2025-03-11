import numpy as np
import polars as pl
import pandas as pd
import spacy
import json
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from utils.functions import read_icd_mapping, contains_both_ltc_types, rename_fields

###############################
# Static data preprocessing
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


def encode_categorical_features(stays: pl.DataFrame) -> pl.DataFrame:
    """Groups and applied one-hot encoding to categorical features.

    Args:
        stays (pl.DataFrame): Stays data.

    Returns:
        pl.DataFrame: Transformed stays data.
    """
    if "gender" in stays.columns:
        stays = transform_gender(stays)
    if "race" in stays.columns:
        stays = transform_race(stays)
    if "marital_status" in stays.columns:
        stays = transform_marital(stays)
    if "insurance" in stays.columns:
        stays = transform_insurance(stays)

    # apply one-hot encoding to integer columns
    stays = stays.to_dummies(
        [
            i
            for i in stays.columns
            if i in ["gender", "race", "marital_status", "insurance"]
        ]
    )

    return stays

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
    medications['medication'] = np.where(medications['medication'].str.contains('n_presc_oxycodone_(immediate_release)', regex=False), 'n_presc_oxycodone', medications['medication'])
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
    admits_last = pl.DataFrame(admits_last)

    return admits_last.lazy() if use_lazy else admits_last
    


###############################
# Notes preprocessing
###############################


def clean_notes(notes: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
    # Remove __
    notes = notes.with_columns(
        subtext=pl.col("subtext").str.replace_all(r"\s___\s", " ")
    )

    return notes


def process_text_to_embeddings(notes: pl.DataFrame) -> dict:
    """Generates dictionary containing embeddings from Bio+Discharge ClinicalBERT (mean vector).
    https://huggingface.co/emilyalsentzer/Bio_Discharge_Summary_BERT

    Args:
        notes (pl.DataFrame): Dataframe containing notes data.

    Returns:
        dict: Dictionary containing hadm_id as keys and average wode embeddings as values.
    """
    embeddings_dict = {}

    nlp = spacy.load("en_core_sci_md")
    tokenizer = AutoTokenizer.from_pretrained(
        "emilyalsentzer/Bio_Discharge_Summary_BERT"
    )
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_Discharge_Summary_BERT")

    for row in tqdm(
        notes.iter_rows(named=True),
        desc="Generating notes embeddings with ClinicalBERT...",
        total=notes.height,
    ):
        hadm_id = row["hadm_id"]
        text = row["subtext"]

        # Turn text into sentences
        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        # Generate embeddings for each sentence
        sentence_embeddings = []
        for sentence in sentences:
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128,
            )
            outputs = model(**inputs)
            sentence_embeddings.append(
                outputs.last_hidden_state.mean(dim=1).detach().numpy()
            )

        if sentence_embeddings:
            embeddings = np.mean(sentence_embeddings, axis=0)
        else:
            embeddings = np.zeros((1, 768))  # Handle case with no sentences

        embeddings_dict[hadm_id] = embeddings

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
