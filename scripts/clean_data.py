import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


NULL_TOKENS = {"", " ", "null", "none", "nan", "n/a", "na", "-", "--"}
LIST_LIKE_COLS = [
    "phone_numbers",
    "websites",
    "specialties",
    "procedure",
    "equipment",
    "capability",
    "affiliation_type_ids",
]
BOOL_LIKE_COLS = ["affiliated_staff_presence", "custom_logo_presence"]
NUMERIC_COLS = [
    "number_doctors",
    "capacity",
    "distinct_social_media_presence_count",
    "number_of_facts_about_the_organization",
    "post_metrics_post_count",
    "engagement_metrics_n_followers",
    "engagement_metrics_n_likes",
    "engagement_metrics_n_engagements",
    "latitude",
    "longitude",
]


def normalize_col_name(col: str) -> str:
    c = col.strip()
    c = re.sub(r"[^a-zA-Z0-9]+", "_", c)
    c = c.strip("_").lower()
    return c


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_col_name(c) for c in df.columns]
    return df


def to_null(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    return np.nan if s.lower() in NULL_TOKENS else s


def clean_phone(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"(?!^\+)[^0-9]", "", s)
    if s and s[0] != "+":
        s = "+" + s
    digits = re.sub(r"\D", "", s)
    if len(digits) < 10:
        return np.nan
    return s


def clean_email(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("\xa0", " ")
    if "[email" in s.lower():
        return np.nan
    if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", s):
        return s.lower()
    return np.nan


def clean_pin(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = re.sub(r"\.0$", "", s)
    s = re.sub(r"\D", "", s)
    if re.fullmatch(r"\d{6}", s):
        return s
    return np.nan


def parse_list_like(x):
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    try:
        v = json.loads(s)
        return v if isinstance(v, list) else [v]
    except Exception:
        return [s]


def clean_bool_like(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return np.nan


def clean_data(input_filepath: str, output_csv: str, output_report: str):
    print(f"Loading: {input_filepath}")
    df_raw = pd.read_csv(input_filepath, dtype=str)
    before_rows = len(df_raw)

    df = normalize_columns(df_raw)

    for c in df.columns:
        df[c] = df[c].map(to_null)

    rename_map = {
        "facilitytypeid": "facility_type_id",
        "operatortypeid": "operator_type_id",
        "affiliationtypeids": "affiliation_type_ids",
        "officialphone": "official_phone",
        "officialwebsite": "official_website",
        "yearestablished": "year_established",
        "numberdoctors": "number_doctors",
        "facebooklink": "facebook_link",
        "twitterlink": "twitter_link",
        "linkedinlink": "linkedin_link",
        "instagramlink": "instagram_link",
        "address_stateorregion": "address_state_or_region",
        "address_ziporpostcode": "address_zip_or_postcode",
        "address_countrycode": "address_country_code",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].str.strip()

    if "facility_type_id" in df.columns:
        df["facility_type_id"] = df["facility_type_id"].str.lower()
        df["facility_type_id"] = df["facility_type_id"].replace({"farmacy": "pharmacy"})

    if "operator_type_id" in df.columns:
        df["operator_type_id"] = df["operator_type_id"].str.lower()

    if "address_country_code" in df.columns:
        df["address_country_code"] = df["address_country_code"].str.upper()

    if "official_phone" in df.columns:
        df["official_phone_clean"] = df["official_phone"].map(clean_phone)

    if "email" in df.columns:
        df["email_clean"] = df["email"].map(clean_email)

    if "address_zip_or_postcode" in df.columns:
        df["pin_code_clean"] = df["address_zip_or_postcode"].map(clean_pin)
        df["pin_invalid"] = df["address_zip_or_postcode"].notna() & df["pin_code_clean"].isna()

    if "address_state_or_region" in df.columns:
        df["address_state_or_region_clean"] = df["address_state_or_region"].str.title()
        df["address_state_or_region_clean"] = df["address_state_or_region_clean"].replace(
            {
                "Orissa": "Odisha",
                "Pondicherry": "Puducherry",
                "Andaman & Nicobar Islands": "Andaman And Nicobar Islands",
                "Jammu & Kashmir": "Jammu And Kashmir",
            }
        )

    parse_error_counts = {}
    for c in LIST_LIKE_COLS:
        if c in df.columns:
            parsed = []
            err = 0
            for v in df[c]:
                try:
                    parsed.append(parse_list_like(v))
                except Exception:
                    parsed.append([])
                    err += 1
            df[c] = [json.dumps(v, ensure_ascii=True) for v in parsed]
            parse_error_counts[c] = err

    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in BOOL_LIKE_COLS:
        if c in df.columns:
            df[c] = df[c].map(clean_bool_like)

    if "latitude" in df.columns and "longitude" in df.columns:
        df["geo_invalid_bbox"] = (
            (df["latitude"] < 6)
            | (df["latitude"] > 38)
            | (df["longitude"] < 68)
            | (df["longitude"] > 98)
        )

    dedup_keys = [k for k in ["name", "address_city", "address_state_or_region_clean"] if k in df.columns]
    if dedup_keys:
        df["is_potential_duplicate"] = df.duplicated(subset=dedup_keys, keep=False)
    else:
        df["is_potential_duplicate"] = False

    before_dedup = len(df)
    df = df.drop_duplicates()
    removed_exact_duplicates = before_dedup - len(df)

    required_cols = [
        c
        for c in [
            "name",
            "facility_type_id",
            "address_city",
            "address_state_or_region_clean",
            "description",
        ]
        if c in df.columns
    ]
    if required_cols:
        df["core_fields_present"] = df[required_cols].notna().sum(axis=1)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    report = {
        "input_rows": int(before_rows),
        "output_rows": int(len(df)),
        "exact_duplicates_removed": int(removed_exact_duplicates),
        "columns": list(df.columns),
        "null_percent_top_15": (
            (df.isna().mean().sort_values(ascending=False).head(15) * 100).round(2).to_dict()
        ),
        "facility_type_counts": (
            df["facility_type_id"].value_counts(dropna=False).to_dict()
            if "facility_type_id" in df.columns
            else {}
        ),
        "pin_invalid_count": int(df["pin_invalid"].sum()) if "pin_invalid" in df.columns else None,
        "invalid_email_count": (
            int(df["email"].notna().sum() - df["email_clean"].notna().sum())
            if "email_clean" in df.columns
            else None
        ),
        "geo_invalid_bbox_count": (
            int(df["geo_invalid_bbox"].sum()) if "geo_invalid_bbox" in df.columns else None
        ),
        "potential_duplicate_count": (
            int(df["is_potential_duplicate"].sum()) if "is_potential_duplicate" in df.columns else None
        ),
        "list_parse_errors": parse_error_counts,
    }

    Path(output_report).parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Done. Clean CSV: {output_csv}")
    print(f"Report: {output_report}")


if __name__ == "__main__":
    raw_data_path = "data/raw/facilities_raw.csv"
    clean_data_path = "data/processed/facilities_clean.csv"
    report_path = "data/processed/cleaning_report.json"

    clean_data(raw_data_path, clean_data_path, report_path)
