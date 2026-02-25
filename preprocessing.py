def _is_nan_like(value):
    try:
        return value != value
    except Exception:
        return False


def _normalize_token(value):
    if value is None or _is_nan_like(value):
        return value

    return (
        str(value)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
    )


def _normalize_categorical(value, mapping):
    if value is None or _is_nan_like(value):
        return value

    token = _normalize_token(value)
    return mapping.get(token, str(value).strip())


def _normalize_boolean_label(value):
    return _normalize_categorical(
        value,
        {
            "0": "False",
            "0_0": "False",
            "false": "False",
            "no": "False",
            "n": "False",
            "1": "True",
            "1_0": "True",
            "true": "True",
            "yes": "True",
            "y": "True",
        },
    )


def _normalize_binary_label(value):
    return _normalize_categorical(
        value,
        {
            "0": 0,
            "0_0": 0,
            "false": 0,
            "no": 0,
            "n": 0,
            "1": 1,
            "1_0": 1,
            "true": 1,
            "yes": 1,
            "y": 1,
        },
    )


def preprocessing(df):
    try:
        df.drop(columns = ["Medication", "año_reco", "fpi", "Unnamed: 0"], inplace = True)
    except:
        df = df

    try:
        df.drop(columns = ["prov_trab"], inplace = True)
    except:
        df = df
    
    try:
        df = df[(df["Age"] != "1_very_young") & (df["Age"] != "6_elderly")].copy()
    except:
        df = df

    try:
        mappings = {
            "PA": {
                "1": "PA_1",
                "1_0": "PA_1",
                "pa1": "PA_1",
                "pa_1": "PA_1",
                "2": "PA_2",
                "2_0": "PA_2",
                "pa2": "PA_2",
                "pa_2": "PA_2",
            },
            "SD": {
                "short": "SD_1_short",
                "sd_1": "SD_1_short",
                "sd_1_short": "SD_1_short",
                "normal": "SD_2_normal",
                "medium": "SD_2_normal",
                "sd_2": "SD_2_normal",
                "sd_2_normal": "SD_2_normal",
                "excessive": "SD_3_excessive",
                "long": "SD_3_excessive",
                "high": "SD_3_excessive",
                "sd_3": "SD_3_excessive",
                "sd_3_excessive": "SD_3_excessive",
            },
            "Age": {
                "2": "age_2_young",
                "2_0": "age_2_young",
                "2_young": "age_2_young",
                "age_2": "age_2_young",
                "young": "age_2_young",
                "3": "age_3_young_adult",
                "3_0": "age_3_young_adult",
                "3_young_adult": "age_3_young_adult",
                "age_3": "age_3_young_adult",
                "young_adult": "age_3_young_adult",
                "4": "age_4_adult",
                "4_0": "age_4_adult",
                "4_adult": "age_4_adult",
                "age_4": "age_4_adult",
                "adult": "age_4_adult",
                "5": "age_5_old_adult",
                "5_0": "age_5_old_adult",
                "5_old_adult": "age_5_old_adult",
                "age_5": "age_5_old_adult",
                "old_adult": "age_5_old_adult",
            },
            "Smoking": {
                "1": "sm_1_not_smoker",
                "1_0": "sm_1_not_smoker",
                "1_not_smoker": "sm_1_not_smoker",
                "sm_1": "sm_1_not_smoker",
                "not_smoker": "sm_1_not_smoker",
                "non_smoker": "sm_1_not_smoker",
                "2": "sm_2_smoker",
                "2_0": "sm_2_smoker",
                "2_smoker": "sm_2_smoker",
                "sm_2": "sm_2_smoker",
                "smoker": "sm_2_smoker",
                "3": "sm_3_ex_smoker",
                "3_0": "sm_3_ex_smoker",
                "3_ex_smoker": "sm_3_ex_smoker",
                "sm_3": "sm_3_ex_smoker",
                "ex_smoker": "sm_3_ex_smoker",
                "former_smoker": "sm_3_ex_smoker",
            },
            "BMI": {
                "1": "bmi_1_underweight",
                "1_0": "bmi_1_underweight",
                "1_underweight": "bmi_1_underweight",
                "bmi_1": "bmi_1_underweight",
                "underweight": "bmi_1_underweight",
                "2": "bmi_2_normal",
                "2_0": "bmi_2_normal",
                "2_normal": "bmi_2_normal",
                "bmi_2": "bmi_2_normal",
                "normal": "bmi_2_normal",
                "3": "bmi_3_overweight",
                "3_0": "bmi_3_overweight",
                "3_overweight": "bmi_3_overweight",
                "bmi_3": "bmi_3_overweight",
                "overweight": "bmi_3_overweight",
                "4": "bmi_4_obese",
                "4_0": "bmi_4_obese",
                "4_obese": "bmi_4_obese",
                "bmi_4": "bmi_4_obese",
                "obese": "bmi_4_obese",
            },
            "Alcohol": {
                "high": "high",
                "low": "low",
            },
            "SES": {
                "0": "ses_0",
                "0_0": "ses_0",
                "ses0": "ses_0",
                "ses_0": "ses_0",
                "1": "ses_1",
                "1_0": "ses_1",
                "ses1": "ses_1",
                "ses_1": "ses_1",
                "2": "ses_2",
                "2_0": "ses_2",
                "ses2": "ses_2",
                "ses_2": "ses_2",
            },
            "Sex": {
                "m": "M",
                "male": "M",
                "w": "W",
                "f": "W",
                "female": "W",
                "woman": "W",
            },
        }

        for col, mapping in mappings.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: _normalize_categorical(x, mapping))

        bool_columns = ["Depression", "Anxiety", "Hyperchol_", "Hyperchol.", "Diabetes", "Hypertension"]
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].apply(_normalize_boolean_label)

        if "CRC" in df.columns:
            df["CRC"] = df["CRC"].apply(_normalize_binary_label)
    except:
        df = df

    return df
