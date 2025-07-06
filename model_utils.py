import pandas as pd
import statsmodels.api as sm
import joblib
import os

try:
    # model_rlm = joblib.load("robust_rlm_model.pkl")
    # Always use absolute path based on file location
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "robust_rlm_model.pkl")
    model_rlm = joblib.load(MODEL_PATH)

except FileNotFoundError:
    model_rlm = None


def extract_inputs_for_prediction(merged_filtered, property_id, target_date=pd.Timestamp('2025-03-31')):
    merged_filtered['MONTH_END'] = pd.to_datetime(merged_filtered['MONTH_END'])
    merged_filtered['OCCUPANCY'] = merged_filtered['OCCUPANCY'].apply(lambda x: 0.01 if x == 0 else x)
    cutoff_date = target_date - pd.DateOffset(years=2)
    df_prop = merged_filtered[merged_filtered['PROPERTY_ID'] == property_id].copy()
    if df_prop.empty:
        raise ValueError(f"PROPERTY_ID {property_id} not found.")
    df_prop['TOTAL_UNITS_IN_SERVICE'] = (
        df_prop['AL_UNITS_IN_SERVICE'] + df_prop['DEM_UNITS_IN_SERVICE'] +
        df_prop['IL_UNITS_IN_SERVICE'] + df_prop['IRF_BEDS_IN_SERVICE'] +
        df_prop['LTACH_BEDS_IN_SERVICE'] + df_prop['MF_UNITS_IN_SERVICE'] +
        df_prop['SNF_BEDS_IN_SERVICE'] + df_prop['SA_UNITS_IN_SERVICE']
    )
    df_prop['REVPOU'] = df_prop['FACILITY_REVENUE_LOCAL_CURRENCY'] / (
        df_prop['TOTAL_UNITS_IN_SERVICE'] * df_prop['OCCUPANCY']
    )
    df_prop['OPEXPOU'] = df_prop['FACILITY_OPEX_LOCAL_CURRENCY'] / (
        df_prop['TOTAL_UNITS_IN_SERVICE'] * df_prop['OCCUPANCY']
    )
    df_prop['EBITDARMPAU'] = (df_prop['REVPOU'] - df_prop['OPEXPOU']) / df_prop['OCCUPANCY']
    capex_positive = df_prop[df_prop['TOTAL_CAPEX'] > 0]
    first_capex_date = capex_positive['MONTH_END'].min() if not capex_positive.empty else pd.NaT
    last_capex_date = capex_positive['MONTH_END'].max() if not capex_positive.empty else pd.NaT
    filtered_cutoff = df_prop[df_prop['MONTH_END'] <= cutoff_date]
    capex_pos_count = filtered_cutoff[filtered_cutoff['TOTAL_CAPEX'] > 0].shape[0]
    total_count_cutoff = filtered_cutoff.shape[0]
    capex_rate = capex_pos_count / total_count_cutoff if total_count_cutoff > 0 else -1
    capex_to_cutoff = df_prop[
        (df_prop['MONTH_END'] <= cutoff_date) & (df_prop['MONTH_END'] >= first_capex_date)
    ]
    if not capex_to_cutoff.empty and df_prop['TOTAL_UNITS_IN_SERVICE'].sum() > 0:
        capex_to_cutoff_sum = capex_to_cutoff['TOTAL_CAPEX'].sum() / df_prop['TOTAL_UNITS_IN_SERVICE'].mean()
    else:
        capex_to_cutoff_sum = -1
    months_since_last_capex = (
        (target_date.year - last_capex_date.year) * 12 + (target_date.month - last_capex_date.month)
    ) if pd.notnull(last_capex_date) else -1
    first_6 = df_prop.sort_values('MONTH_END').head(6)
    revpou_base6m = first_6['REVPOU'].mean() if not first_6.empty else -1
    return {
        "CAPEX_POS_COUNT_2023": capex_pos_count,
        "CAPEX_RATE_2023_imputed": capex_rate,
        "CAPEX_TO_2023_03_imputed": capex_to_cutoff_sum,
        "MONTHS_SINCE_LAST_CAPEX_imputed": months_since_last_capex,
        "REVPOU_base6M": revpou_base6m
    }

def predict_revpou(model, input_dict):
    df_input = pd.DataFrame([input_dict])
    X_input = sm.add_constant(df_input, has_constant='add')
    X_input = X_input[model.model.exog_names]
    prediction = model.predict(X_input)
    return prediction.iloc[0]

def predict_revpou_with_new_capex(model, merged_filtered, property_id, 
                                   investment_amount, num_units, 
                                   capex_date_str, target_date=pd.Timestamp('2025-03-31')):
    capex_date = pd.to_datetime(capex_date_str)
    input_dict = extract_inputs_for_prediction(merged_filtered, property_id, target_date)

    # Step 2: Adjust only if the CAPEX happens before cutoff
    cutoff_date = target_date - pd.DateOffset(years=2)
    capex_date = pd.to_datetime(capex_date_str)
    
    if capex_date <= cutoff_date:
        additional_capex_per_unit = investment_amount / num_units if num_units > 0 else 0
        input_dict["CAPEX_TO_2023_03_imputed"] += additional_capex_per_unit
        input_dict["CAPEX_POS_COUNT_2023"] += 1
        # Recalculate CAPEX_RATE_2023_imputed
        property_df = merged_filtered[
            (merged_filtered['PROPERTY_ID'] == property_id) &
            (merged_filtered['MONTH_END'] <= cutoff_date)
            ]
        total_periods = property_df.shape[0]

        if total_periods > 0:
            input_dict["CAPEX_RATE_2023_imputed"] = input_dict["CAPEX_POS_COUNT_2023"] / total_periods
        else:
            input_dict["CAPEX_RATE_2023_imputed"] = -1
    else:
        # If CAPEX date is outside cutoff, don't include it
        pass


    # additional_capex_per_unit = investment_amount / num_units if num_units > 0 else 0
    # input_dict["CAPEX_TO_2023_03_imputed"] += additional_capex_per_unit
    # input_dict["CAPEX_POS_COUNT_2023"] += 1


    cutoff_date = target_date - pd.DateOffset(years=2)
    property_df = merged_filtered[
        (merged_filtered['PROPERTY_ID'] == property_id) &
        (merged_filtered['MONTH_END'] <= cutoff_date)
    ]
    total_periods = property_df.shape[0]
    input_dict["CAPEX_RATE_2023_imputed"] = input_dict["CAPEX_POS_COUNT_2023"] / total_periods if total_periods > 0 else -1
    return predict_revpou(model, input_dict)
