-- stg_data.sql
-- Casts the seed table (all VARCHAR) to proper data types and renames columns for clarity.

SELECT
    loan_id::UUID                         AS loan_id,
    user_id::INTEGER                      AS user_id,
    target::INTEGER                       AS target,
    global_cardinal::INTEGER              AS global_cardinal,
    cardinal_group                        AS cardinal_group,
    product                               AS product,
    principal_due::NUMERIC                AS principal_due,
    max_amount::NUMERIC                   AS credit_limit,        -- rename for consistency
    util_ratio::NUMERIC                   AS limit_util_ratio,
    tenure::INTEGER                       AS tenure,
    approval_date::DATE                   AS approval_date,
    due_date::DATE                        AS due_date,
    age::INTEGER                          AS age,
    state                                 AS state,
    emp_status                            AS emp_status,
    marital_status                        AS marital_status,
    edu_status                            AS edu_status,
    monthly_income::NUMERIC               AS salary,
    gender                                AS gender,
    acquisition_channel                   AS acquisition_channel,

    -- Bureau availability flag (already clean)
    bureau_availability                   AS bureau_availability,

    -- Bureau summary
    total_bureau_accounts::INTEGER        AS total_bureau_accounts,
    accounts_in_arrears::INTEGER          AS accounts_in_arrears,
    accounts_bad_condition::INTEGER       AS accounts_bad_condition,
    accounts_good_condition::INTEGER      AS accounts_good_condition,
    amount_in_arrears::NUMERIC            AS amount_in_arrears,
    total_outstanding_debt::NUMERIC       AS total_outstanding_debt,
    total_monthly_obligations::NUMERIC    AS total_monthly_obligations,
    bureau_account_rating::INTEGER        AS bureau_account_rating,
    is_thin_file::INTEGER                 AS is_thin_file,
    bureau_risk_category                  AS bureau_risk_category,
    prop_bad::NUMERIC                     AS prop_bad,

    -- Bureau agreements
    has_written_off_account::INTEGER      AS has_written_off_account,
    count_written_off_accounts::INTEGER   AS count_written_off_accounts,
    has_doubtful_account::INTEGER         AS has_doubtful_account,
    distinct_lender_count::INTEGER        AS distinct_lender_count,
    credit_history_months::INTEGER        AS credit_history_months,

    -- Bureau payment history
    max_dpd_ever_bureau::INTEGER          AS max_dpd_ever_bureau,
    max_dpd_last_12m::INTEGER             AS max_dpd_last_12m,
    total_months_delinquent::INTEGER      AS total_months_delinquent,
    has_90plus_dpd_flag::INTEGER          AS has_90plus_dpd_flag,
    has_30plus_dpd_flag::INTEGER          AS has_30plus_dpd_flag,
    months_since_last_delinquency::INTEGER AS months_since_last_delinquency,

    -- Bureau enquiries
    total_enquiry_count::INTEGER          AS total_enquiry_count,
    distinct_enquiring_lenders::INTEGER   AS distinct_enquiring_lenders,

    -- Derived ratios
    bureau_dsti::NUMERIC                  AS bureau_dsti,
    bureau_annual_dti::NUMERIC            AS bureau_annual_dti,
    loan_to_income_ratio::NUMERIC         AS loan_to_income_ratio,
    bureau_health_score::INTEGER          AS bureau_health_score

FROM {{ ref('data') }}