import pandas as pd

def merge_csvs_with_offset(csv_multi, csv_single, out_csv="submission_final.csv"):
    df_multi  = pd.read_csv(csv_multi)
    df_single = pd.read_csv(csv_single)

    if list(df_multi.columns) != list(df_single.columns):
        raise ValueError(f"Inconsistent col name: {df_multi.columns} vs {df_single.columns}")

    offset = len(df_multi)

    try:
        df_single["image_id"] = df_single["image_id"].astype(int) + offset
    except ValueError:
        print("error!")
        df_single["image_id"] = df_single["image_id"].astype(str) + f"_{offset}"

    df_final = pd.concat([df_multi, df_single], axis=0, ignore_index=True)
    df_final.to_csv(out_csv, index=False)
    print(f"✅ merged CSV saved to {out_csv} ({len(df_final)} rows, offset={offset})")

merge_csvs_with_offset("./results/sigliplarge384_multi_lora_proj_head_en_v2_t2_r16_qv.csv", "./results/sigliplarge384_single_lora_proj_head_en_v2_t2_r16_qv.csv", "./results/sigliplarge384_submission_lora_proj_head_en_v2_t2_r16_qv.csv")