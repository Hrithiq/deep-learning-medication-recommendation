def build_ddi_map(ddi_df):
    ddi_map = set()
    for _, row in ddi_df.iterrows():
        ddi_map.add((row['drug1'], row['drug2']))
        ddi_map.add((row['drug2'], row['drug1']))
    return ddi_map


def filter_ddi(meds, ddi_map):
    safe = []
    for m in meds:
        if all((m, s) not in ddi_map for s in safe):
            safe.append(m)
    return safe