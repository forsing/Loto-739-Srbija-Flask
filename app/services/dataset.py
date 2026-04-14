import pandas as pd
from pathlib import Path
from itertools import combinations

# LatLoto RAW (stari format)
RAW_COLUMNS = ["Izlozes Nr.", "Datums", "Izlozētie skaitļi"]


def read_table(path: Path, file_format: str | None = None) -> pd.DataFrame:
    """Čita CSV ili XLSX. Za Loto 7/39 bez zaglavlja: 7 kolona brojeva."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        if file_format == "loto739_samo":
            return pd.read_csv(
                path,
                header=None,
                names=[f"Num{i}" for i in range(1, 8)],
                encoding="utf-8",
            )
        return pd.read_csv(path, encoding="utf-8")
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError("Nepodržan format. Koristite CSV ili XLSX.")


def _canonical_num_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Num1..Num7 (case-insensitive num1..num7)."""
    rename = {}
    for c in df.columns:
        s = str(c).strip()
        low = s.lower().replace(" ", "")
        if low.startswith("num") and len(low) > 3 and low[3:].isdigit():
            rename[c] = "Num" + low[3:]
    return df.rename(columns=rename)


def is_lotto739_num(df: pd.DataFrame) -> bool:
    df2 = _canonical_num_columns(df)
    return all(f"Num{i}" in df2.columns for i in range(1, 8))


def is_latloto_raw(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in RAW_COLUMNS)


def is_prepared(df: pd.DataFrame) -> bool:
    required = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5"]
    return all(col in df.columns for col in required)


def is_lotto739_prepared(df: pd.DataFrame) -> bool:
    cols = [f"n{i}" for i in range(1, 8)]
    return all(c in df.columns for c in cols) and "date" in df.columns


def main_draw_columns(df: pd.DataFrame) -> list[str]:
    """Kolone glavnih brojeva za statistiku (n1..n7 ako postoje, inače n1..n6)."""
    out = []
    for i in range(1, 9):
        c = f"n{i}"
        if c in df.columns:
            out.append(c)
    return out


def _parse_numbers_list(text: str):
    if pd.isna(text):
        return []
    if isinstance(text, (int, float)):
        return [int(text)]

    s = str(text)
    for sep in [",", ";"]:
        s = s.replace(sep, " ")

    parts = [p for p in s.split() if p.strip() != ""]
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            continue
    return nums


def _parse_main_and_bonus(value):
    if pd.isna(value):
        return [], []

    text = str(value)
    if "+" in text:
        left, right = text.split("+", 1)
        mains = _parse_numbers_list(left)
        bonuses = _parse_numbers_list(right)
    else:
        mains = _parse_numbers_list(text)
        bonuses = []

    return mains, bonuses


def _detect_lottery_from_numbers(main_counts, bonus_counts, max_main, max_bonus):
    if max_main <= 48 and all(c == 6 for c in main_counts) and all(b == 1 for b in bonus_counts):
        return "viking"

    if max_main <= 50 and all(c == 5 for c in main_counts) and all(b in (0, 2) for b in bonus_counts):
        return "euro"

    return "unknown"


def normalize_any(df_raw: pd.DataFrame, lottery: str, file_format: str) -> pd.DataFrame:
    if lottery == "loto739":
        if file_format == "loto739_num":
            if not is_lotto739_num(df_raw):
                raise ValueError("Izabran je CSV Num1–Num7, ali u fajlu nedostaju kolone Num1..Num7.")
            df_norm = _normalize_lotto739_num(_canonical_num_columns(df_raw))
        elif file_format == "loto739_samo":
            df_norm = _normalize_lotto739_num(_canonical_num_columns(df_raw))
        elif file_format == "loto739_prepared":
            if not is_lotto739_prepared(df_raw):
                raise ValueError("Pripremljeni format: potrebne kolone date, n1..n7.")
            df_norm = _normalize_lotto739_prepared(df_raw)
        else:
            raise ValueError("Nepoznat file_format za Loto 7/39.")

        _validate_lottery_safety(df_norm, lottery)
        return df_norm

    if file_format == "raw":
        if not is_latloto_raw(df_raw):
            raise ValueError("RAW format izabran, ali nedostaju potrebne kolone.")
        df_norm = _normalize_raw(df_raw)

    elif file_format == "prepared":
        if not is_prepared(df_raw):
            raise ValueError("Pripremljeni format izabran, ali nedostaju kolone.")
        df_norm = _normalize_prepared(df_raw)

    else:
        raise ValueError("Nepoznat file_format.")

    _validate_lottery_safety(df_norm, lottery)

    return df_norm


def _normalize_lotto739_num(df_raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in df_raw.iterrows():
        nums = [int(row[f"Num{i}"]) for i in range(1, 8)]
        if len(set(nums)) != 7:
            raise ValueError(f"Red {idx + 1}: očekuje se 7 različitih brojeva.")
        for n in nums:
            if n < 1 or n > 39:
                raise ValueError(f"Red {idx + 1}: brojevi moraju biti u opsegu 1–39.")

        draw_no = idx + 1
        if "draw_no" in df_raw.columns and pd.notna(row.get("draw_no")):
            try:
                draw_no = int(row["draw_no"])
            except (TypeError, ValueError):
                draw_no = idx + 1

        if "date" in df_raw.columns and pd.notna(row.get("date")):
            date_val = pd.to_datetime(row["date"], dayfirst=True, errors="coerce")
        elif "Datum" in df_raw.columns and pd.notna(row.get("Datum")):
            date_val = pd.to_datetime(row["Datum"], dayfirst=True, errors="coerce")
        else:
            date_val = pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(idx))

        if pd.isna(date_val):
            date_val = pd.Timestamp("1970-01-01") + pd.Timedelta(days=int(idx))

        nums_sorted = sorted(nums)
        rec = {"draw_no": draw_no, "date": date_val}
        for i in range(7):
            rec[f"n{i+1}"] = nums_sorted[i]
        rec["b1"] = None
        rec["b2"] = None
        records.append(rec)

    df_norm = pd.DataFrame(records)
    df_norm = df_norm.sort_values("date").reset_index(drop=True)
    return df_norm


def _normalize_lotto739_prepared(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    cols = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6", "n7", "b1", "b2"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    for c in [f"n{i}" for i in range(1, 8)]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    df["draw_no"] = pd.to_numeric(df["draw_no"], errors="coerce")
    df = df[cols]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _normalize_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    records = []

    for _, row in df_raw.iterrows():
        draw_no = row["Izlozes Nr."]
        date = row["Datums"]

        mains, bonuses = _parse_main_and_bonus(row["Izlozētie skaitļi"])
        mains_sorted = sorted(int(n) for n in mains)
        bonuses_sorted = sorted(int(n) for n in bonuses)

        rec = {
            "draw_no": draw_no,
            "date": pd.to_datetime(date, dayfirst=True),
        }

        for i in range(5):
            rec[f"n{i+1}"] = mains_sorted[i] if i < len(mains_sorted) else None
        rec["n6"] = mains_sorted[5] if len(mains_sorted) > 5 else None

        rec["b1"] = bonuses_sorted[0] if len(bonuses_sorted) > 0 else None
        rec["b2"] = bonuses_sorted[1] if len(bonuses_sorted) > 1 else None

        records.append(rec)

    df_norm = pd.DataFrame(records)
    df_norm = df_norm.sort_values("date").reset_index(drop=True)
    return df_norm


def _normalize_prepared(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    cols = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6", "b1", "b2"]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    for c in ["n1", "n2", "n3", "n4", "n5", "n6", "b1", "b2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df = df[cols]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _validate_lottery_safety(df_norm: pd.DataFrame, lottery: str):
    if lottery == "loto739":
        cols = [f"n{i}" for i in range(1, 8)]
        for ri, row in df_norm.iterrows():
            nums = []
            for c in cols:
                if c not in row or pd.isna(row.get(c)):
                    raise ValueError(f"Red {ri + 1}: nedostaje neki od brojeva n1..n7.")
                nums.append(int(row[c]))
            if len(set(nums)) != 7:
                raise ValueError(f"Red {ri + 1}: brojevi moraju biti svi različiti.")
            if min(nums) < 1 or max(nums) > 39:
                raise ValueError(f"Red {ri + 1}: brojevi moraju biti u opsegu 1–39.")
        return

    main_counts = []
    bonus_counts = []
    max_main = 0
    max_bonus = 0

    for _, row in df_norm.iterrows():
        mains = [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"]]
        if not pd.isna(row.get("n6", pd.NA)):
            mains.append(row["n6"])
        mains_clean = [int(x) for x in mains if not pd.isna(x)]

        bonuses = []
        if "b1" in df_norm.columns and not pd.isna(row.get("b1", pd.NA)):
            bonuses.append(int(row["b1"]))
        if "b2" in df_norm.columns and not pd.isna(row.get("b2", pd.NA)):
            bonuses.append(int(row["b2"]))

        main_counts.append(len(mains_clean))
        bonus_counts.append(len(bonuses))

        if mains_clean:
            max_main = max(max_main, max(mains_clean))
        if bonuses:
            max_bonus = max(max_bonus, max(bonuses))

    detected = _detect_lottery_from_numbers(main_counts, bonus_counts, max_main, max_bonus)

    if lottery == "viking":
        if detected == "euro":
            raise ValueError("Podaci liče na Eurojackpot, a izabran je Viking.")
        if max_main > 48:
            raise ValueError("Glavni brojevi prelaze 48 — nije validno za Viking.")

    elif lottery == "euro":
        if detected == "viking":
            raise ValueError("Podaci liče na Viking, a izabran je Eurojackpot.")
        if max_main > 50:
            raise ValueError("Glavni brojevi prelaze 50 — nije validno za Eurojackpot.")
        if max_bonus > 12:
            raise ValueError("Bonus brojevi prelaze 12 — nije validno za Eurojackpot.")

    else:
        raise ValueError("Nepoznat tip lutrije.")


def get_top_numbers(df_norm, k=5):
    cols = main_draw_columns(df_norm)
    counts: dict[int, int] = {}

    for _, row in df_norm.iterrows():
        for c in cols:
            n = row[c]
            if pd.isna(n):
                continue
            n = int(n)
            counts[n] = counts.get(n, 0) + 1

    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)

    return sorted_counts[:k]


def get_top_combinations(df_norm, comb_size=2, top_k=5):
    cols = main_draw_columns(df_norm)
    counts = {}

    for _, row in df_norm.iterrows():
        clean_nums = []
        for c in cols:
            n = row[c]
            if pd.isna(n):
                continue
            clean_nums.append(int(n))

        if len(clean_nums) < comb_size:
            continue

        for combo in combinations(sorted(clean_nums), comb_size):
            counts[combo] = counts.get(combo, 0) + 1

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_counts[:top_k]
