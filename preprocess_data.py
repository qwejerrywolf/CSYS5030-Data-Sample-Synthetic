"""Standalone preprocessing: private server export -> restricted cleaned dataset.

The private export (`dataset/player_data.csv`) is never modified.  This script applies the
documented cleaning rules, resolves last-connection provinces using the supplied offline
ip2region databases, removes direct identifiers and exports an account-level file that is
safe for restricted university-internal distribution.

Run from the Stage2 workspace:

    .\\.venv\\Scripts\\python.exe preprocess_data.py

Inputs
    dataset/player_data.csv                 private, unchanged
    dataset/ip2region/ip2region_v4.xdb      offline IPv4 database
    dataset/ip2region/ip2region_v6.xdb      offline IPv6 database

Outputs
    dataset/public/player_data_cleaned.csv  de-identified analysis dataset
    outputs/preprocessing_audit.json        aggregate audit, hashes and counts (no identifiers)
    outputs/preprocessing_shared_ip_summary.json
                                             aggregate IP-group context only (no addresses)
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import ip2region.searcher as xdb
import ip2region.util as iputil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dataset/player_data.csv"
PUBLIC = ROOT / "dataset/public/player_data_cleaned.csv"
AUDIT = ROOT / "outputs/preprocessing_audit.json"
SHARED_IP = ROOT / "outputs/preprocessing_shared_ip_summary.json"

EXPECTED_SOURCE_COLUMNS = [
    "UUID", "PlayerID", "FirstJoinTime", "LastQuitTime", "LastIpAddress", "TotalPlayTime"
]
PUBLIC_COLUMNS = [
    "first_join_time_ms", "last_quit_time_ms", "total_play_time_ms",
    "geo_status", "province", "ip_version", "engagement_category", "join_cohort",
]

PROVINCE_PAIRS = [
    ("北京", "Beijing"), ("天津", "Tianjin"), ("河北", "Hebei"), ("山西", "Shanxi"),
    ("内蒙古", "Inner Mongolia"), ("辽宁", "Liaoning"), ("吉林", "Jilin"),
    ("黑龙江", "Heilongjiang"), ("上海", "Shanghai"), ("江苏", "Jiangsu"),
    ("浙江", "Zhejiang"), ("安徽", "Anhui"), ("福建", "Fujian"), ("江西", "Jiangxi"),
    ("山东", "Shandong"), ("河南", "Henan"), ("湖北", "Hubei"), ("湖南", "Hunan"),
    ("广东", "Guangdong"), ("广西", "Guangxi"), ("海南", "Hainan"),
    ("重庆", "Chongqing"), ("四川", "Sichuan"), ("贵州", "Guizhou"),
    ("云南", "Yunnan"), ("西藏", "Tibet"), ("陕西", "Shaanxi"), ("甘肃", "Gansu"),
    ("青海", "Qinghai"), ("宁夏", "Ningxia"), ("新疆", "Xinjiang"),
]
PRIVATE_NETS = [
    ipaddress.ip_network(x)
    for x in ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7"]
]
STATUSES = [
    "mainland", "non_mainland", "private", "reserved_or_non_global", "unresolved",
    "unresolved_mainland_province", "invalid", "query_error",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def make_searchers() -> dict[int, object]:
    searchers: dict[int, object] = {}
    for version, cls in [(4, iputil.IPv4), (6, iputil.IPv6)]:
        path = ROOT / f"dataset/ip2region/ip2region_v{version}.xdb"
        iputil.verify_from_file(str(path))
        searchers[version] = xdb.new_with_buffer(
            cls, iputil.load_content_from_file(str(path))
        )
    return searchers


def map_ip(value: object, searchers: dict[int, object]):
    """Return (status, English province or None, IP version, IPv4-mapped flag).

    No address or exception text is ever returned, displayed or written.
    """
    try:
        addr = ipaddress.ip_address(str(value).strip())
    except ValueError:
        return ("invalid", None, 0, False)
    mapped = addr.version == 6 and addr.ipv4_mapped is not None
    if mapped:
        addr = addr.ipv4_mapped
    version = addr.version
    if any(addr in net for net in PRIVATE_NETS if net.version == version):
        return ("private", None, version, mapped)
    if not addr.is_global or addr.is_multicast or addr.is_reserved:
        return ("reserved_or_non_global", None, version, mapped)
    try:
        region = searchers[version].search(str(addr))
    except Exception:
        return ("query_error", None, version, mapped)
    if not region:
        return ("unresolved", None, version, mapped)
    parts = region.split("|")
    # XDB layouts in use: country|province|city|ISP|ISO or country|0|province|city|ISP.
    if any(
        any(z in p for z in ["香港", "澳门", "澳門", "台湾", "臺灣"])
        or p in ["HK", "MO", "TW", "Hong Kong", "Macao", "Macau", "Taiwan"]
        for p in parts
    ):
        return ("non_mainland", None, version, mapped)
    china = parts[0] in ["中国", "China", "CN", "中华人民共和国"] or parts[-1] == "CN"
    if not china:
        if parts[0] in ["", "0", "未知"]:
            return ("unresolved", None, version, mapped)
        return ("non_mainland", None, version, mapped)
    candidates = {
        en
        for zh, en in PROVINCE_PAIRS
        for p in parts[1:3]
        if p in [zh, zh + "省", zh + "市", en]
        or (p.startswith(zh) and p.endswith("自治区"))
    }
    if len(candidates) == 1:
        return ("mainland", next(iter(candidates)), version, mapped)
    return ("unresolved_mainland_province", None, version, mapped)


def cohorts(times, q: int = 3):
    """Empirical quantile cohorts; equal timestamps go to the earlier cohort."""
    values = np.asarray(times)
    cuts = np.unique(np.quantile(values, np.arange(1, q) / q, method="inverted_cdf"))
    labels = np.searchsorted(cuts, values, side="left").astype(int)
    assert len(np.unique(labels)) == q, "Ties prevent the requested number of cohorts."
    return labels, cuts


def engagement(hours, scale: float = 1.0):
    """Fixed engagement categories [0,1), [1,10), [10,100), [100, inf) hours."""
    return np.searchsorted(
        np.array([1, 10, 100]) * scale, np.asarray(hours), side="right"
    ).astype(int)


def normalise_ip(value: object) -> str:
    addr = ipaddress.ip_address(str(value).strip())
    if addr.version == 6 and addr.ipv4_mapped is not None:
        return str(addr.ipv4_mapped)
    return str(addr)


def main() -> None:
    assert SOURCE.is_file(), f"Private source CSV not found: {SOURCE}"
    assert (ROOT / "dataset/ip2region/ip2region_v4.xdb").is_file()
    assert (ROOT / "dataset/ip2region/ip2region_v6.xdb").is_file()
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC.parent.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        SOURCE,
        encoding="utf-8-sig",
        dtype={"UUID": str, "PlayerID": str, "LastIpAddress": str},
    )
    assert list(raw.columns) == EXPECTED_SOURCE_COLUMNS, "Unexpected source schema."
    assert not raw.UUID.duplicated().any(), (
        "Duplicate account identifiers require manual investigation."
    )

    for col in ["FirstJoinTime", "LastQuitTime", "TotalPlayTime"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    reasons = pd.DataFrame(
        {
            "missing_field": raw.isna().any(axis=1),
            "nonfinite_numeric": ~np.isfinite(
                raw[["FirstJoinTime", "LastQuitTime", "TotalPlayTime"]]
            ).all(axis=1),
            "nonpositive_first_join": raw.FirstJoinTime.le(0),
            "nonpositive_last_quit": raw.LastQuitTime.le(0),
            "negative_playtime": raw.TotalPlayTime.lt(0),
            "reversed_chronology": raw.LastQuitTime.gt(0)
            & raw.LastQuitTime.lt(raw.FirstJoinTime),
            "future_timestamp": raw[["FirstJoinTime", "LastQuitTime"]].max(axis=1).gt(
                pd.Timestamp.now(tz="UTC").timestamp() * 1000
            ),
        }
    )
    invalid = reasons.any(axis=1)
    clean = raw.loc[~invalid].copy().reset_index(drop=True)

    elapsed_ms = clean.LastQuitTime - clean.FirstJoinTime
    assert (clean.TotalPlayTime <= elapsed_ms).all(), (
        "Recorded playtime exceeds elapsed account time; investigate before export."
    )

    # TotalPlayTime is recorded in milliseconds, as confirmed by the plugin developer.
    clean["hours"] = clean.TotalPlayTime / 3_600_000
    clean["joined"] = pd.to_datetime(
        clean.FirstJoinTime, unit="ms", utc=True
    ).dt.tz_convert("Asia/Shanghai")
    clean["quit"] = pd.to_datetime(
        clean.LastQuitTime, unit="ms", utc=True
    ).dt.tz_convert("Asia/Shanghai")
    snapshot_proxy = clean.quit.max()
    clean["exposure_days"] = (snapshot_proxy - clean.joined).dt.total_seconds() / 86400

    searchers = make_searchers()
    try:
        cache = {ip: map_ip(ip, searchers) for ip in raw.LastIpAddress.unique()}
    finally:
        for searcher in searchers.values():
            searcher.close()

    geo = pd.DataFrame(
        [cache[ip] for ip in clean.LastIpAddress],
        columns=["geo_status", "province", "ip_version", "ipv4_mapped"],
    )
    clean = pd.concat([clean, geo], axis=1)
    assert not (clean.geo_status == "query_error").any(), (
        "Offline lookup errors require investigation."
    )

    mainland = clean.geo_status == "mainland"
    assert clean.loc[mainland, "province"].notna().all(), (
        "A mainland record lacks a resolved province."
    )

    # Engagement category is fixed a priori.  Cohort labels use mainland tertiles so the
    # column can be validated against the notebook; non-mainland rows receive the same
    # cutpoint assignment even though they are outside the primary sample.
    clean["engagement_category"] = engagement(clean.hours)
    cohort_all = np.empty(len(clean), dtype=int)
    mainland_labels, cuts = cohorts(clean.loc[mainland, "FirstJoinTime"])
    cohort_all[mainland.to_numpy()] = mainland_labels
    if (~mainland).any():
        cohort_all[(~mainland).to_numpy()] = np.searchsorted(
            cuts, clean.loc[~mainland, "FirstJoinTime"].to_numpy(), side="left"
        )
    clean["join_cohort"] = cohort_all

    public = pd.DataFrame(
        {
            "first_join_time_ms": clean.FirstJoinTime.astype("int64"),
            "last_quit_time_ms": clean.LastQuitTime.astype("int64"),
            "total_play_time_ms": clean.TotalPlayTime.astype("int64"),
            "geo_status": clean.geo_status.astype(str),
            "province": clean.province.where(mainland, "").fillna(""),
            "ip_version": clean.ip_version.astype("int64"),
            "engagement_category": clean.engagement_category.astype("int64"),
            "join_cohort": clean.join_cohort.astype("int64"),
        }
    )
    assert list(public.columns) == PUBLIC_COLUMNS
    assert len(public) == len(clean)
    numeric = [
        "first_join_time_ms", "last_quit_time_ms", "total_play_time_ms",
        "ip_version", "engagement_category", "join_cohort",
    ]
    assert not public[numeric].isna().any().any()
    assert public.geo_status.isin(STATUSES).all()
    assert not public.province.isna().any()

    public.to_csv(PUBLIC, index=False, encoding="utf-8", lineterminator="\n")

    coverage = pd.DataFrame(
        {
            "status": STATUSES,
            "all_source_rows": [
                int((raw.LastIpAddress.map(lambda ip: cache[ip][0]) == status).sum())
                for status in STATUSES
            ],
            "valid_rows": [int((clean.geo_status == status).sum()) for status in STATUSES],
        }
    )
    coverage["valid_percent"] = 100 * coverage.valid_rows / len(clean)

    # Aggregate-only shared-IP context.  The one-account-per-IP resampling cannot be
    # reproduced from the de-identified export and is therefore not part of the notebook.
    primary_ips = clean.loc[mainland, "LastIpAddress"].map(normalise_ip)
    group_sizes = primary_ips.value_counts()
    size_counts = group_sizes.value_counts().sort_index()
    shared_summary = {
        "primary_accounts": int(mainland.sum()),
        "distinct_normalised_last_ip_primary": int(group_sizes.size),
        "groups_with_multiple_accounts": int((group_sizes > 1).sum()),
        "accounts_in_multi_account_groups": int(group_sizes[group_sizes > 1].sum()),
        "largest_group_accounts": int(group_sizes.max()),
        "group_size_histogram": {str(k): int(v) for k, v in size_counts.items()},
        "note": (
            "Aggregate counts only; no addresses or per-account IP keys are retained. "
            "One-account-per-IP resampling requires the private source and is excluded "
            "from the cleaned-data analysis."
        ),
    }

    xdb_meta = []
    for path in sorted((ROOT / "dataset/ip2region").glob("*.xdb")):
        header = iputil.load_header_from_file(str(path))
        xdb_meta.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "format_version": header.version,
                "ip_version": header.ipVersion,
                "created_utc": utc_iso(header.createdAt),
            }
        )

    audit = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": "dataset/player_data.csv",
        "source_bytes": SOURCE.stat().st_size,
        "source_sha256": sha256(SOURCE),
        "source_rows": int(len(raw)),
        "exclusion_reasons": {name: int(count) for name, count in reasons.sum().items()},
        "excluded_rows": int(invalid.sum()),
        "valid_rows": int(len(clean)),
        "primary_mainland_rows": int(mainland.sum()),
        "geographic_exclusions": int((~mainland).sum()),
        "geo_coverage": coverage.to_dict(orient="records"),
        "province_retention_rule": {
            "overall_minimum": 30, "per_cohort_minimum": 10,
            "cohorts": 3, "engagement_thresholds_hours": [1, 10, 100],
        },
        "cleaned_file": "dataset/public/player_data_cleaned.csv",
        "cleaned_bytes": PUBLIC.stat().st_size,
        "cleaned_sha256": sha256(PUBLIC),
        "cleaned_columns": PUBLIC_COLUMNS,
        "rows_written": int(len(public)),
        "unit": {
            "TotalPlayTime": "milliseconds",
            "confirmed_by": "plugin developer",
            "conversion": "TotalPlayTime / 3_600_000 = hours",
            "timestamp_epoch": "Unix milliseconds",
        },
        "xdb": xdb_meta,
        "ip2region_binding": "py-ip2region",
        "shared_ip": "preprocessing_shared_ip_summary.json",
    }
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    SHARED_IP.write_text(json.dumps(shared_summary, indent=2), encoding="utf-8")

    print(f"Source rows: {len(raw)}; excluded: {int(invalid.sum())}; valid: {len(clean)}")
    print(f"Primary mainland rows: {int(mainland.sum())}; geographic exclusions: {int((~mainland).sum())}")
    print("Exclusions by rule:", {k: int(v) for k, v in reasons.sum().items() if v})
    print(f"Wrote {PUBLIC.relative_to(ROOT)} ({PUBLIC.stat().st_size} bytes, sha256={sha256(PUBLIC)})")
    print(f"Wrote {AUDIT.relative_to(ROOT)} and {SHARED_IP.relative_to(ROOT)}")
    print("TotalPlayTime is documented as milliseconds: hours = total_play_time_ms / 3_600_000.")


if __name__ == "__main__":
    main()
