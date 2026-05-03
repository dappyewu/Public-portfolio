"""Data fetcher: Elexon BMRS for GB power data + Yahoo Finance for fuel prices.

Power series (from Elexon Insights API, half-hourly resampled to hourly):

    day_ahead_price    MID dataset, dataProvider=APXMIDP. The N2EXMIDP
                       provider is also returned by the API but has been
                       dormant since the GB auction consolidation (always
                       price=0/volume=0); the loader explicitly skips it.
    load_forecast      /forecast/demand/total/day-ahead — purpose-built
                       day-ahead demand forecast endpoint. Only retains
                       data from ~July 2023 onwards, which bounds the
                       merged frame's start date.
    wind_forecast      /datasets/WINDFOR with publishDateTime filter +
                       12h lead-time gate. Without that gate the dataset
                       returns intraday forecasts that wouldn't have been
                       known at day-ahead auction time.
    solar_forecast     /generation/actual/per-type/wind-and-solar (AGWS)
                       filtered to psrType='Solar', then SHIFTED FORWARD
                       BY 24h. Elexon retired all free day-ahead solar
                       forecast endpoints; the AGWS-shifted-24h series is
                       the closest causally-honest proxy (yesterday's
                       actual at hour T, known by today's auction).

Fuel series (from Yahoo Finance, daily forward-filled to hourly):

    ttf_gas            TTF front-month future (TTF=F). Proxy for NBP gas.
                       Correlation with NBP > 0.95.
    eua_carbon         KEUA — KraneShares European Carbon Allowance ETF.
                       Tracks ICE EUA front-month futures. Different units
                       (USD ETF NAV vs EUR/tCO2) but daily co-movement is
                       what the model uses.

Both fuel series are LAGGED BY ONE DAY before joining to power data, so
the feature at hour T is yesterday's close — unambiguously known before
the 11:00 CET day-ahead auction. No look-ahead leakage. Sundays inherit
Friday's close via forward-fill.

Each Elexon endpoint caps date ranges at 7 days, so all four power-data
fetches are chunked by week. The default 2023-08 -> 2025-12 window is
~125 chunks per dataset; with the polite 0.4s pacing the cold pull takes
~3 minutes for the power layer. Yahoo Finance fetches in one call
regardless of range. Per-dataset parquet caches under ``data/raw/_cache/``
make subsequent runs near-instant.
"""

from __future__ import annotations

import json
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


def _date_chunks(start: str, end: str, chunk_days: int = 7) -> list[tuple[str, str]]:
    """Yield (from_date, to_date) string pairs covering at most `chunk_days` each.

    Elexon's MID endpoint enforces a 7-day max range; the others appear to be
    similar or stricter, so 7 days is the safe default across the board. The
    pairs are inclusive on both ends, so a 7-day chunk runs e.g.
    2021-01-01 -> 2021-01-07, then 2021-01-08 -> 2021-01-14, and so on.
    """
    out = []
    cur = pd.Timestamp(start).to_pydatetime().date()
    end_d = pd.Timestamp(end).to_pydatetime().date()
    step = timedelta(days=chunk_days - 1)  # inclusive both ends
    while cur <= end_d:
        chunk_end = min(cur + step, end_d)
        out.append((cur.isoformat(), chunk_end.isoformat()))
        cur = chunk_end + timedelta(days=1)
    return out


def _request_json(url: str, params: dict, delay_s: float, max_retries: int = 5) -> dict:
    """GET with polite pacing and exponential-backoff retries on transient failures.

    Retries on:
        - Connection errors / DNS failures / timeouts (network hiccups)
        - HTTP 429 (rate limit) and 5xx (server errors)
    Fails fast on 4xx other than 429 — those won't fix themselves.

    Backoff schedule: 2s, 4s, 8s, 16s, 32s.
    """
    last_err = None
    for attempt in range(max_retries):
        time.sleep(delay_s)
        try:
            r = requests.get(url, params=params, timeout=60,
                             headers={"Accept": "application/json"})
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = 2 ** (attempt + 1)
            print(f"    network error (attempt {attempt + 1}/{max_retries}); "
                  f"retrying in {wait}s...")
            last_err = e
            time.sleep(wait)
            continue

        if r.status_code == 200:
            return r.json()

        if r.status_code in (429, 502, 503, 504):
            wait = 2 ** (attempt + 1)
            print(f"    HTTP {r.status_code} (attempt {attempt + 1}/{max_retries}); "
                  f"retrying in {wait}s...")
            time.sleep(wait)
            continue

        # 4xx other than 429: client error, won't fix itself.
        raise RuntimeError(
            f"Elexon API returned {r.status_code} for {url} {params}: {r.text[:300]}"
        )

    raise RuntimeError(
        f"Elexon API failed after {max_retries} retries for {url} {params}. "
        f"Last error: {last_err}"
    )


def _records(payload: dict) -> list[dict]:
    """Elexon wraps results in {'data': [...]}, sometimes with metadata. Extract the list."""
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]
        if "results" in payload and isinstance(payload["results"], list):
            return payload["results"]
    raise RuntimeError(f"Unexpected payload shape: {str(payload)[:300]}")


# --- Per-dataset fetchers --------------------------------------------------

def fetch_mid(api_base: str, start: str, end: str, delay_s: float) -> pd.DataFrame:
    """Day-ahead reference price from MID, dataProvider=APXMIDP. Half-hourly."""
    url = f"{api_base}/datasets/MID"
    frames = []
    for f, t in _date_chunks(start, end):
        print(f"  MID  {f} -> {t}")
        payload = _request_json(url, {"from": f, "to": t, "format": "json"}, delay_s)
        rows = _records(payload)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise RuntimeError("No MID data returned. Check the date range.")
    df = pd.concat(frames, ignore_index=True)

    # Pick the active GB day-ahead reference. APXMIDP (now EPEX SPOT) is the
    # live one; N2EXMIDP rows still appear in the API but always have
    # price=0/volume=0 because that index has been dormant since the GB
    # auction consolidation. Prefer APXMIDP; fall back to N2EXMIDP only if
    # APXMIDP is missing AND N2EXMIDP has any non-zero prices.
    providers = df["dataProvider"].unique() if "dataProvider" in df else []
    chosen = None
    if "APXMIDP" in providers:
        chosen = "APXMIDP"
    elif "N2EXMIDP" in providers:
        n2 = df[df["dataProvider"] == "N2EXMIDP"]
        if (n2["price"] != 0).any():
            chosen = "N2EXMIDP"
    if chosen is None and len(providers):
        chosen = providers[0]
    if chosen is None:
        raise RuntimeError("MID payload had no dataProvider column.")
    print(f"  -> using dataProvider={chosen}")
    df = df[df["dataProvider"] == chosen].copy()

    # Build a UTC half-hourly timestamp from settlementDate + settlementPeriod.
    df["timestamp_utc"] = _settlement_to_utc(df["settlementDate"], df["settlementPeriod"])
    df = df[["timestamp_utc", "price"]].rename(columns={"price": "day_ahead_price"})
    return df.sort_values("timestamp_utc").set_index("timestamp_utc")


def fetch_ndf(api_base: str, start: str, end: str, delay_s: float) -> pd.DataFrame:
    """Day-ahead demand forecast.

    Uses ``/forecast/demand/total/day-ahead`` rather than ``/datasets/NDF``.
    The dataset endpoint returns the *current* live forecast (latest snapshot
    only), so historical fetches collapse to ~70 rows. The dedicated
    day-ahead endpoint returns the day-ahead forecast as it was published —
    one publication per day for the next day's 48 settlement periods.
    Causally honest by construction.
    """
    url = f"{api_base}/forecast/demand/total/day-ahead"
    frames = []
    for f, t in _date_chunks(start, end):
        print(f"  NDF  {f} -> {t}")
        payload = _request_json(url, {"from": f, "to": t, "format": "json"}, delay_s)
        rows = _records(payload)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise RuntimeError("No NDF day-ahead data returned. Check the date range.")
    df = pd.concat(frames, ignore_index=True)

    # The endpoint uses startTime (UTC ISO) directly; settlementDate +
    # settlementPeriod are also present, but startTime avoids the BST/GMT
    # conversion fragility of _settlement_to_utc.
    df["timestamp_utc"] = pd.to_datetime(df["startTime"], utc=True)
    if "publishTime" in df.columns:
        df = df.sort_values("publishTime").drop_duplicates("timestamp_utc", keep="last")
    df = df[["timestamp_utc", "quantity"]].rename(columns={"quantity": "load_forecast"})
    return df.sort_values("timestamp_utc").set_index("timestamp_utc")


def fetch_windfor(api_base: str, start: str, end: str, delay_s: float) -> pd.DataFrame:
    """Wind Forecast (WINDFOR), filtered to forecasts known by auction time.

    The dataset endpoint returns *all* WINDFOR publishes — the same settlement
    period gets dozens of forecasts as publishTime advances. To stay causally
    honest, only forecasts with at least 12 hours of lead time
    (publishTime + 12h <= startTime) are kept; for each settlement period the
    most recent valid forecast is then selected. That's "what a trader would
    have known by the day-ahead auction".

    Without the publishDateTime filter, the dataset endpoint returns only the
    most recent live forecast (~70 rows total) — same gotcha as NDF.
    """
    url = f"{api_base}/datasets/WINDFOR"
    frames = []
    for f, t in _date_chunks(start, end):
        # Publishes that affect the chunk's settlement periods landed roughly
        # within (chunk - 1d, chunk + 6d). The API caps publishDateTime range
        # at 7 days, so the publish window is set to exactly 7 days.
        pub_from = (pd.Timestamp(f) - pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00Z")
        pub_to = pd.Timestamp(t).strftime("%Y-%m-%dT00:00Z")
        print(f"  WIND {f} -> {t}")
        payload = _request_json(
            url,
            {
                "from": f"{f}T00:00Z",
                "to": f"{t}T23:59Z",
                "publishDateTimeFrom": pub_from,
                "publishDateTimeTo": pub_to,
                "format": "json",
            },
            delay_s,
        )
        rows = _records(payload)
        if rows:
            frames.append(pd.DataFrame(rows))
    if not frames:
        raise RuntimeError("No WINDFOR data returned. Check the date range.")
    df = pd.concat(frames, ignore_index=True)
    df["timestamp_utc"] = pd.to_datetime(df["startTime"], utc=True)
    df["publish_utc"] = pd.to_datetime(df["publishTime"], utc=True)

    # Lead-time filter: keep only forecasts with >=12h between publish and
    # delivery. This excludes intraday-updated forecasts that wouldn't have
    # been known to traders before the day-ahead auction (~11:00 CET).
    lead_h = (df["timestamp_utc"] - df["publish_utc"]).dt.total_seconds() / 3600
    df = df[lead_h >= 12]

    # Of the surviving forecasts for each settlement period, keep the most
    # recent — that's the closest publish to "auction-time information cutoff".
    df = df.sort_values("publish_utc").drop_duplicates("timestamp_utc", keep="last")

    qty_col = "generation" if "generation" in df.columns else "quantity"
    df = df[["timestamp_utc", qty_col]].rename(columns={qty_col: "wind_forecast"})
    return df.sort_values("timestamp_utc").set_index("timestamp_utc")


def fetch_solar(api_base: str, start: str, end: str, delay_s: float) -> pd.DataFrame:
    """GB solar from AGWS actuals, lagged by 24 h to avoid auction-time leakage.

    Why this is not a 'forecast': Elexon has retired all of its free day-ahead
    solar forecast endpoints (B1440 / DGWS / the wind-and-solar forecast
    endpoint all return 404 as of investigation). The only free no-auth solar
    series still served is AGWS — actual generation, half-hourly, published
    ~90 minutes after the fact.

    The fix: pull AGWS, then **shift its timestamps forward by 24 hours**
    before joining to the price data. The feature value at hour T is therefore
    "yesterday's actual solar at hour T". At 11:00 CET on day D-1 (when the
    day-ahead auction clears), all of D-2's solar actuals are public, so this
    is causally honest — no peeking at info that would not have been available
    at decision time.

    What this misses vs a real forecast: tomorrow-specific cloud cover. The
    diurnal and seasonal shape is captured well; day-to-day weather variation
    is not. Acceptable trade-off for a free no-auth pipeline; a production
    trading desk would buy a paid forecast.
    """
    url = f"{api_base}/generation/actual/per-type/wind-and-solar"
    frames = []
    empty = pd.DataFrame(
        columns=["solar_forecast"],
        index=pd.DatetimeIndex([], tz="UTC", name="timestamp_utc"),
    )

    # Pull a small pad *before* the requested start so the 24h lag is filled
    # at the boundary; trimmed back to [start, end] at the end.
    pad_start = (pd.Timestamp(start) - pd.Timedelta(days=2)).date().isoformat()

    for f, t in _date_chunks(pad_start, end):
        print(f"  SOL  {f} -> {t}")
        try:
            payload = _request_json(
                url, {"from": f, "to": t, "format": "json"}, delay_s
            )
        except RuntimeError as e:
            if "404" in str(e):
                print("  -> AGWS endpoint returned 404. Solar will be zero-filled.")
                return empty
            raise
        rows = _records(payload)
        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        print("  -> no AGWS rows returned; column will be zero-filled.")
        return empty

    df = pd.concat(frames, ignore_index=True)
    df = df[df["psrType"].str.contains("Solar", case=False, na=False)]
    if df.empty:
        print("  -> AGWS rows had no Solar psrType; column will be zero-filled.")
        return empty

    df["timestamp_utc"] = pd.to_datetime(df["startTime"], utc=True)
    df = df[["timestamp_utc", "quantity"]].rename(columns={"quantity": "solar_forecast"})
    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")
    df = df.set_index("timestamp_utc")

    # === The auction-realism step ===
    # Shift every timestamp forward by 24h so that at hour T the feature value
    # is the actual generation observed at T - 24h, which is known by auction
    # time the day before delivery.
    df.index = df.index + pd.Timedelta(hours=24)

    # Trim to the requested window.
    df = df.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")]
    return df


# --- Helpers ---------------------------------------------------------------

def _settlement_to_utc(dates: pd.Series, periods: pd.Series) -> pd.DatetimeIndex:
    """Convert (settlementDate, settlementPeriod 1-48) into a UTC timestamp.

    Settlement period N covers minutes [(N-1)*30, N*30) of the local day.
    GB uses BST/GMT, so the timestamps are localised to Europe/London first,
    then converted to UTC.
    """
    dates = pd.to_datetime(dates)
    periods = pd.to_numeric(periods, errors="coerce").fillna(0).astype(int)
    minutes = (periods - 1) * 30
    naive = dates + pd.to_timedelta(minutes, unit="m")
    # Localise as GB clock time, then convert to UTC.
    local = naive.dt.tz_localize("Europe/London", ambiguous="NaT", nonexistent="NaT")
    return local.dt.tz_convert("UTC")


# --- Fuel prices (Yahoo Finance) ------------------------------------------

def _fetch_one_ticker(ticker: str, pad_start: str, pad_end: str) -> pd.Series | None:
    """Download a single Yahoo Finance ticker. Returns None if delisted/empty."""
    import yfinance as yf
    try:
        d = yf.download(
            ticker, start=pad_start, end=pad_end,
            progress=False, auto_adjust=False,
        )
    except Exception as e:
        print(f"  WARN  {ticker}: {e}")
        return None
    if d is None or d.empty:
        return None
    s = d["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s if not s.dropna().empty else None


def fetch_fuel_prices(
    start: str,
    end: str,
    ttf_ticker: str = "TTF=F",
    eua_ticker: str = "KEUA",
    lag_days: int = 1,
) -> pd.DataFrame:
    """Daily TTF gas and EUA carbon, lagged and forward-filled to hourly UTC.

    Each ticker is fetched independently. If one is delisted or returns empty,
    its column is filled with NaN and a warning printed — the pipeline can
    proceed with whichever fuel signals are available.

    The lag is the auction-realism step: at 11:00 CET on day D, the trader
    knows yesterday's close (D-1) but not today's settlement (D, set after
    11:00). The fuel series are therefore shifted by ``lag_days`` and forward-
    filled across non-trading days before resampling to hourly.

    Returns a DataFrame indexed by hourly UTC timestamps with columns
    ``ttf_gas`` and ``eua_carbon`` (the latter may be NaN if the ticker is
    unavailable; downstream code should treat carbon as optional).
    """
    print(f"Fetching fuel prices: {ttf_ticker} (gas), {eua_ticker} (carbon)...")
    pad_start = (pd.Timestamp(start) - pd.Timedelta(days=10)).date().isoformat()
    pad_end = (pd.Timestamp(end) + pd.Timedelta(days=2)).date().isoformat()

    ttf = _fetch_one_ticker(ttf_ticker, pad_start, pad_end)
    eua = _fetch_one_ticker(eua_ticker, pad_start, pad_end)

    if ttf is None and eua is None:
        raise RuntimeError(
            "Both fuel tickers returned empty. Yahoo Finance ticker(s) likely "
            f"changed. Tried {ttf_ticker} and {eua_ticker}. Edit config.yaml's "
            "fuel section with current ticker symbols."
        )
    if ttf is None:
        print(f"  WARN  ttf_gas ({ttf_ticker}) returned no data; column will be NaN.")
    if eua is None:
        print(f"  WARN  eua_carbon ({eua_ticker}) returned no data; column will be NaN.")

    # Build the daily frame from whichever series came back.
    daily = pd.DataFrame()
    if ttf is not None:
        daily["ttf_gas"] = ttf
    else:
        daily["ttf_gas"] = pd.NA
    if eua is not None:
        # Align EUA index to TTF's if both exist; pandas concat will outer-join.
        daily = daily.join(eua.rename("eua_carbon"), how="outer")
    else:
        daily["eua_carbon"] = pd.NA

    # Apply the auction-realism lag, then forward-fill across weekends/holidays.
    daily = daily.shift(lag_days).ffill()

    # Promote to UTC-indexed hourly. Yahoo dates are naive; localise to UTC.
    daily.index = pd.to_datetime(daily.index).tz_localize("UTC")
    hourly = daily.resample("1h").ffill()

    hourly = hourly.loc[pd.Timestamp(start, tz="UTC"):pd.Timestamp(end, tz="UTC")]
    print(f"  -> {len(hourly):,} hourly rows of fuel data "
          f"(ttf_gas: {'OK' if ttf is not None else 'MISSING'}, "
          f"eua_carbon: {'OK' if eua is not None else 'MISSING'})")
    return hourly


def _clean_index(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaT index values and de-duplicate the index.

    Why this exists: ``_settlement_to_utc`` uses ``ambiguous='NaT'`` and
    ``nonexistent='NaT'`` to handle BST/GMT transitions. On the day clocks
    go back in October, two settlement periods can collide on the same UTC
    hour, producing NaT entries; on the spring-forward day the missing
    period becomes NaT. Either way, the index ends up with NaT rows and/or
    duplicates, which break ``pd.concat(..., axis=1)`` with InvalidIndexError.

    Belt-and-braces: even fetchers that already drop_duplicates can pick up
    edge cases (e.g. AGWS half-hourly that doesn't perfectly align after the
    24h shift). This helper is the single chokepoint that guarantees a unique
    DateTimeIndex before the concat.
    """
    if df is None or df.empty:
        return df
    if df.index.hasnans:
        df = df[~df.index.isna()]
    if not df.index.is_unique:
        df = df[~df.index.duplicated(keep="last")]
    return df


def _fetch_or_cache(name: str, fn, cache_dir: Path) -> pd.DataFrame:
    """Run `fn()` only if the per-dataset parquet isn't already on disk.

    Layout: ``{raw_dir}/_cache/{name}.parquet``.  This means a partial run
    is recoverable: if MID succeeds and NDF fails, restarting will skip
    MID and pick up NDF where it stopped.

    To force a re-fetch, delete the cache directory.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{name}.parquet"
    if path.exists():
        print(f"  {name} cached at {path.relative_to(cache_dir.parent.parent)}, skipping fetch")
        return pd.read_parquet(path)
    df = fn()
    df.to_parquet(path)
    print(f"  {name} cached -> {path.relative_to(cache_dir.parent.parent)}")
    return df


def fetch_all(
    api_base: str,
    start: str,
    end: str,
    delay_s: float = 0.4,
    fuel_cfg: dict | None = None,
    raw_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Pull power + fuel data, align on a common hourly UTC index, return one frame.

    If ``raw_dir`` is provided, each dataset's intermediate parquet is cached
    under ``raw_dir/_cache/`` and skipped on re-runs. This makes the multi-
    minute fetch resumable across transient network failures.
    """
    cache_dir = Path(raw_dir) / "_cache" if raw_dir else None

    def _wrap(name: str, fn):
        if cache_dir is None:
            return fn()
        return _fetch_or_cache(name, fn, cache_dir)

    print("Fetching MID (price)...")
    price = _wrap("mid",     lambda: fetch_mid(api_base, start, end, delay_s))
    print("Fetching NDF (load forecast)...")
    load  = _wrap("ndf",     lambda: fetch_ndf(api_base, start, end, delay_s))
    print("Fetching WINDFOR (wind forecast)...")
    wind  = _wrap("windfor", lambda: fetch_windfor(api_base, start, end, delay_s))
    print("Fetching B1440 (solar forecast)...")
    solar = _wrap("b1440",   lambda: fetch_solar(api_base, start, end, delay_s))

    # Guarantee unique DateTimeIndex on every frame before joining.
    # Handles DST-transition NaT/duplicates from _settlement_to_utc.
    price = _clean_index(price)
    load  = _clean_index(load)
    wind  = _clean_index(wind)
    solar = _clean_index(solar)

    df = pd.concat([price, load, wind, solar], axis=1)
    df = df.resample("1h").mean()

    if df["solar_forecast"].isna().all():
        df["solar_forecast"] = 0.0
    else:
        df["solar_forecast"] = df["solar_forecast"].fillna(0.0)
    for c in ["load_forecast", "wind_forecast"]:
        df[c] = df[c].ffill(limit=3).bfill(limit=3)

    df = df.dropna(subset=["day_ahead_price", "load_forecast", "wind_forecast"])

    # Fuel layer (cached separately; small, single API call so fast to refetch).
    fuel_cfg = fuel_cfg or {}
    fuel = _wrap(
        "fuel",
        lambda: fetch_fuel_prices(
            start=start, end=end,
            ttf_ticker=fuel_cfg.get("ttf_gas_ticker", "TTF=F"),
            eua_ticker=fuel_cfg.get("eua_carbon_ticker", "KEUA"),
            lag_days=fuel_cfg.get("lag_days", 1),
        ),
    )
    df = df.join(fuel, how="left")

    # Forward/back-fill within available coverage. If a fuel column is entirely
    # absent (Yahoo ticker delisted), fill with 0 — the model still trains on
    # whichever fuel signals survived.
    for c in ["ttf_gas", "eua_carbon"]:
        if c in df.columns:
            if df[c].isna().all():
                print(f"  WARN  {c} is entirely NaN; filling with 0. The model will "
                      f"train without this feature.")
                df[c] = 0.0
            else:
                df[c] = df[c].ffill().bfill()
    return df


def save_raw(df: pd.DataFrame, raw_dir: str | Path, country: str = "GB") -> Path:
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{country}_hourly.parquet"
    df.to_parquet(path)
    return path


def load_raw(raw_dir: str | Path, country: str = "GB") -> pd.DataFrame:
    return pd.read_parquet(Path(raw_dir) / f"{country}_hourly.parquet")
