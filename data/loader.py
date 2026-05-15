"""
loader.py – Lightweight .mat file reader (MATLAB 5.0 / compressed).

Avoids scipy dependency by scanning the binary directly.
All variables are loaded into numpy arrays and cached on first call.
"""

import io
import struct
import zlib
from pathlib import Path
from functools import lru_cache

import numpy as np

# ── low-level MAT-5 element parser ──────────────────────────────────────────

def _parse_matrix(buf: bytes):
    """Return (name, ndarray) from a raw miMATRIX payload, or (name, None)."""
    f = io.BytesIO(buf)

    def _read_sub():
        raw = f.read(8)
        if len(raw) < 8:
            return None, None
        dt, nb = struct.unpack("<II", raw)
        if dt >> 16:                          # small-data format
            sdt = dt & 0xFFFF
            snb = (dt >> 16) & 0xFFFF
            return sdt, raw[4: 4 + snb]
        data = f.read(nb)
        pad = (8 - nb % 8) % 8
        if pad:
            f.read(pad)
        return dt, data

    fdt, fd = _read_sub()
    if fd is None:
        return None, None
    cls = struct.unpack_from("<I", fd, 0)[0] & 0xFF

    ddt, dd = _read_sub()
    if dd is None:
        return None, None
    ndim = len(dd) // 4
    dims = struct.unpack_from("<" + "I" * ndim, dd)

    ndt, nd = _read_sub()
    if nd is None:
        return None, None
    name = nd.decode("ascii", "replace").rstrip("\x00")

    if cls in (5, 6):                         # single / double
        pdt, pd = _read_sub()
        if pd is None:
            return name, None
        dtype = "float64" if cls == 6 else "float32"
        arr = np.frombuffer(pd, dtype=dtype).copy()
        total = 1
        for d in dims:
            total *= d
        if arr.size == total and len(dims) == 2:
            arr = arr.reshape(dims, order="F")
        return name, arr

    return name, None


def _scan_mat5(path: Path) -> dict:
    """Scan a MATLAB 5.0 .mat file and return all double/single arrays."""
    results: dict = {}
    body = path.read_bytes()[128:]          # skip 128-byte file header
    pos = 0

    while pos < len(body) - 8:
        dt, nb = struct.unpack_from("<II", body, pos)

        if dt >> 16:                        # small-data at top level — skip
            pos += 1
            continue

        if dt == 15:                        # miCOMPRESSED
            cdata = body[pos + 8: pos + 8 + nb]
            pad = (8 - nb % 8) % 8
            next_pos = pos + 8 + nb + pad
            try:
                ud = zlib.decompress(cdata)
                upos = 0
                while upos + 8 <= len(ud):
                    udt, unb = struct.unpack_from("<II", ud, upos)
                    if udt == 14:           # miMATRIX inside compressed block
                        name, val = _parse_matrix(ud[upos + 8: upos + 8 + unb])
                        if name and val is not None:
                            results[name] = val
                    upos += 8 + unb + (8 - unb % 8) % 8
            except Exception:
                pass
            pos = next_pos

        elif dt == 14:                      # miMATRIX uncompressed
            name, val = _parse_matrix(body[pos + 8: pos + 8 + nb])
            if name and val is not None:
                results[name] = val
            pad = (8 - nb % 8) % 8
            pos = pos + 8 + nb + pad

        else:
            pos += 1                        # advance byte-by-byte on unknown type

    return results


# ── public API ───────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_names():
    """Return (country_names, tradeable_names, all_sector_names) as lists."""
    def _read(fname):
        return (DATA_DIR / fname).read_text().splitlines()

    countries  = _read("countrynames.txt")
    tradeables = _read("tradeablesnames.txt")
    nontrade   = _read("nontradeablesnames.txt")
    return countries, tradeables, tradeables + nontrade


@lru_cache(maxsize=None)
def load_world():
    """Load world-tariff-change scenario results."""
    return _scan_mat5(DATA_DIR / "Resultados_contrafactual_world.mat")


@lru_cache(maxsize=None)
def load_bloc():
    """Load bloc-only tariff-change scenario results."""
    return _scan_mat5(DATA_DIR / "Resultados_contrafactual_bloc_only.mat")


def build_country_df(selected_indices=None):
    """
    Return a pandas DataFrame with one row per country and columns:

        country, income_world, income_bloc,
        exports_world, exports_bloc,
        tr_change_world, tr_change_bloc,
        wages_world, va_world, va_bloc
    """
    import pandas as pd

    countries, _, _ = load_names()
    W  = load_world()
    B  = load_bloc()

    n = len(countries)

    def _flat(d, key, n):
        v = d.get(key)
        if v is None:
            return np.full(n, np.nan)
        return v.flatten()[:n]

    df = pd.DataFrame({
        "country":        countries,
        # Real income change (factor → %)
        "income_world":   (_flat(W, "Income_all",       n) - 1) * 100,
        "income_bloc":    (_flat(B, "Income_oN",        n) - 1) * 100,
        # Export change (ratio → %)
        "exports_world":  (_flat(W, "change_exports_all", n) - 1) * 100,
        "exports_bloc":   (_flat(B, "change_exports_oN",  n) - 1) * 100,
        # Import change
        "imports_world":  (_flat(W, "change_imports_all", n) - 1) * 100,
        # Tariff revenue change (already a fraction of VA)
        "tr_change_world": _flat(W, "TR_change_all", n) * 100,
        "tr_change_bloc":  _flat(B, "TR_change_oN",  n) * 100,
        # Nominal wages (world scenario, factor → %)
        "wages_world":    (_flat(W, "wf0_all",  n) - 1) * 100,
        # Value added (absolute, counterfactual)
        "va_world":        _flat(W, "VAnp_all", n),
        "va_bloc":         _flat(B, "VAnp_oN",  n),
    })

    if selected_indices is not None:
        df = df.iloc[selected_indices].reset_index(drop=True)

    return df


def build_sector_df(country_idx: int, scenario: str = "world"):
    """
    Return a DataFrame with sector-level export values for one country.
    scenario: 'world' or 'bloc'
    """
    import pandas as pd

    _, tradeables, all_sectors = load_names()
    n_sectors = len(all_sectors)

    if scenario == "world":
        mat = load_world().get("Ejnp_all_out")
    else:
        mat = load_bloc().get("Ejnp_oN_out")

    if mat is None or country_idx >= mat.shape[1]:
        return pd.DataFrame()

    col = mat[:, country_idx][:n_sectors]
    df = pd.DataFrame({
        "sector":      all_sectors[:len(col)],
        "exports":     col,
        "tradeable":   ["Tradeable"] * len(tradeables) + ["Non-Tradeable"] * (len(col) - len(tradeables)),
    })
    return df.sort_values("exports", ascending=False)
