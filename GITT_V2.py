#!/usr/bin/env python3
"""
GITT App — multi-sample GUI
Run: python gitt_app.py --data_dir ~/dataanalysis/gittt/data
"""
import os, glob, argparse, warnings, json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from scipy import stats
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# ── Colours ──────────────────────────────────────────────────────────────────
SAMPLE_COLORS = ['#2E6FBA','#E07B39','#27AE60','#C0392B',
                 '#8E44AD','#16A085','#D4AC0D','#717D7E']
GREEN  = '#27AE60'
RED    = '#C0392B'
F_CONST = 96485.0

# ── Data functions ────────────────────────────────────────────────────────────
def is_eclab_single(path):
    """Return True if path is an EC-Lab all-in-one ASCII file.

    Detected by the presence of the string 'EC-Lab ASCII FILE' in the first
    two lines, which is the standard EC-Lab export header.
    """
    try:
        with open(path, encoding='latin-1', errors='replace') as f:
            head = f.read(200)
        return 'EC-Lab ASCII FILE' in head
    except Exception:
        return False


def parse_eclab_single(path):
    """Parse an EC-Lab all-in-one ASCII export file.

    These files contain both header metadata and all data columns
    (time, voltage, current, capacity) in a single file with format:
      EC-Lab ASCII FILE
      Nb header lines : N
      ... (metadata: mass, Mw, Mion, ne, Qmax, area, volume) ...
      time/s  Ewe/V  I/mA  Capacity/mA.h     <- column header at line N-1
      <data rows>                              <- data from line N onward

    Comma decimal separators (European) are handled automatically.
    Current is in mA — converted to A for consistency with the rest of the app.

    Returns
    -------
    td     : np.ndarray, shape (N,3)  — [V, I_A, T_s]  (same as load_time output)
    cap    : np.ndarray, shape (N,2)  — [Q_mAh, V]      (same as parse_cap output)
    meta   : dict with keys:
               mass_mg, Mw_g_mol, Mion_g_mol, ne, Qmax_mAh,
               area_cm2, vol_cm3  (all float or None if not found)
    """
    import re

    with open(path, encoding='latin-1', errors='replace') as f:
        raw = f.read()
    lines = raw.replace('\r\n', '\n').replace('\r', '\n').split('\n')

    # ── Locate header length ──────────────────────────────────────────────────
    n_hdr = None
    for l in lines[:10]:
        m = re.search(r'Nb header lines\s*:\s*(\d+)', l)
        if m:
            n_hdr = int(m.group(1))
            break
    if n_hdr is None:
        raise ValueError(f"Cannot find 'Nb header lines' in: {path}")

    # ── Parse metadata from header ────────────────────────────────────────────
    def _float(s):
        m2 = re.search(r'[\d]+[,\.][\d]+|[\d]+', s)
        return float(m2.group().replace(',', '.')) if m2 else None

    meta = {'mass_mg': None, 'Mw_g_mol': None, 'Mion_g_mol': None,
            'ne': 1, 'Qmax_mAh': None, 'area_cm2': None, 'vol_cm3': None}
    for l in lines[:n_hdr]:
        if 'Mass of active material' in l:
            meta['mass_mg'] = _float(l)
        elif 'Molecular weight of active material' in l:
            meta['Mw_g_mol'] = _float(l)
        elif 'Atomic weight of intercalated ion' in l:
            meta['Mion_g_mol'] = _float(l)
        elif 'Number of e-' in l:
            v = _float(l)
            if v is not None:
                meta['ne'] = int(v)
        elif 'Battery capacity' in l:
            meta['Qmax_mAh'] = _float(l)
        elif 'Electrode surface area' in l:
            meta['area_cm2'] = _float(l)
        elif 'Volume (V)' in l:
            meta['vol_cm3'] = _float(l)

    # ── Locate column header line (last header line before data) ──────────────
    hdr_line = lines[n_hdr - 1]
    cols = [c.lower().strip() for c in hdr_line.split('\t')]

    def _col(*keywords):
        for ki, c in enumerate(cols):
            if any(k in c for k in keywords):
                return ki
        return None

    col_t = _col('time', 'elapsed')
    col_v = _col('ewe', 'potential', 'voltage', 'ecell', 'e/v')
    col_i = _col('i/ma', 'current', 'iwe', 'i/a')
    col_q = _col('capacity', 'mah', 'ma.h', 'charge', '/q')

    missing = [n for n, c in [('time', col_t), ('voltage', col_v),
                               ('current', col_i), ('capacity', col_q)]
               if c is None]
    if missing:
        raise ValueError(
            f"EC-Lab single file missing columns {missing} in: {path}\n"
            f"  Found columns: {hdr_line.strip()}"
        )

    need = max(col_t, col_v, col_i, col_q) + 1
    td_rows, cap_rows = [], []
    for l in lines[n_hdr:]:
        p = l.strip().split('\t')
        p = [x for x in p if x.strip()]
        if len(p) < need:
            continue
        try:
            row = [float(x.replace(',', '.')) for x in p]
            # I in file is mA — convert to A for app consistency
            td_rows.append([row[col_v], row[col_i] / 1000.0, row[col_t]])
            cap_rows.append([row[col_q], row[col_v]])
        except Exception:
            pass

    if not td_rows:
        raise ValueError(f"No numeric data rows in EC-Lab single file: {path}")

    return np.array(td_rows), np.array(cap_rows), meta

def parse_time(path):
    """Parse a time-domain GITT file into a (N,3) array: col0=V, col1=I, col2=T.

    Scans every line until it finds a tab-separated header containing at least
    one recognisable keyword for voltage, current, AND time.  Columns are then
    reordered so the output is always [V, I, T] regardless of file column order.

    Voltage  keywords: potential, voltage, ewe, ecell, e/v
    Current  keywords: current, iwe, i/a
    Time     keywords: time, elapsed

    Raises ValueError if the header cannot be found (tells user which keywords
    to add so the problem is immediately actionable).
    """
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"Time file is empty: {path}")

    col_v = col_i = col_t = None
    data_start = len(lines)   # will be updated when header found

    for li, line in enumerate(lines):
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        # Try to identify all three columns from this line
        cv = ci = ct = None
        for ci_try, h in enumerate(parts):
            hl = h.lower().strip()
            if any(k in hl for k in ('potential', 'voltage', 'ewe', 'ecell', 'e/v')):
                cv = ci_try
            elif any(k in hl for k in ('current', 'iwe', 'i/a')):
                ci = ci_try
            elif any(k in hl for k in ('time', 'elapsed')):
                ct = ci_try
        if cv is not None and ci is not None and ct is not None:
            col_v, col_i, col_t = cv, ci, ct
            data_start = li + 1
            break

    if col_v is None or col_i is None or col_t is None:
        raise ValueError(
            f"Cannot identify Voltage/Current/Time columns in: {path}\n"
            f"  Header must contain keywords such as:\n"
            f"    Voltage/Potential/Ewe   Current/Iwe   Time/Elapsed\n"
            f"  (tab-separated, all three on the same line)"
        )

    data = []
    need = max(col_v, col_i, col_t) + 1
    for line in lines[data_start:]:
        p = line.strip().split('\t')
        if len(p) < need:
            continue
        try:
            row = [float(x.replace(',', '.')) for x in p]
            data.append([row[col_v], row[col_i], row[col_t]])
        except Exception:
            pass

    if not data:
        raise ValueError(f"No numeric data rows found in time file: {path}")
    return np.array(data)


def parse_cap(path):
    """Parse a capacity file into Q (mAh or mAh/g) and V arrays.

    Scans every line for a tab-separated header containing at least one
    recognisable keyword for capacity AND voltage.  Supports BioLogic format
    (file-path line + column-header line + data) and single-header formats.

    Capacity keywords: capacity, mah, ma.h, charge, q
    Voltage  keywords: ecell, ewe, potential, voltage, e/v

    Detects mAh/g units automatically from the capacity column header.
    Raises ValueError if columns cannot be identified.
    """
    with open(path, encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    if not lines:
        raise ValueError(f"Cap file is empty: {path}")

    col_q = col_v = None
    unit = 'mAh'
    data_start = len(lines)

    for li, line in enumerate(lines):
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        cq = cv = None
        for ci, h in enumerate(parts):
            hl = h.lower().strip()
            if any(k in hl for k in ('capacity', 'mah', 'ma.h', 'charge', '/q')):
                cq = ci
                if 'mah/g' in hl or 'ma.h/g' in hl:
                    unit = 'mAh/g'
            elif any(k in hl for k in ('ecell', 'ewe', 'potential', 'voltage', 'e/v')):
                cv = ci
        if cq is not None and cv is not None:
            col_q, col_v = cq, cv
            data_start = li + 1
            break

    if col_q is None or col_v is None:
        raise ValueError(
            f"Cannot identify Capacity/Voltage columns in: {path}\n"
            f"  Header must contain keywords such as:\n"
            f"    Capacity/mAh/Charge   Ecell/Ewe/Potential/Voltage\n"
            f"  (tab-separated on the same line)"
        )

    data = []
    need = max(col_q, col_v) + 1
    for line in lines[data_start:]:
        p = line.strip().split('\t')
        if len(p) < need:
            continue
        try:
            row = [float(x.replace(',', '.')) for x in p]
            data.append([row[col_q], row[col_v]])
        except Exception:
            pass

    if not data:
        raise ValueError(f"No numeric data rows found in cap file: {path}")
    a = np.array(data)
    return a[:, 0], a[:, 1], unit


def load_time(files):
    # Handles both EC-Lab single-file format and classic two-file format.
    # Single files (EC-Lab ASCII) contain time, voltage, current, capacity
    # in one file; classic format uses separate time and cap files.
    parts=[]; t_off=0.
    for path in sorted(files):
        if is_eclab_single(path):
            d, _cap, _meta = parse_eclab_single(path)
        else:
            d = parse_time(path)
        if d.size == 0: continue
        d[:,2] += t_off; t_off = d[-1,2] + 1.; parts.append(d)
    return np.vstack(parts) if parts else np.zeros((0,3))

def load_cap_with_segs(files, mass_mg=None):
    # For EC-Lab single-file format, extract capacity from the single file.
    # For classic format, parse dedicated cap files as before.
    # Check if any file is a single-format file
    single_files = [p for p in files if is_eclab_single(p)]
    if single_files:
        # Single-file mode: extract Q and V from the all-in-one file
        Qs, Vs, segs = [], [], []; uf = 'mAh'; q_offset = 0.0
        for path in sorted(single_files):
            _td, cap_arr, _meta = parse_eclab_single(path)
            Q = cap_arr[:, 0]; V = cap_arr[:, 1]
            seg_ends = list(np.where(np.diff(Q) < -0.01)[0] + 1) + [len(Q)]
            prev = 0
            for seg_end in seg_ends:
                seg_Q = Q[prev:seg_end]; seg_V = V[prev:seg_end]
                if len(seg_Q) < 20: prev = seg_end; continue
                seg_Q_off = seg_Q + q_offset
                Qs.append(seg_Q_off); Vs.append(seg_V)
                q_offset += seg_Q_off[-1]
                is_disc = seg_V[-1] < seg_V[0]
                segs.append((seg_Q_off.copy(), seg_V.copy(), is_disc))
                prev = seg_end
        Q_all = np.concatenate(Qs) if Qs else np.zeros(0)
        V_all = np.concatenate(Vs) if Vs else np.zeros(0)
        return Q_all, V_all, uf, segs
    # Classic two-file mode below
    """Load cap files, join into continuous Q, and return per-cycle segments.

    Returns: Q_cont, V_cont, unit, segs
    segs: list of (seg_Q, seg_V, is_discharge)
      is_discharge = True  if V decreases over segment (insertion / discharge)
      is_discharge = False if V increases over segment (deinsertion / charge)
    """
    Qs, Vs, segs = [], [], []; uf = 'mAh'; q_offset = 0.0
    for path in sorted(files):
        Q, V, unit = parse_cap(path)
        # Never convert mAh/g→mAh using mass: x = Q/Q_max is dimensionless
        # and mass cancels — the normalization is identical in both units.
        uf = unit
        seg_ends = list(np.where(np.diff(Q) < -0.01)[0] + 1) + [len(Q)]
        prev = 0
        for seg_end in seg_ends:
            seg_Q = Q[prev:seg_end]
            seg_V = V[prev:seg_end]
            if len(seg_Q) < 20:
                prev = seg_end; continue
            seg_Q_off = seg_Q + q_offset
            Qs.append(seg_Q_off); Vs.append(seg_V)
            # FIX Bug 3: advance offset by the offset-adjusted end value, not the raw value
            q_offset += seg_Q_off[-1]
            # Determine direction: V decreases = discharge (sodiation)
            is_disc = seg_V[-1] < seg_V[0]
            segs.append((seg_Q_off.copy(), seg_V.copy(), is_disc))
            prev = seg_end
    Q_all = np.concatenate(Qs) if Qs else np.zeros(0)
    V_all = np.concatenate(Vs) if Vs else np.zeros(0)
    return Q_all, V_all, uf, segs

def load_cap(files, mass_mg=None):
    """Backward-compat wrapper."""
    Q, V, u, _ = load_cap_with_segs(files, mass_mg=mass_mg)
    return Q, V, u

def _strip_boundary_artifact(Q, V, is_discharge):
    """Remove trailing points that reverse the expected V direction.

    BioLogic cap files often record one extra point at the end of each
    half-cycle segment that belongs to the next half-cycle (e.g. V jumps
    back up at the end of a discharge).  This artifact causes a spurious
    spike in the E vs x plot.  We remove at most 3 trailing points.
    """
    if len(Q) < 4:
        return Q, V
    sign = -1 if is_discharge else +1   # expected dV sign per data point
    cut = len(Q)
    while cut > 2 and (len(Q) - cut) < 3:
        dv = (V[cut-1] - V[cut-2]) * sign
        if dv > 0.02:    # correct direction — stop
            break
        if dv < -0.02:   # reversed — potential artifact
            cut -= 1
        else:
            break
    if cut < len(Q):
        return Q[:cut], V[:cut]
    return Q, V


def load_cap_segments(files, mass_mg=None, td=None):
    """Parse cap files and separate discharge / charge segments.

    Direction is determined by current sign from the time file (td):
      discharge = negative current  (cathodic, insertion)  -- works for both anode and cathode
      charge    = positive current  (anodic,  deinsertion)

    If td is not provided, falls back to voltage-trend heuristic (legacy, anode-only).

    Returns
    -------
    disc : dict  {'Q': array, 'V': array, 'Q_max': float, 'unit': str}
    chg  : dict  same structure for the longest charge segment (or None)
    unit : str   'mAh' or 'mAh/g'
    """
    disc_parts_Q=[]; disc_parts_V=[]; chg_candidates=[]
    unit_final='mAh'; q_disc=0.; seg_idx=0  # unit_final only upgraded to mAh/g, never reset

    # If time data available, determine which half-cycles are discharge
    # by computing the dominant current sign in each half-cycle window.
    # Strategy: use td to find the current sign of each half-cycle in order.
    # Cap file segments map sequentially to half-cycles in the time file.
    # We find the sign by looking at the median non-zero current in each
    # time-file segment (separated by sign changes).
    if td is not None and td.shape[0] > 10:
        # File format: [V, I, T]  →  col0=V, col1=I, col2=T
        I_td = td[:,1]; T_td = td[:,2]
        # Find half-cycle boundaries in time file: where current flips sign
        nz = I_td != 0
        signs = np.sign(I_td[nz])
        # sign change positions in the nz-filtered array
        sc = np.where(np.diff(signs) != 0)[0] + 1
        # median current of each half-cycle
        boundaries = [0] + list(sc) + [len(signs)]
        nz_idx = np.where(nz)[0]
        hc_signs = []
        for a,b in zip(boundaries[:-1], boundaries[1:]):
            seg_I = I_td[nz_idx[a:b]]
            if len(seg_I): hc_signs.append(np.sign(np.median(seg_I)))
        # hc_signs[i] = -1 (discharge) or +1 (charge) for half-cycle i
    else:
        hc_signs = None

    for path in sorted(files):
        Q,V,unit=parse_cap(path)
        # Never convert mAh/g→mAh: x = Q/Q_max is dimensionless regardless of unit.
        if unit=='mAh/g': unit_final='mAh/g'

        seg_ends=list(np.where(np.diff(Q)<-0.01)[0]+1)+[len(Q)]
        prev=0
        for seg_end in seg_ends:
            sQ=Q[prev:seg_end]; sV=V[prev:seg_end]
            if len(sQ)<10: prev=seg_end; seg_idx+=1; continue
            Q_rel=sQ-sQ[0]
            # Determine direction: use time-file current sign if available
            # Direction: primary = hc_signs from time file; override with voltage
            # trend when more cap segments than half-cycles (split files where one
            # half-cycle spans multiple cap files causes wrong hc_sign assignment).
            v_trend_disc = sV[-1] < sV[0]   # V going down = discharge (both anodes/cathodes)
            v_range = float(sV.max() - sV.min())
            if hc_signs and seg_idx < len(hc_signs):
                hc_dir = 'discharge' if hc_signs[seg_idx] < 0 else 'charge'
                # Trust voltage trend when it's unambiguous (>0.05V swing) —
                # this correctly handles split cap files where one discharge
                # spans C1 (hc_sign=-1, correct) + C2_seg0 (hc_sign=+1, wrong).
                direction = ('discharge' if v_trend_disc else 'charge') if v_range > 0.05 else hc_dir
            else:
                direction = 'discharge' if v_trend_disc else 'charge'
            if direction=='discharge':
                disc_parts_Q.append(Q_rel+q_disc)
                disc_parts_V.append(sV)
                q_disc+=Q_rel[-1]
            else:
                chg_candidates.append({'Q':Q_rel,'V':sV,'Q_max':Q_rel[-1]})
            prev=seg_end; seg_idx+=1

    if disc_parts_Q:
        dQ=np.concatenate(disc_parts_Q); dV=np.concatenate(disc_parts_V)
        dQ,dV=_strip_boundary_artifact(dQ,dV,is_discharge=True)
        disc={'Q':dQ,'V':dV,'Q_max':float(dQ[-1]) if len(dQ) else 0.,'unit':unit_final}
    else:
        disc={'Q':np.zeros(0),'V':np.zeros(0),'Q_max':0.,'unit':unit_final}

    if chg_candidates:
        best=max(chg_candidates,key=lambda s:s['Q_max'])
        bQ,bV=_strip_boundary_artifact(best['Q'],best['V'],is_discharge=False)
        chg={'Q':bQ,'V':bV,'Q_max':float(bQ[-1]) if len(bQ) else 0.,'unit':unit_final}
    else:
        chg=None

    return disc, chg, unit_final


def find_files(folder):
    """Locate GITT data files in a folder.

    Supports two formats automatically:
      1. EC-Lab single-file ASCII (.txt or .dat with EC-Lab header)
         All data is in one file: time, voltage, current, capacity + metadata.
         Returned as BOTH tf and cf; loaders detect the format internally.
      2. Classic BioLogic two-file format: *time*.dat + *cap*.dat.
    """
    try:
        all_f = sorted(os.listdir(folder))
    except OSError:
        return [], []
    # Priority 1: EC-Lab all-in-one single files
    singles = []
    for f in all_f:
        if f.endswith('.txt') or f.endswith('.dat'):
            fp = os.path.join(folder, f)
            try:
                if is_eclab_single(fp):
                    singles.append(fp)
            except Exception:
                pass
    if singles:
        # Return the same list for both tf and cf; loaders handle this
        return singles, singles
    # Priority 2: classic two-file BioLogic format
    tf = [os.path.join(folder,f) for f in all_f
          if 'time' in f.lower() and f.endswith('.dat')]
    cf = [os.path.join(folder,f) for f in all_f
          if 'cap' in f.lower() and f.endswith('.dat')]
    return sorted(tf), sorted(cf)

def _current_threshold(I):
    """Return an amplitude threshold that separates pulse current from OCP rest.

    Works for both cases:
    - Instrument records exactly 0 A during relaxation (exact zero)
    - Instrument records a small leakage current during relaxation (near-zero)

    Strategy: take the median of the NON-ZERO |I| values — this is the pulse
    current magnitude regardless of what fraction of points are zero.
    Threshold = 5% of that median.  If all values are zero, return 1e-12.
    """
    absI = np.abs(I)
    nonzero = absI[absI > 0]
    if len(nonzero) == 0:
        return 1e-12
    pulse_mag = float(np.median(nonzero))
    return max(pulse_mag * 0.05, 1e-12)


def detect_pulse_duration(td, min_pulses=3):
    """Estimate the GITT pulse duration (tau) from the time-domain current signal.

    Uses a threshold (5% of median pulse current) to distinguish galvanostatic
    pulses from OCP relaxation — works even when the instrument records a small
    non-zero leakage current during rest instead of exactly 0 A.

    Returns float tau in seconds (rounded to nearest second), or None if fewer
    than min_pulses pulses are found.
    """
    if td is None or td.shape[0] < 10:
        return None
    # Array layout after parse_time: col0=V, col1=I, col2=T
    I, T = td[:, 1], td[:, 2]
    thresh = _current_threshold(I)
    active = (np.abs(I) > thresh).astype(int)   # 1 = pulse, 0 = relaxation
    starts = np.where(np.diff(active) == 1)[0] + 1   # relaxation→pulse
    ends   = np.where(np.diff(active) == -1)[0]       # pulse→relaxation
    # If recording started mid-pulse, drop that leading partial pulse
    if len(starts) == 0 or len(ends) == 0:
        return None
    if ends[0] < starts[0]:
        ends = ends[1:]
    if len(starts) == 0 or len(ends) == 0:
        return None
    n = min(len(starts), len(ends))
    starts, ends = starts[:n], ends[:n]
    durations = T[ends] - T[starts]
    # Keep only pulses >= 10 s (filters glitches/transients)
    durations = durations[durations >= 10.]
    if len(durations) < min_pulses:
        return None
    # Round to nearest second — works for any length (30 s, 600 s, 1800 s…)
    tau = round(float(np.median(durations)))
    return max(tau, 10.)


def extract_blocks(td, tfs=60., tfe=540., min_relax=None, tau_s=600.):
    """Extract GITT pulse+relaxation block pairs from the time-domain array.

    min_relax: minimum relaxation duration in seconds to accept a block.
               Defaults to tau_s (the pulse duration) — relaxation should be
               at least as long as the pulse for a valid GITT measurement.
               Set explicitly to override.

    Uses a current threshold (5% of median pulse current) to distinguish
    pulses from OCP relaxation — works when the instrument records a small
    non-zero leakage current during rest rather than exactly 0 A.
    """
    if td.size==0: return []
    # Array layout after parse_time: col0=V, col1=I, col2=T
    V,I,T=td[:,0],td[:,1],td[:,2]
    if min_relax is None:
        min_relax = tau_s          # relaxation must be at least as long as pulse
    thresh = _current_threshold(I)
    active = np.abs(I) > thresh   # True = galvanostatic pulse, False = OCP rest
    # Find transitions from pulse → relaxation (active→inactive)
    tr=np.where(np.diff(active.astype(int))==-1)[0]  # last pulse index before rest
    # If the file starts mid-pulse (active[0] is True), the first transition has
    # no real preceding relaxation — we don't know how long the pulse ran before
    # recording started.  Mark this so we can skip it below.
    file_starts_mid_pulse = len(active) > 0 and active[0]
    blocks=[]
    for idx in tr:
        ps=idx
        while ps>0 and active[ps-1]: ps-=1   # walk back to start of pulse
        # Skip this block if it's the partial first pulse (ps reached row 0
        # because the file started mid-pulse — pulse start time is unknown).
        if ps==0 and file_starts_mid_pulse:
            continue
        end=idx+1
        while end<len(I) and not active[end]: end+=1  # walk forward to end of rest
        # ── Guard: stop relaxation window if current sign flips ──────────────
        # The last pulse of a discharge (or charge) is followed by the first
        # pulse of the opposite half-cycle with no gap between them.
        # When that happens `end` lands on a pulse of opposite sign, so the
        # "relaxation" window [idx+1 : end] actually contains the start of the
        # next half-cycle and V_eq is taken from the wrong phase.
        # Fix: find the sign of this pulse, then trim `end` back to the first
        # point where the current sign changes (even while still below thresh).
        pulse_sign = np.sign(I[ps])          # +1 = charge, -1 = discharge
        relax_slice = I[idx+1:end]           # near-zero current during rest
        if len(relax_slice) > 0:
            # Any sample inside the rest window whose sign is opposite to the
            # pulse AND whose magnitude exceeds thresh means the next half-cycle
            # has already started — cut the window just before that point.
            for rel_i, i_val in enumerate(relax_slice):
                if np.abs(i_val) > thresh and np.sign(i_val) != pulse_sign:
                    end = idx + 1 + rel_i   # trim: exclude this point and beyond
                    break
        Vr=V[idx+1:end]; Tr=T[idx+1:end]
        Vp=V[ps:idx+1]; Tp=T[ps:idx+1]
        if len(Tr)<2 or Tr[-1]-Tr[0]<min_relax or len(Vp)<3: continue
        t_rel=Tp-Tp[0]; fm=(t_rel>0)&(t_rel>=tfs)&(t_rel<=tfe); sq=np.sqrt(t_rel)  # t_rel>0 always excludes IR-drop point at t=0
        # Fall back to all t_rel>0 pts when the window [tfs,tfe] has <3 pts
        # (happens with sparsely sampled data like G172 ~3 pts per pulse)
        if fm.sum() < 3:
            fm = t_rel > 0
        sl=r2=np.nan
        if fm.sum()>=2: sl,_,r,*_=stats.linregress(sq[fm],Vp[fm]); r2=r**2
        t_rx=np.maximum(Tr-Tr[0],0)
        xkc=np.sqrt(t_rx+tau_s)-np.sqrt(t_rx)
        sl_kc=r2_kc=np.nan
        if len(xkc)>=4: sl_kc,_,r,*_=stats.linregress(xkc,Vr); r2_kc=r**2
        # R_el: ohmic step at current switch-on.
        # Only valid when V[ps-1] is a genuine relaxation point immediately
        # before the pulse (time gap ≤ 3× local sampling interval).
        # Sparse data (e.g. G172, ~160 s/pt) can have a large gap so we skip.
        if ps > 0 and abs(I[ps]) > 1e-9:
            dt_step  = T[ps] - T[ps-1]           # time between last relax & first pulse pt
            # Estimate typical sampling interval from the pulse itself
            dt_pulse = (Tp[-1]-Tp[0])/(len(Tp)-1) if len(Tp)>1 else dt_step
            if dt_step <= 3.0 * dt_pulse:         # contiguous — safe to use
                R_el = abs(Vp[0] - V[ps-1]) / abs(I[ps])
            else:                                  # gap too large — not a clean step
                R_el = np.nan
        else:
            R_el = np.nan
        blocks.append({
            'V_eq':Vr[-1],'sl_sqrt':sl,'r2_sqrt':r2,'n_fit':int(fm.sum()),
            'sl_kc':sl_kc,'r2_kc':r2_kc,'R_el':R_el,'I_pulse':I[ps],
            'cycle':'discharge' if I[ps]<0 else 'charge',
            '_Vp':Vp,'_Tp':Tp,'_Vr':Vr,'_Tr':Tr,
            '_t_rel':t_rel,'_sqrt_t':sq,'_fm':fm,'_x_kc':xkc,
        })
    # ── Safety net: remove boundary blocks whose V_eq is inconsistent ───────
    # The sign-change guard above handles most cases.  This pass catches any
    # residual boundary block whose relaxation was still contaminated (e.g.
    # the instrument records a brief zero-current transient between half-cycles
    # that slips through the threshold, so `end` advances one point too far).
    #
    # Rule: for each direction, if the LAST block's V_eq moves in the WRONG
    # direction (charge V_eq should increase; discharge V_eq should decrease)
    # AND the block immediately following it belongs to the opposite half-cycle
    # (or there is no following block), drop the last block of that direction.
    for cyc, wrong_direction in [('charge',    lambda last, prev: last < prev),
                                  ('discharge', lambda last, prev: last > prev)]:
        cyc_blocks = [b for b in blocks if b['cycle'] == cyc]
        if len(cyc_blocks) < 2:
            continue
        last_b = cyc_blocks[-1]
        prev_b = cyc_blocks[-2]
        last_idx = blocks.index(last_b)
        next_b = blocks[last_idx + 1] if last_idx + 1 < len(blocks) else None
        at_boundary = (next_b is None or next_b['cycle'] != cyc)
        if wrong_direction(last_b['V_eq'], prev_b['V_eq']) and at_boundary:
            blocks = [b for b in blocks if b is not last_b]
    return blocks

def build_dedx(disc):
    """Build dE/dx interpolator from the merged discharge V/Q arrays.

    Uses the raw cap data only to establish the V(x) relationship, but
    applies heavy smoothing and unique-x filtering to suppress GITT pulse
    artefacts.  A separate, spike-free interpolator is later rebuilt from
    equilibrium V_eq block values once those are available (see
    build_dedx_from_blocks).
    """
    Q_ref=disc['Q']; V_ref=disc['V']; Q_max=disc['Q_max']
    if Q_max<1e-9 or len(Q_ref)<10: return None,1.
    x_ref=Q_ref/Q_max
    idx=np.argsort(x_ref); xs=x_ref[idx]; Vs=V_ref[idx]
    _,ui=np.unique(xs,return_index=True); xs=xs[ui]; Vs=Vs[ui]
    # Heavy smoothing to suppress GITT pulse steps in the cap file
    wl=min(101,len(xs)//4*2+1); wl=wl if wl%2==1 else wl+1
    Vs_sm=savgol_filter(Vs,wl,2) if len(Vs)>wl else Vs
    dV=np.gradient(Vs_sm); dx=np.gradient(xs)
    with np.errstate(divide='ignore',invalid='ignore'):
        dEdx=np.where(np.abs(dx)>1e-10,dV/dx,np.nan)
    valid=np.isfinite(dEdx)
    if valid.sum()<3: return None,Q_max
    return interp1d(Vs_sm[valid],dEdx[valid],bounds_error=False,
                    fill_value='extrapolate',kind='linear'),Q_max

def build_dedx_from_blocks(blocks):
    """Build a spike-free dE/dx interpolator from block V_eq values.

    The GITT equilibrium voltages are already clean relaxed OCV points —
    differentiating V_eq vs SOC index gives smooth, artefact-free dEdx.
    This replaces the cap-file-based interpolator once blocks are computed.
    Returns (interp_discharge, interp_charge) — either may be None.
    """
    def _make(blist, descending=True):
        blist=[b for b in blist if np.isfinite(b.get('V_eq',np.nan))]
        if len(blist)<4: return None
        blist=sorted(blist, key=lambda b:b['V_eq'], reverse=descending)
        Vs=np.array([b['V_eq'] for b in blist])
        xs=np.linspace(0.,1.,len(Vs))
        wl = min(9, len(Vs) - 1)           # must be < len(Vs)
        wl = max(3, wl if wl % 2 == 1 else wl - 1)   # must be odd and >= polyorder+1
        Vs_sm=savgol_filter(Vs,wl,2) if len(Vs)>wl else Vs
        dEdx=np.gradient(Vs_sm)/np.gradient(xs)
        # Clip wild edge artefacts — only if array is long enough
        if len(dEdx) > 4:
            dEdx[:2]=dEdx[2]; dEdx[-2:]=dEdx[-3]
        return interp1d(Vs_sm,dEdx,bounds_error=False,fill_value='extrapolate',kind='linear')
    disc_blocks=[b for b in blocks if b['cycle']=='discharge']
    chg_blocks =[b for b in blocks if b['cycle']=='charge']
    return _make(disc_blocks,descending=True), _make(chg_blocks,descending=False)

def compute_d(blocks,dedx_interp,I_app,delta_e=1.):
    pref=4*I_app**2/(delta_e**2*F_CONST**2*np.pi)
    # Compute robust minimum slope threshold: median |slope| / 100
    # Blocks with |slope| below this are almost certainly noise (flat pulse/relax)
    # and would produce astronomically large D values that are pure artefacts.
    sl_vals=[abs(b['sl_sqrt']) for b in blocks
             if np.isfinite(b.get('sl_sqrt',np.nan)) and abs(b.get('sl_sqrt',0))>1e-12]
    sl_kc_vals=[abs(b['sl_kc']) for b in blocks
                if np.isfinite(b.get('sl_kc',np.nan)) and abs(b.get('sl_kc',0))>1e-12]
    sl_thresh    = np.median(sl_vals)   /100. if sl_vals    else 1e-8
    sl_kc_thresh = np.median(sl_kc_vals)/100. if sl_kc_vals else 1e-8

    for b in blocks:
        sl=b.get('sl_sqrt',np.nan)
        if np.isfinite(sl) and abs(sl)>sl_thresh:
            dEdx=dedx_interp(b['V_eq']) if dedx_interp else np.nan
            b['dEdx']=float(dEdx) if np.isfinite(dEdx) else np.nan
            b['D_conv']=(pref*(b['dEdx']/sl)**2
                         if np.isfinite(b.get('dEdx',np.nan)) and abs(b.get('dEdx',0))>1e-8
                         else np.nan)
        else:
            b['dEdx']=b['D_conv']=np.nan
        sl_kc=b.get('sl_kc',np.nan)
        if np.isfinite(sl_kc) and abs(sl_kc)>sl_kc_thresh:
            # D_kc uses identical formula to D_conv, but s_xi replaces s_sqrt(t)
            # D_kc = (4*I^2 / (pi * F^2 * delta_e^2)) * (dVeq/dx / s_xi)^2
            dEdx_kc = b.get('dEdx', np.nan)   # already computed above if sl_sqrt valid
            if not np.isfinite(dEdx_kc):      # compute it if D_conv path was skipped
                dEdx_kc = dedx_interp(b['V_eq']) if dedx_interp else np.nan
                dEdx_kc = float(dEdx_kc) if np.isfinite(dEdx_kc) else np.nan
            b['D_kc'] = (pref * (dEdx_kc / sl_kc)**2
                         if np.isfinite(dEdx_kc) and abs(dEdx_kc) > 1e-8
                         else np.nan)
        else:
            b['D_kc'] = np.nan
        b['D_rel'] = b['D_conv']

def rmed(veqs,vals,frac=0.15):
    if len(veqs)<3: return list(veqs),list(vals)
    sv=sorted(zip(veqs,vals)); vv,dd=zip(*sv)
    # Bug 6 fix: keep as numpy arrays so slicing into np.median is fast and correct
    vv=np.array(vv); dd=np.array(dd)
    w=max(3,int(len(vv)*frac))
    return vv.tolist(),[float(np.median(dd[max(0,i-w//2):i+w//2+1])) for i in range(len(dd))]

# ── App ───────────────────────────────────────────────────────────────────────
class GITTApp:
    def __init__(self, root, default_data_dir=None):
        self.root=root
        self.root.title('GITT Analysis App')
        self.root.configure(bg='white')

        # Parameters
        self.data_dir    = tk.StringVar(value=default_data_dir or '')
        self.out_dir     = tk.StringVar(value='')
        self.mass_mg     = tk.DoubleVar(value=7.05)
        self.tau_s       = tk.DoubleVar(value=600.)
        self.t_fit_start = tk.DoubleVar(value=60.)
        self.t_fit_end   = tk.DoubleVar(value=540.)   # overwritten by auto-detect
        self.vlo         = tk.DoubleVar(value=0.0)    # overwritten by auto-detect
        self.vhi         = tk.DoubleVar(value=5.0)    # overwritten by auto-detect
        self.min_r2      = tk.DoubleVar(value=0.85)
        self.delta_e     = tk.DoubleVar(value=1.0)
        self.sigma_clip  = tk.DoubleVar(value=0.0)   # 0 = off
        self.xi_min      = tk.DoubleVar(value=0.0)   # KC fit range min
        self.xi_max      = tk.DoubleVar(value=99.)   # KC fit range max
        self.pulse_from  = tk.IntVar(value=1)         # overlay range start
        self.pulse_to    = tk.IntVar(value=999)       # overlay range end (clamped to n_blocks)
        self.dedx_mode   = tk.StringVar(value='differential')
        self.reverse_disc = tk.BooleanVar(value=True)   # pulse #1 = deepest discharge  # 'differential' | 'finite_diff'
        self.reverse_mode  = tk.StringVar(value='discharge')  # 'none'|'discharge'|'charge'
        self.trim_tlo      = tk.DoubleVar(value=0.0)   # exclude data with time (h) between tlo and thi
        self.trim_thi      = tk.DoubleVar(value=0.0)   # 0=disabled
        # Per-sample per-direction pulse exclusion: name -> {'disc': StringVar, 'chg': StringVar}
        # StringVar holds comma-separated 1-based pulse numbers, e.g. '5,12'
        self._exclude_pulses = {}

        # Sample state: name -> {td, cap_Q, cap_V, cap_unit, blocks, dedx}
        self._samples = {}
        # Checkboxes: name -> BooleanVar
        self._chk_vars = {}
        # Per-sample controls: name -> {'mass': DoubleVar, 'q_norm': StringVar}
        self._sample_controls = {}
        self._chk_widgets = {}
        # Curve data for export: list of dicts
        self._curve_data = []
        # Flag: has tau been auto-set from pulse data for this data directory?
        self._tau_auto_set = False

        self._debounce_id = None
        self._build_ui()

        if default_data_dir:
            self._scan_data_dir()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        # Top bar
        top=tk.Frame(self.root,bg='white',pady=4,padx=8)
        top.pack(side='top',fill='x')
        tk.Label(top,text='Data:',bg='white',font=('sans',10)).pack(side='left')
        tk.Entry(top,textvariable=self.data_dir,width=26,font=('sans',10)).pack(side='left',padx=4)
        tk.Button(top,text='Browse...',command=self._browse,bg='#eee',relief='flat',padx=6).pack(side='left')
        tk.Label(top,text='  Output:',bg='white',font=('sans',10)).pack(side='left')
        tk.Entry(top,textvariable=self.out_dir,width=26,font=('sans',10)).pack(side='left',padx=4)
        tk.Button(top,text='Browse...',command=self._browse_out,bg='#eee',relief='flat',padx=6).pack(side='left')

        self.status_lbl=tk.Label(top,text='No data loaded',bg='white',
                                  fg='gray',font=('sans',9))
        self.status_lbl.pack(side='left',padx=12)

        # Sample checkboxes panel (populated dynamically)
        chk_outer = tk.LabelFrame(self.root, text='Samples', bg='white',
                                   font=('sans', 9, 'bold'), padx=0, pady=3)
        chk_outer.pack(side='top', fill='x', padx=8, pady=(0, 2))
        btn_col = tk.Frame(chk_outer, bg='white')
        btn_col.pack(side='left', padx=4)
        tk.Button(btn_col, text='All', font=('sans', 8), relief='flat',
                  bg='#ddd', command=self._check_all).pack(anchor='w', pady=1)
        tk.Button(btn_col, text='None', font=('sans', 8), relief='flat',
                  bg='#ddd', command=self._check_none).pack(anchor='w', pady=1)
        _chk_cv = tk.Canvas(chk_outer, bg='white', height=54, highlightthickness=0)
        _chk_sb = tk.Scrollbar(chk_outer, orient='horizontal', command=_chk_cv.xview)
        _chk_cv.configure(xscrollcommand=_chk_sb.set)
        _chk_sb.pack(side='bottom', fill='x')
        _chk_cv.pack(side='left', fill='both', expand=True)
        self.chk_frame = tk.Frame(_chk_cv, bg='white')
        _chk_cv.create_window((0, 0), window=self.chk_frame, anchor='nw')
        def _chk_upd(e=None):
            _chk_cv.configure(scrollregion=_chk_cv.bbox('all'))
            _chk_sb.pack_forget() if self.chk_frame.winfo_reqwidth() <= _chk_cv.winfo_width()                 else _chk_sb.pack(side='bottom', fill='x')
        self.chk_frame.bind('<Configure>', _chk_upd)
        chk_outer.bind_all('<Shift-MouseWheel>',
            lambda e: _chk_cv.xview_scroll(int(-1*(e.delta/120)), 'units'))

        # Parameters
        pf=tk.LabelFrame(self.root,text='Parameters',bg='white',
                          font=('sans',10,'bold'),padx=8,pady=3)
        pf.pack(side='top',fill='x',padx=8,pady=(0,3))
        params=[
            ('tau (s)',         self.tau_s,      60., 3600.,60.),
            ('Fit start (s)', self.t_fit_start, 0., 590., 10.),
            ('Fit end (s)',   self.t_fit_end,   10.,7200.,10.),
            ('V_lo (V)',      self.vlo,         0.,  5.0, 0.05),
            ('V_hi (V)',      self.vhi,         0.,  5.0, 0.05),
            ('Min R2',        self.min_r2,      0.,  1.0, 0.05),
            ('delta_e',            self.delta_e,     1.,  4.,  1.),
            ('sigma-clip (0=off)', self.sigma_clip,  0.,  5.0, 0.5),
            ('KC xi_min',          self.xi_min,      0.,  30.,  0.5),
            ('KC xi_max',          self.xi_max,      0.,  99.,  0.5),
        ]
        for col,(label,var,lo,hi,step) in enumerate(params):
            fr=tk.Frame(pf,bg='white'); fr.grid(row=0,column=col,padx=5,sticky='w')
            tk.Label(fr,text=label,bg='white',font=('sans',8)).pack(anchor='w')
            tk.Spinbox(fr,textvariable=var,from_=lo,to=hi,increment=step,
                       width=7,font=('sans',9)).pack()
            # No auto-fire — changes take effect on Replot button only

        # Pulse selector — for Conv. fits and K&C fits tabs
        pls_fr=tk.Frame(pf,bg='white'); pls_fr.grid(row=0,column=len(params),padx=5,sticky='w')
        tk.Label(pls_fr,text='Pulse #',bg='white',font=('sans',8)).pack(anchor='w')
        self._pulse_var=tk.IntVar(value=1)
        self._pulse_spin=tk.Spinbox(pls_fr,textvariable=self._pulse_var,
                                    from_=1,to=200,increment=1,width=5,
                                    font=('sans',9),
                                    command=self._on_pulse_change)
        self._pulse_spin.pack()
        # Single trigger only via command= on spinbox (trace_add removed to avoid double-fire)

        # Pulse range for overlay tabs (all_pulses, all_xi, overpot_curves, relax_curves)
        rng_fr=tk.Frame(pf,bg='white'); rng_fr.grid(row=0,column=len(params)+1,padx=5,sticky='w')
        tk.Label(rng_fr,text='Show pulses',bg='white',font=('sans',8)).pack(anchor='w')
        rng_inner=tk.Frame(rng_fr,bg='white'); rng_inner.pack()
        tk.Label(rng_inner,text='from',bg='white',font=('sans',8)).pack(side='left')
        self._pfrom_spin=tk.Spinbox(rng_inner,textvariable=self.pulse_from,
                                    from_=1,to=999,increment=1,width=4,
                                    font=('sans',9),command=self._replot_debounced)
        self._pfrom_spin.pack(side='left',padx=2)
        tk.Label(rng_inner,text='to',bg='white',font=('sans',8)).pack(side='left')
        self._pto_spin=tk.Spinbox(rng_inner,textvariable=self.pulse_to,
                                  from_=1,to=999,increment=1,width=4,
                                  font=('sans',9),command=self._replot_debounced)
        self._pto_spin.pack(side='left',padx=2)
        # Also fire on keyboard entry (Return key and variable write)
        self._pfrom_spin.bind('<Return>',   lambda e: self._replot_debounced())
        self._pto_spin.bind('<Return>',     lambda e: self._replot_debounced())
        self._pfrom_spin.bind('<FocusOut>', lambda e: self._replot_debounced())
        self._pto_spin.bind('<FocusOut>',   lambda e: self._replot_debounced())
        # Note: trace_add NOT used — command= already fires on arrow clicks
        # Adding trace_add as well causes a double replot on every arrow click

        # dE/dx mode selector
        dx_fr=tk.Frame(pf,bg='white'); dx_fr.grid(row=0,column=len(params)+3,padx=5,sticky='w')
        tk.Label(dx_fr,text='dE/dx mode',bg='white',font=('sans',8)).pack(anchor='w')
        for mode_val, mode_lbl in [('differential','Differential\n(smooth dV/dx)'),
                                    ('finite_diff','Finite diff\n(dE/dx blocks)')]:
            tk.Radiobutton(dx_fr,text=mode_lbl,variable=self.dedx_mode,value=mode_val,
                           bg='white',font=('sans',7),justify='left').pack(anchor='w')

        rev_fr=tk.Frame(pf,bg='white'); rev_fr.grid(row=0,column=len(params)+2,padx=5,sticky='w')
        tk.Label(rev_fr,text='Reverse pulses',bg='white',font=('sans',8)).pack(anchor='w')
        for rv,rl in [('none','None'),('discharge','Discharge'),('charge','Charge'),('both','Both')]:
            tk.Radiobutton(rev_fr,text=rl,variable=self.reverse_mode,value=rv,
                           bg='white',font=('sans',7),justify='left',
                           command=self._on_pulse_change).pack(anchor='w')

        trim_fr=tk.Frame(pf,bg='white'); trim_fr.grid(row=0,column=len(params)+4,padx=5,sticky='w')
        tk.Label(trim_fr,text='Trim time (h)',bg='white',font=('sans',8)).pack(anchor='w')
        trim_inner=tk.Frame(trim_fr,bg='white'); trim_inner.pack()
        tk.Label(trim_inner,text='from',bg='white',font=('sans',7)).pack(side='left')
        tk.Spinbox(trim_inner,textvariable=self.trim_tlo,from_=0.,to=9999.,increment=1.,
                   width=5,font=('sans',8)).pack(side='left',padx=1)
        tk.Label(trim_inner,text='to',bg='white',font=('sans',7)).pack(side='left')
        tk.Spinbox(trim_inner,textvariable=self.trim_thi,from_=0.,to=9999.,increment=1.,
                   width=5,font=('sans',8)).pack(side='left',padx=1)
        tk.Label(trim_fr,text='(0,0=off)',bg='white',fg='#999',font=('sans',7)).pack(anchor='w')

        btn_fr=tk.Frame(pf,bg='white'); btn_fr.grid(row=0,column=len(params)+5,padx=8)
        tk.Button(btn_fr,text='Replot',command=self._replot,
                  bg='#1a6b35',fg='white',font=('sans',10,'bold'),
                  relief='flat',padx=8,pady=2).pack(pady=2)
        tk.Button(btn_fr,text='Export figures',command=self._export_figs,
                  bg='#2c5f8a',fg='white',font=('sans',8),
                  relief='flat',padx=6,pady=2).pack(pady=2)
        tk.Button(btn_fr,text='Export curves',command=self._export_curves,
                  bg='#5c3380',fg='white',font=('sans',8),
                  relief='flat',padx=6,pady=2).pack(pady=2)

        # Notebook
        self.tabs={}

        # ── Section definitions ────────────────────────────────────────────────
        SECTIONS = [
            ('raw',   'Raw Data',                    '#2c5f8a', '#d6e8f5'),
            ('anal',  'Analysis',                    '#5c3380', '#e8d6f5'),
            ('conv',  'Weppner & Huggins (Conv.)',   '#1a6b35', '#d4eede'),
            ('kc',    'Kang & Chueh (K&C)',          '#8a2c2c', '#f5d6d6'),
            ('docs',  'Documentation',               '#4a4a4a', '#e8e8e8'),
        ]
        TAB_SECTIONS = [
            # Raw data
            ('gitt_full',      'GITT curve',            'raw'),
            # Analysis — overview and diagnostics
            ('ocv',            'OCV',                   'anal'),
            ('overpot_curves', 'Overpotential curves',  'anal'),
            ('overpot',        'Overpotential & Rtot',  'anal'),
            ('relax_curves',   'Relaxation curves',     'anal'),
            ('relax_delta',    'Relaxation dE',         'anal'),
            ('relax_kinetics',    'Relax kinetics',        'anal'),
            ('relax_kinetics_pp', 'Relax kinetics / pulse','anal'),
            ('relax_map',         'Relax kinetics map',    'anal'),
            ('relax_map_soc',      'Relax map vs SOD/SOC',  'anal'),
            # Weppner & Huggins (conventional, pulse-based)
            ('all_pulses',     'All pulses (sqt)',      'conv'),
            ('sqrt_fits',      'sqrt(t) fits',          'conv'),
            ('dEdx',           'Slopes & dE/dx',        'conv'),
            ('D_conv',         'D coefficient',         'conv'),
            # Kang & Chueh (relaxation-based, IR-free)
            ('all_xi',         'All xi pulses',         'kc'),
            ('kc_fits',        'xi fits',               'kc'),
            ('kc_slopes',      'xi slopes',             'kc'),
            ('D_kc',           'D coefficient',         'kc'),
        ]

        # ── Custom tab bar (plain tk.Button — cross-platform colour support) ──
        # ttk.Notebook tab background is ignored on many platforms/themes.
        # We replace it with a custom bar of coloured tk.Button widgets that
        # switch a stacked set of tk.Frame pages, giving full colour control.

        SEC_COLOUR = {s[0]: (s[2], s[3]) for s in SECTIONS}
        SEC_COLOUR['docs'] = ('#4a4a4a', 'white')

        # ── Section header bar (coloured section labels) ───────────────────
        sec_bar = tk.Frame(self.root, bg='#d0d0d0', bd=0)
        sec_bar.pack(fill='x', padx=4, pady=(4,0))
        for sec_key, sec_label, sec_dark, sec_light in SECTIONS:
            lbl_fr = tk.Frame(sec_bar, bg=sec_dark, padx=1, pady=1)
            lbl_fr.pack(side='left', padx=(0,2), pady=0)
            tk.Label(lbl_fr, text=sec_label, bg=sec_dark, fg='white',
                     font=('Helvetica','8','bold'), padx=10, pady=3).pack()

        # ── Tab button bar ─────────────────────────────────────────────────
        tab_bar = tk.Frame(self.root, bg='#e8e8e8', bd=0)
        tab_bar.pack(fill='x', padx=4, pady=0)

        # Page container — all frames stacked, only one shown at a time
        page_container = tk.Frame(self.root, bg='white')
        page_container.pack(fill='both', expand=True, padx=4, pady=(0,4))

        self._tab_pages  = {}   # key -> tk.Frame page
        self._tab_btns   = {}   # key -> tk.Button
        self._active_tab = [None]

        # We still expose self.nb for code that calls self.nb.select() /
        # self.nb.index() / self.nb.tab() — keep a dummy ttk.Notebook hidden
        # so those calls don't crash, but we never pack it.
        style = ttk.Style()
        try: style.theme_use('clam')
        except Exception: pass
        self.nb = ttk.Notebook(self.root)   # hidden — never packed

        def _show_tab(key):
            # Hide all pages
            for k, pg in self._tab_pages.items():
                pg.place_forget()
            # Show the selected page
            self._tab_pages[key].place(relx=0, rely=0, relwidth=1, relheight=1)
            # Update button relief: selected = flat+border, others = flat
            for k, btn in self._tab_btns.items():
                sec_k = btn._sec_key
                bg, fg = SEC_COLOUR[sec_k]
                if k == key:
                    btn.config(relief='solid', bd=2,
                               bg=bg, fg=fg, font=('Helvetica',8,'bold'))
                else:
                    # Slightly lighter for unselected
                    btn.config(relief='flat', bd=1,
                               bg=bg, fg=fg, font=('Helvetica',8,'normal'))
            self._active_tab[0] = key
            # Eagerly render the newly visible tab in case it was marked dirty
            # during the last replot (draw_idle deferred its rendering).
            if key in self.tabs:
                try:
                    self.tabs[key]['canvas'].draw()
                except Exception:
                    pass

        all_tab_defs = list(TAB_SECTIONS) + [('docs', 'Documentation', 'docs')]
        for key, label, sec in all_tab_defs:
            bg, fg = SEC_COLOUR.get(sec, ('#888', 'white'))
            btn = tk.Button(tab_bar, text=label,
                            bg=bg, fg=fg,
                            font=('Helvetica', 8, 'normal'),
                            relief='flat', bd=1,
                            padx=6, pady=2,
                            cursor='hand2',
                            command=lambda k=key: _show_tab(k))
            btn._sec_key = sec
            btn.pack(side='left', padx=1, pady=2)
            self._tab_btns[key] = btn

            # Create the page frame
            page = tk.Frame(page_container, bg='white')
            # Coloured top border matching section
            tk.Frame(page, bg=bg, height=3).pack(fill='x', side='top')
            self._tab_pages[key] = page

            if key != 'docs':
                fig = Figure(facecolor='white')
                canvas = FigureCanvasTkAgg(fig, master=page)
                toolbar = NavigationToolbar2Tk(canvas, page)
                toolbar.update()
                canvas.get_tk_widget().pack(fill='both', expand=True)
                self.tabs[key] = {'fig': fig, 'canvas': canvas}

        # Patch self.nb so existing code that calls nb.select()/nb.tab() works
        def _nb_select(widget_or_idx=None):
            if widget_or_idx is None:
                return self._active_tab[0]
            # Accept a tab key string or a widget
            for k, pg in self._tab_pages.items():
                if pg is widget_or_idx or k == widget_or_idx:
                    _show_tab(k); return
        def _nb_index(what):
            if what == 'end': return len(all_tab_defs)
            return 0
        def _nb_tab(idx, **kw): pass
        self.nb.select = _nb_select
        self.nb.index  = _nb_index
        self.nb.tab    = _nb_tab

        # Store for use by other methods, then show first tab
        self._show_tab_fn = _show_tab
        _show_tab(all_tab_defs[0][0])

        # ── Documentation tab content ──────────────────────────────────────
        eq_frame = self._tab_pages['docs']
        self._build_equations_tab(eq_frame)

    # ── Sample management ─────────────────────────────────────────────────────
    # ── Equations reference tab ───────────────────────────────────────────────
    def _build_equations_tab(self, parent):
        """Equations & Methods reference — scrollable, pure-Tk text + matplotlib math."""
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from scipy.signal import savgol_filter
        import matplotlib as mpl

        BG    = 'white';  CARD  = '#f7f8fa';  BDR   = '#dde1e7'
        BLUE  = '#1a56cc';DBLUE = '#0d3080';  ORG   = '#b84800'
        GRN   = '#1a7a1a';RED   = '#aa1111';  GRAY  = '#555555'
        LGRAY = '#999999';BLK   = '#111111';  PAD   = 28

        parent.configure(bg=BG)
        outer = tk.Canvas(parent, bg=BG, highlightthickness=0)
        vsb   = tk.Scrollbar(parent, orient='vertical', command=outer.yview)
        outer.configure(yscrollcommand=vsb.set)
        vsb.pack(side='right', fill='y')
        outer.pack(side='left', fill='both', expand=True)
        inner = tk.Frame(outer, bg=BG)
        wid   = outer.create_window((0,0), window=inner, anchor='nw')
        def _resize(e=None):
            outer.configure(scrollregion=outer.bbox('all'))
            outer.itemconfig(wid, width=outer.winfo_width())
        inner.bind('<Configure>', _resize)
        outer.bind('<Configure>', lambda e: outer.itemconfig(wid, width=e.width))
        outer.bind_all('<MouseWheel>',
            lambda e: outer.yview_scroll(int(-1*(e.delta/120)), 'units'))

        # ── Helpers ───────────────────────────────────────────────────────────
        def sp(h=8):
            tk.Frame(inner, bg=BG, height=h).pack()

        def hl(c=BDR):
            tk.Frame(inner, bg=c, height=1).pack(fill='x', padx=PAD)

        def mpl_text(parent, txt, col, bg, fs=10, bold=False, padx=0, pady=2):
            """Render text via matplotlib canvas so unicode/Greek always display."""
            import textwrap
            lines = []
            for para in str(txt).split('\n'):
                wrapped = textwrap.wrap(para, max(40, int(9.0 / max(fs,8) * 100)))
                lines.extend(wrapped if wrapped else [para])
            n = max(1, len(lines))
            h = max(0.28, n * fs * 0.022)
            fig = Figure(figsize=(9.5, h), facecolor=bg)
            ax  = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off(); ax.set_facecolor(bg)
            step = 1.0 / n
            fw = 'bold' if bold else 'normal'
            for i, line in enumerate(lines):
                ax.text(0.008, 1.0 - (i + 0.5) * step, line,
                        ha='left', va='center', fontsize=fs,
                        fontweight=fw, color=col,
                        transform=ax.transAxes)
            ff = tk.Frame(parent, bg=bg, padx=padx)
            ff.pack(fill='x', pady=pady)
            cv = FigureCanvasTkAgg(fig, master=ff)
            cv.draw(); cv.get_tk_widget().pack(fill='x')



        def render_eq(parent_frame, latex_str, col='#1a56cc', tag=None):
            """Render a LaTeX equation via matplotlib mathtext."""
            fig = Figure(figsize=(8.2, 0.65), facecolor=CARD)
            ax  = fig.add_axes([0,0,1,1])
            ax.set_axis_off(); ax.set_facecolor(CARD)
            ax.text(0.5, 0.5, f'${latex_str}$',
                    ha='center', va='center', fontsize=14, color=col,
                    transform=ax.transAxes)
            if tag:
                ax.text(0.98, 0.5, tag, ha='right', va='center',
                        fontsize=9, color=LGRAY, style='italic',
                        transform=ax.transAxes)
            ef = tk.Frame(parent_frame, bg=CARD,
                          highlightbackground=col, highlightthickness=1)
            ef.pack(fill='x', padx=18, pady=4)
            cv = FigureCanvasTkAgg(fig, master=ef)
            cv.draw(); cv.get_tk_widget().pack(fill='x')

        def embed_fig(fig):
            ff = tk.Frame(inner, bg=BG,
                          highlightbackground=BDR, highlightthickness=1)
            ff.pack(fill='x', padx=PAD, pady=(0,8))
            cv = FigureCanvasTkAgg(fig, master=ff)
            cv.draw(); cv.get_tk_widget().pack(fill='x')

        def sec(txt):
            sp(14); hl('#bbbbbb'); sp(4)
            mpl_text(inner, txt, DBLUE, BG, fs=13, bold=True, padx=PAD, pady=(0,4))
            sp(6)

        def card():
            f = tk.Frame(inner, bg=CARD, highlightbackground=BDR,
                         highlightthickness=1)
            f.pack(fill='x', padx=PAD, pady=(0,8))
            return f

        def ctitle(f, txt, col=GRAY):
            mpl_text(f, txt, col, CARD, fs=10, bold=True, padx=14, pady=(3,2))
            tk.Frame(f, bg=BDR, height=1).pack(fill='x', padx=14)

        def body(f, txt, col=BLK):
            mpl_text(f, txt, col, CARD, fs=10, padx=20, pady=2)

        def notebox(f, txt, col=ORG):
            nf = tk.Frame(f, bg='#fffbf2',
                          highlightbackground='#ddbb66', highlightthickness=1)
            nf.pack(fill='x', padx=18, pady=(2,6))
            mpl_text(nf, txt, col, '#fffbf2', fs=10, padx=8, pady=3)

        def steprow(f, num, title, desc):
            r  = tk.Frame(f, bg=CARD); r.pack(fill='x', padx=14, pady=2)
            tk.Label(r, text=str(num), bg=CARD, fg='#cccccc',
                     font=('Georgia','18','bold'),
                     width=3, anchor='n').pack(side='left', pady=2)
            bf = tk.Frame(r, bg=CARD)
            bf.pack(side='left', fill='x', expand=True, pady=2)
            mpl_text(bf, title, BLK, CARD, fs=11, bold=True, pady=1)
            mpl_text(bf, desc,  GRAY, CARD, fs=10, pady=1)
            tk.Frame(r, bg=BDR, height=1).pack(side='bottom', fill='x')

        def param_card(label, default, rng, affects, what, when, col=BLUE):
            c = card()
            mpl_text(c, f'{label}    default: {default}    range: {rng}',
                     col, CARD, fs=11, bold=True, padx=14, pady=6)
            tk.Frame(c, bg=BDR, height=1).pack(fill='x', padx=14)
            for tag, txt2, tc in [
                ('Affects:',      affects, BLK),
                ('What it does:', what,    GRAY),
                ('When to tune:', when,    ORG),
            ]:
                mpl_text(c, f'{tag}  {txt2}', tc, CARD, fs=10, padx=28, pady=1)
            tk.Frame(c, bg=CARD, height=4).pack()

        mpl_rc = {
            'figure.facecolor':'white','axes.facecolor':'white',
            'axes.edgecolor':'#aaa','axes.labelcolor':'#222',
            'xtick.color':'#444','ytick.color':'#444',
            'grid.color':'#ddd','grid.linestyle':'--',
            'text.color':'#222','legend.facecolor':'white',
            'legend.edgecolor':'#ccc','legend.fontsize':9,
            'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        }
        def sfig(w=9, h=3.0):
            with mpl.rc_context(mpl_rc):
                return Figure(figsize=(w,h), facecolor='white')

        def sym_table(rows):
            """Render symbol table via matplotlib — handles all unicode on any OS."""
            c = card()
            headers = ['Symbol', 'Quantity', 'Unit', 'Notes / source in app']
            col_fracs = [0.12, 0.30, 0.10, 0.48]  # fractional x widths
            row_h = 0.32  # inches per row
            n_rows = len(rows) + 1  # +1 for header
            fig_h = n_rows * row_h + 0.1
            fig = Figure(figsize=(9.5, fig_h), facecolor=CARD)
            ax  = fig.add_axes([0, 0, 1, 1])
            ax.set_axis_off(); ax.set_facecolor(CARD)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            # Draw rows bottom→top (row 0 = bottom, header = top)
            row_h_norm = 1.0 / n_rows
            def cell(row_idx, col_idx, txt, col, bold=False, bg=None):
                y = 1.0 - (row_idx + 0.5) * row_h_norm
                x = sum(col_fracs[:col_idx]) + 0.005
                fw = 'bold' if bold else 'normal'
                ax.text(x, y, txt, ha='left', va='center',
                        fontsize=9, color=col, fontweight=fw,
                        transform=ax.transAxes, clip_on=True)
            # Alternating row backgrounds
            for i in range(n_rows):
                bg_c = '#e8eaf0' if i == 0 else (BG if i % 2 == 1 else CARD)
                y_bot = 1.0 - (i+1)*row_h_norm
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle((0, y_bot), 1, row_h_norm,
                                       facecolor=bg_c, edgecolor='#cccccc',
                                       linewidth=0.5, transform=ax.transAxes))
            # Header
            for ci, (hdr, fc) in enumerate(zip(headers, col_fracs)):
                cell(0, ci, hdr, GRAY, bold=True)
            # Data rows
            for ri, (sym, qty, unit, note) in enumerate(rows):
                r = ri + 1
                cell(r, 0, sym,  DBLUE, bold=True)
                cell(r, 1, qty,  BLK)
                cell(r, 2, unit, GRN)
                cell(r, 3, note, GRAY)
            # Column dividers
            x = 0
            for fc in col_fracs[:-1]:
                x += fc
                ax.axvline(x, color='#cccccc', lw=0.5)
            ff = tk.Frame(c, bg=CARD, padx=14)
            ff.pack(fill='x', pady=6)
            cv = FigureCanvasTkAgg(fig, master=ff)
            cv.draw(); cv.get_tk_widget().pack(fill='x')
            return c

        def param_card(label, default, rng, affects, what, when, col=BLUE):
            c = card()
            mpl_text(c, f'{label}    default: {default}    range: {rng}',
                     col, CARD, fs=11, bold=True, padx=14, pady=6)
            tk.Frame(c, bg=BDR, height=1).pack(fill='x', padx=14)
            for tag, txt2, tc in [
                ('Affects:',      affects, BLK),
                ('What it does:', what,    GRAY),
                ('When to tune:', when,    ORG),
            ]:
                mpl_text(c, f'{tag}  {txt2}', tc, CARD, fs=10, padx=28, pady=1)
            tk.Frame(c, bg=CARD, height=4).pack()

        # ════════════════════════════════════════════════════════════════════
        sp(16)
        hf = tk.Frame(inner, bg=BG); hf.pack(fill='x', padx=PAD, pady=(0,2))
        mpl_text(hf, 'GITT Analysis — Methods & Documentation Reference',
                  DBLUE, BG, fs=17, bold=True, pady=(0,2))
        mpl_text(hf, 'Weppner & Huggins, J. Electrochem. Soc. 124, 1569 (1977)   ×   '
                     'Kang & Chueh, J. Electrochem. Soc. 168, 120504 (2021)   ×   '
                     'Ge et al., ACS Energy Lett. 8, 2738 (2023)',
                 LGRAY, BG, fs=10, pady=(0,2))
        sp(4); hl('#aaaaaa')

        # ── 1. Why relaxation ────────────────────────────────────────────────
        sec('1.  Why Relaxation?  —  Eliminating IR Contamination')
        c = card()
        ctitle(c, 'During the galvanostatic pulse — D_conv method (W&H)', RED)
        body(c, 'During the pulse, IR drop and diffusion signal are mixed and cannot be separated:')
        render_eq(c,
            r'E(t)_{\rm pulse} \;=\; V_{\rm eq} \;+\; I \cdot R_{\rm tot}(t) \;+\; \mathrm{diffusion\ signal}',
            RED, tag='[contaminated]')
        body(c,
            'R_tot changes during the pulse because R_ct is composition-dependent. '
            'Kang & Chueh show that a 17% change in R_tot causes a 4× overestimation of D.', GRAY)
        body(c, 'During open-circuit relaxation (I = 0), the IR drop disappears entirely:')
        render_eq(c,
            r'V(t_{\rm relax}) \;=\; V_{\rm eq} \;+\; \mathrm{diffusion\ signal\ only}',
            GRN, tag='[clean — K&C method]')
        sp(4)

        sec('Figure 1  —  Pulse vs Relaxation')
        fig1 = sfig(9, 2.8)
        t_p=np.linspace(0,1,300); t_r=np.linspace(0,4,300)
        dp=-1.2*np.sqrt(t_p); ir=-0.8-0.3*(1-np.exp(-3*t_p)); op=dp+ir
        xi1=np.sqrt(t_r+1.)-np.sqrt(t_r); or1=1.15*xi1
        ax1=fig1.add_subplot(121)
        ax1.plot(np.sqrt(t_p),dp,'--',color=GRN,lw=1.8,label='True diffusion')
        ax1.plot(np.sqrt(t_p),op,'-',color=RED,lw=2,label='Observed (pulse)')
        ax1.set_xlabel(r'$\sqrt{t}$  (s$^{1/2}$)',fontsize=11)
        ax1.set_ylabel(r'$E - E_0$  (V)',fontsize=11)
        ax1.set_title('Pulse  (D_conv — W&H)',fontsize=11,fontweight='bold')
        ax1.legend(fontsize=9); ax1.grid(True)
        ax1.annotate('IR bias',xy=(0.6,-1.5),xytext=(0.25,-1.05),
                     arrowprops=dict(arrowstyle='->',color=RED),color=RED,fontsize=8)
        ax2=fig1.add_subplot(122)
        ax2.plot(xi1,or1,'-',color=BLUE,lw=2,label=r'V vs $\xi$')
        ax2.set_xlabel(r'$\xi = \sqrt{t_{\rm relax}+\tau} - \sqrt{t_{\rm relax}}$  (s$^{1/2}$)',fontsize=10)
        ax2.set_ylabel(r'$E - E_{\rm eq}$  (V)',fontsize=11)
        ax2.set_title('Relaxation  (D_kc — K&C)',fontsize=11,fontweight='bold')
        ax2.legend(fontsize=9); ax2.grid(True)
        ax2.annotate('linear\nbulk signal',xy=(xi1[80],or1[80]),xytext=(1.2,0.3),
                     arrowprops=dict(arrowstyle='->',color=BLUE),color=BLUE,fontsize=8)
        fig1.tight_layout(pad=1.4); embed_fig(fig1)

        # ── 2. Conventional ──────────────────────────────────────────────────
        sec('2.  Weppner & Huggins (1977)  —  Conventional GITT  (D_conv)')
        c = card()
        ctitle(c, 'Fit V vs sqrt(t) during the pulse in [Fit start, Fit end]', BLUE)
        body(c,
            'The app uses x = Q/Q_max (0 → 1). Molar volume v_M and electrode area S '
            'are not entered — D is reported as D̃·(S/vₘ)² in mol² s⁻¹:')
        render_eq(c,
            r'\tilde{D}_{\rm conv} \;=\; \frac{4\,I^2}{\pi\,F^2\,\delta_e^2} \cdot \left(\frac{\partial V_{\rm eq}/\partial x}{s_{\sqrt{t}}}\right)^{2}',
            BLUE, tag='D_conv in app')
        notebox(c,
            'To get absolute D̃ [cm² s⁻¹]: multiply the reported value by (v_M / S)², '
            'with v_M from XRD lattice parameters and S from BET or geometric electrode area.')
        body(c,
            'The t = 0 point (IR-drop jump at current switch-on) is always excluded from '
            'the fit regardless of Fit start, to avoid slope contamination.', GRAY)
        sp(4)

        sec('Figure 2  —  V vs sqrt(t): slope extraction')
        fig2 = sfig(9, 2.8)
        t2=np.linspace(0.5,9,200); np.random.seed(42)
        v2=1.2-0.18*np.sqrt(t2)+np.random.normal(0,0.003,200)
        msk=(t2>=1.5)&(t2<=6); p2=np.polyfit(np.sqrt(t2[msk]),v2[msk],1)
        ax1=fig2.add_subplot(121)
        ax1.plot(t2,v2,'.',color=BLUE,ms=3,alpha=0.7)
        ax1.axvspan(1.5,6,alpha=0.1,color=BLUE,label='fit window')
        ax1.set_xlabel('t  (s)',fontsize=11); ax1.set_ylabel('V  (V)',fontsize=11)
        ax1.set_title('Raw V(t)',fontsize=11,fontweight='bold')
        ax1.legend(fontsize=9); ax1.grid(True)
        ax2=fig2.add_subplot(122)
        sq2=np.sqrt(t2); ax2.plot(sq2,v2,'.',color=BLUE,ms=3,alpha=0.7,label='Data')
        sqf=np.sqrt(np.linspace(1.5,6,80))
        ax2.plot(sqf,np.polyval(p2,sqf),'-',color=RED,lw=2.5,
                 label=f'$s_{{\\sqrt{{t}}}}$ = {p2[0]:.4f} V/s$^{{1/2}}$')
        ax2.axvspan(np.sqrt(1.5),np.sqrt(6),alpha=0.08,color=BLUE)
        ax2.set_xlabel(r'$\sqrt{t}$  (s$^{1/2}$)',fontsize=11)
        ax2.set_ylabel('V  (V)',fontsize=11)
        ax2.set_title(r'V vs $\sqrt{t}$  $\rightarrow$  extract $s_{\sqrt{t}}$',
                      fontsize=11,fontweight='bold')
        ax2.legend(fontsize=9); ax2.grid(True)
        fig2.tight_layout(pad=1.4); embed_fig(fig2)

        # ── 3. Kang ──────────────────────────────────────────────────────────
        sec('3.  Kang & Chueh (2021)  —  Relaxation Method  (D_kc)')
        c = card()
        ctitle(c, 'Fit E − E_eq vs ξ during relaxation — no IR contamination', ORG)
        body(c, 'After a pulse of duration τ, V relaxes as (semi-infinite planar 1D diffusion):')
        render_eq(c,
            r'V(t_{\rm relax}) - V_{\rm eq} \;=\; \frac{I\,v_M}{z\,F\,S} \cdot \frac{\partial V_{\rm eq}}{\partial x} \cdot \frac{2}{\sqrt{\pi\,\tilde{D}}} \cdot \xi',
            ORG, tag='Kang Eq. 4')
        body(c, r'ξ = √(t_relax + τ) − √t_relax decreases as relaxation proceeds. '
                'V vs ξ is linear in the bulk diffusion regime. App formula:')
        render_eq(c,
            r'\tilde{D}_{\rm kc} \;=\; \frac{4\,I^2}{\pi\,F^2\,\delta_e^2} \cdot \left(\frac{\partial V_{\rm eq}/\partial x}{s_\xi}\right)^{2}',
            ORG, tag='D_kc in app')
        notebox(c,
            'Large ξ = early relaxation (surface transient, NOT bulk diffusion). '
            'Small ξ = late relaxation (bulk diffusion). '
            'Use KC xi_min / KC xi_max to restrict the fit to the linear bulk region. '
            'Check the xi fits tab to identify the linear region visually.')
        sp(4)

        sec('Figure 3  —  V vs ξ: slope extraction  (K&C method)')
        fig3 = sfig(9, 3.0)
        tau3=600.; t3=np.linspace(1,5000,500)
        xi3=np.sqrt(t3+tau3)-np.sqrt(t3)
        rng3=np.random.default_rng(7)
        v3=1.05+0.22*xi3+0.06*np.exp(-t3/40)+rng3.normal(0,0.003,500)
        fmk=(xi3<18)&(xi3>2); pk=np.polyfit(xi3[fmk],v3[fmk],1)
        ax1=fig3.add_subplot(121)
        ax1.plot(t3,v3,'.',color=ORG,ms=3,alpha=0.7)
        ax1.set_xlabel(r'$t_{\rm relax}$  (s)',fontsize=11)
        ax1.set_ylabel('V  (V)',fontsize=11)
        ax1.set_title('Raw relaxation V(t)',fontsize=11,fontweight='bold')
        ax1.grid(True)
        ax2=fig3.add_subplot(122)
        ax2.plot(xi3,v3,'.',color=ORG,ms=3,alpha=0.7,label='Data')
        xif=np.linspace(xi3[fmk].min(),xi3[fmk].max(),80)
        ax2.plot(xif,np.polyval(pk,xif),'-',color=BLUE,lw=2.5,
                 label=f'$s_\\xi$ = {pk[0]:.4f} V/s$^{{1/2}}$')
        ax2.axvspan(xi3[fmk][-1],xi3[fmk][0],alpha=0.08,color=BLUE,label='Bulk fit')
        ax2.annotate('surface\ntransient',xy=(xi3[3],v3[3]),xytext=(19,1.22),
                     arrowprops=dict(arrowstyle='->',color=RED),color=RED,fontsize=8)
        ax2.set_xlabel(r'$\xi = \sqrt{t_{\rm relax}+\tau} - \sqrt{t_{\rm relax}}$  (s$^{1/2}$)',fontsize=10)
        ax2.set_ylabel('V  (V)',fontsize=11)
        ax2.set_title(r'V vs $\xi$  $\rightarrow$  extract $s_\xi$',
                      fontsize=11,fontweight='bold')
        ax2.legend(fontsize=9); ax2.grid(True); ax2.invert_xaxis()
        fig3.tight_layout(pad=1.4); embed_fig(fig3)

        # ── 4. dE/dx ─────────────────────────────────────────────────────────
        sec('4.  Thermodynamic Factor  dV_eq/dx  —  Two Modes')
        c = card()
        ctitle(c, 'How x and dV_eq/dx are computed', DBLUE)
        steprow(c, 1, 'x_block[n] = cumulative pulse charge / Q_max',
                'Q is integrated directly from |I|×dt in the time file (not the cap file, '
                'which also records leakage during relaxation). '
                'x_block[n] = Σ(Q_pulse_k for k=1..n) / Q_max_disc (discharge) '
                'or / Q_max_chg (charge). Each direction uses its own Q_max reference.')
        steprow(c, 2, 'V_eq[n] = last voltage of relaxation after pulse n',
                'True equilibrium OCV at composition x_block[n]. '
                'The first pulse of each half-cycle has no previous V_eq, '
                'so dE/dx and D are undefined for pulse 1 of each direction.')
        body(c, 'Two modes are available via the dE/dx mode radio button in the parameter bar:', GRAY)
        steprow(c, 'A', 'Differential (smooth) — default',
                'V_eq points are smoothed (Savitzky-Golay) and differentiated numerically. '
                'Evaluated at each block V_eq via interpolation. '
                'This is the thermodynamic factor used internally in both D_conv and D_kc.')
        steprow(c, 'B', 'Finite difference ΔE/Δx  (block-to-block)',
                'dE/dx[n] = (V_eq[n] − V_eq[n−1]) / (x[n] − x[n−1]). '
                'Model-free and noisier. Useful to cross-check the smooth estimate. '
                'Neither mode affects D_conv or D_kc values — display only.')
        sp(4)
        fig4 = sfig(9, 2.6)
        x4=np.linspace(0.02,0.98,300)
        v4=1.95-1.85*x4-0.3*np.sin(np.pi*x4)-0.1*np.sin(3*np.pi*x4)
        vs4=savgol_filter(v4,31,2); dv4=np.gradient(vs4)/np.gradient(x4)
        ax1=fig4.add_subplot(121)
        ax1.plot(x4,v4,'-',color=BLUE,lw=2)
        ax1.set_xlabel(r'$x = Q/Q_{\rm max}$',fontsize=11)
        ax1.set_ylabel(r'$V_{\rm eq}$  (V)',fontsize=11)
        ax1.set_title('OCV curve',fontsize=11,fontweight='bold')
        ax1.grid(True)
        ax2=fig4.add_subplot(122)
        ax2.plot(v4,dv4,'-',color=GRN,lw=2)
        ax2.axhline(0,color=LGRAY,lw=0.8,ls='--')
        ax2.set_xlabel(r'$V_{\rm eq}$  (V)',fontsize=11)
        ax2.set_ylabel(r'$\partial V_{\rm eq}/\partial x$  (V)',fontsize=11)
        ax2.set_title(r'Thermodynamic factor $\partial V_{\rm eq}/\partial x$',
                      fontsize=11,fontweight='bold')
        ax2.grid(True)
        fig4.tight_layout(pad=1.4); embed_fig(fig4)

        # ── 5. dV/d(log t) ───────────────────────────────────────────────────
        sec('5.  Relaxation Kinetics  —  dV/d(log t) Process Fingerprinting')
        c = card()
        ctitle(c, 'Model-free identification of multiple processes per relaxation', BLUE)
        body(c,
            'Standard GITT analysis assumes a single solid-state diffusion process. '
            'In co-intercalation, solvated-ion systems, or multi-step insertion materials, '
            'several sequential processes occur with different time constants. '
            'dV/d(log t) reveals them without assuming any functional form:')
        render_eq(c,
            r'\frac{dV}{d\log_{10} t} \;=\; t \cdot \frac{dV}{dt}',
            BLUE, tag='time-domain DRT')
        body(c,
            'Each process (desolvation, solid-state diffusion, staging transition, '
            'electrolyte re-filling) appears as a distinct PEAK. '
            'Peak position on the log-time axis = τ directly. '
            'Peak count = number of distinguishable mechanisms.',
            GRAY)
        notebox(c,
            'Direct EIS analogue: a peak at log₁₀(τ) corresponds to an EIS arc at f = 1/(2πτ). '
            'Cross-validate τ values from dV/d(log t) against EIS semicircle frequencies at the same SOC/V_eq.')
        body(c,
            'Implementation: V(t) is resampled onto a uniform log-time grid (120 points), '
            'differentiated numerically, and smoothed with a Savitzky-Golay filter '
            '(window = n/6, polynomial order 3). The smoother is applied before any '
            'peak finding so the displayed curve always matches what the algorithm sees.',
            GRAY)
        body(c,
            'Three tabs in the Analysis section provide complementary views:',
            GRAY)
        steprow(c, 1, 'Relax kinetics — overlay of all pulses',
                'All relaxation blocks overlaid in one panel per sample. '
                'Discharge top row, charge bottom row. '
                'Colour = V_eq (turbo colourmap). '
                'Useful for a quick visual scan of how the dV/d(log t) shape evolves with SOC.')
        steprow(c, 2, 'Relax kinetics / pulse — single pulse detail',
                'dV/d(log t) for one selected pulse (Pulse # spinbox). '
                'Sample colour, white background, filled area. '
                'Peak τ values annotated in top-left box. '
                'Orange background = 2 or more processes detected. '
                'Updates immediately when Pulse # or Reverse changes.')
        steprow(c, 3, 'Relax kinetics map — y = V_eq (operando-XRD style)',
                'x = log₁₀(τ / s), y = V_eq (V), colour = normalised process fraction. '
                'Blocks sorted by V_eq. See Section 5b below.')
        steprow(c, 4, 'Relax map vs SOD/SOC — y = SOD/SOC %',
                'Identical map but y-axis = SOD % (discharge) or SOC % (charge). '
                'Blocks sorted by x_block so y runs 0→100. '
                'Shows how process τ evolves with state of charge rather than voltage.')
        sp(4)

        # ── 5b. Relax kinetics map ────────────────────────────────────────────
        sec('5b.  Relax Kinetics Maps  —  Operando-XRD-Style 2D Heatmaps')
        c = card()
        ctitle(c, 'Direct analogue of operando XRD waterfall plots', BLUE)
        body(c,
            'In operando XRD, the 2D heatmap shows diffraction intensity vs 2θ (x) '
            'and elapsed time / electrode state (y). Each structural phase appears as '
            'a vertical band at constant 2θ; a shifting band means lattice evolution. '
            'The Relax kinetics maps are the electrochemical equivalent: '
            'x = log₁₀(τ), colour = process strength at each (τ, state) point.')
        body(c,
            'Two y-axis variants are provided:\n'
            'Relax kinetics map — y = V_eq (V): blocks sorted by equilibrium voltage. '
            'Directly comparable with EIS Nyquist arcs measured at the same V_eq.\n'
            'Relax map vs SOD/SOC — y = SOD/SOC %: blocks sorted by state of charge. '
            'Shows whether a process τ is correlated with composition rather than voltage. '
            'Useful when the V_eq–x relationship is non-monotonic (plateau region).',
            GRAY)
        body(c,
            'Colour encoding: each row (one relaxation block) is divided by its own '
            'total area under the dV/d(log t) curve before plotting. The colour therefore '
            'represents the fraction of total relaxation signal at each τ — not the '
            'absolute amplitude. This normalisation makes comparisons across different '
            'V_eq values fair: a block with small total overpotential and a block with '
            'large total overpotential are shown on the same relative scale.',
            GRAY)
        render_eq(c,
            r'Z_{\rm plot}(\tau, V_{\rm eq}) \;=\; '
            r'\frac{|dV/d\log t|(\tau, V_{\rm eq})}{\int |dV/d\log t|\, d\log\tau}',
            BLUE, tag='row-area normalisation')
        body(c,
            'Colormap: blue (low fraction) → cyan → green → yellow → red (high fraction), '
            'matching the operando XRD style. '
            'Colour scale clipped at the 97th percentile to prevent single outlier rows '
            'from compressing the rest of the map.',
            GRAY)
        notebox(c,
            'Reading the map: '
            'a vertical bright band at constant log τ = a process whose time constant '
            'does not change with electrode state (e.g. a fixed-τ desolvation step). '
            'A diagonal or curved band = τ shifts with V_eq (e.g. diffusion-limited process '
            'that slows as the electrode becomes more intercalated). '
            'Comparing discharge and charge maps at the same V_eq reveals hysteresis in τ.')
        sp(4)

        # ── 6. Symbol table ──────────────────────────────────────────────────
        sec('6.  Symbol Table')
        sym_table([
            ('D̃·(S/vₘ)²',
             'Chemical diffusion coefficient × (S/vₘ)²',
             'mol² s⁻¹',
             'Multiply by (vₘ/S)² for absolute D̃ in cm² s⁻¹'),
            ('I',  'Applied current (galvanostatic pulse)',  'A',
             'Median |I| of non-zero samples in time file, per cycle direction'),
            ('F',  'Faraday constant',  'C mol⁻¹',  '96485 C mol⁻¹  (hard-coded)'),
            ('δ_e', 'Mobile species per formula unit', '—',
             'Parameter slider; = 1 for single-ion intercalation (graphite, LCO, LFP…)'),
            ('dVeq/dx', 'Thermodynamic factor', 'V',
             'Smooth derivative of V_eq(x); x = Q/Q_max; used in both D_conv and D_kc'),
            ('s_√t', 'Slope V vs √t  [pulse, W&H]', 'V s⁻½',
             'Linear fit in [Fit start, Fit end]; t=0 IR-drop point always excluded'),
            ('s_ξ',  'Slope V vs ξ  [relaxation, K&C]', 'V s⁻½',
             'ξ = √(t_relax + τ) − √t_relax; fit in [KC xi_min, KC xi_max]'),
            ('τ',   'Pulse duration', 's',
             'Auto-detected as median pulse length from current signal'),
            ('t_relax', 'Time since current interruption', 's',
             't_relax = 0 at pulse end'),
            ('ξ',  'K&C time variable', 's½',
             'Decreases as relaxation proceeds; large ξ = early surface transient'),
            ('V_eq',    'Equilibrium voltage', 'V',
             'Last recorded voltage at end of relaxation window'),
            ('x_block', 'Composition after pulse n', '—',
             'Cumulative pulse charge / Q_max; assigned after each pulse'),
            ('R²',  'Fit quality', '—',
             'R²_sqrt for W&H fit; R²_kc for K&C ξ fit'),
            ('R_el', 'Ohmic resistance estimate', 'Ω',
             '|ΔV|/|I| at current switch-on; NaN if gap > 3× sampling interval'),
            ('Q_max',   'Reference capacity', 'mAh or mAh/g',
             'Max discharge capacity from cap file (discharge) or charge (charge)'),
            ('v_M',     'Molar volume', 'cm³ mol⁻¹',
             'NOT in app — absorbed into dE/dx. Measure from XRD lattice parameters.'),
            ('S',       'Active surface area', 'cm²',
             'NOT in app — absorbed into dE/dx. BET or geometric area of dense pellet.'),
        ])
        sp(6)

        # ── 7. Parameter guide ───────────────────────────────────────────────
        sec('7.  Tunable Parameters  —  What Each One Does')
        c = card()
        ctitle(c, 'Important: parameter changes take effect on Replot only', ORG)
        body(c,
            'All parameter spinboxes (tau, Fit start/end, V_lo/V_hi, Min R², delta_e, '
            'sigma-clip, KC xi_min/max, Trim time, mass) do NOT trigger an automatic replot. '
            'Adjust as many parameters as you like, then click Replot once. '
            'This avoids redundant redraws when changing multiple parameters at the same time.',
            GRAY)
        notebox(c,
            'Exception: Pulse # spinbox and Reverse pulses radio update the three per-pulse tabs '
            'immediately (sqrt fits, xi fits, Relax kinetics / pulse) because those are fast '
            'single-curve draws. Pulse range (Show pulses from/to) also updates overlays '
            'immediately on arrow clicks.')

        param_card('τ (s)  Pulse duration',
            'auto-detected', '10 – 7200 s',
            'D_kc only  (K&C relaxation method)',
            'ξ = √(t_relax + τ) − √t_relax. Auto-detected as median pulse duration '
            'from current signal on load. Has NO effect on D_conv.',
            'Only change if auto-detection gives wrong value '
            '(file starts mid-pulse, or irregular pulse lengths).', ORG)

        param_card('Fit start / Fit end  (s)',
            'Fit start = 60 s;  Fit end = τ (auto)', '0 – 7200 s',
            'D_conv only  (sqrt(t) regression window inside the pulse)',
            'Fit start: excludes early surface transient. '
            'Fit end: excludes finite-size effects. Defaults to τ at load. '
            'The t=0 IR-drop point is always excluded regardless of Fit start.',
            'Increase Fit start if V vs √t curves at early times. '
            'Decrease Fit end if V curves at late times (finite-size effect). '
            'For sparse data, widen to capture more points.', BLUE)

        param_card('V_lo / V_hi  (V)',
            'auto-detected from data', '0 – 5 V',
            'Display filter on D and Slopes plots  (does NOT change any computation)',
            'Hides blocks with V_eq outside [V_lo, V_hi]. '
            'Auto-expanded to cover all samples on load.',
            'Narrow to exclude two-phase plateaus where dVeq/dx → 0 and D diverges. '
            'E.g. set V_lo=0.68 and V_hi=1.6 to analyse only above a 0.65 V plateau.', RED)

        param_card('Min R²', '0.85', '0 – 1',
            'Quality filter on D_conv and D_kc scatter plots',
            'Blocks with R² < Min R² are excluded from D plots. '
            'Low R² = poor linearity = unreliable fit.',
            'Start at 0.85. Raise to 0.95 for cleaner plots. '
            'Lower to 0 to see all blocks for diagnostics.', GRAY)

        param_card('sigma-clip  (0 = off)',
            '0 (off)', '0 – 5 σ',
            'D_conv and D_kc visual filter only  (does NOT affect computation or export)',
            'Removes D values deviating more than N σ from the median in log₁₀(D). '
            'Uses robust MAD estimator (σ_robust = 1.4826 × MAD). '
            'Applied per direction per sample independently.',
            'Leave at 0 first. Try 2–3 σ to suppress diverging outliers near plateaus. '
            'Prefer V_lo/V_hi for clean exclusion of voltage regions.', GRAY)

        param_card('δ_e  Stoichiometric number', '1', '1 – 4',
            'Scales both D_conv and D_kc by 1/δ_e²',
            'Number of mobile ions per formula unit. '
            '= 1 for most intercalation electrodes (graphite, LCO, NMC, LFP, …).',
            'Only change for multi-ion materials (e.g. Na₂M → δ_e = 2). '
            'Wrong value → D off by δ_e².', DBLUE)

        param_card('KC xi_min / KC xi_max', '0 / 99', '0 – 99 s½',
            'K&C fit window for the V vs ξ regression',
            'Restricts which part of the relaxation is used for the K&C slope. '
            'Large ξ = early transient (exclude). Small ξ = bulk diffusion (keep). '
            'Recomputed at every Replot — D_kc always consistent with xi fits tab.',
            'Set xi_max < 15 to exclude early surface transients. '
            'Inspect xi fits tab to find the linear region visually.', ORG)

        param_card('Pulse #  (spinbox)', '1', '1 to N_pulses',
            'sqrt(t) fits,  xi fits,  and Relax kinetics / pulse tabs',
            'Steps through individual blocks for visual inspection. '
            'Does NOT affect any D computation. '
            'Updating the spinbox redraws only the three per-pulse tabs '
            'and stays on whichever tab is currently visible.',
            'Use to verify linearity of fits and identify corrupted blocks.', GRAY)

        param_card('Show pulses from / to  (overlay range)', '1 / 999', '1 to N_pulses',
            'All overlay tabs: All pulses, All xi pulses, Overpotential curves, Relaxation curves, Relax kinetics, Relax kinetics map',
            'Limits which pulse blocks are shown in gradient-overlay and map tabs. '
            'Does not affect computation or single-pulse tabs.',
            'Use when overlays are too crowded to distinguish curves, '
            'or to focus the map on a specific SOC window.', GRAY)

        param_card('Reverse pulses  (radio button)',
            'Discharge', 'None | Discharge | Charge | Both',
            'Per-pulse tabs: sqrt(t) fits, xi fits, Relax kinetics / pulse',
            'Controls the pulse ordering: None = file order (pulse 1 = first recorded). '
            'Discharge = reversed (pulse 1 = most discharged state, lowest V_eq). '
            'Charge = reversed charge direction (pulse 1 = most charged). '
            'Both = both directions reversed. '
            'Useful for comparing charge and discharge at similar V_eq values.',
            'Set to Discharge to align discharge pulse 1 with early charge '
            'at similar potentials. Applies immediately on click.', DBLUE)

        param_card('Trim time (h)  from / to', '0, 0 (off)', '0 – 9999 h',
            'Block exclusion: removes all blocks whose pulse or relaxation overlaps [from, to]',
            'All data in the time window is kept in the raw array — block detection runs on '
            'the complete data. Only blocks that overlap the window are excluded from all '
            'downstream analysis (D, slopes, fits, dV/d(log t)). '
            'The trimmed window is highlighted in red on the GITT Curve tab. '
            'Set both to 0 to disable.',
            'Find the time range by zooming to the artefact on the GITT Curve tab. '
            'Use when you cannot determine the specific pulse number.', RED)

        param_card('Excl disc# / chg#  per sample  (text entry)',
            'empty (none excluded)', 'comma-separated pulse numbers, e.g. 39 or 5,12',
            'Excludes specific pulse blocks per sample per direction from ALL analysis',
            'Entire block (pulse + relaxation) at the given 1-based position is removed '
            'from D_conv, D_kc, slopes, xi fits, sqrt fits, dV/d(log t), and the map. '
            'Excluded blocks are drawn in red on the GITT Curve tab for visual confirmation. '
            'Each sample has its own independent disc and charge exclusion fields. '
            'Takes effect on Replot — no auto-fire.',
            'Find the pulse number in the sqrt(t) fits or Relax kinetics / pulse tab '
            '(anomalous V_eq in title, stepped relaxation). '
            'Enter one or more numbers, then click Replot.', GRN)

        param_card('dE/dx mode  (radio)', 'Differential (smooth)', 'Differential | Finite diff',
            'Bottom row of Slopes & dE/dx tab only — does NOT affect D_conv or D_kc',
            'Differential: Savitzky-Golay smoothed OCV derivative — the thermodynamic '
            'factor used internally in both D methods. '
            'Finite diff: raw block-to-block estimate, model-free and noisier.',
            'Start with Differential. Switch to Finite diff to cross-check.', DBLUE)

        param_card('m for Rtot  per sample  (mg)', '7.05', '0.1 – 200 mg',
            'Overpotential & Rtot tab — right y-axis: R_tot × mass (Ω g)',
            'R_tot = η / I_pulse. Multiplied by mass to normalise. '
            'Does not affect D_conv, D_kc, or x_block.',
            'Set to the active electrode mass. '
            'If 0 or unset, R_tot shown unnormalised in Ω.', GRN)

        # ── 8. Validity conditions ────────────────────────────────────────────
        sec('8.  Validity Conditions')
        c = card()
        ctitle(c, 'Both W&H and K&C methods require these conditions to hold', RED)
        body(c, 'Sign convention: I < 0 = discharge (insertion), I > 0 = charge (deinsertion). '
                'Assigned automatically from median current sign per half-cycle.', GRAY)
        steprow(c,'i',  'Semi-infinite diffusion:  D·τ/L² << 1',
                'τ̂ = D·τ/L² << 0.025 for spheres (Kang Fig. 6c). '
                'Verify after extraction using measured D and particle radius L.')
        steprow(c,'ii', 'Single-phase material (no two-phase plateau)',
                'In a two-phase co-existence region dVeq/dx → 0 and D diverges/scatters. '
                'Exclude plateau voltage ranges using V_lo / V_hi. '
                'See Section 9 for a full explanation.')
        steprow(c,'iii','Diffusion-limited kinetics',
                'In porous composite electrodes, voltage may reflect inter-particle '
                'kinetics rather than bulk diffusion. Dense pellets or cavity '
                'microelectrode measurements are preferred.')
        steprow(c,'iv', 'Constant IR drop  (D_conv only)',
                '17% change in R_tot during pulse → 4× error in D_conv. '
                'D_kc avoids this entirely by analysing zero-current relaxation.')
        steprow(c,'v',  'Good linearity  (R² > 0.95 recommended)',
                'D_kc: fit only the small-ξ (late relaxation) linear region. '
                'Use KC xi_min/xi_max to exclude the surface transient at large ξ.')
        steprow(c,'vi', 'Relaxation long enough',
                'Relaxation should be ≥ pulse duration (τ). '
                'Short relaxation → electrode not equilibrated → V_eq unreliable.')

        # ── 9. Pulse exclusion ────────────────────────────────────────────────
        sec('9.  Pulse Exclusion  —  Removing Artefacts from File Concatenation')
        c = card()
        ctitle(c, 'Problem: channel transfer corrupts one block at the file join', RED)
        body(c,
            'When a GITT experiment is interrupted and resumed on a different potentiostat '
            'channel, the two exported files are concatenated automatically. '
            'The block that spans the file join has a corrupted relaxation: the voltage '
            'jumps at the boundary, so V_eq is wrong. This causes a spike in D, slopes, '
            'and dV/d(log t) for that one block only.')
        body(c,
            'Best fix: use the Excl disc# / chg# entry fields in the Samples panel. '
            'Enter the 1-based pulse number of the corrupted block (e.g. "39"). '
            'The entire block — pulse AND relaxation — is excluded from all analysis.',
            GRAY)
        notebox(c,
            'How to find the pulse number: go to "Relax kinetics / pulse" or "sqrt(t) fits". '
            'Step through with the Pulse # spinbox until you find the block with anomalous '
            'V_eq (shown in the panel title) or a stepped relaxation curve. '
            'Enter that number in Excl disc# and press Enter.')
        body(c,
            'Coarser alternative: the Trim time (h) from/to fields exclude all blocks '
            'whose pulse or relaxation overlaps the specified time window. '
            'Useful when you cannot determine the exact pulse number. '
            'Both mechanisms can be used simultaneously.',
            GRAY)

        # ── 10. D at the plateau ──────────────────────────────────────────────
        sec('10.  D at the Plateau  —  Why Values Scatter and Are Not Meaningful')
        c = card()
        ctitle(c, 'Two-phase region: GITT equations break down', RED)
        body(c,
            'In a two-phase co-existence region (flat OCV plateau), the thermodynamic '
            'factor dVeq/dx → 0 by definition. Both D_conv and D_kc contain this factor:')
        render_eq(c,
            r'\tilde{D} \;\propto\; \left(\frac{dV_{\rm eq}/dx}{s}\right)^2 \;\to\; \text{undefined in plateau}',
            RED, tag='two-phase breakdown')
        body(c,
            'Both numerator (dVeq/dx) and denominator (slope s) go to zero together, '
            'but with different noise characteristics. The slope decreases smoothly '
            'because the pulse is nearly flat. The dVeq/dx, computed from V_eq values '
            'that are all within 1–3 mV of each other, amplifies measurement noise into '
            'large relative fluctuations. The ratio (dVeq/dx / slope)² oscillates wildly.',
            GRAY)
        body(c,
            'This is physically correct — GITT cannot measure D in a two-phase region. '
            'Published papers show a clean single dip only because their plateau is short '
            '(2–3 pulses). A long plateau (10–20+ pulses at nearly the same V_eq) '
            'produces visible scatter. The solution is V_lo/V_hi exclusion, not curve fitting.')
        notebox(c,
            'Practical fix: set V_lo / V_hi to bracket the sloped regions on either side. '
            'For graphite co-intercalation at ~0.65 V: use V_lo=0.68, V_hi=1.6 for the '
            'high-potential region; then re-run with V_lo=0.0, V_hi=0.62 for below the plateau. '
            'D_kc is generally less affected than D_conv because the relaxation slope '
            'is less sensitive to the IR contribution that saturates at the plateau.')

        sp(12); hl('#aaaaaa')
        mpl_text(inner, 'GITT Analysis App  v2.3  ×  Internal Reference',
                 LGRAY, BG, fs=9, padx=PAD, pady=8)



    def _color(self,name):
        names=list(self._chk_vars.keys())
        idx=names.index(name) if name in names else 0
        return SAMPLE_COLORS[idx%len(SAMPLE_COLORS)]

    def _sample_cmap(self, name):
        """Return a LinearSegmentedColormap from near-white to the sample's base colour.

        Each sample gets its own gradient derived from its SAMPLE_COLORS entry,
        so samples are visually distinguishable in the gradient overlay tabs even
        when many pulses are shown at once.
        """
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm
        base_hex = self._color(name)          # e.g. '#2E6FBA'
        r, g, b = mcolors.to_rgb(base_hex)
        # Gradient: very light (high E_eq = end of range) → full colour (low E_eq)
        light = (min(r+0.55, 1.0), min(g+0.55, 1.0), min(b+0.55, 1.0))
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f'smp_{name}', [light, (r, g, b)], N=256)
        return cmap

    def _get_active(self):
        return [n for n,v in self._chk_vars.items() if v.get()]

    def _check_all(self):
        for v in self._chk_vars.values(): v.set(True)
        self._replot()

    def _check_none(self):
        for v in self._chk_vars.values(): v.set(False)
        self._replot()

    def _browse(self):
        d=filedialog.askdirectory(title='Select data folder')
        if d: self.data_dir.set(d); self._scan_data_dir()

    def _browse_out(self):
        d=filedialog.askdirectory(title='Select output folder')
        if d: self.out_dir.set(d)

    def _get_out_dir(self):
        out=self.out_dir.get().strip()
        if not out: out=os.path.join(self.data_dir.get(),'results')
        os.makedirs(out,exist_ok=True); return out

    def _scan_data_dir(self):
        d=self.data_dir.get()
        if not os.path.isdir(d): return
        def _has_dats(p):
            try:
                files = os.listdir(p)
                # Accept .dat (classic) or .txt (EC-Lab single-file)
                return any(f.lower().endswith('.dat') or f.lower().endswith('.txt')
                           for f in files)
            except Exception: return False
        subdirs=sorted([x for x in os.listdir(d)
                        if os.path.isdir(os.path.join(d,x)) and _has_dats(os.path.join(d,x))])
        own_dats = (glob.glob(os.path.join(d,'*.dat')) +
                    glob.glob(os.path.join(d,'*.txt')))
        samples=[]
        if subdirs: samples=subdirs
        if own_dats: samples=['(this folder)']+samples
        if not samples:
            messagebox.showwarning('No data','No data files found.\nPlace EC-Lab .txt files (recommended) or BioLogic *time*.dat + *cap*.dat files in subfolders here.')
            return
        # Build checkboxes
        for w in list(self._chk_widgets.values()): w.destroy()
        self._chk_vars.clear(); self._chk_widgets.clear(); self._samples.clear()
        self._tau_auto_set = False   # reset so first sample auto-sets tau
        for name in samples:
            var=tk.BooleanVar(value=True)
            col=SAMPLE_COLORS[len(self._chk_vars)%len(SAMPLE_COLORS)]
            # Outer frame for this sample's row
            sf=tk.Frame(self.chk_frame,bg='white')
            sf.pack(side='left',padx=4)
            chk=tk.Checkbutton(sf,text=name,variable=var,
                               bg='white',fg=col,font=('sans',10,'bold'),
                               selectcolor='white',activebackground='white',
                               command=self._on_check)
            chk.pack(side='left')
            # Per-sample mass (only relevant when cap file is in mAh/g)
            mass_var=tk.DoubleVar(value=self.mass_mg.get())
            mass_lbl=tk.Label(sf,text='m for Rtot (mg):',bg='white',fg='#555',
                     font=('sans',8))
            mass_lbl.pack(side='left',padx=(4,0))
            me=tk.Spinbox(sf,textvariable=mass_var,from_=0.1,to=200.,
                          increment=0.5,width=5,font=('sans',9),
                          state='normal')
            me.bind('<Return>',lambda e,n=name: None)  # manual replot only
            me.pack(side='left')
            mass_unit_lbl=tk.Label(sf,text='mg',bg='white',fg='#bbb',
                     font=('sans',8))
            mass_unit_lbl.pack(side='left',padx=(1,4))
            # Q_max source toggle with clear label
            # Per-sample pulse exclusion fields
            excl_disc = tk.StringVar(value='')
            excl_chg  = tk.StringVar(value='')
            self._exclude_pulses[name] = {'disc': excl_disc, 'chg': excl_chg}
            tk.Label(sf,text='Excl disc#:',bg='white',fg='#555',font=('sans',8)).pack(side='left',padx=(6,0))
            exc_d_entry=tk.Entry(sf,textvariable=excl_disc,width=6,font=('sans',8))
            exc_d_entry.pack(side='left')
            # no auto-fire — Replot button only
            tk.Label(sf,text='chg#:',bg='white',fg='#555',font=('sans',8)).pack(side='left',padx=(4,0))
            exc_c_entry=tk.Entry(sf,textvariable=excl_chg,width=6,font=('sans',8))
            exc_c_entry.pack(side='left')
            # no auto-fire — Replot button only
            self._chk_vars[name]=var
            self._chk_widgets[name]=sf
            self._sample_controls[name]={'mass':mass_var,
                                         'mass_lbl':mass_lbl,'mass_spin':me,
                                         'mass_unit_lbl':mass_unit_lbl}
            # Pre-load all samples
            self._load_sample(name)
        self._replot()

    def _on_pulse_change(self):
        """Redraw fit tabs when pulse # changes. Stay on current tab."""
        active=self._get_active()
        if not active: return
        try:
            pidx=max(0,self._pulse_var.get()-1)
        except Exception:
            return
        try:
            tfs=float(self.t_fit_start.get()); tfe=float(self.t_fit_end.get())
            tau=float(self.tau_s.get())
            xi_min=float(self.xi_min.get()); xi_max=float(self.xi_max.get())
        except Exception:
            return
        # Find which tab is currently selected so we can stay on it
        cur_tab = self._active_tab[0] if hasattr(self, '_active_tab') else None
        # Redraw all per-pulse tabs (fast — only one pulse each)
        self._draw_sqrt_fits(active, tfs, tfe, pulse_idx=pidx)
        self._draw_kc_fits(active, tau, pulse_idx=pidx, xi_min=xi_min, xi_max=xi_max)
        self._draw_relax_kinetics_perpulse(active)
        self.root.update_idletasks()
        # Stay on whatever tab the user is currently viewing — never force a switch

    def _toggle_qnorm(self, var, name):
        _ST=[('max discharge',       '#e8f0fe','#1a56cc'),
             ('max charge',           '#fde8e8','#cc1a1a'),
             ('theoretical capacity', '#fff3cd','#7a5200')]
        cur=var.get()
        idx=next((i for i,(k,*_) in enumerate(_ST) if k==cur),0)
        k,bg,fg=_ST[(idx+1)%len(_ST)]
        var.set(k)
        ctrl=self._sample_controls.get(name,{})
        # qn_btn removed from UI
        self._replot()

    def _on_check(self):
        self._replot()

    def _load_sample(self,name):
        base=self.data_dir.get()
        folder=base if name=='(this folder)' else os.path.join(base,name)
        self._samples.setdefault(name,{'td':np.zeros((0,3)),
            'disc':{'Q':np.zeros(0),'V':np.zeros(0),'Q_max':0.,'unit':'mAh'},
            'chg':None,'cap_unit':'mAh','blocks':[],'dedx':None})
        tf,cf=find_files(folder)
        if not tf:
            self.status_lbl.config(
                text=f'No data files found in: {folder}  '
                      '(place .txt EC-Lab files or *time*.dat + *cap*.dat there)',
                fg='orange'); return
        try:
            ctrl=self._sample_controls.get(name,{})
            # For single-file format: auto-populate mass from file metadata
            if tf and is_eclab_single(tf[0]):
                try:
                    _,_,file_meta = parse_eclab_single(tf[0])
                    auto_mass = file_meta.get('mass_mg')
                    if auto_mass and auto_mass > 0:
                        if 'mass' in ctrl:
                            try: ctrl['mass'].set(round(auto_mass, 4))
                            except Exception: pass
                        else:
                            self.mass_mg.set(round(auto_mass, 4))
                except Exception:
                    pass
            mass=ctrl['mass'].get() if 'mass' in ctrl else self.mass_mg.get()
            td=load_time(tf)
            # Peek at cap file unit BEFORE loading segments (to configure UI)
            cap_unit_peek = 'mAh'
            if cf and not is_eclab_single(cf[0]):
                try:
                    _,_,cap_unit_peek = parse_cap(cf[0])
                except Exception:
                    pass
            disc,chg,cu=load_cap_segments(cf,mass_mg=mass,td=td)
            self._samples[name]={'td':td,'disc':disc,'chg':chg,
                                  'cap_unit':cu,'blocks':[],'dedx':None}
            # Show/grey the mass spinner based on cap file unit:
            # mAh/g → mass needed to convert to absolute; mAh → mass not needed
            try:
                ctrl = self._sample_controls.get(name, {})
                # mass only needed when file is in mAh, to compute C_th in mAh/g
                # when file is mAh/g, mass cancels in Q/Qmax — never needed
                is_absolute = (cap_unit_peek == 'mAh')
                mass_state  = 'normal' if is_absolute else 'disabled'
                mass_color  = '#888' if is_absolute else '#bbb'
                for wkey in ('mass_lbl', 'mass_spin', 'mass_unit_lbl'):
                    w = ctrl.get(wkey)
                    if w:
                        try: w.config(state=mass_state, fg=mass_color)
                        except Exception:
                            try: w.config(fg=mass_color)
                            except Exception: pass
            except Exception:
                pass
            # ── Auto-detect parameters from data (first sample only) ────────
            if not self._tau_auto_set:
                tau_detected = detect_pulse_duration(td)
                if tau_detected is not None:
                    try:
                        self.tau_s.set(tau_detected)
                        # Fit end = tau (pulse duration); fit start stays 60 s
                        self.t_fit_end.set(tau_detected)
                        self._tau_auto_set = True
                    except Exception:
                        pass

                # Auto-detect V_lo / V_hi: always EXPAND to cover all active samples.
                # When a second sample with a different voltage range is loaded,
                # the existing vlo/vhi must widen to include it, not replace it.
                try:
                    V_data = td[:, 0]
                    V_data = V_data[np.isfinite(V_data)]
                    if len(V_data) > 10:
                        v_min = float(np.min(V_data))
                        v_max = float(np.max(V_data))
                        margin = max((v_max - v_min) * 0.03, 0.05)
                        vlo_new = round(max(0., v_min - margin), 2)
                        vhi_new = round(v_max + margin, 2)
                        # Expand: take the wider of current spinner and new range
                        try:
                            cur_vlo = float(self.vlo.get())
                            cur_vhi = float(self.vhi.get())
                            vlo_new = min(vlo_new, cur_vlo)
                            vhi_new = max(vhi_new, cur_vhi)
                        except Exception:
                            pass
                        self.vlo.set(vlo_new)
                        self.vhi.set(vhi_new)
                        tau_str = f'{tau_detected:.0f} s' if tau_detected else '?'
                        self.status_lbl.config(
                            text=(f'Auto: tau={tau_str}  '
                                  f'V=[{vlo_new:.2f}–{vhi_new:.2f} V]  '
                                  f'fit end={tau_str}'),
                            fg='#1a7a1a')
                except Exception:
                    pass

                # Auto-set C_th from the detected maximum capacity.
                # - File in mAh/g: C_th = Q_max directly (no mass needed)
                # - File in mAh:   C_th = Q_max / mass_kg  (needs mass)
                try:
                    ctrl = self._sample_controls.get(name, {})
                    q_max = max(disc['Q_max'], chg['Q_max'] if chg else 0.)
                    if q_max > 0:
                        if cap_unit_peek == 'mAh/g':
                            # Already specific capacity — use directly, mass irrelevant
                            cth = round(q_max, 1)
                        else:
                            # Absolute mAh — need mass to express as mAh/g for C_th
                            mass = ctrl['mass'].get() if 'mass' in ctrl else self.mass_mg.get()
                            cth = round(q_max / (mass / 1000.), 1) if mass > 0 else 0.
                        if cth > 0 and 'th_cap' in ctrl:
                            ctrl['th_cap'].set(cth)
                except Exception:
                    pass
        except Exception as e:
            self.status_lbl.config(text=f'Error {name}: {e}',fg='red')

    # ── Replot ────────────────────────────────────────────────────────────────
    def _replot_debounced(self):
        if self._debounce_id: self.root.after_cancel(self._debounce_id)
        self._debounce_id=self.root.after(500,self._replot)

    def _replot(self,*_):
        active=self._get_active()
        if not active: return
        try:
            self._curve_data=[]  # reset export buffer
            tfs=float(self.t_fit_start.get()); tfe=float(self.t_fit_end.get())
            vlo=float(self.vlo.get());         vhi=float(self.vhi.get())
            mr2=float(self.min_r2.get());      de =float(self.delta_e.get())
            tau=float(self.tau_s.get())
            sig=float(self.sigma_clip.get())
            xi_min=float(self.xi_min.get())
            xi_max=float(self.xi_max.get())

            for name in active:
                if name not in self._samples: continue
                s=self._samples[name]
                if s['td'].shape[0]==0: continue
                # Reload cap (mass may have changed)
                base=self.data_dir.get()
                folder=base if name=='(this folder)' else os.path.join(base,name)
                _,cf=find_files(folder)
                ctrl=self._sample_controls.get(name,{})
                mass=ctrl['mass'].get() if 'mass' in ctrl else self.mass_mg.get()
                disc,chg,cu=load_cap_segments(cf,mass_mg=mass,td=s['td'])
                # SOD uses disc Q_max, SOC uses chg Q_max — each direction its own reference.
                q_max_disc = disc['Q_max'] if disc and disc['Q_max']>0 else 1.
                q_max_chg  = chg['Q_max']  if chg  and chg['Q_max']>0  else 1.
                q_max_val  = q_max_disc   # kept for legacy draw functions
                s['disc'],s['chg'],s['cap_unit']=disc,chg,cu
                s['q_max_disc']=q_max_disc; s['q_max_chg']=q_max_chg; s['q_max_val']=q_max_val
                s['blocks']=extract_blocks(s['td'],tfs,tfe,tau_s=tau)
                # ── Exclude blocks: time-window trim ─────────────────────────
                try:
                    tlo_h = float(self.trim_tlo.get()); thi_h = float(self.trim_thi.get())
                    if tlo_h < thi_h and thi_h > 0:
                        tlo_s = tlo_h * 3600.; thi_s = thi_h * 3600.
                        def _block_in_trim(b):
                            tp1 = b['_Tp'][-1]; tr0 = b['_Tr'][0]
                            return not (tp1 < tlo_s or tr0 > thi_s)
                        s['blocks'] = [b for b in s['blocks'] if not _block_in_trim(b)]
                except Exception:
                    pass
                # ── Exclude blocks: per-sample per-direction pulse numbers ────
                try:
                    ep = self._exclude_pulses.get(name, {})
                    s['excluded_blocks'] = []   # reset each replot
                    def _parse_excl(sv):
                        """Parse '3,7,12' -> {2,6,11} (0-based indices)."""
                        try:
                            return {int(x.strip())-1 for x in sv.get().split(',') if x.strip().isdigit()}
                        except Exception:
                            return set()
                    excl_d = _parse_excl(ep.get('disc', tk.StringVar()))
                    excl_c = _parse_excl(ep.get('chg',  tk.StringVar()))
                    if excl_d or excl_c:
                        disc_counter = chg_counter = 0
                        kept = []; excluded = []
                        for b in s['blocks']:
                            if b['cycle'] == 'discharge':
                                if disc_counter not in excl_d:
                                    kept.append(b)
                                else:
                                    excluded.append(b)
                                disc_counter += 1
                            else:
                                if chg_counter not in excl_c:
                                    kept.append(b)
                                else:
                                    excluded.append(b)
                                chg_counter += 1
                        s['blocks'] = kept
                        s['excluded_blocks'] = excluded
                    else:
                        s['excluded_blocks'] = []
                except Exception:
                    pass
                max_p=max((len([b for b in s2['blocks'] if b['cycle']=='discharge'])
                           for s2 in self._samples.values() if s2['blocks']),default=1)
                try: self._pulse_spin.config(to=max(1,max_p))
                except Exception: pass
                _ref=dict(disc); _ref['Q_max']=q_max_val
                s['dedx'],_=build_dedx(_ref)
                # Recompute sl_kc/r2_kc within xi window for D_kc
                from scipy import stats as _st3
                xi_lo=max(xi_min,0.); xi_hi=min(xi_max,99.)
                for b in s['blocks']:
                    xkc_b=b['_x_kc']; Vr_b=b['_Vr']
                    win=(xkc_b>=xi_lo)&(xkc_b<=xi_hi) if xi_hi>xi_lo else np.ones(len(xkc_b),bool)
                    if win.sum()>=2:
                        sl_w,_,r_w,*__=_st3.linregress(xkc_b[win],Vr_b[win])
                        b['sl_kc']=sl_w; b['r2_kc']=r_w**2
                # File format: Potential(V), Current(A), Elapsed Time(s)
                I_all=s['td'][:,1]
                I_disc_vals=np.abs(I_all[I_all<0])
                I_chg_vals =np.abs(I_all[I_all>0])
                I_app_disc=float(np.median(I_disc_vals)) if len(I_disc_vals) else 74.65e-6
                I_app_chg =float(np.median(I_chg_vals))  if len(I_chg_vals)  else I_app_disc
                I_app=I_app_disc  # used below per-block; charge blocks use I_app_chg
                # Do NOT call compute_d here — the block-level loop below uses
                # the spike-free dedx_d/dedx_c and correct per-cycle I_app.
                # Rebuild spike-free dedx from equilibrium V_eq values, then recompute D
                dedx_d, dedx_c = build_dedx_from_blocks(s['blocks'])
                s['dedx_d']=dedx_d; s['dedx_c']=dedx_c

                # ── Assign x_block and dEdx_fd ───────────────────────────────
                # x_block[n] = x_before_pulse_n + ΔQ_n/Q_ref
                #            = cumulative sum of (|I_pulse| * dt / 3600 / 1000) / Q_ref
                #
                # Rationale: the composition x after pulse n equals the composition
                # before that pulse plus the fraction of reference capacity passed
                # during that pulse.  This is computed directly from the time file
                # (I×dt) rather than the cap file, which also records Q during
                # relaxation periods (leakage/drift) and would overestimate x.
                #
                # dEdx_fd[n] = (V_eq[n] - V_eq[n-1]) / (x_block[n] - x_block[n-1])
                #            = finite-difference approximation of dV_eq/dx.
                # This is the thermodynamic factor used in both D_conv and D_kc.
                # V_eq[n] is the last recorded voltage of the relaxation after pulse n
                # (true equilibrium OCV at composition x_block[n]).
                # Block 1 has no dEdx_fd (no previous V_eq to difference against).
                # x_block = SOD or SOC in fraction [0,1]
                # SOD: cumulative Q_pulse / Q_max_disc  (discharge own reference)
                # SOC: cumulative Q_pulse / Q_max_chg   (charge own reference)
                # ── Assign x_block ───────────────────────────────────────────
                # x_block[n] = Q_pulse_n / Q_last_pulse  (fraction of total)
                #
                # Q is computed by integrating |I|*dt cumulatively over the full
                # time array.  For each direction (discharge/charge) the offset
                # at the START of that direction's first pulse is subtracted so
                # the two directions are independent.
                #
                # Denominator = Q_cumul at the last pulse end of that direction
                # (includes dropped last pulse if it exists — so OCV dots whose
                # last pulse was dropped stop just before 100%).
                #
                # The bottom E-vs-SOD row uses disc['Q']/disc['Q_max'] which
                # resets to 0 at the start of each half-cycle in the cap file.
                # To make the two rows share the same x-axis we also store
                # q_max_disc_pulse / q_max_chg_pulse (the same denominators)
                # and use them in _draw_ocv for the bottom row.
                td_I = s['td'][:, 1]; td_T = s['td'][:, 2]
                thresh_td = _current_threshold(td_I)
                active_td = np.abs(td_I) > thresh_td

                # Global cumulative Q (direction-independent, all current)
                dT_td = np.diff(td_T)
                I_mid_td = (np.abs(td_I[:-1]) + np.abs(td_I[1:])) / 2.
                Q_cumul_td = np.concatenate([[0.], np.cumsum(I_mid_td * dT_td)]) / 3.6

                # ── Find ALL pulse ends per direction from the time file ──────
                # cyc_blocks only contains KEPT blocks (last pulse dropped if no
                # relaxation).  The denominator must include that dropped pulse so
                # OCV dots stop just before 100 % instead of reaching it.
                # We scan the raw time array for every pulse→rest transition and
                # record the last pulse-end time per direction independently.
                all_tr = np.where(np.diff((active_td).astype(int)) == -1)[0]
                # For each transition find its current sign
                last_disc_pulse_end_T = None
                last_chg_pulse_end_T  = None
                first_disc_pulse_start_T = None
                first_chg_pulse_start_T  = None
                for idx_tr in all_tr:
                    ps_tr = idx_tr
                    while ps_tr > 0 and active_td[ps_tr - 1]: ps_tr -= 1
                    I_sign = np.sign(td_I[ps_tr])
                    if I_sign < 0:   # discharge
                        if first_disc_pulse_start_T is None:
                            first_disc_pulse_start_T = td_T[ps_tr]
                        last_disc_pulse_end_T = td_T[idx_tr]
                    elif I_sign > 0: # charge
                        if first_chg_pulse_start_T is None:
                            first_chg_pulse_start_T = td_T[ps_tr]
                        last_chg_pulse_end_T = td_T[idx_tr]

                # If the file ends while a pulse is still active (no trailing
                # relaxation), the last pulse has no pulse→rest transition and
                # is missed by the loop above.  Detect this and add the
                # end-of-file time as a synthetic pulse end for that direction.
                if active_td[-1]:
                    # Walk back from end to find the start of this final pulse
                    ps_eof = len(active_td) - 1
                    while ps_eof > 0 and active_td[ps_eof - 1]: ps_eof -= 1
                    I_sign_eof = np.sign(td_I[ps_eof])
                    if I_sign_eof < 0:
                        if first_disc_pulse_start_T is None:
                            first_disc_pulse_start_T = td_T[ps_eof]
                        last_disc_pulse_end_T = td_T[-1]
                    elif I_sign_eof > 0:
                        if first_chg_pulse_start_T is None:
                            first_chg_pulse_start_T = td_T[ps_eof]
                        last_chg_pulse_end_T = td_T[-1]

                def _q_at_T(t):
                    if t is None: return 0.
                    row = int(np.clip(np.searchsorted(td_T, t, side='right')-1,
                                      0, len(td_T)-1))
                    return float(Q_cumul_td[row])

                # Per-direction offsets and total Q (including dropped last pulse)
                q_off_disc  = _q_at_T(first_disc_pulse_start_T)
                q_off_chg   = _q_at_T(first_chg_pulse_start_T)
                q_tot_disc  = _q_at_T(last_disc_pulse_end_T) - q_off_disc
                q_tot_chg   = _q_at_T(last_chg_pulse_end_T)  - q_off_chg

                for cyc, cyc_blocks in [
                    ('discharge', [b for b in s['blocks'] if b['cycle']=='discharge']),
                    ('charge',    [b for b in s['blocks'] if b['cycle']=='charge']),
                ]:
                    if not cyc_blocks:
                        continue
                    q_off   = q_off_disc   if cyc == 'discharge' else q_off_chg
                    q_denom = q_tot_disc   if cyc == 'discharge' else q_tot_chg
                    if q_denom < 1e-9: q_denom = 1.
                    key = 'q_max_disc_pulse' if cyc == 'discharge' else 'q_max_chg_pulse'
                    s[key] = float(q_denom)

                    for b in cyc_blocks:
                        Tp = b['_Tp']
                        row = int(np.clip(
                            np.searchsorted(td_T, Tp[-1], side='right') - 1,
                            0, len(td_T) - 1))
                        qe = Q_cumul_td[row] - q_off
                        b['x_block'] = float(np.clip(qe / q_denom, 0., 1.))
                        b['dEdx_fd'] = np.nan   # kept for compatibility
                for b in s['blocks']:
                    cyc_dedx = dedx_d if b['cycle']=='discharge' else dedx_c
                    I_b = I_app_disc if b['cycle']=='discharge' else I_app_chg
                    if cyc_dedx is not None and np.isfinite(b.get('V_eq',np.nan)):
                        dv=float(cyc_dedx(b['V_eq']))
                        b['dEdx']=dv if np.isfinite(dv) else np.nan
                    else:
                        b['dEdx']=np.nan
                    pref=4*I_b**2/(de**2*96485.**2*np.pi)
                    if (np.isfinite(b['sl_sqrt']) and abs(b['sl_sqrt'])>1e-10
                            and np.isfinite(b.get('dEdx',np.nan)) and abs(b.get('dEdx',0))>1e-8):
                        b['D_conv']=pref*(b['dEdx']/b['sl_sqrt'])**2
                    else:
                        b['D_conv']=np.nan
                    # Re-fit sl_kc within the user's [xi_min, xi_max] window.
                    # The value stored from extract_blocks uses the full range —
                    # we must recompute here so D_kc matches what is displayed.
                    xkc = b['_x_kc']; Vr = b['_Vr']
                    xi_lo_r = max(float(xi_min), 0.)
                    xi_hi_r = min(float(xi_max), 99.)
                    win = (xkc >= xi_lo_r) & (xkc <= xi_hi_r) if xi_hi_r > xi_lo_r                           else np.ones(len(xkc), bool)
                    if win.sum() >= 4:
                        sl_kc_w, _, r_w, *_ = stats.linregress(xkc[win], Vr[win])
                        r2_kc_w = r_w ** 2
                    else:
                        sl_kc_w = b.get('sl_kc', np.nan)
                        r2_kc_w = b.get('r2_kc', np.nan)
                    # Update block so plots, exports and D_kc are consistent
                    b['sl_kc']  = float(sl_kc_w)  if np.isfinite(sl_kc_w)  else np.nan
                    b['r2_kc']  = float(r2_kc_w)  if np.isfinite(r2_kc_w)  else np.nan
                    dEdx_kc = b.get('dEdx', np.nan)
                    if not np.isfinite(dEdx_kc) and cyc_dedx is not None and np.isfinite(b.get('V_eq', np.nan)):
                        dv_fb = float(cyc_dedx(b['V_eq']))
                        dEdx_kc = dv_fb if np.isfinite(dv_fb) else np.nan
                    sl_kc = b['sl_kc']
                    if (np.isfinite(sl_kc) and abs(sl_kc) > 1e-10
                            and np.isfinite(dEdx_kc) and abs(dEdx_kc) > 1e-8):
                        b['D_kc'] = pref * (dEdx_kc / sl_kc) ** 2
                    else:
                        b['D_kc'] = np.nan
                    b['D_rel']=b['D_conv']  # keep D_rel in sync with final D_conv
                for b in s['blocks']:
                    self._curve_data.append({
                        'sample':name,'cycle':b['cycle'],
                        'V_eq':round(b['V_eq'],6),
                        'sl_sqrt':b['sl_sqrt'],'r2_sqrt':b['r2_sqrt'],
                        'n_fit':b['n_fit'],
                        'sl_kc':b['sl_kc'],'r2_kc':b['r2_kc'],
                        'dEdx':b.get('dEdx',np.nan),
                        'D_conv':b.get('D_conv',np.nan),
                        'D_kc':b.get('D_kc',np.nan),
                    })

            n=len(active)
            self.status_lbl.config(
                text=f'{n} sample(s) active: {", ".join(active)}',fg='#333')

            self._draw_gitt_full(active,tfs,tfe,vlo,vhi)
            self._draw_all_pulses(active,tfs,tfe)
            self._draw_all_xi(active,tau)
            pidx=max(0,self._pulse_var.get()-1)
            self._draw_sqrt_fits(active,tfs,tfe,pulse_idx=pidx)
            self._draw_kc_fits(active,tau,pulse_idx=pidx,xi_min=xi_min,xi_max=xi_max)
            self._draw_dedx(active,vlo,vhi)
            self._draw_d(active,'D_conv','D_conv',vlo,vhi,mr2,'r2_sqrt',sig,
                         r'Conventional GITT  $\tilde{D}\cdot(S/v_{\mathrm{m}})^2$ / mol$^2$ s$^{-1}$  (Weppner \& Huggins 1977)')
            self._draw_d(active,'D_kc','D_kc',vlo,vhi,mr2,'r2_kc',sig,
                         r'K\&C GITT  $\tilde{D}\cdot(S/v_{\mathrm{m}})^2$ / mol$^2$ s$^{-1}$  (Kang \& Chueh 2021)')
            self._draw_kc_slopes(active,vlo,vhi,mr2)
            self._draw_ocv(active)
            self._draw_overpot_curves(active)
            self._draw_overpotential(active)
            self._draw_relax_curves(active)
            self._draw_relax_delta(active)
            self._draw_relax_kinetics(active)
            self._draw_relax_kinetics_perpulse(active)
            self._draw_relax_map(active)
            self._draw_relax_map_soc(active)

        except Exception as e:
            import traceback; traceback.print_exc()
            self.status_lbl.config(text=f'Error: {e}',fg='red')

    # ── Draw helpers ──────────────────────────────────────────────────────────
    def _clf(self, key):
        """Return a brand-new Figure wired to a fresh FigureCanvasTkAgg widget.

        Destroying and recreating the canvas widget is the only reliable way to
        clear the Tk pixel buffer.  Simply replacing canvas.figure or calling
        clf()+draw() leaves the old raster image visible on platforms where the
        widget is not automatically invalidated on figure replacement.
        """
        from matplotlib.figure import Figure as _MplFig
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg as _Canvas,
            NavigationToolbar2Tk as _Toolbar)

        tab_info = self.tabs[key]
        old_canvas = tab_info['canvas']
        old_fig    = tab_info['fig']

        # Get the parent Tk frame that owns the old canvas widget
        parent_frame = old_canvas.get_tk_widget().master

        # Destroy the old canvas widget and its toolbar cleanly
        try:
            old_fig.clf()
            old_fig.set_canvas(None)
        except Exception:
            pass
        try:
            # Destroy all children of the parent frame (canvas + toolbar)
            for w in parent_frame.winfo_children():
                w.destroy()
        except Exception:
            pass

        # Create fresh figure + canvas + toolbar in the same parent frame
        new_fig    = _MplFig(facecolor='white')
        new_canvas = _Canvas(new_fig, master=parent_frame)
        try:
            toolbar = _Toolbar(new_canvas, parent_frame)
            toolbar.update()
        except Exception:
            pass
        new_canvas.get_tk_widget().pack(fill='both', expand=True)

        tab_info['fig']    = new_fig
        tab_info['canvas'] = new_canvas
        return new_fig

    def _flush(self, key):
        """Render the canvas for tab `key`.

        If the tab is currently visible, draw immediately.
        Otherwise mark it dirty — it will be drawn when the user switches to it.
        This avoids rendering all 18 tab figures on every replot; only the visible
        one is drawn to screen eagerly.  All figures are still updated in memory
        so Export Figures always produces up-to-date output.
        """
        cur = self._active_tab[0] if hasattr(self, '_active_tab') else None
        if key == cur:
            try:
                self.tabs[key]['canvas'].draw()
            except Exception:
                try:
                    self.tabs[key]['canvas'].draw_idle()
                except Exception:
                    pass
        else:
            # Mark dirty: draw_idle schedules a paint when the widget next becomes
            # visible, without blocking the current replot loop.
            try:
                self.tabs[key]['canvas'].draw_idle()
            except Exception:
                pass

    def _draw_gitt_full(self, active, tfs, tfe, vlo, vhi):
        import matplotlib.gridspec as gridspec
        n = len(active)
        n_rows = 1 if n == 1 else 2
        n_cols = 2 if n == 1 else n
        ws = 0.45 if n >= 4 else 0.38 if n == 3 else 0.30
        fig = self._clf('gitt_full')
        fig.set_size_inches(*self._figsize(n_rows, n_cols))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=ws,
                               top=0.88, bottom=0.10, left=0.07, right=0.97)
        for ci, name in enumerate(active):
            s = self._samples[name]; col = self._color(name)
            td = s['td']
            ax1 = fig.add_subplot(gs[0, ci])
            ax2 = fig.add_subplot(gs[0 if n == 1 else 1, 1 if n == 1 else ci])
            if td.shape[0] == 0:
                ax1.set_title(f'{name}  (no data)', fontsize=9, color='#aaa')
                continue
            V, I, T = td[:,0], td[:,1], td[:,2]
            # shade trimmed zones if set
            try:
                tlo = float(self.trim_tlo.get()); thi = float(self.trim_thi.get())
                if tlo < thi:
                    ax1.axvspan(tlo, thi, color='#ffcccc', alpha=0.35, zorder=0, label='trimmed')
                    ax2.axvspan(tlo, thi, color='#ffcccc', alpha=0.35, zorder=0)
            except Exception:
                pass
            ax1.plot(T/3600, V, lw=0.4, color='#ccc', zorder=1)
            for b in s['blocks']:
                Vp, Tp = b['_Vp'], b['_Tp']; fm = b['_fm']
                ax1.plot(Tp/3600, Vp, lw=1, color=col, alpha=0.2, zorder=2)
                if fm.sum() > 0:
                    ax1.plot(Tp[fm]/3600, Vp[fm], lw=2, color=GREEN, alpha=0.85, zorder=3)
            # Draw excluded blocks in red so user can confirm the right pulse was removed
            for b in s.get('excluded_blocks', []):
                Vp, Tp = b['_Vp'], b['_Tp']
                Vr, Tr = b['_Vr'], b['_Tr']
                ax1.plot(Tp/3600, Vp, lw=2.0, color=RED, alpha=0.85, zorder=4)
                ax1.plot(Tr/3600, Vr, lw=1.5, color=RED, alpha=0.55, zorder=4,
                         ls='--')
                # Shade the full block (pulse + relaxation) in red
                t_start = Tp[0]/3600; t_end = Tr[-1]/3600
                ax1.axvspan(t_start, t_end, color=RED, alpha=0.08, zorder=0)
            if vlo: ax1.axhline(vlo, color=RED, ls='--', lw=0.8, alpha=0.5)
            if vhi: ax1.axhline(vhi, color='navy', ls='--', lw=0.8, alpha=0.5)
            ax1.set_ylabel('E (V)', fontsize=9); ax1.grid(True, alpha=0.3)
            ax1.set_title(name, fontsize=9, color=col, fontweight='bold')
            ax2.plot(T/3600, I*1e6, lw=0.5, color=col, alpha=0.7)
            ax2.set_ylabel('I (µA)', fontsize=9); ax2.grid(True, alpha=0.3)
            ax1.set_xlabel('Time (h)', fontsize=9)
            ax2.set_xlabel('Time (h)', fontsize=9)
        fig.suptitle(f'GITT curves  |  green=[{tfs:.0f}–{tfe:.0f}s]',
                     fontsize=10, fontweight='bold', color='#2c5f8a')
        self._flush('gitt_full')

    def _draw_all_pulses(self, active, tfs, tfe):
        import matplotlib.cm as cm, matplotlib.colors as mcolors
        import matplotlib.gridspec as gridspec
        n = max(len(active), 1)
        n_rows = 1 if n == 1 else 2
        n_cols = 2 if n == 1 else n
        ws = 0.45 if n >= 4 else 0.38 if n == 3 else 0.30
        fig = self._clf('all_pulses')
        fig.set_size_inches(*self._figsize(n_rows, n_cols))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=ws,
                               top=0.88, bottom=0.10, left=0.07, right=0.96)
        try:
            p_from = max(1, int(self.pulse_from.get())) - 1
            p_to   = max(1, int(self.pulse_to.get()))
        except Exception:
            p_from, p_to = 0, 999
        rng = f'pulses {p_from+1} to {min(p_to, 999)}'
        cmap = cm.get_cmap('turbo')

        for ci, name in enumerate(active):
            s = self._samples[name]
            for ri, cyc in enumerate(['discharge', 'charge']):
                ax = fig.add_subplot(gs[0 if n == 1 else ri, ri if n == 1 else ci])
                all_bl = [b for b in s['blocks'] if b['cycle'] == cyc]
                bl = all_bl[p_from:p_to]
                veqs_fin = [b.get('V_eq', float('nan')) for b in bl
                            if b.get('V_eq', float('nan')) == b.get('V_eq', float('nan'))]
                vmin = min(veqs_fin) if veqs_fin else 0.
                vmax = max(veqs_fin) if veqs_fin else 1.
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                for b in bl:
                    sq = b['_sqrt_t']; Vp = b['_Vp']; fm = b['_fm']
                    dE = Vp - Vp[0]; nz = b['_t_rel'] > 0
                    veq = b.get('V_eq', float('nan'))
                    c = cmap(norm(veq)) if veq == veq else '#aaa'
                    ax.plot(sq[nz], dE[nz], '-', lw=0.8, color=c, alpha=0.75)
                    if fm.sum() >= 2 and b['sl_sqrt'] == b['sl_sqrt']:
                        xf = np.linspace(sq[fm].min(), sq[fm].max(), 50)
                        ic = np.mean(dE[fm]) - b['sl_sqrt'] * np.mean(sq[fm])
                        ax.plot(xf, b['sl_sqrt']*xf+ic, '--', lw=1.2, color=c, alpha=0.9)
                if bl:
                    sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
                    cb = fig.colorbar(sm, ax=ax,
                                   fraction=max(0.025, 0.06/max(n,1)),
                                   pad=0.01)
                    cb.set_label(r'$E_\mathrm{eq}$ (V)', fontsize=7,
                                 rotation=270, labelpad=8)
                if tfs < tfe:
                    ax.axvspan(np.sqrt(max(tfs,0.001)), np.sqrt(tfe), alpha=0.07, color=GREEN)
                ax.axvline(np.sqrt(max(tfs,0.001)), color=GREEN, ls=':', lw=1.2)
                ax.axvline(np.sqrt(max(tfe,0.001)), color=GREEN, ls=':', lw=1.2)
                ax.axhline(0, color='k', lw=0.5, ls='--', alpha=0.4)
                ax.set_xlabel(r'$\sqrt{t}$ (s$^{1/2}$)', fontsize=9)
                ax.set_ylabel(r'$\Delta E$ (V)', fontsize=9)
                lbl_cyc = 'Discharge' if cyc == 'discharge' else 'Charge'
                ax.set_title(f'{name}  |  {lbl_cyc}', fontsize=9,
                             fontweight='bold', color=self._color(name))
                ax.grid(True, alpha=0.25)

        fig.suptitle(
            f'All pulses  |  sqrt(t)  |  {rng}  |  colour = E_eq (turbo)  |  fit [{tfs:.0f}-{tfe:.0f} s]',
            fontsize=10, fontweight='bold', color='#1a6b35')
        self._flush('all_pulses')


    def _draw_sqrt_fits(self,active,tfs,tfe,pulse_idx=None):
        """2-row layout: discharge (top) / charge (bottom), one column per sample."""
        import matplotlib.gridspec as gridspec
        fig=self._clf('sqrt_fits')
        n=len(active)
        n_rows = 1 if n == 1 else 2
        n_cols = 2 if n == 1 else n
        ws = 0.45 if n >= 4 else 0.38 if n == 3 else 0.30
        if n==0: self._flush('sqrt_fits'); return
        fig.set_size_inches(*self._figsize(n_rows, n_cols))
        gs=gridspec.GridSpec(n_rows, n_cols,figure=fig,hspace=0.45,wspace=ws,
                             top=0.88,bottom=0.10,left=0.07,right=0.97)
        max_pulses=max(
            (len([b for b in self._samples[nm]['blocks'] if b['cycle']=='discharge'])
             for nm in active), default=1)
        if pulse_idx is None: pulse_idx=0
        pidx=int(np.clip(pulse_idx,0,max(0,max_pulses-1)))
        for ci,name in enumerate(active):
            s=self._samples[name]; col=self._color(name)
            disc_bl=[b for b in s['blocks'] if b['cycle']=='discharge']
            chg_bl =[b for b in s['blocks'] if b['cycle']=='charge']
            disc_bl_r = list(reversed(disc_bl)) if self.reverse_mode.get() in ('discharge','both') else list(disc_bl)
            chg_bl_sq = list(reversed(chg_bl)) if self.reverse_mode.get() in ('charge','both') else list(chg_bl)
            for row,(blist,cyc_lbl) in enumerate([(disc_bl_r,'Discharge'),(chg_bl_sq,'Charge')]):
                ax=fig.add_subplot(gs[0 if n == 1 else row, row if n == 1 else ci])
                if pidx>=len(blist):
                    ax.set_title(f'{name} | {cyc_lbl} (no block #{pidx+1})',fontsize=8,color='#aaa'); continue
                b=blist[pidx]
                sq=b['_sqrt_t']; Vp=b['_Vp']; fm=b['_fm']; t_rel=b['_t_rel']
                nz=t_rel>0
                dV=Vp-Vp[0]
                ax.plot(sq[nz],dV[nz],'.',ms=4,color='#cccccc',alpha=0.8,
                        zorder=1,label='all points')
                if fm.sum()>=2:
                    ax.plot(sq[fm],dV[fm],'o',ms=6,color=col,
                            markeredgecolor='#333',markeredgewidth=0.4,
                            zorder=5,label=f'fit [{tfs:.0f}-{tfe:.0f}s]')
                    sl=b['sl_sqrt']; r2=b.get('r2_sqrt',np.nan)
                    if np.isfinite(sl):
                        xf=np.linspace(sq[fm].min(),sq[fm].max(),80)
                        ic=np.mean(dV[fm])-sl*np.mean(sq[fm])
                        ax.plot(xf,sl*xf+ic,'-',color=col,lw=2.0,zorder=6,
                                label=f'R2={r2:.3f}  s={sl:+.5f}')
                # Shade the actual fit window — start from first nonzero t_rel
                # when tfs=0 (fm excludes t=0 to avoid IR-drop point)
                nz_t = t_rel[t_rel>0]
                t_shade_lo = float(nz_t.min()) if len(nz_t) else max(tfs,0.01)
                t_shade_lo = max(tfs, t_shade_lo) if tfs > 0 else t_shade_lo
                ax.axvspan(np.sqrt(t_shade_lo), np.sqrt(max(tfe,0.01)),
                           alpha=0.08, color=col, zorder=0)
                ax.set_xlabel('sqrt(t)  (s^0.5)',fontsize=8)
                ax.set_ylabel(r'$\Delta E$  (V)',fontsize=8)
                veq=b.get('V_eq',np.nan)
                veq_str=f'  E_eq={veq:.4f} V' if np.isfinite(veq) else ''
                lbl='Discharge' if cyc_lbl=='Discharge' else 'Charge'
                ax.set_title(f'{name}  |  {lbl}{veq_str}',
                             fontsize=8,color=col,fontweight='bold')
                ax.legend(fontsize=7,loc='best',framealpha=0.85)
                ax.grid(True,alpha=0.3)
        fig.suptitle(
            f'Conv. sqrt(t) fits  |  Pulse #{pidx+1} of {max_pulses}'
            f'  |  fit window [{tfs:.0f} - {tfe:.0f} s]',
            fontsize=10,fontweight='bold',color='#1a6b35')
        self._flush('sqrt_fits')

    def _draw_all_xi(self, active, tau):
        import matplotlib.cm as cm, matplotlib.colors as mcolors
        import matplotlib.gridspec as gridspec
        n = max(len(active), 1)
        n_rows = 1 if n == 1 else 2
        n_cols = 2 if n == 1 else n
        ws = 0.45 if n >= 4 else 0.38 if n == 3 else 0.30
        fig = self._clf('all_xi')
        fig.set_size_inches(*self._figsize(n_rows, n_cols))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=ws,
                               top=0.88, bottom=0.10, left=0.07, right=0.96)
        try:
            xi_min = float(self.xi_min.get()); xi_max = float(self.xi_max.get())
        except Exception:
            xi_min = 0.; xi_max = 99.
        xi_lo = max(xi_min, 0.); xi_hi = min(xi_max, 99.)
        try:
            p_from = max(1, int(self.pulse_from.get())) - 1
            p_to   = max(1, int(self.pulse_to.get()))
        except Exception:
            p_from, p_to = 0, 999
        rng = f'pulses {p_from+1} to {min(p_to, 999)}'
        cmap = cm.get_cmap('turbo')

        for ci, name in enumerate(active):
            s = self._samples[name]
            for ri, cyc in enumerate(['discharge', 'charge']):
                ax = fig.add_subplot(gs[0 if n == 1 else ri, ri if n == 1 else ci])
                all_bl = [b for b in s['blocks'] if b['cycle'] == cyc and len(b['_x_kc']) >= 2]
                bl = all_bl[p_from:p_to]
                valid_v = [b.get('V_eq', float('nan')) for b in bl
                           if b.get('V_eq', float('nan')) == b.get('V_eq', float('nan'))]
                vmin = min(valid_v) if valid_v else 0.
                vmax = max(valid_v) if valid_v else 1.
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                for b in bl:
                    xkc = b['_x_kc']; Vr = b['_Vr']
                    veq = b.get('V_eq', float('nan'))
                    c = cmap(norm(veq)) if veq == veq else '#aaa'
                    dE = (Vr - veq) * 1000. if veq == veq else Vr * 1000.
                    ax.plot(xkc, dE, '-', lw=0.8, color=c, alpha=0.75)
                    win = (xkc >= xi_lo) & (xkc <= xi_hi) if xi_hi > xi_lo else np.ones(len(xkc), bool)
                    if win.sum() >= 2 and b.get('sl_kc', float('nan')) == b.get('sl_kc', float('nan')):
                        sl_w, ic_w, *_ = stats.linregress(xkc[win], Vr[win])
                        xf = np.linspace(xkc[win].min(), xkc[win].max(), 60)
                        dEf = (sl_w*xf + ic_w - veq)*1000. if veq==veq else (sl_w*xf+ic_w)*1000.
                        ax.plot(xf, dEf, '--', lw=1.2, color=c, alpha=0.9)
                if bl:
                    sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
                    cb = fig.colorbar(sm, ax=ax,
                                   fraction=max(0.025, 0.06/max(n,1)),
                                   pad=0.01)
                    cb.set_label(r'$E_\mathrm{eq}$ (V)', fontsize=7,
                                 rotation=270, labelpad=8)
                ax.invert_xaxis()
                ax.axhline(0, color='#555', lw=0.8, ls='--', alpha=0.5)
                if xi_hi < 99. or xi_lo > 0.:
                    ax.axvspan(xi_lo, xi_hi, alpha=0.07, color='#8a2c2c')
                ax.set_xlabel(r'$\xi=\sqrt{t_{\rm relax}+\tau}-\sqrt{t_{\rm relax}}$ (s$^{1/2}$)', fontsize=9)
                ax.set_ylabel(r'$E-E_\mathrm{eq}$ (mV)', fontsize=9)
                lbl_cyc = 'Discharge' if cyc == 'discharge' else 'Charge'
                ax.set_title(f'{name}  |  {lbl_cyc}', fontsize=9,
                             fontweight='bold', color=self._color(name))
                ax.grid(True, alpha=0.25)

        fig.suptitle(
            f'All xi pulses  |  K&C  |  {rng}  |  colour = E_eq (turbo)  |  tau = {tau:.0f} s',
            fontsize=10, fontweight='bold', color='#8a2c2c')
        self._flush('all_xi')


    def _draw_kc_fits(self,active,tau,pulse_idx=None,xi_min=0.,xi_max=99.):
        """2-row layout: discharge (top) / charge (bottom), one column per sample."""
        import matplotlib.gridspec as gridspec
        fig=self._clf('kc_fits')
        n=len(active)
        n_rows = 1 if n == 1 else 2
        n_cols = 2 if n == 1 else n
        ws = 0.45 if n >= 4 else 0.38 if n == 3 else 0.30
        if n==0: self._flush('kc_fits'); return
        fig.set_size_inches(*self._figsize(n_rows, n_cols))
        gs=gridspec.GridSpec(n_rows, n_cols,figure=fig,hspace=0.45,wspace=ws,
                             top=0.88,bottom=0.10,left=0.07,right=0.97)
        max_pulses=max(
            (len([b for b in self._samples[nm]['blocks'] if b['cycle']=='discharge'])
             for nm in active), default=1)
        if pulse_idx is None: pulse_idx=0
        pidx=int(np.clip(pulse_idx,0,max(0,max_pulses-1)))
        for ci,name in enumerate(active):
            s=self._samples[name]; col=self._color(name)
            disc_bl=[b for b in s['blocks'] if b['cycle']=='discharge']
            chg_bl =[b for b in s['blocks'] if b['cycle']=='charge']
            disc_bl_r = list(reversed(disc_bl)) if self.reverse_mode.get() in ('discharge','both') else list(disc_bl)
            chg_bl_kc = list(reversed(chg_bl)) if self.reverse_mode.get() in ('charge','both') else list(chg_bl)
            for row,(blist,cyc_lbl) in enumerate([(disc_bl_r,'Discharge'),(chg_bl_kc,'Charge')]):
                ax=fig.add_subplot(gs[0 if n == 1 else row, row if n == 1 else ci])
                if pidx>=len(blist):
                    ax.set_title(f'{name} | {cyc_lbl} (no block #{pidx+1})',fontsize=8,color='#aaa'); continue
                b=blist[pidx]
                xkc=b['_x_kc']; Vr=b['_Vr']
                veq=b.get('V_eq',np.nan)
                # Plot E - E_eq so the reference is 0 and curves align across pulses
                dE_r = (Vr - veq)*1000. if np.isfinite(veq) else Vr*1000.
                ax.plot(xkc, dE_r, 'o', ms=4, color=col, alpha=0.85,
                        markeredgecolor='none', label='relaxation data')
                xi_lo=max(xi_min,0.); xi_hi=min(xi_max,99.)
                win=(xkc>=xi_lo)&(xkc<=xi_hi) if xi_hi>xi_lo else np.ones(len(xkc),bool)
                if win.sum()>=2:
                    from scipy import stats as _st
                    sl_w,ic_w,r_w,*_=_st.linregress(xkc[win],Vr[win]); r2_w=r_w**2
                else:
                    sl_w=b['sl_kc']; r2_w=b.get('r2_kc',np.nan)
                    ic_w=np.mean(Vr)-sl_w*np.mean(xkc) if np.isfinite(sl_w) else 0.
                if np.isfinite(sl_w):
                    xf_lo=xkc[win].min() if win.sum()>=2 else xkc.min()
                    xf_hi=xkc[win].max() if win.sum()>=2 else xkc.max()
                    xf=np.linspace(xf_lo,xf_hi,80)
                    dE_fit=(sl_w*xf+ic_w - veq)*1000. if np.isfinite(veq) else (sl_w*xf+ic_w)*1000.
                    ax.plot(xf, dE_fit, '-', color=col, lw=2.0, zorder=6,
                            label=f'R2={r2_w:.3f}  s={sl_w:+.5f} V/s½')
                    if xi_hi<30. or xi_lo>0.:
                        ax.axvspan(xi_lo,min(xi_hi,xkc.max()),alpha=0.10,
                                   color=col,zorder=0,label='fit range')
                ax.axhline(0, color='#555', lw=0.8, ls='--', alpha=0.5)
                ax.invert_xaxis()
                ax.set_xlabel(r'$\xi = \sqrt{t+\tau} - \sqrt{t}$  (s$^{1/2}$)  $\rightarrow$ time $\rightarrow$', fontsize=7)
                ax.set_ylabel(r'$E - E_\mathrm{eq}$  (mV)', fontsize=8)
                veq_str=f'  E_eq={veq:.4f} V' if np.isfinite(veq) else ''
                lbl='Discharge' if cyc_lbl=='Discharge' else 'Charge'
                ax.set_title(f'{name}  |  {lbl}{veq_str}',
                             fontsize=8,color=col,fontweight='bold')
                ax.legend(fontsize=7,loc='best',framealpha=0.85)
                ax.grid(True,alpha=0.3)
        fig.suptitle(
            f'K&C xi fits  |  Pulse #{pidx+1} of {max_pulses}  |  tau = {tau:.0f} s',
            fontsize=10,fontweight='bold',color='#8a2c2c')
        self._flush('kc_fits')

    def _draw_dedx(self, active, vlo, vhi):
        """Slopes and thermodynamic factor dE/dx.

        Top row   : |dE/dsqrt(t)| vs E_eq  (slope of W&H fit per block)
        Bottom row: |dE/dx| vs E_eq

        dE/dx mode (controlled by the UI radio button):
          'differential'  — smooth interpolator from build_dedx_from_blocks,
                            evaluated at each block's V_eq.  Physically this is
                            the gradient of the equilibrium OCV curve dV_eq/dx,
                            which is the thermodynamic factor entering D.
          'finite_diff'   — block-to-block finite difference:
                            dE/dx[n] = (V_eq[n] - V_eq[n-1]) / (x[n] - x[n-1])
                            This is the raw numerical derivative from consecutive
                            equilibrium points and is noisier but model-free.
        """
        try:
            mode = self.dedx_mode.get()
        except Exception:
            mode = 'differential'

        fig = self._clf('dEdx')
        ax1 = fig.add_subplot(221); ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223); ax4 = fig.add_subplot(224)

        for name in active:
            s = self._samples[name]; col = self._color(name)
            for ax_sl, ax_dx, cyc in [(ax1, ax3, 'discharge'), (ax2, ax4, 'charge')]:
                blist = [b for b in s['blocks']
                         if b['cycle'] == cyc
                         and np.isfinite(b.get('V_eq', np.nan))
                         and b['V_eq'] >= vlo and b['V_eq'] <= vhi]

                # ── Top row: |slope_sqrt| vs E_eq ────────────────────────────
                bsl = [b for b in blist if np.isfinite(b['sl_sqrt'])]
                if bsl:
                    veqs = [b['V_eq'] for b in bsl]
                    vals = [abs(b['sl_sqrt']) for b in bsl]
                    ax_sl.semilogy()
                    ax_sl.scatter(veqs, vals, s=20, color=col, alpha=0.55,
                                  edgecolors='none', label=name)
                    if len(veqs) > 4:
                        vv, sm = rmed(veqs, vals)
                        ax_sl.plot(vv, sm, color=col, lw=2)

                # ── Bottom row: |dE/dx| vs E_eq ──────────────────────────────
                bdx = [b for b in blist if np.isfinite(b.get('x_block', np.nan))]
                if not bdx:
                    continue

                if mode == 'differential':
                    # Smooth OCV derivative evaluated at each block's E_eq
                    dedx_fn = s.get('dedx_d') if cyc == 'discharge' else s.get('dedx_c')
                    if dedx_fn is not None:
                        xs   = [b['V_eq'] for b in bdx]          # x = E_eq
                        vals = []
                        for b in bdx:
                            try:
                                v = abs(float(dedx_fn(b['V_eq'])))
                                vals.append(v if np.isfinite(v) and v > 0 else float('nan'))
                            except Exception:
                                vals.append(float('nan'))
                        pairs = [(x, v) for x, v in zip(xs, vals) if np.isfinite(v)]
                        if pairs:
                            xs_p, vals_p = zip(*pairs)
                            ax_dx.semilogy()
                            ax_dx.scatter(list(xs_p), list(vals_p), s=20, color=col,
                                          alpha=0.55, edgecolors='none', label=name)
                            if len(xs_p) > 3:
                                order = np.argsort(xs_p)
                                xs_s  = [xs_p[i] for i in order]
                                vls_s = [vals_p[i] for i in order]
                                vv, sm = rmed(xs_s, vls_s)
                                ax_dx.plot(vv, sm, color=col, lw=2)

                else:  # 'finite_diff'
                    # Raw block-to-block finite difference: ΔE_eq / Δx, plotted vs E_eq
                    prev = None
                    xs_fd = []; vals_fd = []
                    for b in bdx:
                        if prev is not None:
                            dx = b['x_block'] - prev['x_block']
                            dv = b['V_eq']     - prev['V_eq']
                            if abs(dx) > 1e-6:
                                xs_fd.append(b['V_eq'])           # x = E_eq
                                vals_fd.append(abs(dv / dx))
                        prev = b
                    if xs_fd:
                        ax_dx.semilogy()
                        ax_dx.scatter(xs_fd, vals_fd, s=20, color=col,
                                      alpha=0.55, edgecolors='none', label=name)
                        if len(xs_fd) > 3:
                            order = np.argsort(xs_fd)
                            xs_s  = [xs_fd[i] for i in order]
                            vls_s = [vals_fd[i] for i in order]
                            vv, sm = rmed(xs_s, vls_s)
                            ax_dx.plot(vv, sm, color=col, lw=2)

        mode_lbl = (r'$|\partial E_{\rm eq}/\partial x|$ (smooth OCV deriv.)'
                    if mode == 'differential'
                    else r'$|\Delta E_{\rm eq}/\Delta x|$ (block finite diff.)')
        xlabel_v = r'$E_{\mathrm{eq}}$ / V vs. ref'
        for ax, lbl, xl in [
                (ax1, r'$|dE/d\sqrt{t}|$ — Discharge', xlabel_v),
                (ax2, r'$|dE/d\sqrt{t}|$ — Charge',    xlabel_v),
                (ax3, f'|dE/dx| Discharge  ({mode})', xlabel_v),
                (ax4, f'|dE/dx| Charge  ({mode})',    xlabel_v)]:
            ax.set_xlabel(xl, fontsize=9)
            ax.set_ylabel(lbl.split('—')[0].strip().split('(')[0].strip(), fontsize=9)
            ax.set_title(lbl, fontsize=9, fontweight='bold')
            ax.legend(fontsize=7)
            ax.grid(True, which='both', alpha=0.3)
        ax3.set_ylabel(mode_lbl, fontsize=9)
        ax4.set_ylabel(mode_lbl, fontsize=9)
        mode_str = 'Smooth differential dE_eq/dx' if mode == 'differential' else 'Finite difference ΔE_eq/Δx (block-to-block)'
        fig.suptitle(
            f'Slopes  |  top: |dE/dsqrt(t)| vs E_eq  |  bottom: |dE/dx| vs E_eq  |  {mode_str}',
            fontsize=10, fontweight='bold', color='#1a6b35')
        fig.tight_layout()
        self._flush('dEdx')


    def _draw_d(self,active,d_field,tab_key,vlo,vhi,mr2,r2_field,sig,title):
        fig=self._clf(tab_key)
        ax_d=fig.add_subplot(221); ax_c=fig.add_subplot(222)
        ax_dx=fig.add_subplot(223); ax_cx=fig.add_subplot(224)
        for name in active:
            s=self._samples[name]; col=self._color(name)
            for ax,cyc in [(ax_d,'discharge'),(ax_c,'charge')]:
                good=[b for b in s['blocks']
                      if b['cycle']==cyc and b['V_eq']>=vlo and b['V_eq']<=vhi
                      and b.get(r2_field,0)>=mr2
                      and np.isfinite(b.get(d_field,np.nan)) and b.get(d_field,0)>0]
                if not good: continue
                # Top row x-axis: always V_eq (equilibrium potential)
                xs=[b['V_eq'] for b in good]
                Ds=[b[d_field] for b in good]
                # ── sigma-clip on log10(D) using robust MAD estimator ─────────
                if sig > 0 and len(Ds) >= 4:
                    logD = np.log10(Ds)
                    med  = np.median(logD)
                    mad  = np.median(np.abs(logD - med))
                    std  = mad * 1.4826
                    keep = np.abs(logD - med) <= sig * max(std, 0.05)
                    xs = [v for v,k in zip(xs,keep) if k]
                    Ds = [d for d,k in zip(Ds,keep) if k]
                if not xs: continue
                ax.semilogy()
                ax.scatter(xs,Ds,s=22,color=col,alpha=0.6,edgecolors='none')
                if len(xs)>4:
                    order=np.argsort(xs)
                    xs_s=[xs[i] for i in order]; Ds_s=[Ds[i] for i in order]
                    vv,dm=rmed(xs_s,Ds_s)
                    ax.plot(vv,dm,color=col,lw=2.5,label=f'{name} n={len(xs)}')
        # ── Bottom row: D vs SOD/SOC % ───────────────────────────────────────
        for name in active:
            s=self._samples[name]; col=self._color(name)
            for ax,cyc in [(ax_dx,'discharge'),(ax_cx,'charge')]:
                good=[b for b in s['blocks']
                      if b['cycle']==cyc and b['V_eq']>=vlo and b['V_eq']<=vhi
                      and b.get(r2_field,0)>=mr2
                      and np.isfinite(b.get(d_field,np.nan)) and b.get(d_field,0)>0
                      and np.isfinite(b.get('x_block',np.nan))]
                if not good: continue
                xs=[b['x_block']*100. for b in good]   # fraction → SOD/SOC %
                Ds=[b[d_field] for b in good]
                if sig > 0 and len(Ds) >= 4:
                    logD=np.log10(Ds); med=np.median(logD)
                    mad=np.median(np.abs(logD-med)); std=mad*1.4826
                    keep=np.abs(logD-med)<=sig*max(std,0.05)
                    xs=[v for v,k in zip(xs,keep) if k]
                    Ds=[d for d,k in zip(Ds,keep) if k]
                if not xs: continue
                ax.semilogy()
                ax.scatter(xs,Ds,s=22,color=col,alpha=0.6,edgecolors='none')
                if len(xs)>3:
                    order=np.argsort(xs)
                    xs_s=[xs[i] for i in order]; Ds_s=[Ds[i] for i in order]
                    vv,dm=rmed(xs_s,Ds_s)
                    ax.plot(vv,dm,color=col,lw=2.5,label=f'{name} n={len(xs)}')

                sig_lbl=f'   sigma-clip={sig:.1f}σ' if sig>0 else ''
        sig_lbl = ''
        Dlbl=r'$\tilde{D}\cdot(S/v_{\mathrm{m}})^2$ / mol$^2$ s$^{-1}$'
        for ax,lbl in [(ax_d,'Discharge'),(ax_c,'Charge')]:
            ax.set_xlabel(r'$E_{\mathrm{eq}}$ / V vs. ref', fontsize=10)
            ax.set_ylabel(Dlbl, fontsize=9)
            ax.set_title(lbl, fontsize=10); ax.legend(fontsize=7)
            ax.grid(True,which='both',alpha=0.3)
        for ax,lbl,xlbl in [(ax_dx,'Discharge','SOD  (%)'),
                             (ax_cx,'Charge',   'SOC  (%)')]:
            ax.set_xlabel(xlbl, fontsize=10)
            ax.set_ylabel(Dlbl, fontsize=9)
            ax.set_title(lbl+' vs SOD/SOC', fontsize=10); ax.legend(fontsize=7)
            ax.grid(True,which='both',alpha=0.3)
        fig.suptitle(title+sig_lbl, fontsize=10, fontweight='bold',
                     color=('#1a6b35' if 'conv' in tab_key.lower() or 'D_conv' in tab_key else '#8a2c2c'))
        fig.tight_layout(); self._flush(tab_key)
    def _draw_kc_slopes(self,active,vlo,vhi,mr2):
        fig=self._clf('kc_slopes')
        ax_d=fig.add_subplot(221); ax_c=fig.add_subplot(222)
        ax_dx=fig.add_subplot(223); ax_cx=fig.add_subplot(224)
        for name in active:
            s=self._samples[name]; col=self._color(name)
            for ax,ax_x,cyc,xlbl in [
                    (ax_d,ax_dx,'discharge','SOD  (%)'),
                    (ax_c,ax_cx,'charge',   'SOC  (%)')]:
                bl=[b for b in s['blocks'] if b['cycle']==cyc
                    and b['V_eq']>=vlo and b['V_eq']<=vhi
                    and b.get('r2_kc', 0) >= mr2
                    and np.isfinite(b.get('sl_kc',np.nan))
                    and np.isfinite(b.get('x_block',np.nan))]
                if not bl: continue
                veqs=[b['V_eq'] for b in bl]
                xbs =[b['x_block']*100. for b in bl]
                vals=[abs(b['sl_kc']) for b in bl]
                # Top row: vs V_eq
                ax.semilogy()
                ax.scatter(veqs,vals,s=22,color=col,alpha=0.6,edgecolors='none')
                if len(veqs)>4:
                    vv,sm=rmed(veqs,vals)
                    ax.plot(vv,sm,color=col,lw=2.5,label=f'{name} n={len(bl)}')
                # Bottom row: vs SOD/SOC
                ax_x.semilogy()
                ax_x.scatter(xbs,vals,s=22,color=col,alpha=0.6,edgecolors='none')
                if len(xbs)>4:
                    order=np.argsort(xbs)
                    xs_s=[xbs[i] for i in order]; vs_s=[vals[i] for i in order]
                    vv2,sm2=rmed(xs_s,vs_s)
                    ax_x.plot(vv2,sm2,color=col,lw=2.5,label=f'{name} n={len(bl)}')
        for ax,lbl in [(ax_d,'Discharge'),(ax_c,'Charge')]:
            ax.set_xlabel(r'$E_\mathrm{eq}$ / V vs. ref',fontsize=9)
            ax.set_ylabel('|s_KC| (V s⁻½)',fontsize=9)
            ax.set_title(lbl,fontsize=10); ax.legend(fontsize=7)
            ax.grid(True,which='both',alpha=0.3)
        for ax,lbl,xlbl in [(ax_dx,'Discharge','SOD  (%)'),(ax_cx,'Charge','SOC  (%)')]:
            ax.set_xlabel(xlbl,fontsize=9)
            ax.set_ylabel('|s_KC| (V s⁻½)',fontsize=9)
            ax.set_title(f'{lbl} vs SOD/SOC',fontsize=10); ax.legend(fontsize=7)
            ax.grid(True,which='both',alpha=0.3)
        fig.suptitle('K&C slope  ~  1/sqrtD',fontsize=10,fontweight='bold',color='#8a2c2c')
        fig.tight_layout(); self._flush('kc_slopes')


    def _draw_overpot_curves(self, active):
        import matplotlib.cm as cm, matplotlib.colors as mcolors
        import matplotlib.gridspec as gridspec
        C_ANAL = '#5c3380'
        n = max(len(active), 1)
        n_rows = 1 if n == 1 else 2
        n_cols = 2 if n == 1 else n
        ws = 0.45 if n >= 4 else 0.38 if n == 3 else 0.30
        fig = self._clf('overpot_curves')
        fig.set_size_inches(*self._figsize(n_rows, n_cols))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=ws,
                               top=0.88, bottom=0.10, left=0.07, right=0.96)
        try:
            p_from = max(1, int(self.pulse_from.get())) - 1
            p_to   = max(1, int(self.pulse_to.get()))
        except Exception:
            p_from, p_to = 0, 999
        rng = f'pulses {p_from+1} to {min(p_to, 999)}'
        cmap = cm.get_cmap('turbo')

        for ci, name in enumerate(active):
            s = self._samples[name]
            col = self._color(name)
            for ri, cyc in enumerate(['discharge', 'charge']):
                ax = fig.add_subplot(gs[0 if n == 1 else ri, ri if n == 1 else ci])
                all_bl = [b for b in s['blocks'] if b['cycle'] == cyc and len(b['_Vp']) >= 2]
                bl = all_bl[p_from:p_to]
                valid_v = [b.get('V_eq', float('nan')) for b in bl
                           if b.get('V_eq', float('nan')) == b.get('V_eq', float('nan'))]
                vmin = min(valid_v) if valid_v else 0.
                vmax = max(valid_v) if valid_v else 1.
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                for i, b in enumerate(bl):
                    Vp = b['_Vp']; Tp = b['_Tp']
                    t_rel = Tp - Tp[0]
                    veq = b.get('V_eq', float('nan'))
                    c = cmap(norm(veq)) if veq == veq else '#aaa'
                    v_ref = Vp[0]  # default: use pulse start itself
                    if i > 0:
                        prev = bl[i-1]
                        try:
                            gap = b['_Tp'][0] - prev['_Tr'][-1]
                            dur = prev['_Tr'][-1] - prev['_Tr'][0]
                            if gap < max(dur * 0.5, 300.):
                                v_ref = prev.get('V_eq', Vp[0])
                        except Exception:
                            v_ref = prev.get('V_eq', Vp[0])
                    ax.plot(t_rel, (Vp - v_ref) * 1000., '-', lw=0.8, color=c, alpha=0.75)
                ax.axhline(0, color='#555', lw=0.8, ls='--', alpha=0.5)
                if bl:
                    sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
                    cb = fig.colorbar(sm, ax=ax,
                                   fraction=max(0.025, 0.06/max(n,1)),
                                   pad=0.01)
                    cb.set_label(r'$E_\mathrm{eq}$ (V)', fontsize=7,
                                 rotation=270, labelpad=8)
                ax.set_xlabel('Time since pulse start (s)', fontsize=9)
                ax.set_ylabel(r'$\eta = E_{\rm pulse}(t) - E_{\rm eq,prev}$ (mV)', fontsize=9)
                lbl_cyc = 'Discharge' if cyc == 'discharge' else 'Charge'
                ax.set_title(f'{name}  |  {lbl_cyc}', fontsize=9, fontweight='bold', color=col)
                ax.grid(True, alpha=0.25)

        fig.suptitle(
            f'Overpotential curves  |  {rng}  |  colour = E_eq (turbo)',
            fontsize=10, fontweight='bold', color=C_ANAL)
        self._flush('overpot_curves')


    def _draw_relax_curves(self, active):
        import matplotlib.cm as cm, matplotlib.colors as mcolors
        import matplotlib.gridspec as gridspec
        C_ANAL = '#5c3380'
        n = max(len(active), 1)
        n_rows = 1 if n == 1 else 2
        n_cols = 2 if n == 1 else n
        ws = 0.45 if n >= 4 else 0.38 if n == 3 else 0.30
        fig = self._clf('relax_curves')
        fig.set_size_inches(*self._figsize(n_rows, n_cols))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=ws,
                               top=0.88, bottom=0.10, left=0.07, right=0.96)
        try:
            p_from = max(1, int(self.pulse_from.get())) - 1
            p_to   = max(1, int(self.pulse_to.get()))
        except Exception:
            p_from, p_to = 0, 999
        rng = f'pulses {p_from+1} to {min(p_to, 999)}'
        cmap = cm.get_cmap('turbo')

        for ci, name in enumerate(active):
            s = self._samples[name]
            col = self._color(name)
            for ri, cyc in enumerate(['discharge', 'charge']):
                ax = fig.add_subplot(gs[0 if n == 1 else ri, ri if n == 1 else ci])
                all_bl = [b for b in s['blocks'] if b['cycle'] == cyc
                          and len(b['_Vr']) >= 2
                          and b.get('V_eq', float('nan')) == b.get('V_eq', float('nan'))]
                bl = all_bl[p_from:p_to]
                if not bl:
                    ax.set_visible(False); continue
                valid_v = [b['V_eq'] for b in bl]
                vmin = min(valid_v); vmax = max(valid_v)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                for b in bl:
                    Tr = b['_Tr']; Vr = b['_Vr']; veq = b['V_eq']
                    t_rel = Tr - Tr[0]
                    c = cmap(norm(veq))
                    ax.plot(t_rel, (Vr - veq) * 1000., '-', lw=0.8, color=c, alpha=0.75)
                ax.axhline(0, color='#555', lw=0.9, ls='--', alpha=0.6,
                           label=r'$E_\mathrm{eq}$ (equilibrium)')
                sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
                cb = fig.colorbar(sm, ax=ax,
                               fraction=max(0.025, 0.06/max(n,1)),
                               pad=0.01)
                cb.set_label(r'$E_\mathrm{eq}$ (V)', fontsize=7,
                             rotation=270, labelpad=8)
                ax.set_xlabel('Time since pulse end (s)', fontsize=9)
                ax.set_ylabel(r'$E(t) - E_\mathrm{eq}$ (mV)', fontsize=9)
                lbl_cyc = 'Discharge' if cyc == 'discharge' else 'Charge'
                ax.set_title(f'{name}  |  {lbl_cyc}', fontsize=9, fontweight='bold', color=col)
                ax.legend(fontsize=8, loc='upper right', framealpha=0.85)
                ax.grid(True, alpha=0.25)

        fig.suptitle(
            r'Relaxation  |  $E(t)-E_\mathrm{eq}$ (mV)  |  ' + rng + r'  |  colour = $E_\mathrm{eq}$ (gradient per sample)',
            fontsize=10, fontweight='bold', color=C_ANAL)
        self._flush('relax_curves')


    def _draw_relax_delta(self, active):
        """Delta V during relaxation vs SOD/SOC.
        dV_n = V_eq_n - V_relax_start_n (voltage recovery during rest).
        """
        fig = self._clf('relax_delta')
        fig.set_size_inches(14, 6)
        ax_d = fig.add_subplot(121)
        ax_c = fig.add_subplot(122)
        C_ANAL = '#5c3380'
        for name in active:
            s = self._samples[name]
            col = self._color(name)
            blocks = s.get('blocks', [])
            for ax, cyc, xlbl in [
                    (ax_d, 'discharge', 'SOD  (%)'),
                    (ax_c, 'charge',    'SOC  (%)')]:
                bl = [b for b in blocks if b['cycle'] == cyc
                      and len(b['_Vr']) >= 2
                      and len(b['_Vp']) >= 1
                      and np.isfinite(b.get('x_block', np.nan))]
                if not bl: continue
                xbs = [b['x_block'] * 100. for b in bl]
                dVs = [b['_Vr'][-1] - b['_Vr'][0] for b in bl]
                ax.plot(xbs, dVs, 'o-', ms=5, lw=1.5,
                        color=col, alpha=0.85, markeredgewidth=0, label=name)
                ax.axhline(0, color='#aaa', lw=0.8, ls='--')
        for ax, lbl, xlbl in [
                (ax_d, 'Discharge', 'SOD  (%)'),
                (ax_c, 'Charge',    'SOC  (%)')]:
            ax.set_facecolor('white')
            for sp in ax.spines.values():
                sp.set_linewidth(0.8); sp.set_color('#333')
            ax.tick_params(which='major', direction='in', length=5,
                           width=0.8, labelsize=11, top=True, right=True)
            ax.grid(True, which='major', alpha=0.2, linewidth=0.5,
                    color='#888', linestyle='--')
            ax.set_xlabel(xlbl, fontsize=11)
            ax.set_ylabel(r'$\Delta E_\mathrm{relax} = E_\mathrm{eq} - E_\mathrm{relax,0}$  (V)',
                          fontsize=10)
            ax.set_title(f'{lbl}  —  $\\Delta E_{{\\rm relax}}$ vs SOD/SOC',
                         fontsize=11, fontweight='bold', color=C_ANAL)
            ax.legend(fontsize=9, framealpha=0.95, edgecolor='#bbb')
        fig.suptitle(
            r'Relaxation $\Delta E$  =  $E_\mathrm{eq} - E_\mathrm{relax,start}$  |  voltage recovery (no current)',
            fontsize=11, fontweight='bold', color=C_ANAL)
        fig.tight_layout()
        self._flush('relax_delta')

    # ── Figure size helper ─────────────────────────────────────────────────────
    def _figsize(self, n_rows, n_cols, row_h=3.8, col_w=5.2):
        """Return (width, height) fitting the screen, with minimum per panel.
        row_h and col_w are target inches per panel.
        """
        try:
            sh = self.root.winfo_screenheight()
            sw = self.root.winfo_screenwidth()
            max_w = (sw - 60) / 96
            max_h = (sh - 180) / 96
        except Exception:
            max_w, max_h = 14., 9.
        w = min(max(col_w * n_cols, 6.), max_w)
        h = min(max(row_h * n_rows, 4.), max_h)
        return w, h

    # ── Shared dV/d(log t) helper ─────────────────────────────────────────────
    @staticmethod
    def _dvdlogt(b, n_resamp=120):
        """Return (log_t_grid, dvd_mV_per_dec) for one relaxation block, or (None,None).

        Applies a heavier Savitzky-Golay smooth (window = n_resamp//6, min 15 pts)
        to suppress noise before peak detection.  The curve shown is this smoothed
        version so it matches what the peak finder sees.
        """
        from scipy.signal import savgol_filter as _sg
        Vr = b['_Vr']; Tr = b['_Tr']; veq = b['V_eq']
        t_raw = np.maximum(Tr - Tr[0], 1e-2)
        if abs(Vr[0] - veq) < 5e-5:
            return None, None
        t_min = t_raw[t_raw > 0].min(); t_max = t_raw[-1]
        if t_max <= t_min or t_max < 5.:
            return None, None
        log_t = np.linspace(np.log10(max(t_min, 0.5)), np.log10(t_max), n_resamp)
        V_g   = np.interp(10.**log_t, t_raw, Vr)
        dvd   = np.abs(np.gradient(V_g - veq, log_t)) * 1000.   # mV / dec
        # Heavier smooth: window = n_resamp // 6, minimum 15, must be odd
        wl = max(15, (n_resamp // 6) // 2 * 2 + 1)
        wl = min(wl, len(dvd) - 1 if len(dvd) > 1 else 1)
        wl = wl if wl % 2 == 1 else wl - 1
        wl = max(5, wl)
        dvd_sm = _sg(dvd, wl, 3) if len(dvd) > wl else dvd
        return log_t, np.maximum(dvd_sm, 0.)

    @staticmethod
    def _consensus_peaks(blocks_disc, blocks_chg=None, min_frac=0.30, n_resamp=120,
                         veq_tol=0.05):
        """Find peak τ positions that are consistent across pulses and directions.

        Three-tier consensus — a peak is promoted if it satisfies ANY of:

        Tier 1 — Within-direction consistency (strong):
            Peak appears in ≥ min_frac of blocks within discharge OR charge alone.
            These are processes that repeat reliably across the SOC range of one
            direction.

        Tier 2 — Cross-direction consistency at matched V_eq (strongest):
            For each discharge block, find charge blocks with |V_eq difference| <
            veq_tol (same electrode state reached from opposite direction). If a peak
            at a given log-τ appears in both the discharge block AND the matching
            charge block(s), it is flagged as cross-validated. Any peak position
            that is cross-validated in ≥ 30% of matchable pairs is promoted.

        Returns list of (log_tau, mean_amplitude_mV, tier_label) where tier_label
        is 'cross' (cross-direction validated) or 'within' (single-direction).
        Results are sorted by log_tau.
        """
        from scipy.signal import savgol_filter as _sg, find_peaks

        if blocks_chg is None:
            blocks_chg = []

        N_RESAMP = n_resamp

        def _compute(blocks):
            """Return list of (log_t_array, dvd_array, veq) for valid blocks."""
            out = []
            for b in blocks:
                log_t, dvd_sm = GITTApp._dvdlogt(b, n_resamp=N_RESAMP)
                if log_t is None:
                    continue
                out.append((log_t, dvd_sm, b['V_eq']))
            return out

        def _find_block_peaks(log_t, dvd_sm):
            """Return peak log-tau values for one curve."""
            h = max(dvd_sm.max() * 0.12, 1e-6)
            pks, _ = find_peaks(dvd_sm, height=h, distance=6,
                                prominence=dvd_sm.max() * 0.18)
            return [log_t[pi] for pi in pks]

        # Build common reference grid spanning all blocks
        all_lt_min = []; all_lt_max = []
        for b_set in [blocks_disc, blocks_chg]:
            for b in b_set:
                tr = b['_Tr']; t_raw = np.maximum(tr - tr[0], 1e-2)
                t_min = t_raw[t_raw > 0].min(); t_max = t_raw[-1]
                if t_max > t_min and t_max >= 5.:
                    all_lt_min.append(np.log10(max(t_min, 0.5)))
                    all_lt_max.append(np.log10(t_max))
        if not all_lt_min:
            return []
        lt_ref = np.linspace(min(all_lt_min), max(all_lt_max), N_RESAMP)

        # Compute curves for both directions
        curves_d = _compute(blocks_disc)
        curves_c = _compute(blocks_chg)
        all_curves = curves_d + curves_c

        if not all_curves:
            return []

        # ── Tier 1: within-direction presence histogram ───────────────────────
        presence_d = np.zeros(N_RESAMP)
        presence_c = np.zeros(N_RESAMP)
        amplitude  = np.zeros(N_RESAMP)

        def _accumulate(curves, presence):
            for log_t, dvd_sm, veq in curves:
                dvd_on_ref = np.interp(lt_ref, log_t, dvd_sm, left=0., right=0.)
                amplitude[:] += dvd_on_ref
                for lt_val in _find_block_peaks(log_t, dvd_sm):
                    ri = int(np.argmin(np.abs(lt_ref - lt_val)))
                    presence[max(0,ri-2):min(N_RESAMP,ri+3)] += 1

        _accumulate(curves_d, presence_d)
        _accumulate(curves_c, presence_c)

        n_d = max(len(curves_d), 1); n_c = max(len(curves_c), 1)
        frac_d = presence_d / n_d
        frac_c = presence_c / n_c
        mean_amp = amplitude / max(len(all_curves), 1)

        # Smooth presence maps
        wl = max(5, (N_RESAMP // 15) // 2 * 2 + 1)
        frac_d_sm = _sg(frac_d, wl, 2) if len(frac_d) > wl else frac_d
        frac_c_sm = _sg(frac_c, wl, 2) if len(frac_c) > wl else frac_c

        # ── Tier 2: cross-direction at matched V_eq ───────────────────────────
        cross_presence = np.zeros(N_RESAMP)
        n_pairs = 0
        for log_t_d, dvd_d, veq_d in curves_d:
            matching_c = [(log_t_c, dvd_c)
                          for log_t_c, dvd_c, veq_c in curves_c
                          if abs(veq_c - veq_d) <= veq_tol]
            if not matching_c:
                continue
            peaks_d = set()
            for lt_val in _find_block_peaks(log_t_d, dvd_d):
                ri = int(np.argmin(np.abs(lt_ref - lt_val)))
                peaks_d.update(range(max(0,ri-3), min(N_RESAMP,ri+4)))
            for log_t_c, dvd_c in matching_c:
                n_pairs += 1
                for lt_val in _find_block_peaks(log_t_c, dvd_c):
                    ri = int(np.argmin(np.abs(lt_ref - lt_val)))
                    # Only count if discharge also has a peak in this neighbourhood
                    if ri in peaks_d:
                        cross_presence[max(0,ri-2):min(N_RESAMP,ri+3)] += 1
        frac_cross = cross_presence / max(n_pairs, 1)
        frac_cross_sm = _sg(frac_cross, wl, 2) if len(frac_cross) > wl else frac_cross

        # ── Combine: TRUE OR — mark if consistent in disc OR chg OR cross ───────
        # A peak passes if it meets ANY of the three criteria independently.
        # Cross-validated peaks get a separate visual marker but are not required.
        # Use lower threshold for within-direction (0.25) since each direction
        # has fewer blocks; cross threshold even lower (0.10) since pairs are rare.
        THRESH_WITHIN = 0.25
        THRESH_CROSS  = 0.10

        # Combined map: take the max across all three signals
        combined = np.maximum(np.maximum(frac_d_sm, frac_c_sm),
                              frac_cross_sm * (THRESH_WITHIN / max(THRESH_CROSS, 1e-9)))

        cpks, _ = find_peaks(combined,
                             height=THRESH_WITHIN,
                             distance=8,
                             prominence=THRESH_WITHIN * 0.20)

        results = []
        for pi in cpks:
            lt  = lt_ref[pi]
            amp = mean_amp[pi]
            # Mark as confirmed (red) if consistent in discharge, charge, OR cross-validated.
            # Only grey if none of the three checks pass (promoted purely by upweight noise).
            is_cross = (n_pairs > 0 and frac_cross_sm[pi] >= THRESH_CROSS)
            is_disc  = frac_d_sm[pi] >= THRESH_WITHIN
            is_chg   = frac_c_sm[pi] >= THRESH_WITHIN
            tier = 'confirmed' if (is_cross or is_disc or is_chg) else 'weak'
            results.append((lt, amp, tier))
        return sorted(results, key=lambda r: r[0])

    # ── Overlay: all pulses, one panel per sample (discharge|charge) ─────────
    def _draw_relax_kinetics(self, active):
        """dV/d(log t) overlay — 2 rows x N-sample cols, discharge top / charge bottom.
        All pulses overlaid per panel, colour = V_eq (turbo).
        Consensus peaks shown as vertical lines:
          red dashed  = cross-direction validated (same τ at matching V_eq in both disc & chg)
          grey dashed = within-direction only
        """
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.gridspec as gridspec

        C_ANAL  = '#5c3380'
        MIN_PTS = 10
        n = max(len(active), 1)
        n_rows = 1 if n == 1 else 2
        n_cols = 2 if n == 1 else n
        ws = 0.45 if n >= 4 else 0.38 if n == 3 else 0.30
        fig = self._clf('relax_kinetics')
        fig.set_size_inches(*self._figsize(n_rows, n_cols))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.45, wspace=ws,
                               top=0.88, bottom=0.10, left=0.07, right=0.96)

        try:
            p_from = max(1, int(self.pulse_from.get())) - 1
            p_to   = max(1, int(self.pulse_to.get()))
        except Exception:
            p_from, p_to = 0, 999

        cmap = cm.get_cmap('turbo')

        for ci, name in enumerate(active):
            s   = self._samples[name]
            col = self._color(name)

            # Collect both directions for cross-direction consensus
            bl_disc = [b for b in s.get('blocks', [])
                       if b['cycle'] == 'discharge'
                       and len(b['_Vr']) >= MIN_PTS
                       and np.isfinite(b.get('V_eq', np.nan))][p_from:p_to]
            bl_chg  = [b for b in s.get('blocks', [])
                       if b['cycle'] == 'charge'
                       and len(b['_Vr']) >= MIN_PTS
                       and np.isfinite(b.get('V_eq', np.nan))][p_from:p_to]

            # Compute consensus once across BOTH directions for this sample
            cpks = self._consensus_peaks(bl_disc, bl_chg)

            for ri, (cyc, lbl, bl) in enumerate([
                    ('discharge', 'Discharge', bl_disc),
                    ('charge',    'Charge',    bl_chg)]):
                ax = fig.add_subplot(gs[0 if n == 1 else ri, ri if n == 1 else ci])
                if not bl:
                    ax.set_title(f'{name}  |  {lbl}  (no data)', fontsize=8, color='#aaa')
                    continue
                vmin = min(b['V_eq'] for b in bl)
                vmax = max(b['V_eq'] for b in bl)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                for b in bl:
                    log_t, dvd_sm = self._dvdlogt(b)
                    if log_t is None:
                        continue
                    ax.plot(log_t, dvd_sm, '-', lw=1.0,
                            color=cmap(norm(b['V_eq'])), alpha=0.80)

                # No peak markers on overlay — see Relax kinetics map tab
                n_cross = 0; n_within = 0

                sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
                cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
                cb.set_label(r'$E_\mathrm{eq}$  (V)', fontsize=8)
                ax.set_xlabel(r'$\log_{10}(t_\mathrm{relax}\ /\ \mathrm{s})$', fontsize=9)
                ax.set_ylabel(r'$|dV/d\log t|$  (mV dec$^{-1}$)', fontsize=9)
                # Build title annotation
                ax.set_title(f'{name}  |  {lbl}', fontsize=9,
                             fontweight='bold', color=col)
                ax.grid(True, alpha=0.25)
                ax.set_ylim(bottom=0.)
                ax.tick_params(labelsize=8)

        fig.suptitle(r'$dV/d\log t$  |  colour = $E_\mathrm{eq}$ (turbo)  |  '
                     'each curve = one relaxation  |  see Relax kinetics map for peak analysis',
                     fontsize=9, fontweight='bold', color=C_ANAL)
        self._flush('relax_kinetics')

    # ── Per-pulse: one selected pulse, discharge top / charge bottom ──────────
    def _draw_relax_kinetics_perpulse(self, active):
        """dV/d(log t) for a single selected pulse — mirrors the xi fits layout.

        2 rows × N-sample columns.  Top row = discharge, bottom = charge.
        Pulse number selected with the existing Pulse # spinbox.
        Colour = V_eq via turbo colourmap (consistent with the overlay tab).
        Orange background = 2+ peaks detected.  Red dashed lines = peak τ.
        """
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        import matplotlib.gridspec as gridspec
        from scipy.signal import find_peaks

        C_ANAL  = '#5c3380'
        MIN_PTS = 10

        fig = self._clf('relax_kinetics_pp')
        n = len(active)
        n_rows = 1 if n == 1 else 2
        n_cols = 2 if n == 1 else n
        ws = 0.45 if n >= 4 else 0.38 if n == 3 else 0.30
        if n == 0:
            self._flush('relax_kinetics_pp'); return

        fig.set_size_inches(*self._figsize(n_rows, n_cols))

        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                               hspace=0.45, wspace=ws,
                               top=0.88, bottom=0.10,
                               left=0.07, right=0.97)

        # Which pulse index to show (same spinbox as sqrt/kc fits)
        try:
            pidx = max(0, self._pulse_var.get() - 1)
        except Exception:
            pidx = 0

        max_pulses = max(
            (len([b for b in self._samples[nm].get('blocks', [])
                  if b['cycle'] == 'discharge'])
             for nm in active), default=1)
        pidx = int(np.clip(pidx, 0, max(0, max_pulses - 1)))

        cmap = cm.get_cmap('turbo')

        # Collect V_eq range across all samples + both directions for one norm
        all_veq = []
        for name in active:
            for b in self._samples[name].get('blocks', []):
                if np.isfinite(b.get('V_eq', np.nan)):
                    all_veq.append(b['V_eq'])
        vmin = min(all_veq) if all_veq else 0.
        vmax = max(all_veq) if all_veq else 1.
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        def _cell(ax, b, cyc_lbl, name, col):
            log_t, dvd_sm = self._dvdlogt(b)
            veq = b['V_eq']
            if log_t is None:
                ax.set_title(f'{name}  |  {cyc_lbl}  (no data)',
                             fontsize=8, color='#aaa')
                return
            ax.plot(log_t, dvd_sm, '-', lw=2.0, color=col)
            ax.fill_between(log_t, 0., dvd_sm, color=col, alpha=0.15)
            # Use stricter thresholds: height ≥ 15% of max, prominence ≥ 20%
            # This suppresses noise bumps that don't represent real processes.
            h_thresh = max(dvd_sm.max() * 0.15, 1e-6)
            pk_idx, _ = find_peaks(dvd_sm,
                                   height=h_thresh,
                                   distance=7,
                                   prominence=dvd_sm.max() * 0.20)
            tau_strs = []
            for pi in pk_idx:
                ax.axvline(log_t[pi], color='#C0392B', lw=1.4,
                           ls='--', alpha=0.85)
                tau_strs.append(f'\u03c4={10.**log_t[pi]:.0f}s')
            # always white background
            ax.set_facecolor('white')
            ax.set_ylim(bottom=0.)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)
            ax.set_xlabel(r'$\log_{10}(t_\mathrm{relax}\ /\ \mathrm{s})$',
                          fontsize=8)
            ax.set_ylabel(r'$|dV/d\log t|$  (mV dec$^{-1}$)', fontsize=8)
            veq_str = f'{veq:.3f} V' if np.isfinite(veq) else ''
            n_pk = len(pk_idx)
            pk_str = f'  {n_pk}p' if n_pk else ''
            ax.set_title(f'{name}  {cyc_lbl}  {veq_str}{pk_str}',
                         fontsize=7.5, fontweight='bold', color=col, pad=3)
            if tau_strs:
                ax.annotate('  '.join(tau_strs), xy=(0.02, 0.97),
                            xycoords='axes fraction', fontsize=6.5,
                            color='#C0392B', va='top', ha='left',
                            bbox=dict(boxstyle='round,pad=0.2',
                                      fc='white', ec='#C0392B', alpha=0.7, lw=0.8))

        for ci, name in enumerate(active):
            s = self._samples[name]
            col = self._color(name)
            disc_bl = [b for b in s.get('blocks', [])
                       if b['cycle'] == 'discharge'
                       and len(b['_Vr']) >= MIN_PTS
                       and np.isfinite(b.get('V_eq', np.nan))]
            chg_bl  = [b for b in s.get('blocks', [])
                       if b['cycle'] == 'charge'
                       and len(b['_Vr']) >= MIN_PTS
                       and np.isfinite(b.get('V_eq', np.nan))]
            disc_bl_r = list(reversed(disc_bl)) if self.reverse_mode.get() in ('discharge','both') else list(disc_bl)

            chg_bl_r = list(reversed(chg_bl)) if self.reverse_mode.get() in ('charge','both') else list(chg_bl)
            for row, (blist, cyc_lbl) in enumerate(
                    [(disc_bl_r, 'Discharge'), (chg_bl_r, 'Charge')]):
                ax = fig.add_subplot(gs[0 if n == 1 else row, row if n == 1 else ci])
                if pidx >= len(blist):
                    ax.set_title(
                        f'{name}  |  {cyc_lbl}  (no block #{pidx+1})',
                        fontsize=8, color='#aaa')
                    ax.set_facecolor('white')
                    continue
                _cell(ax, blist[pidx], cyc_lbl, name, col)

        fig.suptitle(
            f'$dV/d\\log t$  |  Pulse #{pidx+1} of {max_pulses}  '
            '|  orange bg = 2 processes  |  red dashed = τ',
            fontsize=10, fontweight='bold', color=C_ANAL)
        self._flush('relax_kinetics_pp')

    # ── Shared grid builder for both relax map variants ──────────────────────
    def _relax_map_grid(self, bl, n_resamp=120):
        """Return (y_vals, lt_ref, Z_plot) for a list of blocks.

        y_vals  : 1-D array of the y-axis value for each block (V_eq or x_block %)
        lt_ref  : 1-D log-time grid (n_resamp points)
        Z_plot  : 2-D array (n_blocks × n_resamp), row-area normalised
        Returns (None, None, None) if no valid data.
        """
        MIN_PTS = 10
        lt_mins = []; lt_maxs = []
        for b in bl:
            t_raw = np.maximum(b['_Tr'] - b['_Tr'][0], 1e-2)
            tmin = t_raw[t_raw > 0].min(); tmax = t_raw[-1]
            if tmax > tmin and tmax >= 5.:
                lt_mins.append(np.log10(max(tmin, 0.5)))
                lt_maxs.append(np.log10(tmax))
        if not lt_mins:
            return None, None, None
        lt_ref = np.linspace(min(lt_mins), max(lt_maxs), n_resamp)
        grid_rows = []
        for b in bl:
            log_t, dvd_sm = self._dvdlogt(b, n_resamp=n_resamp)
            if log_t is None:
                grid_rows.append(np.zeros(n_resamp))
            else:
                grid_rows.append(np.interp(lt_ref, log_t, dvd_sm, left=0., right=0.))
        Z = np.array(grid_rows)
        row_areas = Z.sum(axis=1, keepdims=True)
        row_areas[row_areas < 1e-12] = 1.
        return lt_ref, Z / row_areas

    @staticmethod
    def _relax_map_pcolor(ax, fig, lt_ref, y_vals, Z_plot, ylabel, col, name, lbl, n=1):
        """Draw a single relax-map pcolormesh panel and colorbar."""
        from matplotlib.colors import LinearSegmentedColormap
        xrd_cmap = LinearSegmentedColormap.from_list(
            'xrd_style',
            ['#0000aa', '#0066ff', '#00ccff', '#00ffcc',
             '#00ff00', '#ccff00', '#ffff00', '#ffaa00',
             '#ff4400', '#cc0000'], N=256)

        y_arr = np.array(y_vals)
        if len(y_arr) > 1:
            edges = np.concatenate([
                [y_arr[0]  - (y_arr[1]  - y_arr[0])  / 2],
                (y_arr[:-1] + y_arr[1:]) / 2,
                [y_arr[-1] + (y_arr[-1] - y_arr[-2]) / 2]
            ])
        else:
            edges = np.array([y_arr[0] - 0.05, y_arr[0] + 0.05])

        lt_edges = np.append(lt_ref, lt_ref[-1] + (lt_ref[-1] - lt_ref[-2]))
        X_edge, Y_edge = np.meshgrid(lt_edges, edges)

        vmax = float(np.percentile(Z_plot[Z_plot > 0], 97)) if np.any(Z_plot > 0) else 1.
        pcm = ax.pcolormesh(X_edge, Y_edge, Z_plot,
                            cmap=xrd_cmap, vmin=0., vmax=vmax,
                            shading='flat')
        cb = fig.colorbar(pcm, ax=ax,
                          fraction=max(0.025, 0.06/max(n,1)),
                          pad=0.01)
        cb.set_label(r'Norm. $|dV/d\log t|$', fontsize=7,
                     rotation=270, labelpad=10)
        ax.set_xlabel(r'$\log_{10}(t_\mathrm{relax}\ /\ \mathrm{s})$', fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f'{name}  |  {lbl}', fontsize=9, fontweight='bold', color=col)
        ax.tick_params(labelsize=8)
        ax.grid(False)

    # ── 2D relaxation kinetics map — y = V_eq ────────────────────────────────
    def _draw_relax_map(self, active):
        """Operando-XRD-style 2D heatmap: x = log τ, y = V_eq."""
        import matplotlib.gridspec as gridspec
        C_ANAL = '#5c3380'; MIN_PTS = 10; N_RESAMP = 120
        n = max(len(active), 1)
        n_rows = 1 if n == 1 else 2
        n_cols = 2 if n == 1 else n
        ws = 0.45 if n >= 4 else 0.38 if n == 3 else 0.30
        fig = self._clf('relax_map')
        fig.set_size_inches(*self._figsize(n_rows, n_cols, row_h=4.2, col_w=5.5))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.40, wspace=ws,
                               top=0.88, bottom=0.10, left=0.08, right=0.96)
        try:
            p_from = max(1, int(self.pulse_from.get())) - 1
            p_to   = max(1, int(self.pulse_to.get()))
        except Exception:
            p_from, p_to = 0, 999

        for ci, name in enumerate(active):
            s = self._samples[name]; col = self._color(name)
            for ri, (cyc, lbl) in enumerate([('discharge', 'Discharge'),
                                              ('charge',    'Charge')]):
                ax = fig.add_subplot(gs[0 if n == 1 else ri, ri if n == 1 else ci])
                bl = [b for b in s.get('blocks', [])
                      if b['cycle'] == cyc and len(b['_Vr']) >= MIN_PTS
                      and np.isfinite(b.get('V_eq', np.nan))][p_from:p_to]
                if not bl:
                    ax.set_title(f'{name}  |  {lbl}  (no data)', fontsize=8, color='#aaa')
                    continue
                # Sort by V_eq → monotonic y-axis
                bl_s = sorted(bl, key=lambda b: b['V_eq'], reverse=(cyc == 'discharge'))
                y_vals = [b['V_eq'] for b in bl_s]
                lt_ref, Z_plot = self._relax_map_grid(bl_s, N_RESAMP)
                if lt_ref is None:
                    ax.set_title(f'{name}  |  {lbl}  (no data)', fontsize=8, color='#aaa')
                    continue
                self._relax_map_pcolor(ax, fig, lt_ref, y_vals, Z_plot,
                                       r'$E_\mathrm{eq}$  (V)', col, name, lbl, n=n)

        fig.suptitle(r'$dV/d\log t$ map  |  x = log τ,  y = $E_\mathrm{eq}$,  '
                     r'colour = norm. process fraction  |  vertical band = persistent τ',
                     fontsize=9, fontweight='bold', color=C_ANAL)
        self._flush('relax_map')

    # ── 2D relaxation kinetics map — y = SOD/SOC % ───────────────────────────
    def _draw_relax_map_soc(self, active):
        """Same heatmap as relax_map but y-axis = SOD/SOC % instead of V_eq."""
        import matplotlib.gridspec as gridspec
        C_ANAL = '#5c3380'; MIN_PTS = 10; N_RESAMP = 120
        n = max(len(active), 1)
        n_rows = 1 if n == 1 else 2
        n_cols = 2 if n == 1 else n
        ws = 0.45 if n >= 4 else 0.38 if n == 3 else 0.30
        fig = self._clf('relax_map_soc')
        fig.set_size_inches(*self._figsize(n_rows, n_cols, row_h=4.2, col_w=5.5))
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.40, wspace=ws,
                               top=0.88, bottom=0.10, left=0.08, right=0.96)
        try:
            p_from = max(1, int(self.pulse_from.get())) - 1
            p_to   = max(1, int(self.pulse_to.get()))
        except Exception:
            p_from, p_to = 0, 999

        for ci, name in enumerate(active):
            s = self._samples[name]; col = self._color(name)
            for ri, (cyc, lbl, ylbl) in enumerate([
                    ('discharge', 'Discharge', 'SOD (%)'),
                    ('charge',    'Charge',    'SOC (%)')]):
                ax = fig.add_subplot(gs[0 if n == 1 else ri, ri if n == 1 else ci])
                bl = [b for b in s.get('blocks', [])
                      if b['cycle'] == cyc and len(b['_Vr']) >= MIN_PTS
                      and np.isfinite(b.get('V_eq', np.nan))
                      and np.isfinite(b.get('x_block', np.nan))][p_from:p_to]
                if not bl:
                    ax.set_title(f'{name}  |  {lbl}  (no data)', fontsize=8, color='#aaa')
                    continue
                # Sort by x_block % so y-axis runs 0→100 monotonically
                bl_s = sorted(bl, key=lambda b: b['x_block'])
                y_vals = [b['x_block'] * 100. for b in bl_s]
                lt_ref, Z_plot = self._relax_map_grid(bl_s, N_RESAMP)
                if lt_ref is None:
                    ax.set_title(f'{name}  |  {lbl}  (no data)', fontsize=8, color='#aaa')
                    continue
                self._relax_map_pcolor(ax, fig, lt_ref, y_vals, Z_plot,
                                       ylbl, col, name, lbl, n=n)

        fig.suptitle(r'$dV/d\log t$ map  |  x = log τ,  y = SOD/SOC %,  '
                     r'colour = norm. process fraction  |  vertical band = persistent τ',
                     fontsize=9, fontweight='bold', color=C_ANAL)
        self._flush('relax_map_soc')

    def _draw_overpotential(self, active):
        C_ANAL = '#5c3380'
        fig = self._clf('overpot')
        fig.set_size_inches(*self._figsize(2, 2, row_h=4.5, col_w=6.0))

        # One set of shared axes per panel — twinx created once per panel,
        # not once per sample, so all samples share the same right-y scale.
        ax_tl = fig.add_subplot(2, 2, 1)
        ax_tr = fig.add_subplot(2, 2, 2)
        ax_bl = fig.add_subplot(2, 2, 3)
        ax_br = fig.add_subplot(2, 2, 4)
        ax_tl2 = ax_tl.twinx()
        ax_tr2 = ax_tr.twinx()
        ax_bl2 = ax_bl.twinx()
        ax_br2 = ax_br.twinx()
        primary = [ax_tl, ax_tr, ax_bl, ax_br]
        right   = [ax_tl2, ax_tr2, ax_bl2, ax_br2]

        for name in active:
            s      = self._samples[name]
            col    = self._color(name)
            blocks = s.get('blocks', [])
            ctrl   = self._sample_controls.get(name, {})
            mass_g = ctrl['mass'].get() / 1000. if 'mass' in ctrl else 0.

            for ci, cyc in enumerate(['discharge', 'charge']):
                bl = [b for b in blocks
                      if b['cycle'] == cyc
                      and len(b['_Vp']) > 1
                      and np.isfinite(b.get('V_eq', np.nan))
                      and np.isfinite(b.get('x_block', np.nan))]
                if len(bl) < 2:
                    continue

                sod_list=[]; eta_list=[]; R_list=[]; veq_list=[]
                prev_b = None
                for b in bl:
                    sod = b['x_block'] * 100.
                    veq = b['V_eq']
                    if prev_b is not None:
                        # Guard: only compute η if the two blocks are truly consecutive.
                        # A gap (excluded pulse between them) means prev_b's V_eq is
                        # from a different equilibrium state → η would be meaningless.
                        # Check: time between end of prev relaxation and start of this pulse.
                        try:
                            t_prev_end  = prev_b['_Tr'][-1]
                            t_this_start = b['_Tp'][0]
                            # Typical relaxation duration for reference
                            t_relax_dur = prev_b['_Tr'][-1] - prev_b['_Tr'][0]
                            gap = t_this_start - t_prev_end
                            consecutive = gap < max(t_relax_dur * 0.5, 300.)
                        except Exception:
                            consecutive = True
                        if consecutive:
                            e_end = b['_Vp'][-1]
                            eta   = e_end - prev_b['V_eq']
                            I     = b['I_pulse']
                            R_tot = eta / I if abs(I) > 1e-12 else np.nan
                            R_norm = R_tot * mass_g if mass_g > 0 else R_tot
                            sod_list.append(sod)
                            eta_list.append(eta * 1000.)
                            R_list.append(R_norm)
                            veq_list.append(veq)
                    prev_b = b

                if not sod_list:
                    continue

                sod_a = np.array(sod_list)
                eta_a = np.array(eta_list)
                R_a   = np.array(R_list)
                veq_a = np.array(veq_list)
                ok = np.isfinite(eta_a) & np.isfinite(R_a)

                ax_p,  ax_p2 = (ax_tl, ax_tl2) if ci==0 else (ax_tr, ax_tr2)
                ax_b,  ax_b2 = (ax_bl, ax_bl2) if ci==0 else (ax_br, ax_br2)

                ax_p.plot(sod_a[ok], eta_a[ok], 'o-', ms=5, lw=1.5,
                          color=col, alpha=0.85, markeredgewidth=0, label=name)
                ax_b.plot(veq_a[ok], eta_a[ok], 'o-', ms=5, lw=1.5,
                          color=col, alpha=0.85, markeredgewidth=0, label=name)
                ax_p2.plot(sod_a[ok], R_a[ok], 's--', ms=4, lw=1.2,
                           color=col, alpha=0.45, markeredgewidth=0)
                ax_b2.plot(veq_a[ok], R_a[ok], 's--', ms=4, lw=1.2,
                           color=col, alpha=0.45, markeredgewidth=0)

        ctrl_any = next(iter(self._sample_controls.values()), {}) if self._sample_controls else {}
        mass_any = ctrl_any['mass'].get() / 1000. if 'mass' in ctrl_any else 0.
        eta_ylabel = r'$\eta = E_{\rm pulse,end} - E_{\rm eq,prev}$  (mV)'
        R_ylabel = (r'$R_{\rm tot} \cdot m$  / $\Omega\cdot$g'
                    if mass_any > 0 else r'$R_{\rm tot}$  / $\Omega$')

        import matplotlib.lines as mlines
        for ax in primary:
            ax.set_facecolor('white')
            for sp in ax.spines.values():
                sp.set_linewidth(0.8); sp.set_color('#333')
            ax.tick_params(which='major', direction='in', length=4, width=0.8, labelsize=10)
            ax.grid(True, alpha=0.2, linewidth=0.5, color='#888', linestyle='--')
            ax.set_ylabel(eta_ylabel, fontsize=10, color='#333')
        # Build a compact legend outside the plot area to avoid data overlap
        for ax, ax2 in zip(primary, right):
            handles = list(ax.get_lines())
            if handles:
                proxy_eta = mlines.Line2D([],[],color='#555',lw=1.5,marker='o',ms=4,
                                          markeredgewidth=0,label=r'$\eta$ (left, mV)')
                proxy_R   = mlines.Line2D([],[],color='#555',lw=1.2,ls='--',marker='s',ms=3,
                                          markeredgewidth=0,alpha=0.6,label=r'$R_{\rm tot}$ (right)')
                sample_handles = [mlines.Line2D([],[],color=l.get_color(),lw=1.5,
                                               marker='o',ms=4,markeredgewidth=0,
                                               label=l.get_label())
                                  for l in handles if not l.get_label().startswith('_')]
                ax.legend(handles=sample_handles+[proxy_eta,proxy_R],
                          fontsize=7, framealpha=0.9, loc='upper right',
                          bbox_to_anchor=(0.99, 0.99))

        for ax2 in right:
            ax2.tick_params(axis='y', labelsize=9, colors='#666')
            ax2.set_ylabel(R_ylabel, fontsize=10, color='#666')
            ax2.spines['right'].set_color('#666')

        ax_tl.set_xlabel('SOD  (%)', fontsize=10)
        ax_tr.set_xlabel('SOC  (%)', fontsize=10)
        ax_bl.set_xlabel(r'$E_{\rm eq}$  / V vs. ref', fontsize=10)
        ax_br.set_xlabel(r'$E_{\rm eq}$  / V vs. ref', fontsize=10)

        ax_tl.set_title('Discharge  vs SOD', fontsize=10, fontweight='bold', color=C_ANAL, pad=5)
        ax_tr.set_title('Charge  vs SOC',    fontsize=10, fontweight='bold', color=C_ANAL, pad=5)
        ax_bl.set_title('Discharge  vs E_eq', fontsize=10, fontweight='bold', color=C_ANAL, pad=5)
        ax_br.set_title('Charge  vs E_eq',    fontsize=10, fontweight='bold', color=C_ANAL, pad=5)

        mass_note = '  |  R_tot x mass' if mass_any > 0 else '  |  mass not set'
        fig.suptitle(
            r'Overpotential $\eta$ (mV, left axis)  &  $R_{\rm tot}$ (right axis, dashed)' + mass_note,
            fontsize=10, fontweight='bold', color=C_ANAL)
        fig.subplots_adjust(top=0.92, bottom=0.09, left=0.08, right=0.93,
                            hspace=0.38, wspace=0.45)
        self._flush('overpot')

    def _draw_ocv(self, active):
        """Publication-quality OCV vs SOC.

        For each sample:
          - Discharge segments (V decreasing) are joined together,
            SOC_disc = cumulative_Q / Q_max_discharge  (0 = charged, 1 = discharged)
          - Charge segment (V increasing, longest one),
            SOC_chg  = Q / Q_max_charge  (0 = empty, 1 = full)

        Plot style: publication-ready, proper axis labels with subscripts,
        minor ticks, clean grid, 0–2 V y-axis, 0–1 x-axis.
        """
        import matplotlib.ticker as ticker

        fig = self._clf('ocv')
        fig.patch.set_facecolor('white')
        fig.set_size_inches(14, 10)
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)

        # Collect V_eq and x ranges from all blocks across all active samples
        # Done in two passes: first compute x for all blocks, then set axis limits.
        all_veq_d = []; all_veq_c = []; all_x_d = []; all_x_c = []
        for name in active:
            s = self._samples[name]
            blks = s.get('blocks', [])
            disc_s = s.get('disc'); chg_s = s.get('chg')
            Q_ref_s = s.get('q_max_val', None)
            if not Q_ref_s or Q_ref_s < 1e-9:
                ref = (chg_s if s.get('q_norm','')=='max charge' and chg_s else disc_s)
                Q_ref_s = ref['Q_max'] if ref else 1.
            dbl = [b for b in blks if b['cycle']=='discharge' and np.isfinite(b.get('V_eq',np.nan))]
            cbl = [b for b in blks if b['cycle']=='charge'    and np.isfinite(b.get('V_eq',np.nan))]
            if dbl and disc_s and disc_s['Q_max']>1e-6:
                # Use x_block (SOD %) if available, else uniform linspace in %
                if all(np.isfinite(b.get('x_block', np.nan)) for b in dbl):
                    xd_s = [b['x_block']*100. for b in dbl]
                else:
                    xd_s = [i/max(len(dbl)-1,1)*100. for i in range(len(dbl))]
                all_x_d.extend(xd_s)
                all_veq_d.extend(b['V_eq'] for b in dbl)
            if cbl and chg_s and chg_s['Q_max']>1e-6:
                if all(np.isfinite(b.get('x_block', np.nan)) for b in cbl):
                    xc_s = [b['x_block']*100. for b in cbl]
                else:
                    xc_s = [i/max(len(cbl)-1,1)*100. for i in range(len(cbl))]
                all_x_c.extend(xc_s)
                all_veq_c.extend(b['V_eq'] for b in cbl)

        # y-axis: separate ranges for top (V_eq endpoints) and bottom (raw cap)
        all_veq = all_veq_d + all_veq_c
        raw_V_all = []
        for name in active:
            s = self._samples[name]
            for seg_key in ('disc', 'chg'):
                seg = s.get(seg_key)
                if seg and len(seg.get('V',[])) > 0:
                    raw_V_all.extend(seg['V'].tolist())
        # Top row y: from V_eq values only (clean OCV points)
        if all_veq:
            veq_min = min(all_veq); veq_max = max(all_veq)
            m_top = max((veq_max - veq_min) * 0.05, 0.1)
            y_lo_top = max(0., veq_min - m_top); y_hi_top = veq_max + m_top
        else:
            y_lo_top, y_hi_top = 0.0, 5.0
        # Bottom row y: from raw cap (includes pulse dips)
        all_V_bot = all_veq + raw_V_all if raw_V_all else all_veq
        if all_V_bot:
            v_min = min(all_V_bot); v_max = max(all_V_bot)
            m_bot = max((v_max - v_min) * 0.03, 0.1)
            y_lo_bot = max(0., v_min - m_bot); y_hi_bot = v_max + m_bot
        else:
            y_lo_bot, y_hi_bot = 0.0, 5.0
        # Shared settings for _style_ax (uses y_lo, y_hi)
        y_lo = y_lo_top; y_hi = y_hi_top  # top row default
        v_span = y_hi_top - y_lo_top
        major_step = 0.5 if v_span <= 3 else (1.0 if v_span <= 6 else 2.0)
        minor_step = major_step / 5

        # x-axis: extend to fit all data (charge can exceed 1.0 when Q_chg > Q_disc)
        x_max_d = max(all_x_d) if all_x_d else 100.
        x_max_c = max(all_x_c) if all_x_c else 100.
        x_hi_d = min(round(x_max_d + 5., -1), 100.)   # SOD — round to nearest 10
        x_hi_c = 100.                                   # SOC always goes to 100%

        def _style_ax(ax, x_hi):
            ax.set_facecolor('white')
            for sp in ax.spines.values():
                sp.set_linewidth(0.8); sp.set_color('#333')
            ax.tick_params(which='major', direction='in', length=5,
                           width=0.8, labelsize=11, top=True, right=True)
            ax.tick_params(which='minor', direction='in', length=3,
                           width=0.6, top=True, right=True)
            ax.grid(True, which='major', alpha=0.2, linewidth=0.5,
                    color='#888', linestyle='--')
            ax.set_xlim(0., x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.yaxis.set_major_locator(ticker.MultipleLocator(major_step))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(minor_step))
            # x_hi is in % (0-100 scale) — space ticks at multiples of 10%
            x_tick = 10.  # major tick every 10%
            ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(x_tick/2))

        _style_ax(ax1, x_hi_d)
        _style_ax(ax2, x_hi_c)

        plotted_d = set(); plotted_c = set()

        for name in active:
            s   = self._samples[name]
            col = self._color(name)
            blocks = s.get('blocks', [])
            disc   = s.get('disc')
            chg    = s.get('chg')
            # Both curves share the same Q_ref so x-axes are comparable.
            Q_ref = s.get('q_max_val', None)
            if not Q_ref or Q_ref < 1e-9:
                ref_cap = (chg if s.get('q_norm','') == 'max charge' and chg else disc)
                Q_ref   = ref_cap['Q_max'] if ref_cap else 1.

            disc_bl = [b for b in blocks if b['cycle'] == 'discharge'
                       and np.isfinite(b.get('V_eq', np.nan))]
            chg_bl  = [b for b in blocks if b['cycle'] == 'charge'
                       and np.isfinite(b.get('V_eq', np.nan))]

            def _veq_to_x(bl, seg, **kwargs):
                """Map each block to x = Q(V_pulse_end)/Q_ref.

                Uses x_block stored on each block (cap Q interpolated at the
                voltage at the END of the pulse — the correct composition
                assignment: x is 'after pulse n', not before).
                Falls back to uniform linspace if x_block not available.
                """
                if not bl or seg is None or seg['Q_max'] < 1e-6:
                    return None, None
                is_discharge = kwargs.get('is_discharge', True)
                bl_s = sorted(bl, key=lambda b: b['V_eq'], reverse=is_discharge)
                veqs = np.array([b['V_eq'] for b in bl_s])
                # Use x_block if available (assigned in _replot from cap interpolation)
                if all(np.isfinite(b.get('x_block', np.nan)) for b in bl_s):
                    x = np.array([b['x_block'] for b in bl_s]) * 100.  # → SOD/SOC %
                    x = np.clip(x, 0., 100.)
                else:
                    x = np.linspace(0., 100., len(bl_s))
                return veqs, x
            # ── Discharge OCV (left panel) ──────────────────────────────────
            veqs, x = _veq_to_x(disc_bl, disc, is_discharge=True)
            if veqs is not None:
                order = np.argsort(x)
                lbl = name if name not in plotted_d else None
                ax1.plot(x[order], veqs[order], 'o-', ms=5, lw=1.5,
                         color=col, alpha=0.9, label=lbl, markeredgewidth=0)
                plotted_d.add(name)

            # ── Charge OCV (right panel) ─────────────────────────────────────
            veqs, x = _veq_to_x(chg_bl, chg, is_discharge=False)
            if veqs is not None:
                order = np.argsort(x)
                lbl = name if name not in plotted_c else None
                ax2.plot(x[order], veqs[order], 'o-', ms=5, lw=1.5,
                         color=col, alpha=0.9, label=lbl, markeredgewidth=0)
                plotted_c.add(name)

        # ── Axis labels — publication style ──
        ylabel = r'$E_\mathrm{WE/RE}$  /  V vs. ref'

        ax1.set_xlabel('SOD  (%)', fontsize=11)
        ax1.set_ylabel(ylabel, fontsize=12)
        ax1.set_title('Discharge OCV  (Vₑⁱ from relaxation endpoints)', fontsize=11, fontweight='bold', pad=6)
        if plotted_d:
            ax1.legend(fontsize=10, framealpha=0.95, edgecolor='#bbb',
                       handlelength=1.5, loc='upper right')

        ax2.set_xlabel('SOC  (%)', fontsize=11)
        ax2.set_ylabel(ylabel, fontsize=12)
        ax2.set_title('Charge OCV  (Vₑⁱ from relaxation endpoints)', fontsize=11, fontweight='bold', pad=6)
        if plotted_c:
            ax2.legend(fontsize=10, framealpha=0.95, edgecolor='#bbb',
                       handlelength=1.5, loc='best')

        # ── Row 2: Full E vs x (raw cap, same x-axis as top row) ──────────
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        for name in active:
            s   = self._samples[name]
            col = self._color(name)
            disc = s.get('disc'); chg = s.get('chg')
            lbl_d = name if name in plotted_d else None
            lbl_c = name if name in plotted_c else None

            # Bottom row: full raw cap E vs SOD/SOC %
            # Each direction uses its own Q_max as reference
            if disc and disc['Q_max'] > 1e-6 and len(disc['Q']) > 1:
                x_d = disc['Q'] / disc['Q_max'] * 100.   # SOD %
                idx_d = np.argsort(x_d)
                ax3.plot(x_d[idx_d], disc['V'][idx_d], '-',
                         lw=0.8, color=col, alpha=0.7, label=lbl_d)

            if chg and chg['Q_max'] > 1e-6 and len(chg['Q']) > 1:
                x_c = chg['Q'] / chg['Q_max'] * 100.    # SOC %
                idx_c = np.argsort(x_c)
                ax4.plot(x_c[idx_c], chg['V'][idx_c], '-',
                         lw=0.8, color=col, alpha=0.7, label=lbl_c)

        for ax, title, xl, x_hi_ax in [
                (ax3, 'Discharge  (E vs SOD)', 'SOD  (%)', 100.),
                (ax4, 'Charge  (E vs SOC)',    'SOC  (%)', 100.)]:
            ax.set_facecolor('white')
            for sp in ax.spines.values():
                sp.set_linewidth(0.8); sp.set_color('#333')
            ax.tick_params(which='major', direction='in', length=5,
                           width=0.8, labelsize=10, top=True, right=True)
            ax.tick_params(which='minor', direction='in', length=3,
                           width=0.6, top=True, right=True)
            ax.grid(True, which='major', alpha=0.2, linewidth=0.5,
                    color='#888', linestyle='--')
            ax.set_xlim(0., x_hi_ax)
            ax.set_ylim(y_lo_bot, y_hi_bot)   # full raw cap range
            ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
            ax.set_xlabel(xl, fontsize=11)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=11, fontweight='bold', pad=6)
            ax.legend(fontsize=9, framealpha=0.95, edgecolor='#bbb')

        fig.suptitle(
            r'Open circuit voltage  |  top: $V_{\mathrm{eq}}$ (relaxation endpoints)   '
            r'bottom: full $E$ vs SOD/SOC  (all cap data)',
            fontsize=12, fontweight='bold', y=1.01)
        fig.tight_layout()
        self._flush('ocv')

    def _export_figs(self):
        active = self._get_active()
        if not active:
            messagebox.showwarning('Nothing active', 'Select at least one sample.')
            return
        base = self._get_out_dir()

        # (tab_key, subdir, filename)
        fig_map = [
            # GITT raw
            ('gitt_full',         'GITT',                                'gitt_curves.png'),
            ('all_pulses',        'GITT/all_pulses_sqrt_t',              'all_pulses_sqrt_t.png'),
            ('all_xi',            'GITT/all_xi_pulses',                  'all_xi_pulses.png'),
            # Analysis
            ('ocv',               'Analysis/OCV',                        'OCV.png'),
            ('overpot_curves',    'Analysis/Overpotential_curves',       'overpotential_curves.png'),
            ('overpot',           'Analysis/Overpotential_Rtot',         'overpotential_Rtot.png'),
            ('relax_curves',      'Analysis/Relaxation_curves',          'relaxation_curves.png'),
            ('relax_delta',       'Analysis/Relaxation_dE',              'relaxation_dE.png'),
            ('relax_kinetics',    'Analysis/Relax_kinetics',             'relax_kinetics.png'),
            ('relax_kinetics_pp', 'Analysis/Relax_kinetics',             'relax_kinetics_per_pulse.png'),
            ('relax_map',         'Analysis/Relax_map',                  'relax_map_Veq.png'),
            ('relax_map_soc',     'Analysis/Relax_map_SOD_SOC',          'relax_map_SOD_SOC.png'),
            # Weppner & Huggins
            ('dEdx',              'Weppner_Huggins',                     'slopes_dEdx.png'),
            ('sqrt_fits',         'Weppner_Huggins',                     'sqrt_t_fits.png'),
            ('D_conv',            'Weppner_Huggins',                     'D_conventional.png'),
            # Kang & Chueh
            ('kc_fits',           'Kang_Chueh',                         'xi_fits.png'),
            ('kc_slopes',         'Kang_Chueh',                         'xi_slopes.png'),
            ('D_kc',              'Kang_Chueh',                         'D_kang_chueh.png'),
        ]

        saved = []
        for tab_key, subdir, fname in fig_map:
            if tab_key not in self.tabs:
                continue
            out_dir = os.path.join(base, *subdir.split('/'))
            os.makedirs(out_dir, exist_ok=True)
            fpath = os.path.join(out_dir, fname)
            try:
                self.tabs[tab_key]['fig'].savefig(
                    fpath, dpi=150, bbox_inches='tight', facecolor='white')
                saved.append(os.path.join(subdir, fname))
            except Exception as e:
                print(f'Figure save failed [{tab_key}]: {e}')

        messagebox.showinfo('Figures saved',
            f'Saved {len(saved)} figures to:\n{base}\n\n' +
            '\n'.join(f'  {f}' for f in saved[:20]) +
            (f'\n  ... and {len(saved)-20} more' if len(saved) > 20 else ''))

    def _export_curves(self):
        """Export ALL plot data to CSVs, organised in subdirectories by section.

        Directory layout:
          <out>/GITT/
                    gitt_curves.csv
                    all_pulses_sqrt_t/<sample>_all_pulses_sqrtt.csv
                    all_xi_pulses/<sample>_all_xi_pulses.csv
          <out>/Analysis/
                    OCV/
                    Overpotential_curves/
                    Overpotential_Rtot/
                    Relaxation_curves/
                    Relaxation_dE/
                    Relax_kinetics/      ← dV/dlogt + peak-tau summary
                    Relax_map/           ← 2D heatmap rows=Veq
                    Relax_map_SOD_SOC/   ← 2D heatmap rows=SOD%
          <out>/Weppner_Huggins/
          <out>/Kang_Chueh/
        """
        if not self._curve_data:
            messagebox.showwarning('No data', 'Run a plot first (press Replot).')
            return

        import math
        from scipy.signal import find_peaks as _fp

        active   = self._get_active()
        base     = self._get_out_dir()
        N_RESAMP = 120   # must match _dvdlogt / _relax_map_grid
        saved    = []    # relative paths for summary dialog

        # ── filter settings (same as displayed) ──────────────────────────────
        try:
            mr2 = float(self.min_r2.get())
            vlo = float(self.vlo.get())
            vhi = float(self.vhi.get())
            sig = float(self.sigma_clip.get())
        except Exception:
            mr2, vlo, vhi, sig = 0.85, 0.0, 2.0, 0.0

        try:
            tau = float(self.tau_s.get())
        except Exception:
            tau = 600.

        # ── helpers ───────────────────────────────────────────────────────────

        def _mkdir(*parts):
            d = os.path.join(base, *parts)
            os.makedirs(d, exist_ok=True)
            return d

        def _write_csv(path, datasets):
            """datasets = list of (x_name, y_name, x_list, y_list).
            Side-by-side XY pairs, Origin-compatible."""
            if not datasets:
                return
            maxlen = max(len(d[2]) for d in datasets)
            header = ','.join(f'{d[0]},{d[1]}' for d in datasets)
            rows   = [header]
            for i in range(maxlen):
                cols = []
                for _, _, xs, ys in datasets:
                    cols.append(f'{xs[i]:.8g},{ys[i]:.8g}'
                                if i < len(xs) else ',')
                rows.append(','.join(cols))
            with open(path, 'w') as fp:
                fp.write('\n'.join(rows))
            saved.append(os.path.relpath(path, base))

        def _write_2d_csv(path, lt_ref, y_vals, y_col_name, Z, sample_name):
            """2D heatmap CSV: first col = row label, rest = log_t values."""
            with open(path, 'w') as fp:
                fp.write(f'{y_col_name}_{sample_name},' +
                         ','.join(f'{lt:.4f}' for lt in lt_ref) + '\n')
                for rl, row in zip(y_vals, Z):
                    fp.write(f'{rl:.6g},' +
                             ','.join(f'{v:.6g}' for v in row) + '\n')
            saved.append(os.path.relpath(path, base))

        def _blocks_xy(cyc, x_field, y_field, r2_field=None,
                       filter_fn=None, allow_negative=False):
            """Return {sample: (xs, ys)} with current filters applied."""
            out_d = {}
            for r in self._curve_data:
                if r['cycle'] != cyc:
                    continue
                x = r.get(x_field, float('nan'))
                y = r.get(y_field, float('nan'))
                if not np.isfinite(x):
                    continue
                if allow_negative:
                    if not (np.isfinite(y) and y != 0):
                        continue
                else:
                    if not (np.isfinite(y) and y > 0):
                        continue
                veq = r.get('V_eq', x)
                if not (vlo <= veq <= vhi):
                    continue
                if r2_field:
                    r2v = r.get(r2_field, float('nan'))
                    if not (np.isfinite(r2v) and r2v >= mr2):
                        continue
                if filter_fn and not filter_fn(r):
                    continue
                nm = r['sample']
                out_d.setdefault(nm, ([], []))
                out_d[nm][0].append(x)
                out_d[nm][1].append(y)
            result = {}
            for nm, (xs, ys) in out_d.items():
                if not xs:
                    continue
                if sig > 0 and len(ys) >= 4:
                    pos = [(x, y) for x, y in zip(xs, ys)
                           if (y > 0 if not allow_negative else y != 0)]
                    if not pos:
                        continue
                    xs2, ys2 = zip(*pos)
                    xs2, ys2 = list(xs2), list(ys2)
                    logy = [math.log10(abs(y)) for y in ys2]
                    med  = sorted(logy)[len(logy) // 2]
                    mad  = sorted(abs(l - med) for l in logy)[len(logy) // 2]
                    std  = mad * 1.4826
                    keep = [abs(l - med) <= sig * max(std, 0.05) for l in logy]
                    xs   = [x for x, k in zip(xs2, keep) if k]
                    ys   = [y for y, k in zip(ys2, keep) if k]
                if xs:
                    pairs = sorted(zip(xs, ys))
                    result[nm] = ([p[0] for p in pairs],
                                  [p[1] for p in pairs])
            return result

        # =====================================================================
        # SECTION 1 — GITT
        # =====================================================================
        gitt_dir = _mkdir('GITT')

        # 1a. Full GITT V(t) and I(t)
        datasets = []
        for nm in active:
            td = self._samples.get(nm, {}).get('td')
            if td is None or td.shape[0] == 0:
                continue
            T = td[:, 2].tolist()
            V = td[:, 0].tolist()
            I = (td[:, 1] * 1e6).tolist()   # A → µA
            datasets += [
                (f'{nm}_time_s', f'{nm}_V_vs_ref', T, V),
                (f'{nm}_time_s', f'{nm}_I_uA',     T, I),
            ]
        if datasets:
            _write_csv(os.path.join(gitt_dir, 'gitt_curves.csv'), datasets)

        # 1b. All pulses sqrt(t) — one CSV per sample
        sqt_dir = _mkdir('GITT', 'all_pulses_sqrt_t')
        for nm in active:
            s = self._samples.get(nm, {})
            datasets = []
            for cyc in ('discharge', 'charge'):
                bl = [b for b in s.get('blocks', [])
                      if b['cycle'] == cyc and len(b['_Tp']) >= 2]
                for bi, b in enumerate(bl, 1):
                    t_rel = b['_Tp'] - b['_Tp'][0]
                    veq   = b.get('V_eq', float('nan'))
                    lbl   = f'{cyc[0].upper()}{bi}_Veq{veq:.3f}V'
                    datasets.append((f'{lbl}_sqrtt_s05', f'{lbl}_V',
                                     np.sqrt(t_rel).tolist(),
                                     b['_Vp'].tolist()))
            if datasets:
                _write_csv(os.path.join(sqt_dir,
                           f'{nm}_all_pulses_sqrtt.csv'), datasets)

        # 1c. All xi relaxation — one CSV per sample
        xi_dir = _mkdir('GITT', 'all_xi_pulses')
        for nm in active:
            s = self._samples.get(nm, {})
            datasets = []
            for cyc in ('discharge', 'charge'):
                bl = [b for b in s.get('blocks', [])
                      if b['cycle'] == cyc and len(b['_x_kc']) >= 2]
                for bi, b in enumerate(bl, 1):
                    veq = b.get('V_eq', float('nan'))
                    dE  = ((b['_Vr'] - veq) * 1000.).tolist() \
                          if np.isfinite(veq) else (b['_Vr'] * 1000.).tolist()
                    lbl = f'{cyc[0].upper()}{bi}_Veq{veq:.3f}V'
                    datasets.append((f'{lbl}_xi_s05', f'{lbl}_dE_mV',
                                     b['_x_kc'].tolist(), dE))
            if datasets:
                _write_csv(os.path.join(xi_dir,
                           f'{nm}_all_xi_pulses.csv'), datasets)

        # =====================================================================
        # SECTION 2 — Analysis
        # =====================================================================

        # 2a. OCV
        ocv_dir = _mkdir('Analysis', 'OCV')
        for cyc_key, cyc_lbl in [('disc', 'discharge'), ('chg', 'charge')]:
            datasets = []
            for nm in active:
                s   = self._samples.get(nm, {})
                seg = s.get(cyc_key, {})
                if not (seg and 'Q' in seg and 'V' in seg):
                    continue
                Q    = np.array(seg['Q'])
                V    = np.array(seg['V'])
                Qmax = s.get('q_max_val') or Q.max()
                if Qmax <= 0 or len(Q) < 2:
                    continue
                x   = Q / Qmax
                idx = np.argsort(x)
                datasets.append((f'{nm}_x_norm', f'{nm}_Veq_V',
                                 x[idx].tolist(), V[idx].tolist()))
            if datasets:
                _write_csv(os.path.join(ocv_dir,
                           f'OCV_{cyc_lbl}.csv'), datasets)

        # OCV from clean block endpoints
        for cyc_lbl in ('discharge', 'charge'):
            datasets = []
            for nm in active:
                bl = [b for b in self._samples.get(nm, {}).get('blocks', [])
                      if b['cycle'] == cyc_lbl
                      and np.isfinite(b.get('V_eq', float('nan')))
                      and np.isfinite(b.get('x_block', float('nan')))]
                if not bl:
                    continue
                xs = [b['x_block'] * 100. for b in bl]
                ys = [b['V_eq'] for b in bl]
                xlbl = 'SOD_pct' if cyc_lbl == 'discharge' else 'SOC_pct'
                datasets.append((f'{nm}_{xlbl}', f'{nm}_Veq_V', xs, ys))
            if datasets:
                _write_csv(os.path.join(ocv_dir,
                           f'OCV_Veq_blocks_{cyc_lbl}.csv'), datasets)

        # 2b. Overpotential curves — η(t) per pulse, one CSV per sample
        op_dir = _mkdir('Analysis', 'Overpotential_curves')
        for cyc in ('discharge', 'charge'):
            for nm in active:
                s      = self._samples.get(nm, {})
                bl_all = [b for b in s.get('blocks', [])
                          if b['cycle'] == cyc and len(b['_Vp']) >= 2
                          and np.isfinite(b.get('V_eq', float('nan')))]
                datasets = []
                prev_b   = None
                for bi, b in enumerate(bl_all, 1):
                    t_rel  = (b['_Tp'] - b['_Tp'][0]).tolist()
                    v_ref  = (prev_b['V_eq']
                              if prev_b is not None
                              and np.isfinite(prev_b.get('V_eq', float('nan')))
                              else b['_Vp'][0])
                    eta    = ((b['_Vp'] - v_ref) * 1000.).tolist()
                    veq    = b.get('V_eq', float('nan'))
                    lbl    = f'pulse{bi}_Veq{veq:.3f}V'
                    datasets.append((f'{lbl}_t_s', f'{lbl}_eta_mV',
                                     t_rel, eta))
                    prev_b = b
                if datasets:
                    _write_csv(os.path.join(op_dir,
                               f'{nm}_overpot_curves_{cyc}.csv'), datasets)

        # 2c. Overpotential & Rtot summary
        opR_dir = _mkdir('Analysis', 'Overpotential_Rtot')
        for cyc in ('discharge', 'charge'):
            xlbl = 'SOD_pct' if cyc == 'discharge' else 'SOC_pct'
            ds_sod_eta=[]; ds_sod_R=[]; ds_veq_eta=[]; ds_veq_R=[]
            for nm in active:
                s      = self._samples.get(nm, {})
                ctrl   = self._sample_controls.get(nm, {})
                mass_g = ctrl['mass'].get() / 1000. if 'mass' in ctrl else 0.
                bl = [b for b in s.get('blocks', [])
                      if b['cycle'] == cyc and len(b['_Vp']) > 1
                      and np.isfinite(b.get('V_eq', float('nan')))
                      and np.isfinite(b.get('x_block', float('nan')))]
                sod_l=[]; eta_l=[]; R_l=[]; veq_l=[]
                prev_b = None
                for b in bl:
                    if prev_b is not None:
                        try:
                            gap    = b['_Tp'][0] - prev_b['_Tr'][-1]
                            dur    = prev_b['_Tr'][-1] - prev_b['_Tr'][0]
                            consec = gap < max(dur * 0.5, 300.)
                        except Exception:
                            consec = True
                        if consec:
                            eta   = (b['_Vp'][-1] - prev_b['V_eq'])
                            I_p   = b['I_pulse']
                            R_tot = eta / I_p if abs(I_p) > 1e-12 else float('nan')
                            R_n   = R_tot * mass_g if mass_g > 0 else R_tot
                            sod_l.append(b['x_block'] * 100.)
                            eta_l.append(eta * 1000.)
                            R_l.append(R_n)
                            veq_l.append(b['V_eq'])
                    prev_b = b
                if sod_l:
                    ds_sod_eta.append((f'{nm}_{xlbl}', f'{nm}_eta_mV',
                                       sod_l, eta_l))
                    ds_sod_R.append(  (f'{nm}_{xlbl}', f'{nm}_Rtot',
                                       sod_l, R_l))
                    ds_veq_eta.append((f'{nm}_Veq_V',  f'{nm}_eta_mV',
                                       veq_l, eta_l))
                    ds_veq_R.append(  (f'{nm}_Veq_V',  f'{nm}_Rtot',
                                       veq_l, R_l))
            if ds_sod_eta:
                _write_csv(os.path.join(opR_dir,
                           f'overpot_Rtot_vs_{xlbl}_{cyc}.csv'),
                           ds_sod_eta + ds_sod_R)
            if ds_veq_eta:
                _write_csv(os.path.join(opR_dir,
                           f'overpot_Rtot_vs_Veq_{cyc}.csv'),
                           ds_veq_eta + ds_veq_R)

        # 2d. Relaxation curves — (V−Veq)(t) per pulse, one CSV per sample
        rc_dir = _mkdir('Analysis', 'Relaxation_curves')
        for cyc in ('discharge', 'charge'):
            for nm in active:
                s  = self._samples.get(nm, {})
                bl = [b for b in s.get('blocks', [])
                      if b['cycle'] == cyc and len(b['_Vr']) >= 2
                      and np.isfinite(b.get('V_eq', float('nan')))]
                datasets = []
                for bi, b in enumerate(bl, 1):
                    t_rel = (b['_Tr'] - b['_Tr'][0]).tolist()
                    dV    = ((b['_Vr'] - b['V_eq']) * 1000.).tolist()
                    veq   = b['V_eq']
                    lbl   = f'pulse{bi}_Veq{veq:.3f}V'
                    datasets.append((f'{lbl}_t_s', f'{lbl}_dV_mV',
                                     t_rel, dV))
                if datasets:
                    _write_csv(os.path.join(rc_dir,
                               f'{nm}_relax_curves_{cyc}.csv'), datasets)

        # 2e. Relaxation ΔE (total voltage recovery per pulse)
        rdE_dir = _mkdir('Analysis', 'Relaxation_dE')
        for cyc in ('discharge', 'charge'):
            datasets = []
            for nm in active:
                bl = [b for b in self._samples.get(nm, {}).get('blocks', [])
                      if b['cycle'] == cyc and len(b['_Vr']) >= 2
                      and np.isfinite(b.get('x_block', float('nan')))]
                if not bl:
                    continue
                xlbl = 'SOD_pct' if cyc == 'discharge' else 'SOC_pct'
                xs   = [b['x_block'] * 100. for b in bl]
                ys   = [b['_Vr'][-1] - b['_Vr'][0] for b in bl]
                datasets.append((f'{nm}_{xlbl}',
                                 f'{nm}_dV_relax_V', xs, ys))
            if datasets:
                _write_csv(os.path.join(rdE_dir,
                           f'relax_delta_{cyc}.csv'), datasets)

        # 2f. Relax kinetics — dV/d(log t) curves + peak-tau summary
        rk_dir = _mkdir('Analysis', 'Relax_kinetics')
        for cyc in ('discharge', 'charge'):
            # Per-sample: all pulses dV/dlogt
            for nm in active:
                s  = self._samples.get(nm, {})
                bl = [b for b in s.get('blocks', [])
                      if b['cycle'] == cyc and len(b['_Vr']) >= 10
                      and np.isfinite(b.get('V_eq', float('nan')))]
                datasets = []
                for bi, b in enumerate(bl, 1):
                    log_t, dvd_sm = GITTApp._dvdlogt(b, n_resamp=N_RESAMP)
                    if log_t is None:
                        continue
                    veq = b['V_eq']
                    lbl = f'pulse{bi}_Veq{veq:.3f}V'
                    datasets.append((f'{lbl}_log10t',
                                     f'{lbl}_dVdlogt_mVdec',
                                     log_t.tolist(), dvd_sm.tolist()))
                if datasets:
                    _write_csv(os.path.join(rk_dir,
                               f'{nm}_relax_kinetics_{cyc}.csv'), datasets)

            # Cross-sample: dominant peak log_tau vs Veq
            datasets = []
            for nm in active:
                s  = self._samples.get(nm, {})
                bl = [b for b in s.get('blocks', [])
                      if b['cycle'] == cyc and len(b['_Vr']) >= 10
                      and np.isfinite(b.get('V_eq', float('nan')))]
                veqs=[]; peak_taus=[]; peak_amps=[]
                for b in bl:
                    log_t, dvd_sm = GITTApp._dvdlogt(b, n_resamp=N_RESAMP)
                    if log_t is None:
                        continue
                    h    = max(dvd_sm.max() * 0.15, 1e-6)
                    pks, props = _fp(dvd_sm, height=h, distance=7,
                                     prominence=dvd_sm.max() * 0.20)
                    if len(pks) > 0:
                        main = pks[np.argmax(dvd_sm[pks])]
                        veqs.append(b['V_eq'])
                        peak_taus.append(log_t[main])
                        peak_amps.append(float(dvd_sm[main]))
                if veqs:
                    datasets.append((f'{nm}_Veq_V',
                                     f'{nm}_peak_log10tau',
                                     veqs, peak_taus))
                    datasets.append((f'{nm}_Veq_V',
                                     f'{nm}_peak_amp_mVdec',
                                     veqs, peak_amps))
            if datasets:
                _write_csv(os.path.join(rk_dir,
                           f'peak_log_tau_{cyc}.csv'), datasets)

        # 2g. Relax map 2D — y = Veq
        rm_dir = _mkdir('Analysis', 'Relax_map')
        for cyc in ('discharge', 'charge'):
            for nm in active:
                s  = self._samples.get(nm, {})
                bl = [b for b in s.get('blocks', [])
                      if b['cycle'] == cyc and len(b['_Vr']) >= 10
                      and np.isfinite(b.get('V_eq', float('nan')))]
                if not bl:
                    continue
                bl_s   = sorted(bl, key=lambda b: b['V_eq'],
                                 reverse=(cyc == 'discharge'))
                lt_ref, Z = self._relax_map_grid(bl_s, N_RESAMP)
                if lt_ref is None:
                    continue
                y_vals = [b['V_eq'] for b in bl_s]
                _write_2d_csv(
                    os.path.join(rm_dir,
                                 f'{nm}_relax_map_Veq_{cyc}.csv'),
                    lt_ref, y_vals, 'Veq_V', Z, nm)

        # 2h. Relax map 2D — y = SOD/SOC%
        rms_dir = _mkdir('Analysis', 'Relax_map_SOD_SOC')
        for cyc in ('discharge', 'charge'):
            for nm in active:
                s  = self._samples.get(nm, {})
                bl = [b for b in s.get('blocks', [])
                      if b['cycle'] == cyc and len(b['_Vr']) >= 10
                      and np.isfinite(b.get('V_eq', float('nan')))
                      and np.isfinite(b.get('x_block', float('nan')))]
                if not bl:
                    continue
                bl_s   = sorted(bl, key=lambda b: b['x_block'])
                lt_ref, Z = self._relax_map_grid(bl_s, N_RESAMP)
                if lt_ref is None:
                    continue
                y_vals = [b['x_block'] * 100. for b in bl_s]
                ylbl   = 'SOD_pct' if cyc == 'discharge' else 'SOC_pct'
                _write_2d_csv(
                    os.path.join(rms_dir,
                                 f'{nm}_relax_map_{ylbl}_{cyc}.csv'),
                    lt_ref, y_vals, ylbl, Z, nm)

        # =====================================================================
        # SECTION 3 — Weppner & Huggins
        # =====================================================================
        wh_dir = _mkdir('Weppner_Huggins')

        for cyc in ('discharge', 'charge'):
            # D_conv vs Veq
            data = _blocks_xy(cyc, 'V_eq', 'D_conv', r2_field='r2_sqrt')
            ds = [(f'{nm}_Veq_V', f'{nm}_Dconv_mol2s',
                   list(xs), list(ys))
                  for nm in active if nm in data
                  for xs, ys in [data[nm]]]
            if ds:
                _write_csv(os.path.join(wh_dir,
                           f'D_conv_{cyc}.csv'), ds)

            # D_conv vs SOD/SOC
            data2 = {}
            for r in self._curve_data:
                if r['cycle'] != cyc:
                    continue
                if not (np.isfinite(r.get('D_conv', float('nan')))
                        and r.get('D_conv', 0) > 0):
                    continue
                if not np.isfinite(r.get('x_block', float('nan'))):
                    continue
                nm2 = r['sample']
                data2.setdefault(nm2, ([], []))
                data2[nm2][0].append(r['x_block'] * 100.)
                data2[nm2][1].append(r['D_conv'])
            xlbl2 = 'SOD_pct' if cyc == 'discharge' else 'SOC_pct'
            ds2 = [(f'{nm}_{xlbl2}', f'{nm}_Dconv_mol2s',
                    list(xs), list(ys))
                   for nm in active if nm in data2
                   for xs, ys in [data2[nm]]]
            if ds2:
                _write_csv(os.path.join(wh_dir,
                           f'D_conv_vs_{xlbl2}_{cyc}.csv'), ds2)

            # slope sqrt(t)
            data = _blocks_xy(cyc, 'V_eq', 'sl_sqrt', r2_field='r2_sqrt',
                              filter_fn=lambda r: abs(r.get('sl_sqrt', 0)) > 0,
                              allow_negative=True)
            ds = [(f'{nm}_Veq_V', f'{nm}_slope_sqrtt_V_s05',
                   list(xs), [abs(y) for y in ys])
                  for nm in active if nm in data
                  for xs, ys in [data[nm]]]
            if ds:
                _write_csv(os.path.join(wh_dir,
                           f'slope_sqrt_t_{cyc}.csv'), ds)

            # dEdx
            data = _blocks_xy(cyc, 'V_eq', 'dEdx', r2_field='r2_sqrt',
                              filter_fn=lambda r: abs(r.get('dEdx', 0)) > 0)
            ds = [(f'{nm}_Veq_V', f'{nm}_dEdx_V',
                   list(xs), [abs(y) for y in ys])
                  for nm in active if nm in data
                  for xs, ys in [data[nm]]]
            if ds:
                _write_csv(os.path.join(wh_dir,
                           f'dEdx_{cyc}.csv'), ds)

        # dEdx vs V smooth curve
        for cyc_key, cyc_lbl in [('disc', 'discharge'), ('chg', 'charge')]:
            datasets = []
            for nm in active:
                s       = self._samples.get(nm, {})
                seg     = s.get(cyc_key, {})
                dk      = 'dedx_d' if cyc_key == 'disc' else 'dedx_c'
                dedx_fn = s.get(dk) or s.get('dedx')
                if not (seg and 'Q' in seg and 'V' in seg and dedx_fn):
                    continue
                V   = np.array(seg['V'])
                idx = np.argsort(V)
                vs  = V[idx].tolist()
                dv  = []
                for vi in vs:
                    try:
                        v = float(dedx_fn(vi))
                        dv.append(abs(v) if np.isfinite(v) else None)
                    except Exception:
                        dv.append(None)
                vf = [v for v, d in zip(vs, dv) if d is not None]
                df = [d for d in dv if d is not None]
                if vf:
                    datasets.append((f'{nm}_Veq_V', f'{nm}_dEdx_V',
                                     vf, df))
            if datasets:
                _write_csv(os.path.join(wh_dir,
                           f'dEdx_vs_V_{cyc_lbl}.csv'), datasets)

        # OCV from blocks (also needed for W&H context)
        for cyc_key, cyc_lbl in [('disc', 'discharge'), ('chg', 'charge')]:
            datasets = []
            for nm in active:
                s   = self._samples.get(nm, {})
                seg = s.get(cyc_key, {})
                if not (seg and 'Q' in seg and 'V' in seg):
                    continue
                Q    = np.array(seg['Q'])
                V    = np.array(seg['V'])
                Qmax = s.get('q_max_val') or Q.max()
                if Qmax <= 0 or len(Q) < 2:
                    continue
                x   = Q / Qmax
                idx = np.argsort(x)
                datasets.append((f'{nm}_x_norm', f'{nm}_Veq_V',
                                 x[idx].tolist(), V[idx].tolist()))
            if datasets:
                _write_csv(os.path.join(wh_dir,
                           f'OCV_{cyc_lbl}.csv'), datasets)

        # =====================================================================
        # SECTION 4 — Kang & Chueh
        # =====================================================================
        kc_dir = _mkdir('Kang_Chueh')

        for cyc in ('discharge', 'charge'):
            # D_kc vs Veq
            data = _blocks_xy(cyc, 'V_eq', 'D_kc', r2_field='r2_kc')
            ds = [(f'{nm}_Veq_V', f'{nm}_Dkc_mol2s',
                   list(xs), list(ys))
                  for nm in active if nm in data
                  for xs, ys in [data[nm]]]
            if ds:
                _write_csv(os.path.join(kc_dir,
                           f'D_kc_{cyc}.csv'), ds)

            # D_kc vs SOD/SOC
            data3 = {}
            for r in self._curve_data:
                if r['cycle'] != cyc:
                    continue
                if not (np.isfinite(r.get('D_kc', float('nan')))
                        and r.get('D_kc', 0) > 0):
                    continue
                if not np.isfinite(r.get('x_block', float('nan'))):
                    continue
                nm3 = r['sample']
                data3.setdefault(nm3, ([], []))
                data3[nm3][0].append(r['x_block'] * 100.)
                data3[nm3][1].append(r['D_kc'])
            xlbl3 = 'SOD_pct' if cyc == 'discharge' else 'SOC_pct'
            ds3 = [(f'{nm}_{xlbl3}', f'{nm}_Dkc_mol2s',
                    list(xs), list(ys))
                   for nm in active if nm in data3
                   for xs, ys in [data3[nm]]]
            if ds3:
                _write_csv(os.path.join(kc_dir,
                           f'D_kc_vs_{xlbl3}_{cyc}.csv'), ds3)

            # slope xi
            data = _blocks_xy(cyc, 'V_eq', 'sl_kc', r2_field='r2_kc',
                              filter_fn=lambda r: abs(r.get('sl_kc', 0)) > 0,
                              allow_negative=True)
            ds = [(f'{nm}_Veq_V', f'{nm}_slope_xi_V_s05',
                   list(xs), [abs(y) for y in ys])
                  for nm in active if nm in data
                  for xs, ys in [data[nm]]]
            if ds:
                _write_csv(os.path.join(kc_dir,
                           f'slope_xi_{cyc}.csv'), ds)

        # ── Done ─────────────────────────────────────────────────────────────
        if saved:
            messagebox.showinfo('Exported',
                f'Saved {len(saved)} CSV files to:\n{base}\n\n' +
                '\n'.join(f'  {f}' for f in saved[:30]) +
                (f'\n  ... and {len(saved)-30} more'
                 if len(saved) > 30 else ''))
        else:
            messagebox.showwarning('Nothing exported',
                'No data passed the quality filters. Try lowering Min R2.')

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser=argparse.ArgumentParser(description='GITT GUI App')
    parser.add_argument('--data_dir',default=None)
    args=parser.parse_args()
    root=tk.Tk()
    root.geometry('1350x850')
    GITTApp(root,default_data_dir=args.data_dir)
    root.mainloop()

if __name__=='__main__': main()
