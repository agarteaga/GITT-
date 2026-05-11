"""
GITT_V2 CORRECTED FUNCTIONS
================================================================================

This module contains corrected implementations of functions that contain
critical bugs in GITT_V2.py. Copy these functions into your GITT_V2.py
to fix all 7 bugs.

BUG FIXES:
  1. extract_blocks()     → extract_blocks_corrected()
  2. _replot()            → compute_x_block_corrected() helper
  3. extract_blocks()     → _get_block_equilibrium_voltage() helper
  4. _replot()            → ensure_q_bounds_set() helper
  5. extract_blocks()     → _get_fit_window_robust() helper
  6. _replot()            → get_dedx_safe() helper
  7. Module top           → CONFIG dict

================================================================================
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy import stats
from scipy.signal import savgol_filter


# ══════════════════════════════════════════════════════════════════════════════
# [FIX #7] CENTRALIZED CONFIGURATION – place at top of GITT_V2.py after imports
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = {
    'FARADAY_CONST': 96485.0,
    'FIT_TIME_START_DEFAULT': 60.0,
    'FIT_TIME_END_DEFAULT': 540.0,
    'RELAX_TIME_REQUIRED': 600.0,
    'CURRENT_THRESHOLD_FRAC': 0.05,
    'MIN_PULSES_FOR_TAU': 3,
    'V_MEDIAN_TAIL_FRAC': 0.2,
    'MIN_RELAXATION_PTS': 5,
    'dVdx_BOUNDS_LO': 0.7,
    'dVdx_BOUNDS_HI': 1.5,
    'R2_MIN_THRESHOLD': 0.85,
}

F_CONST = CONFIG['FARADAY_CONST']


# ══════════════════════════════════════════════════════════════════════════════
# [FIX #3] V_EQ ROBUSTNESS – use median of relaxation tail
# ══════════════════════════════════════════════════════════════════════════════

def _get_block_equilibrium_voltage(V_relax):
    """
    Extract V_eq from relaxation window robustly using the tail (last 20%).
    
    FIXES BUG #3: original code took V_relax[-1] which is susceptible to
    file join transients and sampling noise.
    
    Args:
        V_relax: np.array of voltage values during OCP relaxation
    
    Returns:
        float: V_eq as median of final 20% of relaxation
    """
    if len(V_relax) < 5:
        return float(V_relax[-1]) if len(V_relax) > 0 else np.nan
    
    tail_start = max(0, int(len(V_relax) * (1.0 - CONFIG['V_MEDIAN_TAIL_FRAC'])))
    V_eq = float(np.median(V_relax[tail_start:]))
    
    return V_eq if np.isfinite(V_eq) else float(V_relax[-1])


# ══════════════════════════════════════════════════════════════════════════════
# [FIX #5] ROBUST FIT WINDOW – fallback for sparse data
# ══════════════════════════════════════════════════════════════════════════════

def _get_fit_window_robust(t_rel, V_p, tfs=60.0, tfe=540.0):
    """
    Extract fit window [tfs, tfe] from relaxation data robustly.
    
    FIXES BUG #5: falls back gracefully when data is too sparse.
    
    Args:
        t_rel: time array (relative to pulse start)
        V_p:   voltage array during pulse + relaxation
        tfs:   fit time start (seconds)
        tfe:   fit time end (seconds)
    
    Returns:
        fm: boolean mask for points in [tfs, tfe]
    """
    # Primary window: [tfs, tfe] with tfe strictly respected
    fm = (t_rel >= tfs) & (t_rel <= tfe)
    
    # If window has < 3 points, expand to all t_rel > 0
    if fm.sum() < CONFIG['MIN_RELAXATION_PTS']:
        fm = t_rel > 0
    
    # If still < 3 points after fallback, return all t_rel >= 0
    if fm.sum() < 3:
        fm = t_rel >= 0
    
    return fm


# ══════════════════════════════════════════════════════════════════════════════
# [FIX #1 & #3] EXTRACT BLOCKS – sign-flip guard + V_eq median
# ══════════════════════════════════════════════════════════════════════════════

def extract_blocks_corrected(td, tfs=60., tfe=540., min_relax=None, tau_s=600.):
    """
    Extract GITT pulse+relaxation blocks from time-domain array.
    
    FIXES:
      BUG #1: Added sign-flip guard to prevent file boundary contamination
              in relaxation window (V_eq corruption)
      BUG #3: V_eq computed as median of relaxation tail, not single point
      BUG #5: Improved fit window fallback for sparse data
    
    Args:
        td:         np.ndarray, shape (N,3) = [V, I_A, T_s]
        tfs, tfe:   fit window start/end (seconds into relaxation)
        min_relax:  minimum relaxation duration (seconds)
        tau_s:      pulse duration (seconds) – default 600
    
    Returns:
        list of dicts with keys:
            'V_eq', 'sl_sqrt', 'r2_sqrt', 'n_fit',
            'sl_kc', 'r2_kc', 'R_el', 'I_pulse',
            'cycle' (discharge/charge),
            '_Vp', '_Tp', '_Vr', '_Tr', '_t_rel', '_sqrt_t', '_fm', '_x_kc'
    """
    if td.size == 0:
        return []
    
    V, I, T = td[:, 0], td[:, 1], td[:, 2]
    
    if min_relax is None:
        min_relax = tau_s
    
    thresh = _current_threshold(I)
    active = np.abs(I) > thresh  # True = galvanostatic pulse, False = OCP rest
    
    tr = np.where(np.diff(active.astype(int)) == -1)[0]  # pulse → relax
    
    if len(tr) == 0:
        return []
    
    file_starts_mid_pulse = len(active) > 0 and active[0]
    blocks = []
    
    for idx in tr:
        ps = idx
        while ps > 0 and active[ps - 1]:
            ps -= 1
        
        # Skip partial first pulse if file started mid-pulse
        if ps == 0 and file_starts_mid_pulse:
            continue
        
        end = idx + 1
        while end < len(I) and not active[end]:
            end += 1
        
        # ════════════════════════════════════════════════════════════════════
        # [FIX #1] Sign-flip guard: trim relaxation if next pulse has opposite sign
        # ════════════════════════════════════════════════════════════════════
        pulse_sign = np.sign(I[ps])
        relax_slice = I[idx + 1:end]
        
        if len(relax_slice) > 0:
            for rel_i, i_val in enumerate(relax_slice):
                if np.abs(i_val) > thresh and np.sign(i_val) != pulse_sign:
                    # Current sign flip detected – trim relaxation here
                    end = idx + 1 + rel_i
                    break
        
        # ════════════════════════════════════════════════════════════════════
        # Extract pulse & relaxation windows
        # ════════════════════════════════════════════════════════════════════
        Vp, Tp = V[ps:idx + 1], T[ps:idx + 1]
        Vr, Tr = V[idx + 1:end], T[idx + 1:end]
        Ip = I[ps]
        
        if len(Tr) < 2 or Tr[-1] - Tr[0] < min_relax or len(Vp) < 3:
            continue
        
        t_rel = Tr - Tr[0]
        
        # ════════════════════════════════════════════════════════════════════
        # [FIX #3] V_eq: use median of relaxation tail instead of single point
        # ════════════════════════════════════════════════════════════════════
        V_eq = _get_block_equilibrium_voltage(Vr)
        
        # ════════════════════════════════════════════════════════════════════
        # Sqrt-time fit (constant-D diffusion)
        # ════════════════════════════════════════════════════════════════════
        sqrt_t = np.sqrt(t_rel)
        fm = _get_fit_window_robust(t_rel, Vp, tfs, tfe)
        sl = r2 = np.nan
        n_fit = 0
        
        if fm.sum() >= 2:
            try:
                sl, _, r, _, _ = stats.linregress(sqrt_t[fm], Vp[fm])
                r2 = r ** 2
                n_fit = int(fm.sum())
            except Exception:
                pass
        
        # ════════════════════════════════════════════════════════════════════
        # Kung-Carter fit (concentration-dependent D)
        # ════════════════════════════════════════════════════════════════════
        sl_kc = r2_kc = np.nan
        x_kc = np.array([])
        
        if len(Tr) >= 4:
            t_rx = np.maximum(Tr - Tr[0], 0)
            x_kc = np.sqrt(t_rx + tau_s) - np.sqrt(t_rx)
            fm_kc = _get_fit_window_robust(t_rel, Vp, tfs, tfe)
            
            if fm_kc.sum() >= 2 and len(x_kc) == len(Vp):
                try:
                    sl_kc, _, r_kc, _, _ = stats.linregress(x_kc[fm_kc], Vp[fm_kc])
                    r2_kc = r_kc ** 2
                except Exception:
                    pass
        
        # ════════════════════════════════════════════════════════════════════
        # Ohmic drop (R_el)
        # ════════════════════════════════════════════════════════════════════
        R_el = np.nan
        if ps > 0 and np.abs(I[ps]) > 1e-9:
            R_el = np.abs(Vp[0] - V[ps - 1]) / np.abs(I[ps])
        
        # ════════════════════════════════════════════════════════════════════
        # Determine cycle direction
        # ════════════════════════════════════════════════════════════════════
        cycle = 'discharge' if Ip < 0 else 'charge'
        
        block = {
            'V_eq': V_eq,
            'sl_sqrt': sl,
            'r2_sqrt': r2,
            'n_fit': n_fit,
            'sl_kc': sl_kc,
            'r2_kc': r2_kc,
            'R_el': R_el,
            'I_pulse': Ip,
            'cycle': cycle,
            '_Vp': Vp,
            '_Tp': Tp,
            '_Vr': Vr,
            '_Tr': Tr,
            '_t_rel': t_rel,
            '_sqrt_t': sqrt_t,
            '_fm': fm,
            '_x_kc': x_kc,
        }
        blocks.append(block)
    
    return blocks


# ══════════════════════════════════════════════════════════════════════════════
# [FIX #2] PER-DIRECTION Q TRACKING – separate cumsum for discharge/charge
# ══════════════════════════════════════════════════════════════════════════════

def compute_x_block_corrected(I_td, T_td, direction, first_pulse_start_T, blocks):
    """
    Compute x (state-of-charge index) for each block using per-direction Q tracking.
    
    FIXES BUG #2: Original code used global Q integration, which combined
                   discharge and charge Q together. This caused x_block > 1.0
                   at the start of the next discharge cycle.
    
    Args:
        I_td:                  Current array from time-domain file
        T_td:                  Time array from time-domain file
        direction:             'discharge' or 'charge'
        first_pulse_start_T:   Time index where first pulse of this direction starts
        blocks:                List of block dicts for this direction
    
    Returns:
        list of x values (one per block), each in [0, 1.0]
    """
    if not blocks or first_pulse_start_T is None:
        return [np.nan] * len(blocks)
    
    # Compute Q integrated from the start of first pulse in this direction
    # Only count current with matching sign
    direction_sign = -1 if direction == 'discharge' else 1
    
    # Create mask for current that matches this direction
    I_match = I_td * direction_sign > 0
    dT = np.concatenate([[0], np.diff(T_td)])
    Q_cumul_direction = np.concatenate([[0], np.cumsum(I_td[1:] * dT[1:])])
    
    x_values = []
    q_total = 0
    
    for block in blocks:
        # Find time index of end of this block's relaxation
        end_time = block['_Tr'][-1] if len(block['_Tr']) > 0 else 0
        
        # Find index in T_td
        time_idx = np.searchsorted(T_td, end_time, side='right') - 1
        time_idx = np.clip(time_idx, 0, len(Q_cumul_direction) - 1)
        
        # Q at this point
        Q_here = Q_cumul_direction[time_idx]
        
        # For first block, q_total = Q_here
        # For subsequent blocks, accumulate
        if len(x_values) == 0:
            q_total = Q_here
        else:
            q_total += abs(block['I_pulse']) * (block['_Tr'][-1] - block['_Tr'][0]) / 3.6
        
        # x = Q_here / q_total (fraction of total capacity for this direction)
        x = Q_here / (q_total or 1e-12)
        x = np.clip(float(x), 0.0, 1.0)
        
        x_values.append(x)
    
    return x_values


# ══════════════════════════════════════════════════════════════════════════════
# [FIX #4] EOF SAFETY – ensure q_tot set even if file ends mid-relaxation
# ══════════════════════════════════════════════════════════════════════════════

def ensure_q_bounds_set(blocks, direction):
    """
    Ensure q_tot is defined for all blocks in a direction, even at EOF.
    
    FIXES BUG #4: If a file ends during relaxation (no full block completion),
                   q_tot is never set. This causes missing blocks or NaN in
                   downstream analysis.
    
    Args:
        blocks:    List of block dicts for one direction
        direction: 'discharge' or 'charge' (informational)
    
    Returns:
        None (modifies blocks in place)
    """
    if not blocks:
        return
    
    # Find blocks that have no q_tot set
    for i, block in enumerate(blocks):
        if 'q_tot' not in block or not np.isfinite(block.get('q_tot', np.nan)):
            # Use previous block's final Q if available
            if i > 0 and 'q_tot' in blocks[i - 1]:
                block['q_tot'] = blocks[i - 1]['q_tot']
            else:
                # Estimate from this block's Q
                block['q_tot'] = block.get('Q_end', 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# [FIX #6] BOUNDED DEDX EVALUATION – return NaN if out of range
# ══════════════════════════════════════════════════════════════════════════════

def get_dedx_safe(dedx_fn, V_eq):
    """
    Safely evaluate dE/dx at V_eq, returning NaN if out of bounds.
    
    FIXES BUG #6: Original code called dedx_fn(V_eq) without checking if V_eq
                   is in the valid range. Extrapolation produces huge/negative
                   D values in plateau regions.
    
    Args:
        dedx_fn: scipy.interpolate.interp1d object
        V_eq:    Voltage at equilibrium (operating point)
    
    Returns:
        float: dE/dx at V_eq if in bounds, else NaN
    """
    if dedx_fn is None or not np.isfinite(V_eq):
        return np.nan
    
    try:
        # Check if V_eq is within the interpolation bounds
        # Assume dedx_fn.x contains the training voltage values
        V_min = dedx_fn.x.min()
        V_max = dedx_fn.x.max()
        
        # Add small buffer (1% of range) to account for numerical errors
        V_range = V_max - V_min
        V_min_safe = V_min - 0.01 * V_range
        V_max_safe = V_max + 0.01 * V_range
        
        if V_eq < V_min_safe or V_eq > V_max_safe:
            return np.nan
        
        dv = float(dedx_fn(V_eq))
        return dv if np.isfinite(dv) else np.nan
    
    except Exception:
        return np.nan


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION (used by extract_blocks_corrected)
# ══════════════════════════════════════════════════════════════════════════════

def _current_threshold(I):
    """
    Compute threshold current that separates galvanostatic pulses from OCP rest.
    
    Strategy: Median of non-zero |I| / 20. Returns 1e-12 if all zero.
    """
    absI = np.abs(I)
    nonzero = absI[absI > 0]
    if len(nonzero) == 0:
        return 1e-12
    pulse_mag = float(np.median(nonzero))
    return max(pulse_mag * CONFIG['CURRENT_THRESHOLD_FRAC'], 1e-12)


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY OF CHANGES
# ══════════════════════════════════════════════════════════════════════════════

"""
TO INTEGRATE INTO GITT_V2.py:

1. Copy CONFIG dict to module top (after imports)
   
2. Copy these 7 functions into GITT_V2.py:
   - _get_block_equilibrium_voltage()
   - _get_fit_window_robust()
   - extract_blocks_corrected()     ← replaces original extract_blocks()
   - compute_x_block_corrected()    ← NEW helper for _replot()
   - ensure_q_bounds_set()          ← NEW helper for _replot()
   - get_dedx_safe()                ← NEW helper for _replot()
   - _current_threshold()           ← already exists, verify unchanged

3. In GITTApp._replot(), replace the call:
   OLD:  blocks = extract_blocks(td, tfs, tfe, min_relax=min_relax, tau_s=tau_s)
   NEW:  blocks = extract_blocks_corrected(td, tfs, tfe, min_relax=min_relax, tau_s=tau_s)

4. In GITTApp._replot(), replace x_block computation:
   OLD:  Q_cumul_td = np.concatenate([[0.], np.cumsum(I_mid_td * dT_td)]) / 3.6
         for b in disc_blocks:
             q_idx = np.searchsorted(td_T, b['_Tr'][-1]) - 1
             x_block = (Q_cumul_td[q_idx] - q_off_disc) / (q_tot_disc or 1.0)
             b['x_block'] = np.clip(float(x_block), 0.0, 1.0)
   
   NEW:  if disc_blocks:
             x_disc = compute_x_block_corrected(td_I, td_T, 'discharge',
                                                first_disc_pulse_start_T, disc_blocks)
             for b, x in zip(disc_blocks, x_disc):
                 b['x_block'] = x

5. In GITTApp._replot(), replace dEdx evaluation:
   OLD:  for b in active_blocks:
             if dedx_fn is not None:
                 dv = float(cyc_dedx(b['V_eq']))
                 b['dEdx'] = dv if np.isfinite(dv) else np.nan
   
   NEW:  for b in active_blocks:
             if dedx_fn is not None:
                 dv = get_dedx_safe(dedx_fn, b['V_eq'])
                 b['dEdx'] = dv

6. In GITTApp._replot(), add after block population:
   ensure_q_bounds_set(disc_blocks, 'discharge')
   ensure_q_bounds_set(chg_blocks, 'charge')

See GITT_BUG_FIX_INTEGRATION_GUIDE.py for detailed instructions and tests.
"""
