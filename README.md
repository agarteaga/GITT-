# GITT Analysis App — v2.3

Multi-sample desktop GUI for Galvanostatic Intermittent Titration Technique analysis.
Implements Weppner & Huggins (1977) (D_conv) and Kang & Chueh (2021) (D_kc),

---

## Requirements

| Package | Version |
|---------|---------|
| Python | ≥ 3.9 |
| tkinter | bundled (see below) |
| numpy | ≥ 1.23 |
| scipy | ≥ 1.9 |
| matplotlib | ≥ 3.6 |

```bash
pip install numpy scipy matplotlib
# tkinter:
# Ubuntu/Debian : sudo apt-get install python3-tk
# macOS         : brew install python-tk
# Windows       : included with python.org installer
```

---

## Input files

### Recommended — EC-Lab single-file ASCII (`.txt`)
One `.txt` per sample from EC-Lab. Contains time, voltage, current, capacity,
and header metadata (mass, Mw, area…). Multiple files in a folder are
concatenated alphabetically.

```
data/
├── SampleA/
│   └── SampleA_GITT.txt
├── SampleB/
│   └── SampleB_part1.txt
│   └── SampleB_part2.txt
```

Point the app at `data/` — each subfolder becomes one named sample.

### Also supported — BioLogic two-file format (`.dat`)
One `*time*.dat` (time/voltage/current) + one `*cap*.dat` (capacity/voltage) per folder.
Column order detected automatically from header keywords.

---

## Running

```bash
python GITT_V2.py
python GITT_V2.py --data_dir /path/to/data
```

---

## Interface overview

### Samples panel (top)
One row per sample. Controls per sample:
- **Checkbox** — include/exclude from all plots
- **m for Rtot (mg)** — active mass for R_tot normalisation (Overpotential & Rtot tab)
- **Excl disc#** — comma-separated 1-based discharge pulse numbers to exclude (e.g. `39` or `5,12`)
- **chg#** — same for charge direction

All per-sample fields take effect on **Replot** only.
### Parameter bar

> **All parameters take effect only when you click Replot.**

| Parameter | Default | Description |
|-----------|---------|-------------|
| tau (s) | auto | Pulse duration; auto-detected from current signal |
| Fit start (s) | 60 | Start of √t fit window (W&H); excludes early surface transient |
| Fit end (s) | τ (auto) | End of √t fit window; auto-set to τ at load |
| V_lo / V_hi | auto | Voltage filter on D and slope plots (display only, no effect on computation) |
| Min R² | 0.85 | Minimum R² to include a block in D plots |
| delta_e | 1 | Stoichiometric number δe; scales D by 1/δe² |
| sigma-clip | 0 (off) | MAD outlier rejection on log₁₀(D); 0 = disabled |
| KC xi_min / xi_max | 0 / 99 | Fit window on ξ axis for K&C method (s½) |
| Pulse # | 1 | Which pulse to show in per-pulse tabs (updates immediately) |
| Show pulses from / to | 1 / 999 | Range for all overlay and map tabs |
| Reverse pulses | Discharge | Ordering in per-pulse tabs: None / Discharge / Charge / Both (updates immediately) |
| Trim time (h) from / to | 0, 0 (off) | Exclude all blocks overlapping this time window |
| dE/dx mode | Differential | Smooth OCV derivative or finite-diff ΔE/Δx (display only) |

---

## Tabs

### 🔵 Raw Data
| Tab | Content |
|-----|---------|
| GITT curve | E(t) and I(t) per sample; W&H fit window in green.

### 🟣 Analysis
| Tab | Content |
|-----|---------|
| OCV | V_eq (relaxation endpoints) vs SOD/SOC % + raw E vs SOD/SOC|
| Overpotential curves | η(t) = E_pulse − E_eq,prev (mV)|
| Overpotential & Rtot | η (mV, left) and R_tot·m (Ω·g, right) vs SOD/SOC and V_eq|
| Relaxation curves | E(t) − E_eq (mV); all pulses overlaid|
| Relaxation ΔE | Total voltage recovery (E_eq − E_relax,start) vs SOD/SOC|
| Relax kinetics| dV/d(log t) overlay|
| Relax kinetics / pulse | dV/d(log t) for selected Pulse|
| Relax kinetics map | 2D heatmap|

### 🟩 Weppner & Huggins (Conv.)
| Tab | Content |
|-----|---------|
| All pulses (sqt) | ΔE vs √t; all pulses overlaid|
| sqrt(t) fits | Single pulse (Pulse #); fit line and R² |
| Slopes & dE/dx | \|dE/d√t\| vs V_eq (top) + \|dE/dx\| vs V_eq (bottom) |
| D coefficient | D̃_conv vs V_eq and vs SOD/SOC % |

### 🟥 Kang & Chueh (K&C)
| Tab | Content |
|-----|---------|
| All xi pulses | E − E_eq vs ξ; all pulses overlaid|
| xi fits | Single pulse (Pulse #); fit line and R² |
| xi slopes | \|s_KC\| vs V_eq and vs SOD/SOC % |
| D coefficient | D̃_KC vs V_eq and vs SOD/SOC % |

### ⚫ Documentation
Scrollable built-in reference: rendered LaTeX equations, method descriptions,
symbol table, full parameter guide, relax kinetics map theory, validity conditions,
pulse exclusion guide, and D behaviour at the plateau.

---

## Exporting

| Button | Output |
|--------|--------|
| Export figures | PNG (150 dpi), all tabs → `<data_dir>/results/` |
| Export curves | Origin-ready CSV (one X,Y pair per dataset) → same folder |

---

## References

- Weppner, W. & Huggins, R. A. *J. Electrochem. Soc.* **124**, 1569–1578 (1977)
- Kang, S. & Chueh, W. C. *J. Electrochem. Soc.* **168**, 120504 (2021)
