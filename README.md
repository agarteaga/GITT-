# GITT_V2 — Galvanostatic Intermittent Titration Technique Analysis GUI

A multi-sample desktop application for analysing GITT electrochemical data. Implements both the Weppner & Huggins (1977) conventional method and the Kang & Chueh (2021) relaxation-based method for extracting solid-state diffusion coefficients, with a complete suite of diagnostic visualisation tabs.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| tkinter | bundled with Python (see Installation) |
| numpy | ≥ 1.23 |
| scipy | ≥ 1.9 |
| matplotlib | ≥ 3.6 |

---

## Installation

```bash
pip install -e .
```

### Linux (Ubuntu / Debian)
```bash
sudo apt-get install python3-tk
pip install -e .
```

### macOS
```bash
brew install python-tk   # only if using Homebrew Python
pip install -e .
```

### Windows
tkinter ships with the standard python.org installer:
```bash
pip install -e .
```

---

## Running

```bash
# Open with no data pre-loaded:
python GITT_V2.py

# Pre-load a data directory on startup:
python GITT_V2.py --data_dir /path/to/your/data

# If installed via setup.py:
gitt --data_dir /path/to/your/data
```

---

## Data Format

The app reads BioLogic-style `.dat` files. Both file types must be placed in the same folder — the app detects them automatically by the keywords `time` and `cap` in the filenames.

### Time file (`*time*.dat`)
Tab-separated. Header line must contain `Potential` (or `Ewe`/`Voltage`), `Current` (or `Iwe`), and `Time` (or `Elapsed`) — all on the same line. Comma or period decimal separators are both accepted.

```
Potential (V)    Current (A)    Elapsed Time (s)
1.959867         -7.46483E-05   231930.99
...
```

### Capacity file (`*cap*.dat`)
Tab-separated. Header must contain a capacity keyword (`Capacity`, `mAh`, `mA.h`, `Charge`) and a voltage keyword (`Ecell`, `Ewe`, `Potential`, `Voltage`). Units `mAh/g` are detected automatically.

```
Capacity/mA.h    Ecell/V
4.147E-08        1.95987
...
```

### Multi-file experiments
Files split across multiple recordings (e.g. `G170_time_1.dat`, `G170_time_2.dat`) are sorted and concatenated automatically.

### Folder structure
```
data/
├── SampleA/
│   ├── SampleA_time_1.dat
│   ├── SampleA_cap_1.dat
│   └── ...
├── SampleB/
│   └── ...
└── *.dat   ← files directly in root shown as "(this folder)"
```

---

## Interface Overview

### Global parameter bar (top)

| Parameter | Default | Description |
|---|---|---|
| **tau (s)** | auto | GITT pulse duration in seconds; auto-detected from current signal |
| **Fit start (s)** | 60 | Start of the sqrt(t) fit window (W&H); excludes early surface transients |
| **Fit end (s)** | auto (= tau) | End of the fit window; excludes finite-size effects |
| **E_lo / E_hi** | auto | Voltage filter on D and dE/dx plots (display only, no effect on fitting) |
| **Min R²** | 0.85 | Minimum R² to include a block in D calculations |
| **delta_e** | 1 | Stoichiometric number δe (formula units per mobile ion); scales D by 1/δe² |
| **sigma-clip** | 0 (off) | Robust MAD outlier rejection on log₁₀(D) in scatter plots; 0 = disabled |
| **KC xi_min / xi_max** | 0 / 99 | Fit-window limits for the K&C ξ axis; restrict to linear bulk region |
| **Pulse #** | 1 | Which pulse to show in single-pulse fit tabs (sqrt fits, xi fits) |
| **Show pulses from / to** | 1 / 999 | Pulse range for gradient overlay tabs (All pulses, All xi, Relaxation curves, Overpotential curves) |
| **dE/dx mode** | Differential | Bottom row of Slopes & dE/dx tab: smooth OCV derivative, or block-to-block finite difference dE/dx |

Parameters auto-detect from the first loaded sample (τ, E range) and update plots immediately when changed.

### Per-sample controls (sidebar)

Each sample checkbox row has a **m for Rtot (mg)** spinner. This sets the active electrode mass used to normalise R_tot in the Overpotential & Rtot tab (right y-axis shows R_tot × mass in Ω·g). It does not affect D_conv, D_kc, or SOD/SOC calculations.

---

## Tabs

Sections appear in this order: **Raw Data → Analysis → Weppner & Huggins → Kang & Chueh → Documentation**

### 🔵 Raw Data
| Tab | Content |
|---|---|
| **GITT curve** | Full E(t) and I(t) traces per sample; fit window highlighted in green |

### 🟣 Analysis
| Tab | Content |
|---|---|
| **OCV** | Top: E_eq (relaxation endpoints) vs SOD/SOC %; bottom: full E vs SOD/SOC from cap file |
| **Overpotential curves** | One row per sample: η(t) = E_pulse(t) − E_eq,prev (mV) overlaid; gradient coloured by E_eq (per-sample colour scheme); pulse range selectable |
| **Overpotential & Rtot** | 4 dual-y panels: left y = η (mV), right y = R_tot·m (Ω·g). Top row vs SOD/SOC %, bottom row vs E_eq |
| **Relaxation curves** | One row per sample: E(t) − E_eq (mV) overlaid, converging to 0 at equilibrium; gradient coloured by E_eq; pulse range selectable |
| **Relaxation ΔE** | Total voltage recovery per pulse (E_eq − E_relax,start) vs SOD/SOC % |

### 🟩 Weppner & Huggins (Conventional)
| Tab | Content |
|---|---|
| **All pulses (sqt)** | One row per sample: ΔE vs √t overlaid; gradient coloured by E_eq (per-sample colour scheme); fit lines shown; pulse range selectable |
| **sqrt(t) fits** | One column per sample: ΔE vs √t for Pulse # with fit line and R² |
| **Slopes & dE/dx** | Top: \|dE/d√t\| vs E_eq; bottom: \|dE/dx\| vs SOD/SOC % — dE/dx mode selectable (smooth differential or block finite difference) |
| **D coefficient** | D̃_conv vs E_eq (top) and vs SOD/SOC % (bottom); R² and sigma-clip filters applied |

### 🟥 Kang & Chueh (K&C)
| Tab | Content |
|---|---|
| **All xi pulses** | One row per sample: E−E_eq (mV) vs ξ overlaid; gradient coloured by E_eq (per-sample colour scheme); pulse range selectable |
| **xi fits** | One column per sample: E−E_eq vs ξ for Pulse # with fit line and R² |
| **xi slopes** | \|s_KC\| vs E_eq (top) and vs SOD/SOC % (bottom) |
| **D coefficient** | D̃_KC vs E_eq (top) and vs SOD/SOC % (bottom); R² and sigma-clip filters applied |

### ⚫ Documentation
Built-in equations reference with rendered LaTeX, method descriptions, parameter guidance, symbol table, and validity conditions.

---

## Physics

### SOD / SOC normalisation

State of Discharge/Charge is computed from the time file by trapezoidal integration of |I|·dt, normalised by the total pulse charge **including the last pulse** (even if it has no trailing relaxation and is excluded from D calculations):

```
x_block[n] = Σ_{k=1..n} |I|·dt_k  /  Σ_{k=1..N_all} |I|·dt_k
```

Discharge and charge are normalised independently from their own first pulse start. If the recording ends mid-pulse, the file end-time is used as the last pulse end, ensuring charge reaches 100% SOC.

### Conventional GITT — Weppner & Huggins (1977)

Fit ΔE vs √t during the galvanostatic pulse in window [Fit start, Fit end]:

```
D̃_conv · (S/Vm)² = (4 I² / π F² δe²) · (dE_eq/dx / s_√t)²
```

where s_√t is the slope of ΔE vs √t. The slope is recomputed at every replot within the current fit window.

### K&C GITT — Kang & Chueh (2021)

Fit E − E_eq vs ξ = √(t_relax + τ) − √t_relax during open-circuit relaxation, within [KC xi_min, KC xi_max]:

```
D̃_KC · (S/Vm)² = (4 I² / π F² δe²) · (dE_eq/dx / s_ξ)²
```

The slope s_ξ is recomputed within the xi window at every replot, so D_kc is always consistent with what the xi fits tab displays.

Both methods report the lumped product D̃·(S/Vm)² in mol² s⁻¹. To extract D, the electroactive area S and molar volume Vm must be known independently.

The K&C method is preferred because the relaxation signal is free of IR contamination (R_tot = 0 at I = 0), whereas the pulse signal mixes diffusion with ohmic and charge-transfer resistances.

### dE/dx — two modes

The thermodynamic factor |dE_eq/dx| is shown in the bottom row of the Slopes & dE/dx tab:

- **Differential (smooth dV/dx)** — default. All V_eq block values are smoothed (Savitzky-Golay) and differentiated to give a continuous curve. This is the same value used internally in D_conv and D_kc.
- **Finite difference ΔE/Δx** — raw block-to-block estimate: (V_eq[n] − V_eq[n−1]) / (x[n] − x[n−1]). Model-free and noisier; useful to cross-check the smooth result.

Changing the mode only affects the Slopes & dE/dx tab display; D values are not affected.

---

## Exporting

| Button | Output |
|---|---|
| **Export figures** | PNG images (150 dpi) of all visible plots → `<data_dir>/results/` |
| **Export curves** | Origin-ready CSV files, one X,Y pair per dataset → same folder |

---

## Tips

- **Pulse range** (`Show pulses from / to`) limits which pulses appear in the gradient overlay tabs (All pulses, All xi pulses, Overpotential curves, Relaxation curves). Useful when all pulses together are too dense to read. Each sample keeps its own per-sample colour gradient.
- **Colour coding** in gradient tabs: each sample gets a gradient built from its own base colour (blue → sample 1, orange → sample 2, green → sample 3, etc.), matching the solid colours used in D plots and OCV.
- **Pulse #** controls the single-pulse fit tabs (sqrt fits, xi fits); changing it redraws only those tabs.
- **KC xi_min / xi_max**: set xi_max < 15 to exclude surface transients and fit only the linear bulk relaxation. D_kc updates immediately when you change these.
- **E_lo / E_hi**: exclude noisy or two-phase voltage regions from D plots without affecting the fit computation.
- **Sigma-clip** (e.g. 2.5): robust outlier rejection on log₁₀(D) in scatter plots; set to 0 to see all data.
- Multiple samples overlay automatically — uncheck any sample to hide it.

---

## References

1. Weppner, W. & Huggins, R. A. *Determination of the kinetic parameters of mixed-conducting electrodes and application to the system Li₃Sb.* J. Electrochem. Soc. **124**, 1569–1578 (1977).
2. Kang, S. & Chueh, W. C. *Galvanostatic Intermittent Titration Technique Reinvented: Part I. A Critical Review.* J. Electrochem. Soc. **168**, 120504 (2021).
