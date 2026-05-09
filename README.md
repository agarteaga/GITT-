# GITT_V2 — Galvanostatic Intermittent Titration Technique Analysis GUI

A multi-sample desktop application for analysing GITT electrochemical data. Implements both the Weppner & Huggins (1977) conventional method and the Kang & Chueh (2021) relaxation-based method for extracting solid-state diffusion coefficients.

---

## Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.9 |
| tkinter | bundled with Python (see Installation) |
| numpy | ≥ 1.23 |
| scipy | ≥ 1.9 |
| matplotlib | ≥ 3.6 |


## Input Files

### ✅ Recommended: EC-Lab single-file ASCII (.txt)

Export one `.txt` file per sample from EC-Lab. Everything is in one place — time, voltage, current, capacity, and all sample metadata in the header. Just put the file in its own folder; the app reads and fills in everything automatically.


**Folder structure:**
```
data/
├── SampleA/
│   └── SampleA_GITT.txt        ← one file, everything inside
├── SampleB/
│   └── SampleB_part1.txt       ← multiple files sorted and concatenated
│   └── SampleB_part2.txt
```

Point the app at `data/` — each subfolder becomes a named sample.

---

### Also supported: BioLogic two-file format (.dat)

Classic BioLogic export with a `*time*.dat` file (time/voltage/current) and a `*cap*.dat` file (capacity/voltage) in the same folder. Columns detected by keyword; any order accepted.

---

## Interface Overview

### Global parameter bar (top)

| Parameter | Default | Description |
|---|---|---|
| **tau (s)** | auto | GITT pulse duration; auto-detected from current signal |
| **Fit start (s)** | 60 | Start of √t fit window (W&H); excludes early surface transient |
| **Fit end (s)** | auto | End of fit window; auto-set to tau at load |
| **E_lo / E_hi** | auto | Voltage filter on D and dE/dx display plots (does not affect fits) |
| **Min R²** | 0.85 | Minimum R² to include a block in D calculations |
| **delta_e** | 1 | Stoichiometric number δe; scales D by 1/δe² |
| **sigma-clip** | 0 | Robust MAD outlier rejection on log₁₀(D) scatter; 0 = off |
| **KC xi_min / xi_max** | 0 / 99 | Fit window for K&C ξ axis (s½) |
| **Pulse #** | 1 | Which pulse to show in single-pulse fit tabs |
| **Show pulses from / to** | 1 / 999 | Pulse range for gradient overlay tabs |
| **dE/dx mode** | Differential | Bottom row of Slopes & dE/dx: smooth OCV derivative or block finite difference (both plotted vs E_eq) |

### Per-sample controls (sample sidebar)

Each sample row has an **m for Rtot (mg)** spinner — the active electrode mass for normalising R_tot in the Overpotential & Rtot tab. For EC-Lab single files this is auto-populated from the file header.

---

## Tabs

**Section order: Raw Data → Analysis → Weppner & Huggins → Kang & Chueh → Documentation**

### 🔵 Raw Data
| Tab | Content |
|---|---|
| **GITT curve** | Full E(t) and I(t) traces; W&H fit window highlighted in green |

### 🟣 Analysis
| Tab | Content |
|---|---|
| **OCV** | E_eq (relaxation endpoints) vs SOD/SOC %; full E vs SOD/SOC from cap |
| **Overpotential curves** | η(t) = E_pulse − E_eq,prev (mV) overlaid; gradient by E_eq; pulse range selectable |
| **Overpotential & Rtot** | 4 dual-y panels: η (mV, left) and R_tot·m (Ω·g, right); top row vs SOD/SOC %, bottom row vs E_eq |
| **Relaxation curves** | E(t) − E_eq (mV) overlaid, converging to 0; gradient by E_eq; pulse range selectable |
| **Relaxation ΔE** | Total voltage recovery (E_eq − E_relax,start) vs SOD/SOC % |

### 🟩 Weppner & Huggins (Conventional)
| Tab | Content |
|---|---|
| **All pulses (sqt)** | One panel per sample: ΔE vs √t overlaid; gradient by E_eq; pulse range selectable |
| **sqrt(t) fits** | Single-pulse ΔE vs √t for Pulse #; fit line and R² |
| **Slopes & dE/dx** | Top row: \|dE/d√t\| vs E_eq. Bottom row: \|dE/dx\| vs E_eq — both rows share the same x-axis. dE/dx mode selectable |
| **D coefficient** | D̃_conv vs E_eq (top) and vs SOD/SOC % (bottom) |

### 🟥 Kang & Chueh (K&C)
| Tab | Content |
|---|---|
| **All xi pulses** | One panel per sample: E−E_eq (mV) vs ξ overlaid; gradient by E_eq; pulse range selectable |
| **xi fits** | Single-pulse E−E_eq vs ξ for Pulse #; fit line and R² |
| **xi slopes** | \|s_KC\| vs E_eq (top) and vs SOD/SOC % (bottom) |
| **D coefficient** | D̃_KC vs E_eq (top) and vs SOD/SOC % (bottom) |

### ⚫ Documentation
Built-in reference: equations with rendered LaTeX, method descriptions, parameter guide, file format details, symbol table, validity conditions.

---

## Physics

### SOD / SOC normalisation

```
x_block[n] = Σ|I|dt (pulses 1..n) / Σ|I|dt (all pulses including last)
```

Discharge and charge normalised independently. If the file ends mid-pulse (no trailing relaxation), the end-of-file time is used as the last pulse end — ensuring SOC reaches 100%.

### Conventional GITT — Weppner & Huggins (1977)

Fit ΔE vs √t during the pulse in [Fit start, Fit end]:

```
D̃_conv · (S/Vm)² = (4 I² / π F² δe²) · (dE_eq/dx / s_√t)²
```

### K&C GITT — Kang & Chueh (2021)

Fit E − E_eq vs ξ = √(t_relax + τ) − √t_relax in [KC xi_min, KC xi_max]:

```
D̃_KC · (S/Vm)² = (4 I² / π F² δe²) · (dE_eq/dx / s_ξ)²
```

Both methods report D̃·(S/Vm)² in mol² s⁻¹. The K&C method is preferred: relaxation is free of IR contamination. s_ξ is recomputed within the xi window at every replot so D_kc is always consistent with the xi fits tab.

### dE/dx modes (Slopes & dE/dx tab, bottom row)

Both modes plot |dE/dx| vs **E_eq** — the same x-axis as the top row — so the slope and thermodynamic factor can be compared at the same voltage.

- **Differential (smooth)** — default: Savitzky-Golay smooth + differentiate V_eq(x). This is the thermodynamic factor used internally in D_conv and D_kc.
- **Finite diff ΔE/Δx** — raw (V_eq[n] − V_eq[n−1]) / (x[n] − x[n−1]). Model-free, noisier; useful to cross-check.

Neither mode affects D_conv or D_kc values.

---

## Exporting

| Button | Output |
|---|---|
| **Export figures** | PNG (150 dpi) of all visible plots → `<data_dir>/results/` |
| **Export curves** | Origin-ready CSV files, one X,Y pair per dataset → same folder |

---

## References

1. Weppner, W. & Huggins, R. A. *J. Electrochem. Soc.* **124**, 1569–1578 (1977).
2. Kang, S. & Chueh, W. C. *J. Electrochem. Soc.* **168**, 120504 (2021).
