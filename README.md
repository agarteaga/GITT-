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

All per-sample fields take effect on **Replot** only — no auto-fire.

### Parameter bar

> **All parameters take effect only when you click Replot.**
> Adjust multiple parameters freely, then click once.
> Exception: **Pulse #** and **Reverse pulses** update the three per-pulse tabs immediately (fast single-curve draws).

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
| GITT curve | E(t) and I(t) per sample; W&H fit window in green; trimmed zone in red; excluded (red) pulses highlighted |

### 🟣 Analysis
| Tab | Content |
|-----|---------|
| OCV | V_eq (relaxation endpoints) vs SOD/SOC % + raw E vs SOD/SOC |
| Overpotential curves | η(t) = E_pulse − E_eq,prev (mV); all pulses overlaid; turbo colour = V_eq |
| Overpotential & Rtot | η (mV, left) and R_tot·m (Ω·g, right) vs SOD/SOC and V_eq. Gap-checked: excluded pulses do not create fake η spikes |
| Relaxation curves | E(t) − E_eq (mV); all pulses overlaid; turbo colour = V_eq |
| Relaxation ΔE | Total voltage recovery (E_eq − E_relax,start) vs SOD/SOC |
| **Relax kinetics** | dV/d(log t) overlay; all pulses per sample; discharge top / charge bottom; turbo colour = V_eq |
| **Relax kinetics / pulse** | dV/d(log t) for selected Pulse #; discharge top / charge bottom; τ annotated; orange bg = 2+ processes |
| **Relax kinetics map** | Operando-XRD-style 2D heatmap — see below |

All overlay tabs: 2 rows (discharge / charge) × N-sample columns.

#### Relax kinetics map — detail

x = log₁₀(τ / s), y = V_eq (V), colour = normalised process fraction.

Each row (one relaxation block) is divided by its own total area under the
dV/d(log t) curve before plotting:

$$Z_\text{plot}(\tau, V_\text{eq}) = \frac{|dV/d\log t|(\tau, V_\text{eq})}{\int |dV/d\log t|\, d\log\tau}$$

This row-area normalisation puts all blocks on the same relative scale regardless
of absolute overpotential amplitude — a block at 0.1 V and one at 1.0 V are
directly comparable in the map.

**Colormap:** blue → cyan → green → yellow → red (operando XRD style).
Colour scale clipped at the 97th percentile to prevent outlier saturation.

**Reading the map:**
- Vertical bright band at constant log τ = process with fixed time constant (e.g. desolvation)
- Diagonal/curved band = τ shifts with electrode state (e.g. diffusion-limited process that slows on intercalation)
- Comparing discharge vs charge panels at the same V_eq reveals kinetic hysteresis

### 🟩 Weppner & Huggins (Conv.)
| Tab | Content |
|-----|---------|
| All pulses (sqt) | ΔE vs √t; all pulses overlaid; turbo colour = V_eq |
| sqrt(t) fits | Single pulse (Pulse #); fit line and R² |
| Slopes & dE/dx | \|dE/d√t\| vs V_eq (top) + \|dE/dx\| vs V_eq (bottom) |
| D coefficient | D̃_conv vs V_eq and vs SOD/SOC % |

### 🟥 Kang & Chueh (K&C)
| Tab | Content |
|-----|---------|
| All xi pulses | E − E_eq vs ξ; all pulses overlaid; turbo colour = V_eq |
| xi fits | Single pulse (Pulse #); fit line and R² |
| xi slopes | \|s_KC\| vs V_eq and vs SOD/SOC % |
| D coefficient | D̃_KC vs V_eq and vs SOD/SOC % |

### ⚫ Documentation
Scrollable built-in reference: rendered LaTeX equations, method descriptions,
symbol table, full parameter guide, relax kinetics map theory, validity conditions,
pulse exclusion guide, and D behaviour at the plateau.

---

## Physics

### D_conv — Weppner & Huggins (1977)
Fit ΔE vs √t during the pulse in [Fit start, Fit end]:

$$\tilde{D}_\mathrm{conv} \cdot (S/v_m)^2 = \frac{4I^2}{\pi F^2 \delta_e^2} \left(\frac{dV_\mathrm{eq}/dx}{s_{\sqrt{t}}}\right)^2$$

### D_kc — Kang & Chueh (2021)
Fit E − V_eq vs ξ = √(t_relax + τ) − √t_relax during relaxation:

$$\tilde{D}_\mathrm{kc} \cdot (S/v_m)^2 = \frac{4I^2}{\pi F^2 \delta_e^2} \left(\frac{dV_\mathrm{eq}/dx}{s_\xi}\right)^2$$

Both report D̃·(S/v_m)² in mol² s⁻¹. Multiply by (v_m/S)² for absolute D̃ in cm² s⁻¹.
**D_kc is preferred** — zero-current relaxation has no IR contamination.

### dV/d(log t) relaxation fingerprinting
$$\frac{dV}{d\log_{10} t} = t \cdot \frac{dV}{dt}$$

Each electrochemical process appears as a distinct **peak**. Peak position = τ.
Cross-validate with EIS: peak at log₁₀(τ) ↔ EIS arc at f = 1/(2πτ).

---

## Pulse exclusion

Enter 1-based pulse numbers in the **Excl disc#** / **chg#** fields per sample.
The excluded block (pulse + relaxation) is:
- Removed from all analysis (D, slopes, fits, dV/d(log t), map)
- Drawn in **red** on the GITT Curve tab for visual confirmation
- Skipped in overpotential/R_tot calculation to prevent fake spikes

Click **Replot** after editing exclusion fields. Multiple values: `5,12,39`.

**Finding the pulse number:** step through the Pulse # spinbox on "Relax kinetics / pulse"
or "sqrt(t) fits" until you find the block with anomalous V_eq or stepped relaxation.

**Coarser alternative:** use the Trim time (h) from/to fields to exclude all blocks
overlapping a time window. Both mechanisms can be combined.

---

## D values at the OCV plateau — expected behaviour

In a two-phase region, dV_eq/dx → 0 and D is undefined. Both D_conv and D_kc scatter
wildly — this is physically correct. Use **V_lo / V_hi** to exclude the plateau voltage
range and analyse the single-phase regions on either side separately.

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
- Ge, K. et al. *ACS Energy Lett.* **8**, 2738–2745 (2023)
