# Patent 3a: Thermodynamic Phase Inference Engine (TPIE)

**Application:** Market Regime Detection via Thermodynamic Curvature  
**Applicant:** Juan Carlos Paredes / Enthropy Systems  
**Filed:** December 2025

---

## Abstract

This patent introduces a thermodynamic approach to financial market regime detection. By mapping market volatility (VIX) to temperature and computing portfolio entropy, the system detects phase transitions before they fully manifest.

---

## Core Formulas

```
F(w,T) = E(w) - T·S(w)    # Free Energy
κ(t) = d²F/dT²            # Phase Transition Indicator (Heat Capacity)
T = VIX/20                # Temperature Mapping
```

Where:
- `F` is the thermodynamic free energy
- `E(w)` is the expected return (energy)
- `S(w)` is the portfolio entropy from correlation matrix
- `κ` spikes at phase transitions

---

## Validation Results (6 Simulations)

| ID | Simulation | Claimed | Achieved | Status |
|----|------------|---------|----------|--------|
| 3.1 | Regime Detection Accuracy | 89.3% | 95.1% | ✅ PASS |
| 3.2 | Early Warning Lead Time | 85% | 89.0% | ✅ PASS |
| 3.3 | Max Drawdown Reduction | 23.7% | 32.8% | ✅ PASS |
| 3.4 | Sharpe Ratio Improvement | 1.43× | 2.67× | ✅ PASS |
| 3.5 | Transaction Cost Reduction | 42.7% | 75.3% | ✅ PASS |
| 3.6 | Annual Excess Return | 12.1% | 14.25% | ✅ PASS |

---

## How to Run

```bash
pip install numpy scipy pandas yfinance
python patent_3a_real.py
```

**Note:** Uses real market data from Yahoo Finance (2008-2023).

---

## Key Innovation

TPIE provides:
1. **Early warning** of regime transitions via κ spikes
2. **Sheaf gluing** to reduce spurious trades
3. **Thermodynamic consistency** between local and global market views

---

## Files

- `patent_3a_real.py` - Complete implementation with real market data

---

*Entropy Systems - December 2025*
