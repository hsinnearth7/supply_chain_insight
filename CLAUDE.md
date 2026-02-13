# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChainInsight — End-to-end supply chain inventory analytics platform. Generates a dirty 10k-row CSV, applies an 8-step ETL cleaning pipeline, then runs statistical analysis, advanced optimization (EOQ, Monte Carlo), and 30 ML algorithms (20 applied, 10 documented). Outputs a cleaned CSV with 3 derived fields and 23 publication-ready charts. Bilingual project: every script has both Chinese and English versions.

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline
python generate_data.py
python "clean_data .py"
python "chart_0_Inventory Health & Stockout Risk.py.txt"
python chart_01_to_08_statistical_analysis.py.txt
python chart_09_to_14_advanced_supply_chain.py.txt
python chart_15_to_22_ai_algorithms_analysis.py.txt
```

No build step. No test framework. Dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `squarify`. Optional: `xgboost`. `re` is stdlib.

## File Structure

| File | Description |
|------|-------------|
| `generate_data.py` | Synthetic dirty data generator (10,050 rows, 7 categories, 7 vendors) |
| `clean_data .py` / `clean_data_en.py.txt` | ETL pipeline (CN / EN) |
| `chart_0_Inventory Health & Stockout Risk.py.txt` / `.en.py.txt` | Inventory health dashboard: KPI cards, ABC Pareto, stockout alert (CN / EN) |
| `chart_01_to_08_statistical_analysis.py.txt` / `_en.py.txt` | Statistical analysis: correlation, distribution, ANOVA, Chi-square, regression, risk profiling, outlier detection (CN / EN) |
| `chart_09_to_14_advanced_supply_chain.py.txt` / `_en.py.txt` | Advanced optimization: EOQ, vendor radar, treemap, Monte Carlo, reorder gap, demand variability (CN / EN) |
| `chart_15_to_22_ai_algorithms_analysis.py.txt` / `_en.py.txt` | 30 AI algorithms: classification, regression, clustering, PCA/t-SNE, anomaly detection, genetic algorithm (CN / EN) |

## Architecture

### ETL Pipeline (`clean_supply_chain_data` in `clean_data .py`)

**Extract:** Reads `Supply_Chain_Inventory_Dirty_10k.csv` via pandas.

**Transform (8 steps):**
1. `Product_ID` — strip whitespace
2. `Category` — strip + capitalize (deduplicates variant spellings)
3. `Unit_Cost_Raw` → `Unit_Cost` — regex extraction from mixed formats (USD, $, commas, "Quote Pending" → NaN)
4. `Current_Stock_Raw` → `Current_Stock` — coerce to numeric, clamp negatives to 0
5. Null handling — stock nulls → 0; cost nulls → category median, then global median fallback
6. `Vendor_Name` — strip whitespace
7. Validation — clip numeric fields to valid ranges (demand/safety ≥ 0, lead time ≥ 1)
8. Derived fields:
   - `Reorder_Point` = Daily_Demand_Est × Lead_Time_Days + Safety_Stock_Target
   - `Stock_Status` = Out of Stock / Low Stock / Normal Stock (based on stock vs reorder point)
   - `Inventory_Value` = Current_Stock × Unit_Cost

**Load:** Writes 11-column `Supply_Chain_Inventory_Clean.csv` and prints a data quality summary.

### Analysis Modules

**Inventory Health (chart_0):** DSI, ABC classification (Pareto 80/95), stockout alert table, KPI dashboard.

**Statistical Analysis (chart_01~08):** Pearson/Spearman correlation with p-values, distribution + QQ plots + Shapiro-Wilk, vendor box plots with ANOVA/Kruskal-Wallis, category×vendor heatmaps with Chi-square, OLS regression with 95% CI, category risk profiling, Z-score/IQR outlier detection, pair plots.

**Advanced Optimization (chart_09~14):** EOQ = sqrt(2DS/H) with S=$50, H=25% of unit cost; vendor radar (multi-dimensional normalized scoring); inventory treemap + nested donut (uses squarify); Monte Carlo stockout simulation (5,000 runs); reorder gap waterfall + replenishment priority; demand CV, safety stock funnel, service level curve, lead time risk heatmap.

**AI/ML (chart_15~22):** 10 classifiers (#2-10, #21) predicting Stock_Status; feature importance (RF, GB, DT, LR); 3 regressors (#1, #4, #8) predicting Inventory_Value; clustering (#11 K-Means, #12 Hierarchical, #13 DBSCAN, #26 K-Means++); dimensionality reduction (#14 PCA, #15 t-SNE); anomaly detection (#27 Autoencoder/MLP, #28 Isolation Forest); optimization (#30 Genetic Algorithm); 30-algorithm suitability matrix.

## Data Schema

**Input columns (8):** Product_ID, Category, Unit_Cost_Raw, Current_Stock_Raw, Daily_Demand_Est, Safety_Stock_Target, Vendor_Name, Lead_Time_Days

**Output columns (11):** Product_ID, Category, Unit_Cost, Current_Stock, Daily_Demand_Est, Safety_Stock_Target, Vendor_Name, Lead_Time_Days, Reorder_Point, Stock_Status, Inventory_Value

## Data Details

**Categories (7):** Electronics, Home, Food, Shipping, Office, Apparel, Industrial
**Vendors (7):** Tokyo Electronics, Fukuoka Logistics, Hokkaido Foods, Kyoto Crafts, Osaka Supplies, Nagoya Parts, Sapporo Steel
**Dirty data issues:** Mixed cost formats, negative stock, "pcs" suffix, whitespace, case inconsistency, ~5% nulls, 50 duplicates

## Output Artifacts

- `Supply_Chain_Inventory_Clean.csv` — 10,050 rows × 11 columns
- 23 PNG charts: `chart_0_*.png`, `chart_01_*.png` through `chart_22_*.png`
