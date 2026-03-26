"""Statistical Analysis Pipeline — Charts 01-08 + Inventory Health (Chart 0)."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway, kruskal, pearsonr, spearmanr

from app.config import (
    ABC_THRESHOLD_A,
    ABC_THRESHOLD_B,
    CHART_BG_COLOR,
    CHART_DPI,
    CHART_TEXT_COLOR,
    CHARTS_DIR,
    DSI_SENTINEL,
    RISK_LEVEL_BINS,
    RISK_LEVEL_LABELS,
    SHAPIRO_SAMPLE_LIMIT,
    STATUS_COLORS,
    SUPPLY_RISK_WEIGHTS,
)
from app.pipeline.enrichment import enrich_base

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Runs all statistical analyses and generates charts."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else CHARTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.chart_paths = []

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns needed for statistical analysis."""
        df = enrich_base(df)
        w_lt, w_cov, w_demand = SUPPLY_RISK_WEIGHTS
        lt_max = df["Lead_Time_Days"].max() or 1
        demand_max = df["Daily_Demand_Est"].max() or 1
        df["Supply_Risk_Score"] = (
            df["Lead_Time_Days"] / lt_max * w_lt
            + (1 - df["Stock_Coverage_Ratio"].clip(0, 3) / 3) * w_cov
            + df["Daily_Demand_Est"] / demand_max * w_demand
        )
        return df

    def enrich_inventory_health(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add DSI, ABC classification, and stockout alert fields."""
        df = df.copy()
        if "DSI" not in df.columns:
            df["DSI"] = np.where(
                df["Daily_Demand_Est"] > 0,
                df["Current_Stock"] / df["Daily_Demand_Est"],
                DSI_SENTINEL,
            )
        df["Days_to_Deplete"] = df["DSI"]
        df["Suggested_Reorder"] = (df["Reorder_Point"] - df["Current_Stock"]).clip(lower=0)

        # ABC Classification
        df_sorted = df.sort_values("Inventory_Value", ascending=False).reset_index(drop=True)
        total_value = max(df_sorted["Inventory_Value"].sum(), 1e-9)
        df_sorted["Cumulative_Pct"] = df_sorted["Inventory_Value"].cumsum() / total_value * 100
        df_sorted["ABC_Class"] = np.where(
            df_sorted["Cumulative_Pct"] <= ABC_THRESHOLD_A, "A",
            np.where(df_sorted["Cumulative_Pct"] <= ABC_THRESHOLD_B, "B", "C"),
        )
        df = df.merge(df_sorted[["Product_ID", "ABC_Class"]], on="Product_ID", how="left")
        return df

    def run_all(self, df: pd.DataFrame) -> dict:
        """Run all statistical analyses and return results dict."""
        logger.info("Statistical analysis started")
        df_stats = self.enrich(df)
        df_health = self.enrich_inventory_health(df_stats)

        self._compute_kpis(df_health)
        self.plot_inventory_health(df_health)
        self.plot_correlation(df_stats)
        self.plot_distributions(df_stats)
        self.plot_vendor_analysis(df_stats)
        self.plot_cross_analysis(df_stats)
        self.plot_regression(df_stats)
        self.plot_category_risk(df_stats)
        self.plot_outlier_risk(df_stats)
        self.plot_pairplot(df_stats)

        logger.info("Statistical analysis completed — %d charts", len(self.chart_paths))
        return {
            "kpis": self.results.get("kpis", {}),
            "chart_paths": [str(p) for p in self.chart_paths],
        }

    def _compute_kpis(self, df: pd.DataFrame):
        avg_dsi = df.loc[df["DSI"] < 999, "DSI"].mean()
        total_skus = df["Product_ID"].nunique()
        oos_count = (df["Stock_Status"] == "Out of Stock").sum()
        self.results["kpis"] = {
            "inventory_turnover": round(365 / avg_dsi, 1) if avg_dsi > 0 else 0,
            "avg_dsi": round(avg_dsi, 1) if not np.isnan(avg_dsi) else 0.0,
            "oos_rate": round(oos_count / total_skus * 100, 1) if total_skus > 0 else 0.0,
            "oos_count": int(oos_count),
            "total_skus": int(total_skus),
            "slow_moving_value": float(df.loc[df["DSI"] > 90, "Inventory_Value"].sum()),
            "total_inventory_value": float(df["Inventory_Value"].sum()),
            "abc_a_count": int((df["ABC_Class"] == "A").sum()),
            "abc_b_count": int((df["ABC_Class"] == "B").sum()),
            "abc_c_count": int((df["ABC_Class"] == "C").sum()),
        }

    # ------------------------------------------------------------------
    # Chart 0: Inventory Health Dashboard
    # ------------------------------------------------------------------
    def plot_inventory_health(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_0_inventory_health.png"
        plt.rcParams.update({"font.size": 9, "axes.titlesize": 12, "axes.titleweight": "bold"})
        sns.set_style("whitegrid")

        fig = plt.figure(figsize=(22, 16), facecolor=CHART_BG_COLOR)
        fig.suptitle("Inventory Health & Stockout Risk Analysis",
                     fontsize=20, fontweight="bold", y=0.98, color=CHART_TEXT_COLOR)
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1, 3, 3],
                      hspace=0.35, wspace=0.25, left=0.06, right=0.96, top=0.93, bottom=0.04)

        # KPI Cards
        ax_kpi = fig.add_subplot(gs[0, :])
        ax_kpi.axis("off")
        kpis = self.results["kpis"]
        kpi_cards = [
            ("Inventory Turnover", f'{kpis["inventory_turnover"]}x', "365 / Avg DSI", "#2E86C1"),
            ("Avg DSI (Days)", f'{kpis["avg_dsi"]:.0f}', "Days Sales of Inventory", "#1ABC9C"),
            ("OOS Rate", f'{kpis["oos_rate"]:.1f}%', f'{kpis["oos_count"]} / {kpis["total_skus"]} SKUs', "#E74C3C"),
            ("Slow-Moving Value", f'${kpis["slow_moving_value"]:,.0f}', "DSI > 90 days", "#F39C12"),
            ("Total Inventory Value", f'${kpis["total_inventory_value"]:,.0f}', "All SKUs", "#8E44AD"),
        ]
        card_width, gap = 0.17, 0.02
        start_x = 0.5 - (len(kpi_cards) * card_width + (len(kpi_cards) - 1) * gap) / 2
        for i, (title, value, subtitle, color) in enumerate(kpi_cards):
            cx = start_x + i * (card_width + gap) + card_width / 2
            rect = plt.Rectangle((start_x + i * (card_width + gap), 0.05), card_width, 0.9,
                                 transform=ax_kpi.transAxes, facecolor="white",
                                 edgecolor=color, linewidth=2.5, zorder=2, clip_on=False)
            ax_kpi.add_patch(rect)
            bar = plt.Rectangle((start_x + i * (card_width + gap), 0.82), card_width, 0.13,
                                transform=ax_kpi.transAxes, facecolor=color, zorder=3, clip_on=False)
            ax_kpi.add_patch(bar)
            ax_kpi.text(cx, 0.88, title, transform=ax_kpi.transAxes,
                        ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=4)
            ax_kpi.text(cx, 0.52, value, transform=ax_kpi.transAxes,
                        ha="center", va="center", fontsize=18, fontweight="bold", color=color, zorder=4)
            ax_kpi.text(cx, 0.20, subtitle, transform=ax_kpi.transAxes,
                        ha="center", va="center", fontsize=8, color="#7F8C8D", zorder=4)

        # Scatter — Efficiency Matrix
        ax_scatter = fig.add_subplot(gs[1, 0])
        mask = (df["DSI"] < 999) & (df["Inventory_Value"] > 0)
        plot_df = df[mask].copy()
        color_map = {"A": "#E74C3C", "B": "#F39C12", "C": "#3498DB"}
        for cls in ["C", "B", "A"]:
            sub = plot_df[plot_df["ABC_Class"] == cls]
            ax_scatter.scatter(sub["DSI"], sub["Inventory_Value"], c=color_map[cls],
                               label=f"Class {cls} ({len(sub)})", alpha=0.55, s=20,
                               edgecolors="white", linewidth=0.3)
        dsi_median = plot_df["DSI"].median()
        val_median = plot_df["Inventory_Value"].median()
        ax_scatter.axvline(dsi_median, color="#7F8C8D", ls="--", lw=1, alpha=0.7)
        ax_scatter.axhline(val_median, color="#7F8C8D", ls="--", lw=1, alpha=0.7)
        quad_labels = [
            (dsi_median * 0.35, val_median * 1.6, "Cash Cow\n(High Value, Low DSI)", "#27AE60"),
            (dsi_median * 1.6, val_median * 1.6, "Dead Stock Risk\n(High Value, High DSI)", "#E74C3C"),
            (dsi_median * 0.35, val_median * 0.4, "Stockout Risk\n(Low Value, Low DSI)", "#F39C12"),
            (dsi_median * 1.6, val_median * 0.4, "Low Priority\n(Low Value, High DSI)", "#95A5A6"),
        ]
        for qx, qy, qlabel, qcolor in quad_labels:
            ax_scatter.text(qx, qy, qlabel, fontsize=8, ha="center", va="center",
                            color=qcolor, fontweight="bold", alpha=0.85,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=qcolor, alpha=0.7))
        ax_scatter.set_xlabel("DSI (Days Sales of Inventory)")
        ax_scatter.set_ylabel("Inventory Value ($)")
        ax_scatter.set_title("A. Inventory Efficiency Matrix")
        ax_scatter.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax_scatter.legend(loc="upper right", fontsize=8)

        # Pareto — ABC by Category
        ax_pareto = fig.add_subplot(gs[1, 1])
        cat_value = df.groupby("Category")["Inventory_Value"].sum().sort_values(ascending=False)
        cum_pct = cat_value.cumsum() / cat_value.sum() * 100
        ax_pareto.bar(range(len(cat_value)), cat_value.values,
                      color=["#E74C3C" if p <= 80 else "#F39C12" if p <= 95 else "#3498DB"
                             for p in cum_pct.values],
                      edgecolor="white", linewidth=0.5)
        ax_pareto.set_xticks(range(len(cat_value)))
        ax_pareto.set_xticklabels(cat_value.index, rotation=30, ha="right", fontsize=8)
        ax_pareto.set_ylabel("Inventory Value ($)")
        ax_pareto.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax_line = ax_pareto.twinx()
        ax_line.plot(range(len(cum_pct)), cum_pct.values, color=CHART_TEXT_COLOR, marker="o", ms=5, lw=2, zorder=5)
        ax_line.set_ylabel("Cumulative %")
        ax_line.set_ylim(0, 105)
        ax_line.axhline(80, color="#E74C3C", ls="--", lw=1, alpha=0.6)
        ax_line.axhline(95, color="#F39C12", ls="--", lw=1, alpha=0.6)
        ax_pareto.legend(handles=[
            Patch(facecolor="#E74C3C", label="A Class (<=80%)"),
            Patch(facecolor="#F39C12", label="B Class (80-95%)"),
            Patch(facecolor="#3498DB", label="C Class (>95%)"),
        ], loc="center right", fontsize=8)
        ax_pareto.set_title("B. Pareto Chart — ABC Inventory Analysis")

        # Top 30 SKUs
        ax_sku = fig.add_subplot(gs[2, 0])
        sku_value = df[["Product_ID", "Inventory_Value", "ABC_Class"]].sort_values(
            "Inventory_Value", ascending=False).head(30)
        sku_cum = sku_value["Inventory_Value"].cumsum() / df["Inventory_Value"].sum() * 100
        bar_colors = [color_map.get(c, "#3498DB") for c in sku_value["ABC_Class"]]
        ax_sku.bar(range(len(sku_value)), sku_value["Inventory_Value"].values,
                   color=bar_colors, edgecolor="white", linewidth=0.3)
        ax_sku.set_xticks(range(len(sku_value)))
        ax_sku.set_xticklabels(sku_value["Product_ID"].values, rotation=90, fontsize=6)
        ax_sku.set_ylabel("Inventory Value ($)")
        ax_sku.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax_sku_line = ax_sku.twinx()
        ax_sku_line.plot(range(len(sku_cum)), sku_cum.values, color=CHART_TEXT_COLOR, marker=".", ms=3, lw=1.5)
        ax_sku_line.set_ylabel("Cumulative % of Total")
        ax_sku_line.set_ylim(0, max(sku_cum.values) * 1.1)
        ax_sku.set_title("C. Top 30 SKUs by Inventory Value")

        # Stockout Alert Table
        ax_table = fig.add_subplot(gs[2, 1])
        ax_table.axis("off")
        alert_df = df[
            (df["Days_to_Deplete"] < df["Lead_Time_Days"]) & (df["Stock_Status"] != "Out of Stock")
        ].sort_values("Days_to_Deplete").head(20).copy()
        if len(alert_df) < 20:
            oos = df[df["Stock_Status"] == "Out of Stock"].head(20 - len(alert_df))
            alert_df = pd.concat([alert_df, oos])
        table_data = alert_df[[
            "Product_ID", "Category", "Current_Stock", "Safety_Stock_Target",
            "Days_to_Deplete", "Lead_Time_Days", "Suggested_Reorder", "Stock_Status",
        ]].copy()
        table_data.columns = ["SKU", "Category", "Stock", "Safety\nStock",
                              "Days to\nDeplete", "Lead\nTime", "Reorder\nQty", "Status"]
        table_data["Stock"] = table_data["Stock"].apply(lambda x: f"{x:,.0f}")
        table_data["Safety\nStock"] = table_data["Safety\nStock"].apply(lambda x: f"{x:,.0f}")
        table_data["Days to\nDeplete"] = table_data["Days to\nDeplete"].apply(
            lambda x: f"{x:.0f}" if x < 999 else "N/A")
        table_data["Lead\nTime"] = table_data["Lead\nTime"].apply(lambda x: f"{x:.0f}")
        table_data["Reorder\nQty"] = table_data["Reorder\nQty"].apply(lambda x: f"{x:,.0f}")
        table = ax_table.table(cellText=table_data.values, colLabels=table_data.columns,
                               cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1, 1.3)
        for j in range(len(table_data.columns)):
            table[0, j].set_facecolor(CHART_TEXT_COLOR)
            table[0, j].set_text_props(color="white", fontweight="bold")
        for i in range(len(table_data)):
            status = alert_df.iloc[i]["Stock_Status"]
            row_color = {"Out of Stock": "#FADBD8", "Low Stock": "#FEF9E7"}.get(status, "#EAFAF1")
            for j in range(len(table_data.columns)):
                table[i + 1, j].set_facecolor(row_color)
        ax_table.set_title("D. Stockout Alert Report (Top 20 At-Risk SKUs)",
                           fontsize=12, fontweight="bold", pad=15)

        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 0: %s", save_path)

    # ------------------------------------------------------------------
    # Chart 1: Correlation Matrix
    # ------------------------------------------------------------------
    def plot_correlation(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_01_correlation_matrix.png"
        numeric_cols = ["Unit_Cost", "Current_Stock", "Daily_Demand_Est", "Safety_Stock_Target",
                        "Lead_Time_Days", "Reorder_Point", "Inventory_Value", "DSI",
                        "Stock_Coverage_Ratio", "Supply_Risk_Score"]
        labels = ["Unit Cost", "Stock Qty", "Daily Demand", "Safety Stock", "Lead Time",
                  "Reorder Pt", "Inv. Value", "DSI", "Coverage Ratio", "Risk Score"]
        data = df[numeric_cols].replace([np.inf, -np.inf], np.nan).dropna()
        n = len(numeric_cols)
        pearson_corr = data.corr(method="pearson")
        spearman_corr = data.corr(method="spearman")
        pval_matrix = pd.DataFrame(np.zeros((n, n)), index=numeric_cols, columns=numeric_cols)
        for i in range(n):
            for j in range(n):
                if i != j:
                    _, p = pearsonr(data[numeric_cols[i]], data[numeric_cols[j]])
                    pval_matrix.iloc[i, j] = p

        fig, axes = plt.subplots(1, 2, figsize=(22, 9), facecolor=CHART_BG_COLOR)
        fig.suptitle("Correlation Analysis — Pearson vs Spearman",
                     fontsize=18, fontweight="bold", y=1.0, color=CHART_TEXT_COLOR)
        for idx, (corr, title) in enumerate([
            (pearson_corr, "Pearson Correlation (Linear)"),
            (spearman_corr, "Spearman Correlation (Monotonic / Rank)"),
        ]):
            ax = axes[idx]
            mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                        center=0, vmin=-1, vmax=1, linewidths=0.5,
                        xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={"shrink": 0.8})
            if idx == 0:
                for i in range(n):
                    for j in range(i):
                        if pval_matrix.iloc[i, j] > 0.05:
                            ax.text(j + 0.5, i + 0.5, "ns", ha="center", va="bottom",
                                    fontsize=6, color="gray")
            ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
            ax.tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 1: %s", save_path)

    # ------------------------------------------------------------------
    # Chart 2: Distribution Analysis
    # ------------------------------------------------------------------
    def plot_distributions(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_02_distribution_analysis.png"
        metrics = [
            ("Unit_Cost", "Unit Cost ($)", "#2E86C1"),
            ("Current_Stock", "Current Stock (units)", "#27AE60"),
            ("Daily_Demand_Est", "Daily Demand Est.", "#E74C3C"),
            ("Lead_Time_Days", "Lead Time (days)", "#8E44AD"),
            ("Inventory_Value", "Inventory Value ($)", "#F39C12"),
            ("DSI", "Days Sales of Inventory", "#1ABC9C"),
        ]
        fig, axes = plt.subplots(3, 4, figsize=(24, 15), facecolor=CHART_BG_COLOR)
        fig.suptitle("Distribution Analysis — Histograms, KDE & Q-Q Plots",
                     fontsize=18, fontweight="bold", y=1.0, color=CHART_TEXT_COLOR)
        for i, (col, label, color) in enumerate(metrics):
            row = i // 2
            col_idx = (i % 2) * 2
            data = df[col].replace([np.inf, -np.inf], np.nan).dropna()
            if col == "DSI":
                data = data[data < 999]
            ax_hist = axes[row, col_idx]
            ax_hist.hist(data, bins=50, density=True, alpha=0.6, color=color, edgecolor="white")
            kde_x = np.linspace(data.min(), data.max(), 300)
            if data.std() > 0:
                kde = stats.gaussian_kde(data)
                ax_hist.plot(kde_x, kde(kde_x), color=CHART_TEXT_COLOR, lw=2)
            ax_hist.set_title(label, fontsize=11, fontweight="bold")
            ax_hist.set_ylabel("Density")
            skew_val = data.skew()
            kurt_val = data.kurtosis()
            _, shapiro_p = stats.shapiro(data.sample(min(SHAPIRO_SAMPLE_LIMIT, len(data)), random_state=42))
            stat_text = (f"n={len(data):,}\nmean={data.mean():.1f}\nstd={data.std():.1f}\n"
                         f"skew={skew_val:.2f}\nkurt={kurt_val:.2f}\nShapiro p={shapiro_p:.4f}")
            ax_hist.text(0.97, 0.95, stat_text, transform=ax_hist.transAxes, fontsize=7,
                         va="top", ha="right", family="monospace",
                         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.85))
            ax_qq = axes[row, col_idx + 1]
            osm, osr = stats.probplot(data, dist="norm")[:2]
            ax_qq.scatter(osm[0], osm[1], s=3, alpha=0.4, color=color)
            ax_qq.plot(osm[0], osr[0] * osm[0] + osr[1], "r-", lw=1.5)
            ax_qq.set_title(f"Q-Q Plot: {label}", fontsize=10)
            ax_qq.set_xlabel("Theoretical Quantiles")
            ax_qq.set_ylabel("Sample Quantiles")
            normality = "Normal" if shapiro_p > 0.05 else "Non-Normal"
            norm_color = "#27AE60" if shapiro_p > 0.05 else "#E74C3C"
            ax_qq.text(0.05, 0.92, normality, transform=ax_qq.transAxes, fontsize=9,
                       fontweight="bold", color=norm_color,
                       bbox=dict(boxstyle="round", fc="white", ec=norm_color))
        plt.tight_layout()
        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 2: %s", save_path)

    # ------------------------------------------------------------------
    # Chart 3: Vendor Performance
    # ------------------------------------------------------------------
    def plot_vendor_analysis(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_03_vendor_performance.png"
        metrics = [
            ("Unit_Cost", "Unit Cost ($)"), ("Current_Stock", "Stock Quantity"),
            ("Lead_Time_Days", "Lead Time (days)"), ("Inventory_Value", "Inventory Value ($)"),
            ("DSI", "DSI (days)"), ("Supply_Risk_Score", "Supply Risk Score"),
        ]
        fig, axes = plt.subplots(2, 3, figsize=(24, 14), facecolor=CHART_BG_COLOR)
        fig.suptitle("Vendor Performance Comparison — Box Plots with Statistical Tests",
                     fontsize=18, fontweight="bold", y=1.0, color=CHART_TEXT_COLOR)
        vendor_order = df.groupby("Vendor_Name")["Inventory_Value"].sum().sort_values(ascending=False).index
        palette = sns.color_palette("Set2", n_colors=len(vendor_order))
        for idx, (col, label) in enumerate(metrics):
            ax = axes[idx // 3, idx % 3]
            plot_data = df.copy()
            if col == "DSI":
                plot_data = plot_data[plot_data["DSI"] < 999]
            sns.boxplot(data=plot_data, x="Vendor_Name", y=col, order=vendor_order,
                        palette=palette, ax=ax, fliersize=2, linewidth=0.8)
            ax.set_xticklabels([v.replace(" ", "\n") for v in vendor_order], fontsize=7)
            ax.set_xlabel("")
            ax.set_ylabel(label, fontsize=9)
            groups = [g[col].dropna().values for _, g in plot_data.groupby("Vendor_Name")]
            if col == "DSI":
                groups = [g[g < 999] for g in groups]
            try:
                h_stat, kw_p = kruskal(*groups)
                f_stat, anova_p = f_oneway(*groups)
                test_text = f"ANOVA: F={f_stat:.1f}, p={anova_p:.2e}\nKruskal: H={h_stat:.1f}, p={kw_p:.2e}"
                sig_color = "#E74C3C" if kw_p < 0.05 else "#27AE60"
            except Exception:
                test_text = "Test N/A"
                sig_color = "#95A5A6"
            ax.text(0.02, 0.97, test_text, transform=ax.transAxes, fontsize=7, va="top",
                    family="monospace", bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=sig_color, alpha=0.9))
            if col == "Inventory_Value":
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        plt.tight_layout()
        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 3: %s", save_path)

    # ------------------------------------------------------------------
    # Chart 4: Cross Analysis
    # ------------------------------------------------------------------
    def plot_cross_analysis(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_04_category_vendor_heatmap.png"

        # Shorten vendor names for better layout (e.g. "Tokyo Electronics" -> "Tokyo")
        df = df.copy()
        df["_Vendor_Short"] = df["Vendor_Name"].str.split().str[0]

        analyses = [
            ("Inventory_Value", "sum", "Total Inventory Value ($)", "YlOrRd"),
            ("Current_Stock", "mean", "Avg Stock Quantity", "Blues"),
            ("Lead_Time_Days", "mean", "Avg Lead Time (days)", "Purples"),
            ("Supply_Risk_Score", "mean", "Avg Supply Risk Score", "RdYlGn_r"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor=CHART_BG_COLOR)
        fig.suptitle("Category × Vendor Cross Analysis — Heatmaps with Chi-Square Test",
                     fontsize=16, fontweight="bold", color=CHART_TEXT_COLOR)

        for idx, (col, agg, title, cmap) in enumerate(analyses):
            ax = axes[idx // 2, idx % 2]
            ax.set_facecolor("white")
            if col not in df.columns:
                ax.set_title(f"{title} (N/A)", fontsize=12, fontweight="bold")
                ax.text(0.5, 0.5, "Column not available", ha="center", va="center",
                        transform=ax.transAxes, fontsize=11, color=CHART_TEXT_COLOR)
                continue
            pivot = df.pivot_table(values=col, index="Category", columns="_Vendor_Short",
                                   aggfunc=agg, fill_value=0)
            annot_fmt = ".0f" if "Value" in title or "Stock" in title else ".2f"
            sns.heatmap(pivot, annot=True, fmt=annot_fmt, cmap=cmap, linewidths=0.5,
                        ax=ax, cbar_kws={"shrink": 0.8},
                        annot_kws={"fontsize": 8})
            ax.set_title(title, fontsize=12, fontweight="bold", pad=8, color=CHART_TEXT_COLOR)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(labelsize=9, colors=CHART_TEXT_COLOR)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        ct = pd.crosstab(df["Category"], df["Stock_Status"])
        chi2, chi_p, dof, _ = chi2_contingency(ct)
        sig_text = "Significant" if chi_p < 0.05 else "Not Significant"
        fig.text(0.5, 0.01, f"Chi-Square: χ²={chi2:.1f}, df={dof}, p={chi_p:.2e} → {sig_text}",
                 ha="center", fontsize=10, color=CHART_TEXT_COLOR,
                 bbox=dict(boxstyle="round,pad=0.5", fc="#EBF5FB", ec="#2E86C1"))
        self.results["chi_square"] = {"chi2": chi2, "p": chi_p, "dof": dof}

        plt.subplots_adjust(top=0.93, bottom=0.06, hspace=0.35, wspace=0.30)
        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 4: %s", save_path)

    # ------------------------------------------------------------------
    # Chart 5: Regression Analysis
    # ------------------------------------------------------------------
    def plot_regression(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_05_regression_analysis.png"
        fig, axes = plt.subplots(2, 2, figsize=(20, 16), facecolor=CHART_BG_COLOR)
        fig.suptitle("Regression & Relationship Analysis",
                     fontsize=18, fontweight="bold", y=1.0, color=CHART_TEXT_COLOR)
        pairs = [
            ("Daily_Demand_Est", "Current_Stock", "Daily Demand", "Current Stock", "#2E86C1"),
            ("Lead_Time_Days", "Supply_Risk_Score", "Lead Time (days)", "Supply Risk Score", "#E74C3C"),
            ("Unit_Cost", "Inventory_Value", "Unit Cost ($)", "Inventory Value ($)", "#8E44AD"),
            ("Safety_Stock_Target", "Current_Stock", "Safety Stock Target", "Current Stock", "#27AE60"),
        ]
        for idx, (xcol, ycol, xlabel, ylabel, color) in enumerate(pairs):
            ax = axes[idx // 2, idx % 2]
            x = df[xcol].replace([np.inf, -np.inf], np.nan).dropna()
            y = df.loc[x.index, ycol].replace([np.inf, -np.inf], np.nan)
            valid = x.notna() & y.notna()
            x, y = x[valid].values, y[valid].values
            sample_idx = np.random.RandomState(42).choice(len(x), min(3000, len(x)), replace=False)
            ax.scatter(x[sample_idx], y[sample_idx], s=8, alpha=0.3, color=color, edgecolors="none")
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = slope * x_fit + intercept
            ax.plot(x_fit, y_fit, "r-", lw=2, label="OLS Fit")
            n = len(x)
            x_mean = x.mean()
            se = std_err * np.sqrt(1/n + (x_fit - x_mean)**2 / np.sum((x - x_mean)**2))
            t_crit = stats.t.ppf(0.975, n - 2)
            ax.fill_between(x_fit, y_fit - t_crit * se, y_fit + t_crit * se,
                            alpha=0.15, color="red", label="95% CI")
            r_pearson, p_pearson = pearsonr(x, y)
            r_spearman, p_spearman = spearmanr(x, y)
            stat_text = (f"OLS: y = {slope:.2f}x + {intercept:.1f}\nR2 = {r_value**2:.4f}\n"
                         f"Pearson r = {r_pearson:.3f} (p={p_pearson:.2e})\n"
                         f"Spearman rho = {r_spearman:.3f} (p={p_spearman:.2e})\nn = {n:,}")
            ax.text(0.03, 0.97, stat_text, transform=ax.transAxes, fontsize=8, va="top",
                    family="monospace", bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.9))
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(f"{xlabel} vs {ylabel}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=8, loc="lower right")
            if "Value" in ylabel:
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
        plt.tight_layout()
        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 5: %s", save_path)

    # ------------------------------------------------------------------
    # Chart 6: Category Risk Profile
    # ------------------------------------------------------------------
    def plot_category_risk(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_06_category_risk_profile.png"
        fig = plt.figure(figsize=(22, 14), facecolor=CHART_BG_COLOR)
        fig.suptitle("Category Risk Profile — Stock Status, Coverage & Risk Distribution",
                     fontsize=18, fontweight="bold", y=1.0, color=CHART_TEXT_COLOR)
        gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3, left=0.06, right=0.96, top=0.92, bottom=0.06)
        cat_order = df.groupby("Category")["Inventory_Value"].sum().sort_values(ascending=False).index

        ax1 = fig.add_subplot(gs[0, 0])
        ct = pd.crosstab(df["Category"], df["Stock_Status"], normalize="index") * 100
        ct = ct.loc[cat_order, ["Normal Stock", "Low Stock", "Out of Stock"]]
        ct.plot(kind="barh", stacked=True, ax=ax1,
                color=[STATUS_COLORS[c] for c in ct.columns], edgecolor="white", linewidth=0.5)
        ax1.set_xlabel("% of SKUs")
        ax1.set_title("Stock Status by Category", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=7, loc="lower right")
        ax1.invert_yaxis()

        ax2 = fig.add_subplot(gs[0, 1])
        plot_df = df[df["Stock_Coverage_Ratio"] < 10].copy()
        sns.violinplot(data=plot_df, y="Category", x="Stock_Coverage_Ratio",
                       order=cat_order, palette="Set2", ax=ax2, inner="quartile", density_norm="width", cut=0)
        ax2.axvline(1.0, color="#E74C3C", ls="--", lw=1.5, label="Safety Stock = 1x")
        ax2.set_xlabel("Stock Coverage Ratio")
        ax2.set_title("Safety Stock Coverage by Category", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=8)

        ax3 = fig.add_subplot(gs[0, 2])
        palette_cat = sns.color_palette("tab10", n_colors=len(cat_order))
        for i, cat in enumerate(cat_order):
            subset = df[df["Category"] == cat]["Supply_Risk_Score"].dropna()
            subset.plot(kind="kde", ax=ax3, label=cat, color=palette_cat[i], lw=1.5)
        ax3.set_xlabel("Supply Risk Score")
        ax3.set_title("Risk Score Distribution by Category", fontsize=12, fontweight="bold")
        ax3.legend(fontsize=7, loc="upper right")

        ax4 = fig.add_subplot(gs[1, 0])
        sns.boxplot(data=df, y="Category", x="Lead_Time_Days", order=cat_order,
                    palette="Set2", ax=ax4, fliersize=2, linewidth=0.8)
        ax4.set_xlabel("Lead Time (days)")
        ax4.set_title("Lead Time Distribution by Category", fontsize=12, fontweight="bold")
        groups = [g["Lead_Time_Days"].dropna().values for _, g in df.groupby("Category")]
        h_stat, kw_p = kruskal(*groups)
        ax4.text(0.97, 0.05, f"Kruskal-Wallis\nH={h_stat:.1f}, p={kw_p:.2e}",
                 transform=ax4.transAxes, fontsize=8, ha="right", va="bottom", family="monospace",
                 bbox=dict(boxstyle="round", fc="white", ec="#8E44AD", alpha=0.9))

        ax5 = fig.add_subplot(gs[1, 1])
        cat_stats = df.groupby("Category").agg(
            oos_rate=("Stock_Status", lambda x: (x == "Out of Stock").mean() * 100),
            avg_lead=("Lead_Time_Days", "mean"),
            total_value=("Inventory_Value", "sum"),
        ).loc[cat_order]
        ax5.scatter(cat_stats["avg_lead"], cat_stats["oos_rate"],
                    s=cat_stats["total_value"] / cat_stats["total_value"].max() * 800,
                    c=range(len(cat_stats)), cmap="Set2", edgecolors=CHART_TEXT_COLOR, linewidth=1.5, alpha=0.8)
        for i, cat in enumerate(cat_stats.index):
            ax5.annotate(cat, (cat_stats["avg_lead"].iloc[i], cat_stats["oos_rate"].iloc[i]),
                         fontsize=8, fontweight="bold", ha="center", va="bottom", xytext=(0, 8),
                         textcoords="offset points")
        ax5.set_xlabel("Avg Lead Time (days)")
        ax5.set_ylabel("Out of Stock Rate (%)")
        ax5.set_title("OOS Rate vs Lead Time", fontsize=12, fontweight="bold")

        ax6 = fig.add_subplot(gs[1, 2])
        vendor_order = df.groupby("Vendor_Name")["Inventory_Value"].sum().sort_values(ascending=False).index
        ct_v = pd.crosstab(df["Vendor_Name"], df["Stock_Status"], normalize="index") * 100
        ct_v = ct_v.loc[vendor_order, ["Normal Stock", "Low Stock", "Out of Stock"]]
        ct_v.plot(kind="barh", stacked=True, ax=ax6,
                  color=[STATUS_COLORS[c] for c in ct_v.columns], edgecolor="white", linewidth=0.5)
        ax6.set_xlabel("% of SKUs")
        ax6.set_title("Stock Status by Vendor", fontsize=12, fontweight="bold")
        ax6.legend(fontsize=7, loc="lower right")
        ax6.invert_yaxis()

        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 6: %s", save_path)

    # ------------------------------------------------------------------
    # Chart 7: Outlier & Risk
    # ------------------------------------------------------------------
    def plot_outlier_risk(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_07_outlier_risk_analysis.png"
        fig = plt.figure(figsize=(20, 16), facecolor=CHART_BG_COLOR)
        fig.suptitle("Outlier Detection & Risk Segmentation",
                     fontsize=18, fontweight="bold", y=0.97, color=CHART_TEXT_COLOR)
        gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.25, left=0.07, right=0.95, top=0.92, bottom=0.06)

        # Z-Score
        ax1 = fig.add_subplot(gs[0, 0])
        inv_values = df["Inventory_Value"].dropna()
        z_scores = np.abs(stats.zscore(inv_values))
        normal_mask = z_scores < 3
        outlier_mask = z_scores >= 3
        ax1.scatter(range(normal_mask.sum()), inv_values[normal_mask].sort_values().values,
                    s=3, alpha=0.3, color="#3498DB", label=f"Normal ({normal_mask.sum():,})")
        ax1.scatter(range(normal_mask.sum(), normal_mask.sum() + outlier_mask.sum()),
                    inv_values[outlier_mask].sort_values().values,
                    s=15, alpha=0.8, color="#E74C3C", label=f"Outlier |z|>=3 ({outlier_mask.sum():,})")
        ax1.set_ylabel("Inventory Value ($)")
        ax1.set_title("Z-Score Outlier Detection", fontsize=12, fontweight="bold")
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.legend(fontsize=9)

        # IQR
        ax2 = fig.add_subplot(gs[0, 1])
        check_cols = ["Inventory_Value", "Current_Stock", "Daily_Demand_Est", "Unit_Cost"]
        cat_outliers = {}
        for cat in df["Category"].unique():
            total_out = 0
            for col in check_cols:
                subset = df[df["Category"] == cat][col].dropna()
                q1, q3 = subset.quantile(0.25), subset.quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    total_out += ((subset < q1 - 1.5 * iqr) | (subset > q3 + 1.5 * iqr)).sum()
            cat_outliers[cat] = total_out
        cats = sorted(cat_outliers, key=cat_outliers.get, reverse=True)
        counts = [cat_outliers[c] for c in cats]
        bars = ax2.barh(cats, counts, color=sns.color_palette("Reds_r", n_colors=len(cats)), edgecolor="white")
        for bar, c in zip(bars, counts, strict=False):
            ax2.text(bar.get_width() + max(counts) * 0.02, bar.get_y() + bar.get_height() / 2,
                     str(c), va="center", fontsize=9, fontweight="bold")
        ax2.set_xlabel("Total Outliers (IQR Method)")
        ax2.set_title("Outlier Count by Category", fontsize=12, fontweight="bold")
        ax2.invert_yaxis()

        # Risk Quadrant
        ax3 = fig.add_subplot(gs[1, 0])
        plot_df = df[(df["Stock_Coverage_Ratio"] < 10) & (df["Demand_Intensity"] > 0)].copy()
        sample = plot_df.sample(min(3000, len(plot_df)), random_state=42)
        for status, color in STATUS_COLORS.items():
            sub = sample[sample["Stock_Status"] == status]
            ax3.scatter(sub["Demand_Intensity"], sub["Stock_Coverage_Ratio"],
                        s=12, alpha=0.4, color=color, label=status, edgecolors="none")
        di_med = plot_df["Demand_Intensity"].median()
        ax3.axvline(di_med, color="gray", ls="--", lw=1, alpha=0.5)
        ax3.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.5)
        ax3.set_xlabel("Demand Intensity")
        ax3.set_ylabel("Stock Coverage Ratio")
        ax3.set_title("Risk Quadrant", fontsize=12, fontweight="bold")
        ax3.legend(fontsize=8)

        # Risk Level
        ax4 = fig.add_subplot(gs[1, 1])
        df_risk = df.copy()
        df_risk["Risk_Level"] = pd.cut(df_risk["Supply_Risk_Score"],
                                        bins=RISK_LEVEL_BINS,
                                        labels=RISK_LEVEL_LABELS)
        risk_cat = pd.crosstab(df_risk["Risk_Level"], df_risk["Category"])
        risk_cat.plot(kind="bar", stacked=True, ax=ax4, colormap="Set2", edgecolor="white", linewidth=0.5)
        ax4.set_xlabel("Risk Level")
        ax4.set_ylabel("Number of SKUs")
        ax4.set_title("Risk Level Distribution by Category", fontsize=12, fontweight="bold")
        ax4.legend(fontsize=7, loc="upper left", title="Category")
        ax4.tick_params(axis="x", rotation=0)

        plt.savefig(save_path, dpi=CHART_DPI, facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 7: %s", save_path)

    # ------------------------------------------------------------------
    # Chart 8: Pair Plot
    # ------------------------------------------------------------------
    def plot_pairplot(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_08_pairplot_regression.png"
        cols = ["Unit_Cost", "Current_Stock", "Daily_Demand_Est", "Lead_Time_Days", "Inventory_Value"]
        labels = {"Unit_Cost": "Unit Cost", "Current_Stock": "Stock",
                  "Daily_Demand_Est": "Demand", "Lead_Time_Days": "Lead Time",
                  "Inventory_Value": "Inv. Value"}
        sample = df[cols + ["Stock_Status"]].dropna().sample(min(2000, len(df)), random_state=42)
        palette = {"Normal Stock": "#27AE60", "Low Stock": "#F39C12", "Out of Stock": "#E74C3C"}
        g = sns.pairplot(sample, hue="Stock_Status", palette=palette,
                         plot_kws={"s": 8, "alpha": 0.4, "edgecolor": "none"},
                         diag_kws={"alpha": 0.5}, height=2.5)
        g.figure.suptitle("Pair Plot — Key Variables by Stock Status",
                          fontsize=16, fontweight="bold", y=1.02, color=CHART_TEXT_COLOR)
        g.figure.set_facecolor(CHART_BG_COLOR)
        for ax_row in g.axes:
            for ax in ax_row:
                xlabel = ax.get_xlabel()
                ylabel = ax.get_ylabel()
                if xlabel in labels:
                    ax.set_xlabel(labels[xlabel], fontsize=8)
                if ylabel in labels:
                    ax.set_ylabel(labels[ylabel], fontsize=8)
        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor=g.figure.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 8: %s", save_path)
