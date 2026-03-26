"""Supply Chain Optimization Pipeline — Charts 09-14."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats

from app.config import (
    CHART_BG_COLOR,
    CHART_DPI,
    CHART_TEXT_COLOR,
    CHARTS_DIR,
    HOLDING_RATE,
    MONTE_CARLO_SIMS,
    ORDERING_COST,
)
from app.pipeline.enrichment import enrich_base

logger = logging.getLogger(__name__)


class SupplyChainAnalyzer:
    """Advanced supply chain optimization — EOQ, Monte Carlo, Vendor Radar, etc."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else CHARTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self.chart_paths = []

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add supply-chain specific derived columns."""
        df = enrich_base(df)
        df["Annual_Demand"] = df["Daily_Demand_Est"] * 365
        df["Reorder_Gap"] = df["Current_Stock"] - df["Reorder_Point"]
        df["Holding_Cost"] = df["Unit_Cost"] * HOLDING_RATE
        df["EOQ"] = np.where(
            (df["Annual_Demand"] > 0) & (df["Holding_Cost"] > 0),
            np.sqrt(2 * df["Annual_Demand"] * ORDERING_COST / df["Holding_Cost"]), 0)
        df["Annual_Orders"] = np.where(df["EOQ"] > 0, df["Annual_Demand"] / df["EOQ"], 0)
        df["Total_Inventory_Cost"] = (
            (df["EOQ"] / 2) * df["Holding_Cost"] + df["Annual_Orders"] * ORDERING_COST)
        return df

    def run_all(self, df: pd.DataFrame) -> dict:
        """Run all supply chain analyses."""
        logger.info("Supply chain analysis started")
        df_sc = self.enrich(df)

        self.plot_eoq_analysis(df_sc)
        self.plot_vendor_radar(df_sc)
        self.plot_inventory_treemap(df_sc)
        self.plot_monte_carlo(df_sc)
        self.plot_reorder_waterfall(df_sc)
        self.plot_demand_variability(df_sc)

        logger.info("Supply chain analysis completed — %d charts", len(self.chart_paths))
        return {
            "results": self.results,
            "chart_paths": [str(p) for p in self.chart_paths],
        }

    # ------------------------------------------------------------------
    # Chart 9: EOQ Analysis
    # ------------------------------------------------------------------
    def plot_eoq_analysis(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_09_eoq_analysis.png"
        fig = plt.figure(figsize=(22, 14), facecolor=CHART_BG_COLOR)
        fig.suptitle("EOQ Analysis — Economic Order Quantity Optimization",
                     fontsize=18, fontweight="bold", y=0.98, color=CHART_TEXT_COLOR)
        gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.30,
                      left=0.06, right=0.96, top=0.92, bottom=0.06)
        valid = df[(df["EOQ"] > 0) & (df["EOQ"] < df["EOQ"].quantile(0.99))].copy()
        if valid.empty:
            plt.close()
            logger.warning("Chart 9: No valid EOQ data, skipping")
            return
        cat_order = valid.groupby("Category")["EOQ"].median().sort_values(ascending=False).index

        ax1 = fig.add_subplot(gs[0, 0])
        sns.boxplot(data=valid, y="Category", x="EOQ", order=cat_order, palette="Set2", ax=ax1, fliersize=1)
        ax1.set_xlabel("EOQ (units)")
        ax1.set_title("EOQ Distribution by Category", fontsize=12, fontweight="bold")

        ax2 = fig.add_subplot(gs[0, 1])
        sample = valid.sample(min(2000, len(valid)), random_state=42)
        ax2.scatter(sample["Annual_Demand"], sample["EOQ"], s=8, alpha=0.3, c="#2E86C1", edgecolors="none")
        x_theory = np.linspace(1, sample["Annual_Demand"].max(), 200)
        avg_hc = valid["Holding_Cost"].median()
        y_theory = np.sqrt(2 * x_theory * ORDERING_COST / avg_hc)
        ax2.plot(x_theory, y_theory, "r-", lw=2, label=f"Theoretical EOQ\n(H=${avg_hc:.0f}, S=${ORDERING_COST})")
        ax2.set_xlabel("Annual Demand (units)")
        ax2.set_ylabel("EOQ (units)")
        ax2.set_title("EOQ vs Annual Demand", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=8)

        ax3 = fig.add_subplot(gs[0, 2])
        cost_by_cat = valid.groupby("Category").agg(
            Holding=("Holding_Cost", lambda x: (valid.loc[x.index, "EOQ"] / 2 * x).sum()),
            Ordering=("Annual_Orders", lambda x: (x * ORDERING_COST).sum()),
        ).loc[cat_order]
        (cost_by_cat / 1e6).plot(kind="barh", stacked=True, ax=ax3,
                                  color=["#3498DB", "#E74C3C"], edgecolor="white")
        ax3.set_xlabel("Total Annual Cost ($M)")
        ax3.set_title("Inventory Cost Breakdown", fontsize=12, fontweight="bold")
        ax3.legend(fontsize=8, title="Cost Type")
        ax3.invert_yaxis()

        ax4 = fig.add_subplot(gs[1, 0])
        example = valid.iloc[0]
        D, S, H = example["Annual_Demand"], ORDERING_COST, example["Holding_Cost"]
        Q_range = np.linspace(10, D * 0.5, 500)
        holding_costs = (Q_range / 2) * H
        ordering_costs = (D / Q_range) * S
        total_costs = holding_costs + ordering_costs
        eoq_val = example["EOQ"]
        ax4.plot(Q_range, holding_costs, "--", color="#3498DB", lw=1.5, label="Holding Cost")
        ax4.plot(Q_range, ordering_costs, "--", color="#E74C3C", lw=1.5, label="Ordering Cost")
        ax4.plot(Q_range, total_costs, "-", color=CHART_TEXT_COLOR, lw=2.5, label="Total Cost")
        ax4.axvline(eoq_val, color="#27AE60", ls=":", lw=2)
        ax4.plot(eoq_val, total_costs.min(), "o", color="#27AE60", ms=10, zorder=5)
        ax4.annotate(f"EOQ = {eoq_val:.0f}\nMin Cost = ${total_costs.min():,.0f}",
                     xy=(eoq_val, total_costs.min()), xytext=(eoq_val * 1.5, total_costs.min() * 1.3),
                     fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="->", color="#27AE60"),
                     bbox=dict(boxstyle="round", fc="#EAFAF1", ec="#27AE60"))
        ax4.set_xlabel("Order Quantity (Q)")
        ax4.set_ylabel("Annual Cost ($)")
        ax4.set_title(f"EOQ Cost Curve — {example['Product_ID']}", fontsize=12, fontweight="bold")
        ax4.legend(fontsize=8)

        ax5 = fig.add_subplot(gs[1, 1])
        valid["Orders_Per_Year"] = valid["Annual_Orders"].clip(upper=52)
        sns.histplot(valid["Orders_Per_Year"], bins=40, kde=True, ax=ax5, color="#8E44AD", alpha=0.6)
        ax5.axvline(valid["Orders_Per_Year"].median(), color="red", ls="--", lw=1.5,
                    label=f'Median: {valid["Orders_Per_Year"].median():.1f} orders/yr')
        ax5.set_xlabel("Optimal Orders per Year")
        ax5.set_title("Reorder Frequency Distribution", fontsize=12, fontweight="bold")
        ax5.legend(fontsize=8)

        ax6 = fig.add_subplot(gs[1, 2])
        monthly_q = valid["Annual_Demand"] / 12
        monthly_cost = (monthly_q / 2) * valid["Holding_Cost"] + 12 * ORDERING_COST
        savings = (monthly_cost - valid["Total_Inventory_Cost"]).clip(lower=0)
        savings_by_cat = savings.groupby(valid["Category"]).sum().loc[cat_order] / 1e6
        bars = ax6.barh(savings_by_cat.index, savings_by_cat.values,
                        color=sns.color_palette("YlOrRd_r", len(savings_by_cat)), edgecolor="white")
        for bar, val in zip(bars, savings_by_cat.values, strict=False):
            ax6.text(bar.get_width() + savings_by_cat.max() * 0.02,
                     bar.get_y() + bar.get_height() / 2,
                     f"${val:.2f}M", va="center", fontsize=9, fontweight="bold")
        ax6.set_xlabel("Potential Annual Savings ($M)")
        ax6.set_title("EOQ Savings Potential", fontsize=12, fontweight="bold")
        ax6.invert_yaxis()

        plt.savefig(save_path, dpi=CHART_DPI, facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 9: %s", save_path)

    # ------------------------------------------------------------------
    # Chart 10: Vendor Radar
    # ------------------------------------------------------------------
    def plot_vendor_radar(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_10_vendor_radar.png"
        fig = plt.figure(figsize=(22, 10), facecolor=CHART_BG_COLOR)
        fig.suptitle("Vendor Performance Radar — Multi-Dimensional Comparison",
                     fontsize=18, fontweight="bold", y=1.0, color=CHART_TEXT_COLOR)
        metrics = {
            "Avg Unit Cost": ("Unit_Cost", "mean", True),
            "Stock Coverage": ("Stock_Coverage_Ratio", "mean", False),
            "OOS Rate %": ("Stock_Status", lambda x: (x == "Out of Stock").mean() * 100, True),
            "Avg Lead Time": ("Lead_Time_Days", "mean", True),
            "Avg Inv. Value": ("Inventory_Value", "mean", False),
            "SKU Count": ("Product_ID", "count", False),
        }
        vendor_scores = {}
        for vendor in df["Vendor_Name"].unique():
            vdf = df[df["Vendor_Name"] == vendor]
            scores = {}
            for name, (col, agg, _) in metrics.items():
                scores[name] = agg(vdf[col]) if callable(agg) else vdf[col].agg(agg)
            vendor_scores[vendor] = scores
        score_df = pd.DataFrame(vendor_scores).T
        norm_df = score_df.copy()
        for name, (_, _, invert) in metrics.items():
            col_min, col_max = score_df[name].min(), score_df[name].max()
            if col_max > col_min:
                if invert:
                    norm_df[name] = 1 - (score_df[name] - col_min) / (col_max - col_min)
                else:
                    norm_df[name] = (score_df[name] - col_min) / (col_max - col_min)
            else:
                norm_df[name] = 0.5
        self.results["vendor_rankings"] = norm_df.mean(axis=1).sort_values(ascending=False).to_dict()

        categories_list = list(metrics.keys())
        N = len(categories_list)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        colors = sns.color_palette("Set2", n_colors=len(norm_df))
        vendors = norm_df.index.tolist()
        n_vendors = len(vendors)
        cols_layout = 4
        rows_layout = (n_vendors + cols_layout - 1) // cols_layout
        gs_radar = GridSpec(rows_layout, cols_layout + 1, figure=fig,
                            width_ratios=[1] * cols_layout + [0.6],
                            hspace=0.4, wspace=0.3, left=0.03, right=0.97, top=0.88, bottom=0.05)
        for idx, vendor in enumerate(vendors):
            r, c = idx // cols_layout, idx % cols_layout
            ax = fig.add_subplot(gs_radar[r, c], polar=True)
            values = norm_df.loc[vendor].tolist() + [norm_df.loc[vendor].tolist()[0]]
            ax.fill(angles, values, alpha=0.25, color=colors[idx])
            ax.plot(angles, values, "o-", ms=4, lw=1.5, color=colors[idx])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories_list, fontsize=6)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.25, 0.5, 0.75])
            ax.set_yticklabels(["25%", "50%", "75%"], fontsize=5, color="gray")
            ax.set_title(vendor.replace(" ", "\n"), fontsize=9, fontweight="bold", pad=12, color=colors[idx])

        ax_rank = fig.add_subplot(gs_radar[:, cols_layout])
        ax_rank.axis("off")
        overall = norm_df.mean(axis=1).sort_values(ascending=False)
        ax_rank.set_title("Overall\nRanking", fontsize=11, fontweight="bold", pad=10)
        for i, (vendor, score) in enumerate(overall.items()):
            medal = ["#1", "#2", "#3"][i] if i < 3 else f"#{i+1}"
            y = 0.9 - i * 0.12
            ax_rank.text(0.1, y, medal, transform=ax_rank.transAxes, fontsize=14, va="center")
            ax_rank.text(0.35, y, vendor.split()[0], transform=ax_rank.transAxes,
                         fontsize=8, va="center", fontweight="bold")
            bar_width = score * 0.55
            rect = plt.Rectangle((0.35, y - 0.025), bar_width, 0.04,
                                 transform=ax_rank.transAxes,
                                 facecolor=colors[vendors.index(vendor)], alpha=0.7)
            ax_rank.add_patch(rect)
            ax_rank.text(0.35 + bar_width + 0.02, y, f"{score:.2f}",
                         transform=ax_rank.transAxes, fontsize=8, va="center")

        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 10: %s", save_path)

    # ------------------------------------------------------------------
    # Chart 11: Treemap
    # ------------------------------------------------------------------
    def plot_inventory_treemap(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_11_inventory_treemap.png"
        fig, axes = plt.subplots(1, 2, figsize=(22, 10), facecolor=CHART_BG_COLOR)
        fig.suptitle("Inventory Value Composition — Treemap & Sunburst View",
                     fontsize=18, fontweight="bold", y=1.0, color=CHART_TEXT_COLOR)

        import squarify

        cat_val = df.groupby("Category")["Inventory_Value"].sum().sort_values(ascending=False)
        total = cat_val.sum()
        labels = [f"{cat}\n${val/1e6:.1f}M\n({val/total*100:.1f}%)" for cat, val in cat_val.items()]
        colors_tree = sns.color_palette("Set2", n_colors=len(cat_val))
        ax1 = axes[0]
        squarify.plot(sizes=cat_val.values, label=labels, color=colors_tree,
                      alpha=0.85, ax=ax1, text_kwargs={"fontsize": 9, "fontweight": "bold"})
        ax1.set_title("Inventory Value Treemap by Category", fontsize=13, fontweight="bold")
        ax1.axis("off")

        ax2 = axes[1]
        cat_order = cat_val.index.tolist()
        vendor_palette = sns.color_palette("Pastel2", n_colors=df["Vendor_Name"].nunique())
        vendor_list = df["Vendor_Name"].unique().tolist()
        inner_vals, inner_colors = [], []
        for cat in cat_order:
            cat_df = df[df["Category"] == cat]
            vendor_val = cat_df.groupby("Vendor_Name")["Inventory_Value"].sum().sort_values(ascending=False)
            for vendor, val in vendor_val.items():
                inner_vals.append(val)
                inner_colors.append(vendor_palette[vendor_list.index(vendor)])
        wedges1, _ = ax2.pie(cat_val.values, radius=1.1, colors=colors_tree,
                             wedgeprops=dict(width=0.35, edgecolor="white", linewidth=2), startangle=90)
        for _i, (w, cat) in enumerate(zip(wedges1, cat_order, strict=False)):
            ang = (w.theta2 + w.theta1) / 2
            x, y = 1.3 * np.cos(np.radians(ang)), 1.3 * np.sin(np.radians(ang))
            ax2.text(x, y, cat, ha="center", va="center", fontsize=8, fontweight="bold")
        ax2.pie(inner_vals, radius=0.75, colors=inner_colors,
                wedgeprops=dict(width=0.30, edgecolor="white", linewidth=0.5), startangle=90)
        ax2.text(0, 0, f"Total\n${total/1e9:.2f}B", ha="center", va="center",
                 fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax2.set_title("Nested Donut: Category (Outer) x Vendor (Inner)", fontsize=13, fontweight="bold")
        legend_handles = [Patch(facecolor=vendor_palette[i], label=v) for i, v in enumerate(vendor_list)]
        ax2.legend(handles=legend_handles, loc="lower right", fontsize=7, title="Vendors", bbox_to_anchor=(1.15, 0))

        plt.tight_layout()
        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 11: %s", save_path)

    # ------------------------------------------------------------------
    # Chart 12: Monte Carlo Simulation
    # ------------------------------------------------------------------
    def plot_monte_carlo(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_12_monte_carlo_stockout.png"
        fig = plt.figure(figsize=(22, 14), facecolor=CHART_BG_COLOR)
        fig.suptitle("Monte Carlo Simulation — Stockout Probability During Lead Time",
                     fontsize=18, fontweight="bold", y=0.98, color=CHART_TEXT_COLOR)
        gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.30,
                      left=0.06, right=0.96, top=0.92, bottom=0.06)
        rng = np.random.RandomState(42)
        categories = df.groupby("Category")["Inventory_Value"].sum().sort_values(ascending=False).index[:6]
        mc_results = {}

        for idx, cat in enumerate(categories):
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            cat_df = df[(df["Category"] == cat) & (df["Daily_Demand_Est"] > 0)]
            demand_mean = cat_df["Daily_Demand_Est"].mean()
            demand_std = cat_df["Daily_Demand_Est"].std()
            if np.isnan(demand_std) or demand_std == 0:
                demand_std = max(demand_mean * 0.1, 1.0)  # Fallback: 10% CV
            lt_mean = cat_df["Lead_Time_Days"].mean()
            lt_std = cat_df["Lead_Time_Days"].std()
            if np.isnan(lt_std) or lt_std == 0:
                lt_std = max(lt_mean * 0.1, 1.0)
            avg_stock = cat_df["Current_Stock"].mean()
            avg_safety = cat_df["Safety_Stock_Target"].mean()
            sim_lt = rng.normal(lt_mean, max(lt_std, 1), MONTE_CARLO_SIMS).clip(1, 60).astype(int)
            sim_demand = rng.normal(demand_mean, max(demand_std, 1), MONTE_CARLO_SIMS).clip(0)
            sim_total = sim_lt * sim_demand
            stockout_mask = sim_total > avg_stock
            stockout_pct = stockout_mask.mean() * 100
            mc_results[cat] = stockout_pct
            ax.hist(sim_total[~stockout_mask], bins=60, alpha=0.6, color="#27AE60", label="Fulfilled", density=True)
            ax.hist(sim_total[stockout_mask], bins=60, alpha=0.6, color="#E74C3C", label="Stockout", density=True)
            ax.axvline(avg_stock, color="#2E86C1", ls="--", lw=2, label=f"Avg Stock: {avg_stock:,.0f}")
            ax.axvline(avg_safety, color="#F39C12", ls=":", lw=1.5, label=f"Safety Stock: {avg_safety:,.0f}")
            kde_x = np.linspace(sim_total.min(), sim_total.max(), 300)
            kde = stats.gaussian_kde(sim_total)
            ax.plot(kde_x, kde(kde_x), color=CHART_TEXT_COLOR, lw=1.5)
            ax.text(0.97, 0.95,
                    f"Stockout Prob: {stockout_pct:.1f}%\nDemand={demand_mean:.0f}/day\n"
                    f"LT={lt_mean:.0f} days\nn={MONTE_CARLO_SIMS:,}",
                    transform=ax.transAxes, fontsize=7, va="top", ha="right", family="monospace",
                    bbox=dict(boxstyle="round", fc="white", ec="#E74C3C" if stockout_pct > 30 else "#27AE60"))
            ax.set_xlabel("Total Demand During Lead Time")
            ax.set_ylabel("Density")
            ax.set_title(f"{cat} — Stockout Risk: {stockout_pct:.1f}%", fontsize=11, fontweight="bold",
                         color="#E74C3C" if stockout_pct > 30 else CHART_TEXT_COLOR)
            ax.legend(fontsize=6, loc="upper left")

        self.results["monte_carlo"] = mc_results
        plt.savefig(save_path, dpi=CHART_DPI, facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 12: %s", save_path)

    # ------------------------------------------------------------------
    # Chart 13: Reorder Gap Waterfall
    # ------------------------------------------------------------------
    def plot_reorder_waterfall(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_13_reorder_gap_waterfall.png"
        fig = plt.figure(figsize=(22, 14), facecolor=CHART_BG_COLOR)
        fig.suptitle("Reorder Gap Analysis — Replenishment Priority",
                     fontsize=18, fontweight="bold", y=0.98, color=CHART_TEXT_COLOR)
        gs = GridSpec(2, 2, figure=fig, hspace=0.30, wspace=0.28,
                      left=0.06, right=0.96, top=0.92, bottom=0.06)
        cat_order = df.groupby("Category")["Inventory_Value"].sum().sort_values(ascending=False).index

        ax1 = fig.add_subplot(gs[0, 0])
        sns.violinplot(data=df, y="Category", x="Reorder_Gap", order=cat_order,
                       palette="RdYlGn", ax=ax1, inner="quartile", density_norm="width", cut=0)
        ax1.axvline(0, color="#E74C3C", ls="--", lw=2, label="Reorder Point Line")
        ax1.set_xlabel("Stock - Reorder Point")
        ax1.set_title("Reorder Gap by Category", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(gs[0, 1])
        urgent = df[df["Reorder_Gap"] < 0].nsmallest(25, "Reorder_Gap").sort_values("Reorder_Gap")
        colors_bar = ["#E74C3C" if s == "Out of Stock" else "#F39C12" for s in urgent["Stock_Status"]]
        ax2.barh(range(len(urgent)), urgent["Reorder_Gap"].values, color=colors_bar, edgecolor="white")
        ax2.set_yticks(range(len(urgent)))
        ax2.set_yticklabels(urgent["Product_ID"].values, fontsize=6)
        ax2.set_xlabel("Stock Deficit (units below Reorder Point)")
        ax2.set_title("Top 25 Most Urgent SKUs", fontsize=12, fontweight="bold")
        ax2.invert_yaxis()
        ax2.legend(handles=[Patch(facecolor="#E74C3C", label="Out of Stock"),
                            Patch(facecolor="#F39C12", label="Low Stock")], fontsize=8)

        ax3 = fig.add_subplot(gs[1, 0])
        reorder_df = df[df["Reorder_Gap"] < 0].copy()
        reorder_df["Urgency"] = -reorder_df["Reorder_Gap"]
        sample = reorder_df.sample(min(2000, len(reorder_df)), random_state=42)
        for status, color in [("Out of Stock", "#E74C3C"), ("Low Stock", "#F39C12")]:
            sub = sample[sample["Stock_Status"] == status]
            if len(sub) > 0:
                ax3.scatter(sub["Urgency"], sub["Unit_Cost"],
                            s=sub["Annual_Demand"] / sub["Annual_Demand"].max() * 100,
                            alpha=0.4, color=color, label=status, edgecolors="none")
        ax3.set_xlabel("Urgency (units below Reorder Point)")
        ax3.set_ylabel("Unit Cost ($)")
        ax3.set_title("Replenishment Priority", fontsize=12, fontweight="bold")
        ax3.legend(fontsize=8)

        ax4 = fig.add_subplot(gs[1, 1])
        reorder_df["Replenishment_Cost"] = reorder_df["Urgency"] * reorder_df["Unit_Cost"]
        replen_by_cat = reorder_df.groupby("Category").agg(
            Total_Cost=("Replenishment_Cost", "sum"), SKU_Count=("Product_ID", "count")
        ).sort_values("Total_Cost", ascending=False)
        bars = ax4.bar(range(len(replen_by_cat)), replen_by_cat["Total_Cost"].values / 1e6,
                       color=sns.color_palette("Reds_r", len(replen_by_cat)), edgecolor="white")
        ax4.set_xticks(range(len(replen_by_cat)))
        ax4.set_xticklabels(replen_by_cat.index, rotation=30, ha="right", fontsize=8)
        ax4.set_ylabel("Est. Replenishment Cost ($M)")
        ax4.set_title("Replenishment Budget by Category", fontsize=12, fontweight="bold")
        for _i, (bar, count) in enumerate(zip(bars, replen_by_cat["SKU_Count"], strict=False)):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{count} SKUs", ha="center", va="bottom", fontsize=8, fontweight="bold")

        plt.savefig(save_path, dpi=CHART_DPI, facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 13: %s", save_path)

    # ------------------------------------------------------------------
    # Chart 14: Demand Variability & Safety Stock
    # ------------------------------------------------------------------
    def plot_demand_variability(self, df: pd.DataFrame):
        save_path = self.output_dir / "chart_14_demand_safety_stock.png"
        fig = plt.figure(figsize=(22, 14), facecolor=CHART_BG_COLOR)
        fig.suptitle("Demand Variability & Safety Stock Adequacy",
                     fontsize=18, fontweight="bold", y=0.98, color=CHART_TEXT_COLOR)
        gs = GridSpec(2, 3, figure=fig, hspace=0.30, wspace=0.30,
                      left=0.06, right=0.96, top=0.92, bottom=0.06)

        cat_stats = df.groupby("Category").agg(
            demand_mean=("Daily_Demand_Est", "mean"), demand_std=("Daily_Demand_Est", "std"),
            cost_mean=("Unit_Cost", "mean"), avg_safety=("Safety_Stock_Target", "mean"),
            avg_stock=("Current_Stock", "mean"), avg_lead=("Lead_Time_Days", "mean"),
            sku_count=("Product_ID", "count"),
            oos_rate=("Stock_Status", lambda x: (x == "Out of Stock").mean() * 100),
        )
        cat_stats["CV"] = np.where(cat_stats["demand_mean"] > 0,
                                    cat_stats["demand_std"] / cat_stats["demand_mean"], 0.0)
        cat_stats = cat_stats.sort_values("CV", ascending=False)

        ax1 = fig.add_subplot(gs[0, 0])
        bars = ax1.barh(cat_stats.index, cat_stats["CV"],
                        color=sns.color_palette("YlOrRd", len(cat_stats)), edgecolor="white")
        for bar, cv in zip(bars, cat_stats["CV"], strict=False):
            ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                     f"{cv:.3f}", va="center", fontsize=9, fontweight="bold")
        ax1.set_xlabel("Coefficient of Variation")
        ax1.set_title("Demand Variability (CV)", fontsize=12, fontweight="bold")
        ax1.invert_yaxis()

        ax2 = fig.add_subplot(gs[0, 1])
        levels = [
            (">3x Safety Stock", (df["Stock_Coverage_Ratio"] > 3).sum()),
            ("2-3x Safety Stock", ((df["Stock_Coverage_Ratio"] > 2) & (df["Stock_Coverage_Ratio"] <= 3)).sum()),
            ("1-2x Safety Stock", ((df["Stock_Coverage_Ratio"] > 1) & (df["Stock_Coverage_Ratio"] <= 2)).sum()),
            ("0.5-1x (At Risk)", ((df["Stock_Coverage_Ratio"] > 0.5) & (df["Stock_Coverage_Ratio"] <= 1)).sum()),
            ("<0.5x (Critical)", ((df["Stock_Coverage_Ratio"] > 0) & (df["Stock_Coverage_Ratio"] <= 0.5)).sum()),
            ("Zero Stock", (df["Current_Stock"] == 0).sum()),
        ]
        labels_f, counts = zip(*levels, strict=False)
        colors_f = ["#27AE60", "#82E0AA", "#F9E79F", "#F5B041", "#E74C3C", "#8B0000"]
        total_skus = len(df)
        for i, (label, count, color) in enumerate(zip(labels_f, counts, colors_f, strict=False)):
            ax2.barh(i, count, color=color, edgecolor="white", height=0.7)
            ax2.text(count + max(counts) * 0.02, i,
                     f"{count:,} ({count/total_skus*100:.1f}%)", va="center", fontsize=8, fontweight="bold")
        ax2.set_yticks(range(len(labels_f)))
        ax2.set_yticklabels(labels_f, fontsize=8)
        ax2.set_xlabel("Number of SKUs")
        ax2.set_title("Safety Stock Coverage Funnel", fontsize=12, fontweight="bold")
        ax2.invert_yaxis()

        ax3 = fig.add_subplot(gs[0, 2])
        coverage_ratios = np.linspace(0, 5, 200)
        pct_covered = [(df["Stock_Coverage_Ratio"] >= r).mean() * 100 for r in coverage_ratios]
        ax3.plot(coverage_ratios, pct_covered, color="#2E86C1", lw=2.5)
        ax3.fill_between(coverage_ratios, pct_covered, alpha=0.1, color="#2E86C1")
        for target_pct, label in [(95, "95% SL"), (90, "90% SL"), (80, "80% SL")]:
            idx_arr = np.argmin(np.abs(np.array(pct_covered) - target_pct))
            ratio_at = coverage_ratios[idx_arr]
            ax3.axhline(target_pct, color="gray", ls=":", lw=0.8, alpha=0.5)
            ax3.plot(ratio_at, target_pct, "ro", ms=6)
            ax3.annotate(f"{label}\n(>={ratio_at:.1f}x)", xy=(ratio_at, target_pct),
                         xytext=(ratio_at + 0.5, target_pct + 3), fontsize=8,
                         arrowprops=dict(arrowstyle="->", color="red"))
        ax3.set_xlabel("Min Stock Coverage Ratio")
        ax3.set_ylabel("% of SKUs Meeting Target")
        ax3.set_title("Service Level Curve", fontsize=12, fontweight="bold")
        ax3.set_xlim(0, 5)
        ax3.set_ylim(0, 105)

        ax4 = fig.add_subplot(gs[1, 0])
        scatter = ax4.scatter(cat_stats["CV"], cat_stats["oos_rate"],
                              s=cat_stats["sku_count"] / cat_stats["sku_count"].max() * 500,
                              c=cat_stats["cost_mean"], cmap="YlOrRd",
                              edgecolors=CHART_TEXT_COLOR, linewidth=1.5, zorder=5)
        for i, cat in enumerate(cat_stats.index):
            ax4.annotate(cat, (cat_stats["CV"].iloc[i], cat_stats["oos_rate"].iloc[i]),
                         fontsize=8, fontweight="bold", ha="center", va="bottom",
                         xytext=(0, 8), textcoords="offset points")
        plt.colorbar(scatter, ax=ax4, label="Avg Unit Cost ($)", shrink=0.7)
        ax4.set_xlabel("Demand CV")
        ax4.set_ylabel("Out of Stock Rate (%)")
        ax4.set_title("CV vs OOS Rate", fontsize=12, fontweight="bold")

        ax5 = fig.add_subplot(gs[1, 1])
        df_lt = df.copy()
        df_lt["LT_Bucket"] = pd.cut(df_lt["Lead_Time_Days"], bins=[0, 7, 14, 21, 30],
                                     labels=["1-7d", "8-14d", "15-21d", "22-30d"])
        lt_oos = pd.crosstab(df_lt["Category"], df_lt["LT_Bucket"],
                              values=df_lt["Stock_Status"].apply(lambda x: 1 if x == "Out of Stock" else 0),
                              aggfunc="mean") * 100
        sns.heatmap(lt_oos, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5,
                    ax=ax5, cbar_kws={"label": "OOS Rate (%)", "shrink": 0.7})
        ax5.set_title("OOS Rate: Category x Lead Time", fontsize=12, fontweight="bold")

        ax6 = fig.add_subplot(gs[1, 2])
        sample = df[df["Daily_Demand_Est"] > 0].sample(min(2000, len(df)), random_state=42)
        for status, color in [("Normal Stock", "#27AE60"), ("Low Stock", "#F39C12"), ("Out of Stock", "#E74C3C")]:
            sub = sample[sample["Stock_Status"] == status]
            ax6.scatter(sub["Daily_Demand_Est"], sub["Safety_Stock_Target"],
                        s=8, alpha=0.3, color=color, label=status, edgecolors="none")
        x_line = np.linspace(1, sample["Daily_Demand_Est"].max(), 100)
        avg_lt = df["Lead_Time_Days"].mean()
        # Standard safety stock: Z * sigma_demand * sqrt(LT)
        # Using demand * 0.5 as proxy for sigma_demand (CV ~= 0.5 assumption)
        theoretical_ss = 1.96 * np.sqrt(avg_lt) * x_line * 0.5
        ax6.plot(x_line, theoretical_ss, "r--", lw=1.5, label=f"Theoretical SS (Z=1.96, LT={avg_lt:.0f}d)")
        ax6.set_xlabel("Daily Demand")
        ax6.set_ylabel("Safety Stock Target")
        ax6.set_title("Demand vs Safety Stock", fontsize=12, fontweight="bold")
        ax6.legend(fontsize=7, loc="upper left")

        plt.savefig(save_path, dpi=CHART_DPI, facecolor=fig.get_facecolor())
        plt.close()
        self.chart_paths.append(save_path)
        logger.info("Chart 14: %s", save_path)
