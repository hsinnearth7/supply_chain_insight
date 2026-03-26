"""ML Analysis Pipeline — Charts 15-22 (30 AI Algorithms)."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline as SKPipeline

    # ConvergenceWarning is suppressed locally around sklearn calls, not globally

# Supervised classifiers
# Unsupervised
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA

# Anomaly detection
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Preprocessing / model selection / metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from app.config import (
    CHART_BG_COLOR,
    CHART_DPI,
    CHART_TEXT_COLOR,
    CHARTS_DIR,
    GA_CROSSOVER_RATE,
    GA_GENERATIONS,
    GA_MUTATION_RATE,
    GA_POPULATION,
)
from app.pipeline.enrichment import enrich_base

logger = logging.getLogger(__name__)

# Classification features — exclude circular predictors that encode the target
CLASSIFICATION_FEATURES = [
    "Unit_Cost", "Current_Stock", "Daily_Demand_Est",
    "Safety_Stock_Target", "Lead_Time_Days",
    "Demand_Intensity", "Category_Enc", "Vendor_Enc",
]

# Regression features — can include derived columns for value prediction
REGRESSION_FEATURES = CLASSIFICATION_FEATURES + [
    "Reorder_Point", "DSI", "Stock_Coverage_Ratio",
]

# All feature columns (for clustering/dimensionality reduction)
# NOTE: Inventory_Value is the regression target — excluded to prevent leakage.
# Current_Stock and Unit_Cost in REGRESSION_FEATURES create partial leakage
# with Inventory_Value (= Current_Stock * Unit_Cost), but are kept because
# they carry independent signal for clustering and classification tasks.
FEATURE_COLS = REGRESSION_FEATURES


class MLAnalyzer:
    """Runs all ML analyses (charts 15-22) and generates charts."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else CHARTS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict = {}
        self.chart_paths: list = []

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def enrich(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add DSI, Stock_Coverage_Ratio, Demand_Intensity and label-encoded columns."""
        df = enrich_base(df)

        le_cat = LabelEncoder()
        df["Category_Enc"] = le_cat.fit_transform(df["Category"].astype(str))

        le_vendor = LabelEncoder()
        df["Vendor_Enc"] = le_vendor.fit_transform(df["Vendor_Name"].astype(str))

        le_status = LabelEncoder()
        df["Stock_Status_Enc"] = le_status.fit_transform(df["Stock_Status"].astype(str))

        # Store encoders on instance for later use
        self._le_cat = le_cat
        self._le_vendor = le_vendor
        self._le_status = le_status

        return df

    def _prepare_classification_arrays(self, df: pd.DataFrame):
        """Return (clean_df, X, y_class) for classification tasks."""
        clean = df.dropna(subset=CLASSIFICATION_FEATURES + ["Stock_Status_Enc"]).copy()
        X = clean[CLASSIFICATION_FEATURES].values
        y_class = clean["Stock_Status_Enc"].values
        return clean, X, y_class

    def _prepare_regression_arrays(self, df: pd.DataFrame):
        """Return (clean_df, X, y_reg) for regression tasks."""
        clean = df.dropna(subset=REGRESSION_FEATURES + ["Inventory_Value"]).copy()
        X = clean[REGRESSION_FEATURES].values
        y_reg = clean["Inventory_Value"].values
        return clean, X, y_reg

    def _prepare_arrays(self, df: pd.DataFrame):
        """Return (clean_df, X, y_class, y_reg) for clustering/DR tasks."""
        clean = df.dropna(subset=FEATURE_COLS + ["Stock_Status_Enc"]).copy()
        X = clean[FEATURE_COLS].values
        y_class = clean["Stock_Status_Enc"].values
        y_reg = clean["Inventory_Value"].values
        return clean, X, y_class, y_reg

    # ------------------------------------------------------------------
    # run_all — orchestrator
    # ------------------------------------------------------------------

    def run_all(self, df: pd.DataFrame) -> dict:
        """Run all ML analyses, return dict with results and chart_paths."""
        logger.info("ML analysis started")
        df_ml = self.enrich(df)
        self.chart_paths = []

        path15 = self.plot_classification(df_ml)
        path16 = self.plot_feature_importance(df_ml)
        path17 = self.plot_regression_analysis(df_ml)
        path18 = self.plot_clustering(df_ml)
        path19 = self.plot_dimensionality_reduction(df_ml)
        path20 = self.plot_anomaly_detection(df_ml)
        path21 = self.plot_genetic_algorithm(df_ml)
        path22 = self.plot_algorithm_overview(df_ml)

        self.chart_paths = [path15, path16, path17, path18, path19, path20, path21, path22]
        logger.info("ML analysis complete — %d charts generated", len(self.chart_paths))
        return {"results": self.results, "chart_paths": self.chart_paths}

    # ------------------------------------------------------------------
    # Chart 15 — Classification comparison
    # ------------------------------------------------------------------

    def plot_classification(self, df: pd.DataFrame) -> str:
        """Chart 15: Compare 9 classifiers on Stock_Status prediction."""
        logger.info("chart_15: classification comparison")
        clean, X, y_class = self._prepare_classification_arrays(df)

        # Split BEFORE scaling to prevent data leakage
        class_counts = pd.Series(y_class).value_counts()
        if class_counts.min() < 5:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_class, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_class, test_size=0.2, random_state=42, stratify=y_class
            )

        # Fit scaler only on training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(kernel="rbf", probability=True, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "Naive Bayes": GaussianNB(),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
            "MLP (ANN)": MLPClassifier(
                hidden_layer_sizes=(128, 64), max_iter=300,
                random_state=42, early_stopping=True,
            ),
        }

        clf_results = {}
        trained_models = {}
        for name, clf in classifiers.items():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                # Use sklearn Pipeline for CV to prevent scaler leakage
                # CV only on training data to avoid test-set contamination
                pipe = SKPipeline([("scaler", StandardScaler()), ("clf", clf)])
                cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")

                # Train on scaled train set for test accuracy
                clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred)

            clf_results[name] = {
                "accuracy": acc,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
                "y_pred": y_pred,
            }
            trained_models[name] = clf
            logger.debug("  %s — acc=%.4f CV=%.4f±%.4f", name, acc, cv_scores.mean(), cv_scores.std())

        # Store for use by plot_feature_importance
        self._clf_results = clf_results
        self._trained_models = trained_models
        self._y_class = y_class
        self._y_test_cls = y_test
        self._scaler_cls = scaler
        self._classification_features = CLASSIFICATION_FEATURES
        self.results["classification"] = {k: {kk: vv for kk, vv in v.items() if kk != "y_pred"}
                                           for k, v in clf_results.items()}

        # --- Plot ---
        names = list(clf_results.keys())
        accs = [clf_results[n]["accuracy"] for n in names]
        cv_means = [clf_results[n]["cv_mean"] for n in names]
        cv_stds = [clf_results[n]["cv_std"] for n in names]
        sorted_idx = np.argsort(accs)[::-1]
        names_s = [names[i] for i in sorted_idx]
        accs_s = [accs[i] for i in sorted_idx]
        cv_s = [cv_means[i] for i in sorted_idx]
        cvstd_s = [cv_stds[i] for i in sorted_idx]

        fig = plt.figure(figsize=(24, 16), facecolor=CHART_BG_COLOR)
        fig.suptitle(
            "Supervised Learning — Classification of Stock Status (9 Classifiers)",
            fontsize=18, fontweight="bold", y=0.98, color=CHART_TEXT_COLOR,
        )
        gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30,
                      left=0.06, right=0.96, top=0.92, bottom=0.06)

        # (1) Test accuracy bar
        ax1 = fig.add_subplot(gs[0, 0])
        bar_colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(names_s)))
        bars = ax1.barh(range(len(names_s)), accs_s, color=bar_colors[::-1], edgecolor="white")
        ax1.set_yticks(range(len(names_s)))
        ax1.set_yticklabels(names_s, fontsize=8.5)
        for bar, acc in zip(bars, accs_s, strict=False):
            ax1.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                     f"{acc:.4f}", va="center", fontsize=8, fontweight="bold",
                     color=CHART_TEXT_COLOR)
        ax1.set_xlabel("Test Accuracy")
        ax1.set_title("Test Accuracy Ranking", fontsize=13, fontweight="bold", color=CHART_TEXT_COLOR)
        ax1.set_xlim(max(0, min(accs_s) - 0.06), 1.01)
        ax1.invert_yaxis()
        ax1.set_facecolor(CHART_BG_COLOR)

        # (2) 5-fold CV with error bars
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.barh(range(len(names_s)), cv_s, xerr=cvstd_s,
                 color="#3498DB", alpha=0.75, edgecolor="white", capsize=4)
        ax2.set_yticks(range(len(names_s)))
        ax2.set_yticklabels(names_s, fontsize=8.5)
        for i, (cv, std) in enumerate(zip(cv_s, cvstd_s, strict=False)):
            ax2.text(cv + std + 0.004, i, f"{cv:.4f}±{std:.4f}",
                     va="center", fontsize=7.5, fontweight="bold", color=CHART_TEXT_COLOR)
        ax2.set_xlabel("5-Fold CV Accuracy")
        ax2.set_title("Cross-Validation Scores", fontsize=13, fontweight="bold", color=CHART_TEXT_COLOR)
        ax2.set_xlim(max(0, min(cv_s) - 0.06), 1.01)
        ax2.invert_yaxis()
        ax2.set_facecolor(CHART_BG_COLOR)

        # (3) Test vs CV scatter (overfitting check)
        ax3 = fig.add_subplot(gs[0, 2])
        scatter_colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
        for i, name in enumerate(names):
            ax3.scatter(clf_results[name]["accuracy"], clf_results[name]["cv_mean"],
                        s=110, color=scatter_colors[i], zorder=5,
                        edgecolors=CHART_TEXT_COLOR, linewidth=1)
            ax3.annotate(name.split(" ")[0], (clf_results[name]["accuracy"], clf_results[name]["cv_mean"]),
                         fontsize=7, ha="center", va="bottom", xytext=(0, 6),
                         textcoords="offset points", color=CHART_TEXT_COLOR)
        all_vals = accs + cv_means
        lims = [max(0, min(all_vals) - 0.03), min(1.0, max(all_vals) + 0.03)]
        ax3.plot(lims, lims, "r--", lw=1.2, alpha=0.6, label="y=x (no overfit)")
        ax3.set_xlabel("Test Accuracy")
        ax3.set_ylabel("CV Accuracy")
        ax3.set_title("Overfitting Check: Test vs CV", fontsize=13, fontweight="bold", color=CHART_TEXT_COLOR)
        ax3.legend(fontsize=8)
        ax3.set_facecolor(CHART_BG_COLOR)

        # (4-6) Top-3 confusion matrices
        class_names = self._le_status.classes_
        top3 = names_s[:3]
        for idx, name in enumerate(top3):
            ax = fig.add_subplot(gs[1, idx])
            cm = confusion_matrix(y_test, clf_results[name]["y_pred"])
            cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=class_names, yticklabels=class_names,
                        linewidths=0.5, cbar_kws={"shrink": 0.7})
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    ax.text(j + 0.5, i + 0.72, f"({cm_pct[i, j]:.1f}%)",
                            ha="center", va="center", fontsize=7, color="gray")
            ax.set_xlabel("Predicted", color=CHART_TEXT_COLOR)
            ax.set_ylabel("Actual", color=CHART_TEXT_COLOR)
            ax.set_title(f"{name}\nAccuracy: {clf_results[name]['accuracy']:.4f}",
                         fontsize=11, fontweight="bold", color=CHART_TEXT_COLOR)
            ax.set_facecolor(CHART_BG_COLOR)

        save_path = str(self.output_dir / "chart_15_classification_comparison.png")
        plt.savefig(save_path, dpi=CHART_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Chart 16 — Feature importance
    # ------------------------------------------------------------------

    def plot_feature_importance(self, df: pd.DataFrame) -> str:
        """Chart 16: RF, GB, DT feature importances + LR coefficient magnitudes."""
        logger.info("chart_16: feature importance")
        clean, X, y_class = self._prepare_classification_arrays(df)

        # Re-train fresh models if classification wasn't run yet
        if not hasattr(self, "_trained_models"):
            class_counts = pd.Series(y_class).value_counts()
            if class_counts.min() < 5:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_class, test_size=0.2, random_state=42
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_class, test_size=0.2, random_state=42, stratify=y_class
                )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            models = {
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
                "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            }
            for m in models.values():
                m.fit(X_train_scaled, y_train)
            self._trained_models = models

        feature_labels = [
            "Unit Cost", "Curr. Stock", "Daily Demand", "Safety Stock",
            "Lead Time", "Demand Intens.", "Category", "Vendor",
        ]

        fig, axes = plt.subplots(2, 2, figsize=(22, 14), facecolor=CHART_BG_COLOR)
        fig.suptitle(
            "Feature Importance — Which Factors Drive Stock Status?",
            fontsize=18, fontweight="bold", y=1.01, color=CHART_TEXT_COLOR,
        )

        importance_specs = [
            ("Random Forest",       "feature_importances_", "#27AE60", axes[0, 0]),
            ("Gradient Boosting",   "feature_importances_", "#E74C3C", axes[0, 1]),
            ("Decision Tree",       "feature_importances_", "#3498DB", axes[1, 0]),
        ]

        for name, attr, color, ax in importance_specs:
            ax.set_facecolor(CHART_BG_COLOR)
            model = self._trained_models.get(name)
            if model is None:
                ax.axis("off")
                continue
            imp = getattr(model, attr)
            s_idx = np.argsort(imp)
            ax.barh([feature_labels[i] for i in s_idx], imp[s_idx],
                    color=color, alpha=0.82, edgecolor="white")
            for i, v in enumerate(imp[s_idx]):
                ax.text(v + imp.max() * 0.01, i, f"{v:.4f}",
                        va="center", fontsize=8, fontweight="bold", color=CHART_TEXT_COLOR)
            ax.set_xlabel("Importance")
            ax.set_title(f"{name} — Feature Importance", fontsize=12,
                         fontweight="bold", color=CHART_TEXT_COLOR)

        # Logistic Regression coefficients
        ax4 = axes[1, 1]
        ax4.set_facecolor(CHART_BG_COLOR)
        lr = self._trained_models.get("Logistic Regression")
        if lr is not None:
            coef_abs = np.abs(lr.coef_).mean(axis=0)
            s_idx = np.argsort(coef_abs)
            ax4.barh([feature_labels[i] for i in s_idx], coef_abs[s_idx],
                     color="#8E44AD", alpha=0.82, edgecolor="white")
            for i, v in enumerate(coef_abs[s_idx]):
                ax4.text(v + coef_abs.max() * 0.01, i, f"{v:.4f}",
                         va="center", fontsize=8, fontweight="bold", color=CHART_TEXT_COLOR)
            ax4.set_xlabel("|Coefficient| (avg across classes)")
            ax4.set_title("Logistic Regression — Coefficient Magnitude",
                          fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)

        plt.tight_layout()
        save_path = str(self.output_dir / "chart_16_feature_importance.png")
        plt.savefig(save_path, dpi=CHART_DPI, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        logger.info("Saved %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Chart 17 — Regression analysis
    # ------------------------------------------------------------------

    def plot_regression_analysis(self, df: pd.DataFrame) -> str:
        """Chart 17: LinearRegression, RF Regressor, GB Regressor predicting Inventory_Value."""
        logger.info("chart_17: regression analysis")
        clean, X, y_reg = self._prepare_regression_arrays(df)

        # Split BEFORE scaling to prevent data leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        regressors = {
            "Linear Regression": LinearRegression(),
            "RF Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
            "GB Regressor": GradientBoostingRegressor(n_estimators=100, random_state=42),
        }

        colors = ["#2E86C1", "#27AE60", "#E74C3C"]
        reg_results = {}
        for (name, reg), color in zip(regressors.items(), colors, strict=False):
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            reg_results[name] = {"rmse": rmse, "r2": r2, "y_pred": y_pred, "color": color}
            logger.debug("  %s — R2=%.4f RMSE=%.2f", name, r2, rmse)

        self.results["regression"] = {k: {"r2": v["r2"], "rmse": v["rmse"]} for k, v in reg_results.items()}

        fig = plt.figure(figsize=(24, 14), facecolor=CHART_BG_COLOR)
        fig.suptitle(
            "Regression Analysis — Predicting Inventory Value (Linear, RF, GB)",
            fontsize=18, fontweight="bold", y=0.98, color=CHART_TEXT_COLOR,
        )
        gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30,
                      left=0.06, right=0.96, top=0.92, bottom=0.06)

        rng = np.random.RandomState(42)

        # Row 1: Actual vs Predicted scatter for each model
        for idx, (name, res) in enumerate(reg_results.items()):
            ax = fig.add_subplot(gs[0, idx])
            ax.set_facecolor(CHART_BG_COLOR)
            sample = rng.choice(len(y_test), min(2000, len(y_test)), replace=False)
            ax.scatter(y_test[sample], res["y_pred"][sample],
                       s=5, alpha=0.3, color=res["color"], edgecolors="none")
            max_val = max(y_test.max(), res["y_pred"].max())
            ax.plot([0, max_val], [0, max_val], "r--", lw=1.5, label="Perfect")
            ax.set_xlabel("Actual Inventory Value ($)", color=CHART_TEXT_COLOR)
            ax.set_ylabel("Predicted ($)", color=CHART_TEXT_COLOR)
            ax.set_title(f"{name}\nR²={res['r2']:.4f}   RMSE=${res['rmse']:,.0f}",
                         fontsize=11, fontweight="bold", color=CHART_TEXT_COLOR)
            ax.legend(fontsize=8)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

        # Row 2 left: R² comparison
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_facecolor(CHART_BG_COLOR)
        r2_vals = [reg_results[n]["r2"] for n in reg_results]
        reg_names = list(reg_results.keys())
        reg_colors = [reg_results[n]["color"] for n in reg_names]
        bars = ax4.barh(reg_names, r2_vals, color=reg_colors, alpha=0.82, edgecolor="white")
        for bar, r2 in zip(bars, r2_vals, strict=False):
            ax4.text(bar.get_width() + 0.008, bar.get_y() + bar.get_height() / 2,
                     f"{r2:.4f}", va="center", fontsize=10, fontweight="bold",
                     color=CHART_TEXT_COLOR)
        ax4.set_xlabel("R² Score")
        ax4.set_title("Model Comparison — R²", fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax4.invert_yaxis()

        # Row 2 centre: RMSE comparison
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_facecolor(CHART_BG_COLOR)
        rmse_vals = [reg_results[n]["rmse"] for n in reg_names]
        bars = ax5.barh(reg_names, rmse_vals, color=reg_colors, alpha=0.82, edgecolor="white")
        for bar, rmse in zip(bars, rmse_vals, strict=False):
            ax5.text(bar.get_width() + max(rmse_vals) * 0.02,
                     bar.get_y() + bar.get_height() / 2,
                     f"${rmse:,.0f}", va="center", fontsize=10, fontweight="bold",
                     color=CHART_TEXT_COLOR)
        ax5.set_xlabel("RMSE ($)")
        ax5.set_title("Model Comparison — RMSE (lower is better)",
                      fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax5.invert_yaxis()

        # Row 2 right: Residual distribution for best model
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor(CHART_BG_COLOR)
        best_name = max(reg_results, key=lambda n: reg_results[n]["r2"])
        residuals = y_test - reg_results[best_name]["y_pred"]
        ax6.hist(residuals, bins=60, density=True, alpha=0.6, color="#8E44AD", edgecolor="white")
        kde_x = np.linspace(residuals.min(), residuals.max(), 300)
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(residuals)
        ax6.plot(kde_x, kde(kde_x), color=CHART_TEXT_COLOR, lw=2, label="KDE")
        ax6.axvline(0, color="red", ls="--", lw=1.5, label="Zero")
        ax6.set_xlabel("Residual ($)", color=CHART_TEXT_COLOR)
        ax6.set_ylabel("Density", color=CHART_TEXT_COLOR)
        ax6.set_title(f"Residual Distribution — {best_name}",
                      fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax6.legend(fontsize=8)
        skew_val = pd.Series(residuals).skew()
        ax6.text(0.97, 0.95,
                 f"Mean: ${residuals.mean():,.0f}\nStd: ${residuals.std():,.0f}\nSkew: {skew_val:.2f}",
                 transform=ax6.transAxes, fontsize=9, va="top", ha="right",
                 family="monospace", color=CHART_TEXT_COLOR,
                 bbox=dict(boxstyle="round", fc="white", ec="#8E44AD"))

        save_path = str(self.output_dir / "chart_17_regression_prediction.png")
        plt.savefig(save_path, dpi=CHART_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Chart 18 — Clustering
    # ------------------------------------------------------------------

    def plot_clustering(self, df: pd.DataFrame) -> str:
        """Chart 18: Elbow, silhouette, K-Means++, Hierarchical, DBSCAN, cluster vs stock status."""
        logger.info("chart_18: clustering analysis")
        clean, X, _, _ = self._prepare_arrays(df)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca_2d = PCA(n_components=2, random_state=42)
        X_2d = pca_2d.fit_transform(X_scaled)

        fig = plt.figure(figsize=(24, 16), facecolor=CHART_BG_COLOR)
        fig.suptitle(
            "Unsupervised Learning — Clustering Analysis (K-Means++, Hierarchical, DBSCAN)",
            fontsize=16, fontweight="bold", y=0.98, color=CHART_TEXT_COLOR,
        )
        gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.28,
                      left=0.06, right=0.96, top=0.92, bottom=0.06)

        # (1) Elbow + silhouette
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(CHART_BG_COLOR)
        inertias, sil_scores = [], []
        K_range = range(2, 11)
        for k in K_range:
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, km.labels_, sample_size=3000, random_state=42))

        ax1.plot(K_range, inertias, "bo-", lw=2, label="Inertia (SSE)")
        ax1.set_xlabel("Number of Clusters (k)", color=CHART_TEXT_COLOR)
        ax1.set_ylabel("Inertia", color="#2E86C1")
        ax1.tick_params(axis="y", labelcolor="#2E86C1")

        ax1b = ax1.twinx()
        ax1b.plot(K_range, sil_scores, "r^--", lw=2, label="Silhouette Score")
        ax1b.set_ylabel("Silhouette Score", color="#E74C3C")
        ax1b.tick_params(axis="y", labelcolor="#E74C3C")

        best_k = list(K_range)[int(np.argmax(sil_scores))]
        ax1.axvline(best_k, color="gray", ls=":", lw=1.5, label=f"Best k={best_k}")
        ax1.set_title(f"Elbow Method + Silhouette (Best k={best_k})",
                      fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5, loc="center right")

        # (2) K-Means++ result
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(CHART_BG_COLOR)
        km_best = KMeans(n_clusters=best_k, init="k-means++", n_init=10, random_state=42)
        km_labels = km_best.fit_predict(X_scaled)
        sil_km = silhouette_score(X_scaled, km_labels, sample_size=3000, random_state=42)
        ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=km_labels, cmap="Set2",
                    s=5, alpha=0.4, edgecolors="none")
        centers_2d = pca_2d.transform(km_best.cluster_centers_)
        ax2.scatter(centers_2d[:, 0], centers_2d[:, 1], c="red", marker="X",
                    s=200, edgecolors="black", linewidth=2, zorder=10, label="Centroids")
        ax2.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)", color=CHART_TEXT_COLOR)
        ax2.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)", color=CHART_TEXT_COLOR)
        ax2.set_title(f"K-Means++ (k={best_k}) — Silhouette={sil_km:.3f}",
                      fontsize=11, fontweight="bold", color=CHART_TEXT_COLOR)
        ax2.legend(fontsize=8)

        # (3) Hierarchical clustering
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor(CHART_BG_COLOR)
        hc = AgglomerativeClustering(n_clusters=best_k)
        hc_labels = hc.fit_predict(X_scaled)
        sil_hc = silhouette_score(X_scaled, hc_labels, sample_size=3000, random_state=42)
        ax3.scatter(X_2d[:, 0], X_2d[:, 1], c=hc_labels, cmap="Set2",
                    s=5, alpha=0.4, edgecolors="none")
        ax3.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)", color=CHART_TEXT_COLOR)
        ax3.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)", color=CHART_TEXT_COLOR)
        ax3.set_title(f"Hierarchical Clustering (k={best_k}) — Silhouette={sil_hc:.3f}",
                      fontsize=11, fontweight="bold", color=CHART_TEXT_COLOR)

        # (4) DBSCAN
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_facecolor(CHART_BG_COLOR)
        db = DBSCAN(eps=2.5, min_samples=10)
        db_labels = db.fit_predict(X_scaled)
        n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
        n_noise = int((db_labels == -1).sum())
        noise_mask = db_labels == -1
        ax4.scatter(X_2d[~noise_mask, 0], X_2d[~noise_mask, 1],
                    c=db_labels[~noise_mask], cmap="Set2", s=5, alpha=0.4, edgecolors="none")
        ax4.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1],
                    c="red", s=8, alpha=0.6, marker="x", label=f"Noise ({n_noise:,})")
        ax4.set_xlabel("PC1", color=CHART_TEXT_COLOR)
        ax4.set_ylabel("PC2", color=CHART_TEXT_COLOR)
        ax4.set_title(f"DBSCAN — {n_clusters_db} clusters, {n_noise:,} noise pts",
                      fontsize=11, fontweight="bold", color=CHART_TEXT_COLOR)
        ax4.legend(fontsize=8)

        # (5) Cluster vs actual Stock_Status
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_facecolor(CHART_BG_COLOR)
        status_vals = clean["Stock_Status"].values
        ct = pd.crosstab(
            pd.Series(km_labels, name="Cluster"),
            pd.Series(status_vals, name="Stock_Status"),
            normalize="index",
        ) * 100
        for col in ["Normal Stock", "Low Stock", "Out of Stock"]:
            if col not in ct.columns:
                ct[col] = 0.0
        ct = ct[["Normal Stock", "Low Stock", "Out of Stock"]]
        ct.plot(kind="bar", stacked=True, ax=ax5,
                color=["#27AE60", "#F39C12", "#E74C3C"], edgecolor="white")
        ax5.set_xlabel("K-Means++ Cluster", color=CHART_TEXT_COLOR)
        ax5.set_ylabel("% Stock Status", color=CHART_TEXT_COLOR)
        ax5.set_title("K-Means++ Clusters vs Actual Stock Status",
                      fontsize=11, fontweight="bold", color=CHART_TEXT_COLOR)
        ax5.legend(fontsize=8)
        ax5.tick_params(axis="x", rotation=0)

        # (6) Cluster feature profiles
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor(CHART_BG_COLOR)
        profiles = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
        profiles["Cluster"] = km_labels
        profile_mean = profiles.groupby("Cluster").mean()
        # Labels must match FEATURE_COLS order:
        # Unit_Cost, Current_Stock, Daily_Demand_Est, Safety_Stock_Target,
        # Lead_Time_Days, Demand_Intensity, Category_Enc, Vendor_Enc,
        # Reorder_Point, DSI, Stock_Coverage_Ratio
        short_labels = ["Cost", "Stock", "Demand", "Safety", "LeadT",
                        "DemInt", "Cat", "Vend", "ReOrd", "DSI", "CovR"]
        x_pos = np.arange(len(short_labels))
        width = 0.8 / best_k
        cluster_colors = sns.color_palette("Set2", best_k)
        for i in range(best_k):
            ax6.bar(x_pos + i * width, profile_mean.iloc[i].values, width,
                    label=f"Cluster {i}", color=cluster_colors[i], edgecolor="white")
        ax6.set_xticks(x_pos + width * (best_k - 1) / 2)
        ax6.set_xticklabels(short_labels, fontsize=7, rotation=35, ha="right")
        ax6.set_ylabel("Standardized Mean", color=CHART_TEXT_COLOR)
        ax6.set_title("Cluster Feature Profiles (Standardized)",
                      fontsize=11, fontweight="bold", color=CHART_TEXT_COLOR)
        ax6.legend(fontsize=7, loc="upper right", ncol=2)
        ax6.axhline(0, color="gray", ls="--", lw=0.5)

        self.results["clustering"] = {
            "best_k": best_k,
            "kmeans_silhouette": round(sil_km, 4),
            "hierarchical_silhouette": round(sil_hc, 4),
            "dbscan_clusters": n_clusters_db,
            "dbscan_noise": n_noise,
        }

        save_path = str(self.output_dir / "chart_18_clustering_analysis.png")
        plt.savefig(save_path, dpi=CHART_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Chart 19 — Dimensionality reduction
    # ------------------------------------------------------------------

    def plot_dimensionality_reduction(self, df: pd.DataFrame) -> str:
        """Chart 19: PCA explained variance, PCA 2D, t-SNE 2D projections."""
        logger.info("chart_19: dimensionality reduction")
        clean, X, _, _ = self._prepare_arrays(df)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        status_colors = {"Normal Stock": "#27AE60", "Low Stock": "#F39C12", "Out of Stock": "#E74C3C"}
        statuses = clean["Stock_Status"].values
        categories = clean["Category"].values
        cat_unique = sorted(clean["Category"].unique())
        cat_palette = sns.color_palette("Set2", len(cat_unique))

        fig = plt.figure(figsize=(24, 14), facecolor=CHART_BG_COLOR)
        fig.suptitle(
            "Dimensionality Reduction — PCA & t-SNE Visualization",
            fontsize=18, fontweight="bold", y=0.98, color=CHART_TEXT_COLOR,
        )
        gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.28,
                      left=0.06, right=0.96, top=0.92, bottom=0.06)

        # (1) PCA explained variance
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(CHART_BG_COLOR)
        pca_full = PCA()
        pca_full.fit(X_scaled)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_) * 100
        n_components = len(cum_var)
        ax1.bar(range(1, n_components + 1), pca_full.explained_variance_ratio_ * 100,
                alpha=0.6, color="#3498DB", label="Individual")
        ax1.plot(range(1, n_components + 1), cum_var, "ro-", lw=2, label="Cumulative")
        ax1.axhline(90, color="gray", ls="--", lw=1, alpha=0.6)
        n_90 = int(np.argmax(cum_var >= 90)) + 1
        ax1.axvline(n_90, color="#E74C3C", ls=":", lw=1.5)
        ax1.text(n_90 + 0.2, 50, f"{n_90} PCs\nfor 90%", fontsize=8, color="#E74C3C")
        ax1.set_xlabel("Principal Component", color=CHART_TEXT_COLOR)
        ax1.set_ylabel("Explained Variance (%)", color=CHART_TEXT_COLOR)
        ax1.set_title(f"PCA Explained Variance ({n_90} PCs → 90%)",
                      fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax1.legend(fontsize=8)

        # (2) PCA 2D — Stock Status
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(CHART_BG_COLOR)
        pca_2d = PCA(n_components=2)
        X_pca = pca_2d.fit_transform(X_scaled)
        for status, color in status_colors.items():
            mask = statuses == status
            ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], s=5, alpha=0.3,
                        color=color, label=status, edgecolors="none")
        ax2.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)", color=CHART_TEXT_COLOR)
        ax2.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)", color=CHART_TEXT_COLOR)
        ax2.set_title("PCA 2D — Stock Status", fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax2.legend(fontsize=8, markerscale=3)

        # (3) PCA 2D — Category
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor(CHART_BG_COLOR)
        for i, cat in enumerate(cat_unique):
            mask = categories == cat
            ax3.scatter(X_pca[mask, 0], X_pca[mask, 1], s=5, alpha=0.3,
                        color=cat_palette[i], label=cat, edgecolors="none")
        ax3.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)", color=CHART_TEXT_COLOR)
        ax3.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)", color=CHART_TEXT_COLOR)
        ax3.set_title("PCA 2D — Category", fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax3.legend(fontsize=7, markerscale=3, ncol=2)

        # t-SNE subsample
        sample_size = min(5000, len(X_scaled))
        rng = np.random.RandomState(42)
        idx_s = rng.choice(len(X_scaled), sample_size, replace=False)
        X_sample = X_scaled[idx_s]
        status_sample = statuses[idx_s]
        cat_sample = categories[idx_s]

        logger.info("  Running t-SNE (n=%d)…", sample_size)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        X_tsne = tsne.fit_transform(X_sample)

        # (4) t-SNE — Stock Status
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_facecolor(CHART_BG_COLOR)
        for status, color in status_colors.items():
            mask = status_sample == status
            ax4.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=5, alpha=0.4,
                        color=color, label=status, edgecolors="none")
        ax4.set_xlabel("t-SNE 1", color=CHART_TEXT_COLOR)
        ax4.set_ylabel("t-SNE 2", color=CHART_TEXT_COLOR)
        ax4.set_title(f"t-SNE 2D — Stock Status (n={sample_size:,})",
                      fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax4.legend(fontsize=8, markerscale=3)

        # (5) t-SNE — Category
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_facecolor(CHART_BG_COLOR)
        for i, cat in enumerate(cat_unique):
            mask = cat_sample == cat
            ax5.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=5, alpha=0.4,
                        color=cat_palette[i], label=cat, edgecolors="none")
        ax5.set_xlabel("t-SNE 1", color=CHART_TEXT_COLOR)
        ax5.set_ylabel("t-SNE 2", color=CHART_TEXT_COLOR)
        ax5.set_title("t-SNE 2D — Category", fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax5.legend(fontsize=7, markerscale=3, ncol=2)

        # (6) PCA loading plot
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor(CHART_BG_COLOR)
        loadings = pca_2d.components_.T  # shape (n_features, 2)
        feat_short = ["Cost", "Stock", "Demand", "Safety", "LeadT",
                      "DemInt", "Cat", "Vend", "ReOrd", "DSI", "CovR"]
        for i, (lx, ly) in enumerate(loadings):
            ax6.arrow(0, 0, lx * 3, ly * 3, head_width=0.08, head_length=0.05,
                      fc="#E74C3C", ec="#E74C3C", alpha=0.75)
            ax6.text(lx * 3.4, ly * 3.4, feat_short[i], fontsize=8.5,
                     fontweight="bold", ha="center", va="center", color=CHART_TEXT_COLOR)
        circle = plt.Circle((0, 0), 1, fill=False, color="gray", ls="--", lw=1)
        ax6.add_patch(circle)
        ax6.set_xlim(-4.2, 4.2)
        ax6.set_ylim(-4.2, 4.2)
        ax6.axhline(0, color="gray", lw=0.5)
        ax6.axvline(0, color="gray", lw=0.5)
        ax6.set_xlabel("PC1 Loading", color=CHART_TEXT_COLOR)
        ax6.set_ylabel("PC2 Loading", color=CHART_TEXT_COLOR)
        ax6.set_title("PCA Loading Plot — Feature Contributions",
                      fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax6.set_aspect("equal")

        self.results["dim_reduction"] = {
            "pca_components_for_90pct": n_90,
            "pc1_variance": round(float(pca_2d.explained_variance_ratio_[0]), 4),
            "pc2_variance": round(float(pca_2d.explained_variance_ratio_[1]), 4),
        }

        save_path = str(self.output_dir / "chart_19_pca_tsne.png")
        plt.savefig(save_path, dpi=CHART_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Chart 20 — Anomaly detection
    # ------------------------------------------------------------------

    def plot_anomaly_detection(self, df: pd.DataFrame) -> str:
        """Chart 20: Isolation Forest + MLP Autoencoder anomaly detection."""
        logger.info("chart_20: anomaly detection")
        clean, X, _, _ = self._prepare_arrays(df)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        pca_2d = PCA(n_components=2)
        X_2d = pca_2d.fit_transform(X_scaled)

        fig = plt.figure(figsize=(24, 14), facecolor=CHART_BG_COLOR)
        fig.suptitle(
            "Anomaly Detection — Isolation Forest & Autoencoder (80/20 holdout evaluation)",
            fontsize=18, fontweight="bold", y=0.98, color=CHART_TEXT_COLOR,
        )
        gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30,
                      left=0.06, right=0.96, top=0.92, bottom=0.06)

        # --- Train / Test Split for fair evaluation ---
        X_train, X_test, idx_train, idx_test = train_test_split(
            X_scaled, np.arange(len(X_scaled)),
            test_size=0.2, random_state=42,
        )
        X_2d_test = X_2d[idx_test]   # PCA coords for test set only

        # --- Isolation Forest (train on train set, evaluate on test set) ---
        iso = IsolationForest(contamination=0.05, random_state=42, n_estimators=200)
        iso.fit(X_train)
        iso_labels = iso.predict(X_test)   # 1=normal, -1=anomaly
        iso_anomaly = iso_labels == -1
        iso_scores = iso.score_samples(X_test)

        # (1) Isolation Forest scatter (test set only)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(CHART_BG_COLOR)
        ax1.scatter(X_2d_test[~iso_anomaly, 0], X_2d_test[~iso_anomaly, 1],
                    s=3, alpha=0.2, color="#3498DB",
                    label=f"Normal ({(~iso_anomaly).sum():,})", edgecolors="none")
        ax1.scatter(X_2d_test[iso_anomaly, 0], X_2d_test[iso_anomaly, 1],
                    s=15, alpha=0.7, color="#E74C3C", marker="x",
                    label=f"Anomaly ({iso_anomaly.sum():,})")
        ax1.set_xlabel("PC1", color=CHART_TEXT_COLOR)
        ax1.set_ylabel("PC2", color=CHART_TEXT_COLOR)
        ax1.set_title(f"Isolation Forest (test set) — {iso_anomaly.sum():,} anomalies detected",
                      fontsize=11, fontweight="bold", color=CHART_TEXT_COLOR)
        ax1.legend(fontsize=8)

        # (2) Isolation Forest score distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(CHART_BG_COLOR)
        ax2.hist(iso_scores[~iso_anomaly], bins=60, alpha=0.6, color="#3498DB",
                 density=True, label="Normal")
        ax2.hist(iso_scores[iso_anomaly], bins=30, alpha=0.6, color="#E74C3C",
                 density=True, label="Anomaly")
        thresh_iso = np.percentile(iso_scores, 5)
        ax2.axvline(thresh_iso, color="red", ls="--", lw=1.5, label=f"5th pct ({thresh_iso:.3f})")
        ax2.set_xlabel("Anomaly Score (lower = more anomalous)", color=CHART_TEXT_COLOR)
        ax2.set_ylabel("Density", color=CHART_TEXT_COLOR)
        ax2.set_title("Isolation Forest — Score Distribution",
                      fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax2.legend(fontsize=8)

        # (3) Feature comparison anomaly vs normal (test set)
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor(CHART_BG_COLOR)
        feat_short = ["Cost", "Stock", "Demand", "Safety", "LeadT",
                      "DemInt", "Cat", "Vend", "ReOrd", "DSI", "CovR"]
        X_test_orig = X[idx_test]
        df_feat = pd.DataFrame(X_test_orig, columns=FEATURE_COLS)
        df_feat["Anomaly"] = iso_anomaly
        normal_means = df_feat[~df_feat["Anomaly"]][FEATURE_COLS].mean()
        anomaly_means = df_feat[df_feat["Anomaly"]][FEATURE_COLS].mean()
        ratio = anomaly_means / normal_means.replace(0, 1e-9)
        s_idx = np.argsort(np.abs(ratio.values - 1))[::-1]
        ratio_sorted = ratio.values[s_idx]
        labels_sorted = [feat_short[i] for i in s_idx]
        bar_colors = ["#E74C3C" if abs(r - 1) > 0.3 else "#3498DB" for r in ratio_sorted]
        bars = ax3.barh(labels_sorted, ratio_sorted, color=bar_colors, alpha=0.82, edgecolor="white")
        ax3.axvline(1.0, color="gray", ls="--", lw=1.5)
        for bar, r in zip(bars, ratio_sorted, strict=False):
            ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                     f"{r:.2f}x", va="center", fontsize=8, fontweight="bold",
                     color=CHART_TEXT_COLOR)
        ax3.set_xlabel("Anomaly / Normal Mean Ratio", color=CHART_TEXT_COLOR)
        ax3.set_title("Feature Comparison: Anomaly vs Normal",
                      fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax3.invert_yaxis()

        # --- Autoencoder (train on train set, evaluate on test set) ---
        ae = MLPRegressor(hidden_layer_sizes=(32, 8, 32), max_iter=200,
                          random_state=42, early_stopping=True, activation="relu")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            ae.fit(X_train, X_train)
        X_recon = ae.predict(X_test)
        recon_error = np.mean((X_test - X_recon) ** 2, axis=1)
        threshold_ae = float(np.percentile(recon_error, 95))
        ae_anomaly = recon_error > threshold_ae

        # (4) Autoencoder scatter (test set only)
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_facecolor(CHART_BG_COLOR)
        ax4.scatter(X_2d_test[~ae_anomaly, 0], X_2d_test[~ae_anomaly, 1],
                    s=3, alpha=0.2, color="#27AE60",
                    label=f"Normal ({(~ae_anomaly).sum():,})", edgecolors="none")
        ax4.scatter(X_2d_test[ae_anomaly, 0], X_2d_test[ae_anomaly, 1],
                    s=15, alpha=0.7, color="#E74C3C", marker="x",
                    label=f"Anomaly ({ae_anomaly.sum():,})")
        ax4.set_xlabel("PC1", color=CHART_TEXT_COLOR)
        ax4.set_ylabel("PC2", color=CHART_TEXT_COLOR)
        ax4.set_title(f"Autoencoder (test set) — {ae_anomaly.sum():,} anomalies (MSE > P95)",
                      fontsize=11, fontweight="bold", color=CHART_TEXT_COLOR)
        ax4.legend(fontsize=8)

        # (5) Autoencoder reconstruction error distribution
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_facecolor(CHART_BG_COLOR)
        ax5.hist(recon_error[~ae_anomaly], bins=60, alpha=0.6, color="#27AE60",
                 density=True, label="Normal")
        ax5.hist(recon_error[ae_anomaly], bins=30, alpha=0.6, color="#E74C3C",
                 density=True, label="Anomaly")
        ax5.axvline(threshold_ae, color="red", ls="--", lw=1.5,
                    label=f"Threshold P95={threshold_ae:.3f}")
        ax5.set_xlabel("Reconstruction Error (MSE)", color=CHART_TEXT_COLOR)
        ax5.set_ylabel("Density", color=CHART_TEXT_COLOR)
        ax5.set_title("Autoencoder — Reconstruction Error Distribution",
                      fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax5.legend(fontsize=8)

        # (6) Method agreement bar chart
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor(CHART_BG_COLOR)
        both = iso_anomaly & ae_anomaly
        only_iso = iso_anomaly & ~ae_anomaly
        only_ae = ~iso_anomaly & ae_anomaly
        neither = ~iso_anomaly & ~ae_anomaly
        venn_data = {
            "Both Methods": int(both.sum()),
            "Only Isolation Forest": int(only_iso.sum()),
            "Only Autoencoder": int(only_ae.sum()),
            "Normal (Both)": int(neither.sum()),
        }
        bar_cols = ["#8B0000", "#E74C3C", "#F39C12", "#27AE60"]
        bars = ax6.barh(list(venn_data.keys()), list(venn_data.values()),
                        color=bar_cols, alpha=0.82, edgecolor="white")
        n_total = len(X_test)
        for bar, val in zip(bars, venn_data.values(), strict=False):
            pct = val / n_total * 100
            ax6.text(bar.get_width() + max(venn_data.values()) * 0.02,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:,} ({pct:.1f}%)", va="center", fontsize=9,
                     fontweight="bold", color=CHART_TEXT_COLOR)
        ax6.set_xlabel("Number of SKUs", color=CHART_TEXT_COLOR)
        ax6.set_title("Anomaly Detection Method Agreement",
                      fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax6.invert_yaxis()

        agreement = float((iso_anomaly == ae_anomaly).mean() * 100)
        fig.text(
            0.5, 0.005,
            (f"Agreement: {agreement:.1f}%  |  "
             f"Isolation Forest: {iso_anomaly.sum():,}  |  "
             f"Autoencoder: {ae_anomaly.sum():,}  |  "
             f"Both: {both.sum():,}"),
            ha="center", fontsize=10, color=CHART_TEXT_COLOR,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FADBD8", ec="#E74C3C"),
        )

        self.results["anomaly"] = {
            "isolation_forest_anomalies": int(iso_anomaly.sum()),
            "autoencoder_anomalies": int(ae_anomaly.sum()),
            "agreement_pct": round(agreement, 2),
            "both_methods": int(both.sum()),
        }

        save_path = str(self.output_dir / "chart_20_anomaly_detection.png")
        plt.savefig(save_path, dpi=CHART_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Chart 21 — Genetic Algorithm
    # ------------------------------------------------------------------

    def plot_genetic_algorithm(self, df: pd.DataFrame) -> str:
        """Chart 21: GA optimises safety stock multipliers per category."""
        logger.info("chart_21: genetic algorithm")
        clean, _, _, _ = self._prepare_arrays(df)

        rng = np.random.RandomState(42)
        categories = sorted(clean["Category"].unique())
        n_cats = len(categories)

        # Category-level aggregates
        cat_stats = {}
        for cat in categories:
            sub = clean[clean["Category"] == cat]
            cat_stats[cat] = {
                "demand_mean": float(sub["Daily_Demand_Est"].mean()),
                "cost_mean": float(sub["Unit_Cost"].mean()),
                "safety_mean": float(sub["Safety_Stock_Target"].mean()),
                "lead_mean": float(sub["Lead_Time_Days"].mean()),
                "sku_count": len(sub),
            }

        def fitness(chromosome):
            total = 0.0
            for i, cat in enumerate(categories):
                s = cat_stats[cat]
                ss = s["safety_mean"] * chromosome[i]
                holding_cost = ss * s["cost_mean"] * 0.25 * s["sku_count"]
                lt_demand = s["demand_mean"] * s["lead_mean"]
                stockout_cost = max(0.0, (lt_demand * 0.5 - ss)) * s["cost_mean"] * 2 * s["sku_count"]
                total += holding_cost + stockout_cost
            return total

        # GA hyper-parameters (from config)
        POP_SIZE = GA_POPULATION
        N_GEN = GA_GENERATIONS
        MUT_RATE = GA_MUTATION_RATE
        CROSS_RATE = GA_CROSSOVER_RATE
        LOW, HIGH = 0.5, 3.0

        population = rng.uniform(LOW, HIGH, (POP_SIZE, n_cats))
        best_fitness_hist: list = []
        avg_fitness_hist: list = []
        best_chrom_hist: list = []

        for _gen in range(N_GEN):
            scores = np.array([fitness(ind) for ind in population])
            best_idx = int(np.argmin(scores))
            best_fitness_hist.append(float(scores[best_idx]))
            avg_fitness_hist.append(float(scores.mean()))
            best_chrom_hist.append(population[best_idx].copy())

            # Elitism + tournament selection
            new_pop = [population[best_idx].copy()]
            for _ in range(POP_SIZE - 1):
                i, j = rng.randint(0, POP_SIZE, 2)
                winner = population[i] if scores[i] < scores[j] else population[j]
                new_pop.append(winner.copy())

            # Single-point crossover
            for i in range(1, POP_SIZE - 1, 2):
                if rng.random() < CROSS_RATE:
                    pt = rng.randint(1, n_cats)
                    new_pop[i][pt:], new_pop[i + 1][pt:] = (
                        new_pop[i + 1][pt:].copy(),
                        new_pop[i][pt:].copy(),
                    )

            # Gaussian mutation
            for i in range(1, POP_SIZE):
                for j in range(n_cats):
                    if rng.random() < MUT_RATE:
                        new_pop[i][j] = float(
                            np.clip(new_pop[i][j] + rng.normal(0, 0.2), LOW, HIGH)
                        )

            population = np.array(new_pop)

        # Final evaluation
        final_scores = np.array([fitness(ind) for ind in population])
        best_solution = population[int(np.argmin(final_scores))]
        current_cost = fitness(np.ones(n_cats))
        optimal_cost = fitness(best_solution)
        savings = current_cost - optimal_cost

        self.results["genetic_algorithm"] = {
            "categories": categories,
            "best_multipliers": {cat: round(float(best_solution[i]), 4)
                                 for i, cat in enumerate(categories)},
            "current_cost": round(current_cost, 2),
            "optimal_cost": round(optimal_cost, 2),
            "savings": round(savings, 2),
            "savings_pct": round(savings / current_cost * 100, 2) if current_cost > 0 else 0.0,
        }

        # ---- Plotting ----
        fig = plt.figure(figsize=(24, 14), facecolor=CHART_BG_COLOR)
        fig.suptitle(
            f"Genetic Algorithm — Safety Stock Multiplier Optimisation by Category\n"
            f"(pop={POP_SIZE}, gen={N_GEN}, mut={MUT_RATE}, cross={CROSS_RATE})",
            fontsize=16, fontweight="bold", y=0.98, color=CHART_TEXT_COLOR,
        )
        gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30,
                      left=0.06, right=0.96, top=0.92, bottom=0.06)

        cat_colors = sns.color_palette("Set2", n_cats)

        # (1) Convergence curve
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(CHART_BG_COLOR)
        ax1.plot(best_fitness_hist, "b-", lw=2, label="Best Cost")
        ax1.plot(avg_fitness_hist, "r--", lw=1.5, alpha=0.7, label="Avg Cost")
        ax1.set_xlabel("Generation", color=CHART_TEXT_COLOR)
        ax1.set_ylabel("Total Cost ($)", color=CHART_TEXT_COLOR)
        ax1.set_title("GA Convergence Curve", fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
        ax1.legend(fontsize=8)
        ax1.text(0.97, 0.97,
                 f"Pop={POP_SIZE}\nGen={N_GEN}\nMut={MUT_RATE}\nCross={CROSS_RATE}",
                 transform=ax1.transAxes, fontsize=8, va="top", ha="right",
                 family="monospace", color=CHART_TEXT_COLOR,
                 bbox=dict(boxstyle="round", fc="white", ec="gray"))

        # (2) Optimal multipliers per category
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(CHART_BG_COLOR)
        bars = ax2.bar(categories, best_solution, color=cat_colors, edgecolor="white")
        ax2.axhline(1.0, color="red", ls="--", lw=1.5, label="Current baseline (1.0x)")
        for bar, val in zip(bars, best_solution, strict=False):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.03, f"{val:.2f}x",
                     ha="center", fontsize=9, fontweight="bold", color=CHART_TEXT_COLOR)
        ax2.set_ylabel("Safety Stock Multiplier", color=CHART_TEXT_COLOR)
        ax2.set_title("Optimal Safety Stock Multipliers per Category",
                      fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax2.legend(fontsize=8)
        ax2.tick_params(axis="x", rotation=30)
        ax2.set_ylim(0, HIGH + 0.4)

        # (3) Cost savings comparison
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_facecolor(CHART_BG_COLOR)
        ax3.bar(["Current\n(1.0x all)", "GA Optimised"],
                [current_cost / 1e6, optimal_cost / 1e6],
                color=["#E74C3C", "#27AE60"], edgecolor="white", width=0.5)
        ax3.text(0, current_cost / 1e6 * 1.02, f"${current_cost/1e6:.1f}M",
                 ha="center", fontsize=11, fontweight="bold", color=CHART_TEXT_COLOR)
        ax3.text(1, optimal_cost / 1e6 * 1.02, f"${optimal_cost/1e6:.1f}M",
                 ha="center", fontsize=11, fontweight="bold", color=CHART_TEXT_COLOR)
        ax3.set_ylabel("Total Cost ($M)", color=CHART_TEXT_COLOR)
        pct = (savings / current_cost * 100) if current_cost > 0 else 0.0
        ax3.set_title(
            f"Cost Savings: ${savings/1e6:.2f}M ({pct:.1f}%)",
            fontsize=12, fontweight="bold", color="#27AE60",
        )

        # (4) Cost breakdown by category (holding vs stockout) — optimised
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_facecolor(CHART_BG_COLOR)
        holding_costs, stockout_costs = [], []
        for i, cat in enumerate(categories):
            s = cat_stats[cat]
            ss = s["safety_mean"] * best_solution[i]
            h = ss * s["cost_mean"] * 0.25 * s["sku_count"]
            lt_demand = s["demand_mean"] * s["lead_mean"]
            so = max(0.0, (lt_demand * 0.5 - ss) * s["cost_mean"] * 2 * s["sku_count"])
            holding_costs.append(h / 1e6)
            stockout_costs.append(so / 1e6)
        x_pos = np.arange(n_cats)
        ax4.bar(x_pos, holding_costs, label="Holding Cost", color="#3498DB", edgecolor="white")
        ax4.bar(x_pos, stockout_costs, bottom=holding_costs,
                label="Stockout Cost", color="#E74C3C", edgecolor="white")
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(categories, fontsize=8, rotation=30, ha="right")
        ax4.set_ylabel("Cost ($M)", color=CHART_TEXT_COLOR)
        ax4.set_title("Cost Breakdown by Category (Optimised)",
                      fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax4.legend(fontsize=8)

        # (5) Gene evolution across generations
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_facecolor(CHART_BG_COLOR)
        hist_arr = np.array(best_chrom_hist)   # shape (N_GEN, n_cats)
        for i, cat in enumerate(categories):
            ax5.plot(hist_arr[:, i], label=cat, lw=1.5, alpha=0.85,
                     color=cat_colors[i])
        ax5.axhline(1.0, color="gray", ls="--", lw=1)
        ax5.set_xlabel("Generation", color=CHART_TEXT_COLOR)
        ax5.set_ylabel("Safety Stock Multiplier", color=CHART_TEXT_COLOR)
        ax5.set_title("Gene Evolution Across Generations",
                      fontsize=12, fontweight="bold", color=CHART_TEXT_COLOR)
        ax5.legend(fontsize=7, ncol=2)

        # (6) Recommendation summary table
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_facecolor(CHART_BG_COLOR)
        ax6.axis("off")
        table_data = []
        for i, cat in enumerate(categories):
            s = cat_stats[cat]
            curr_ss = s["safety_mean"]
            opt_ss = curr_ss * best_solution[i]
            change = (best_solution[i] - 1.0) * 100
            direction = "UP" if change >= 0 else "DWN"
            table_data.append([
                cat,
                f"{curr_ss:.0f}",
                f"{opt_ss:.0f}",
                f"{direction} {abs(change):.0f}%",
                f"{best_solution[i]:.2f}x",
            ])
        tbl = ax6.table(
            cellText=table_data,
            colLabels=["Category", "Curr. SS", "Opt. SS", "Change", "Multiplier"],
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1, 1.65)
        for j in range(5):
            tbl[0, j].set_facecolor("#2E86C1")
            tbl[0, j].set_text_props(color="white", fontweight="bold")
        for row in range(1, len(table_data) + 1):
            bg = "#EBF5FB" if row % 2 == 0 else "white"
            for col in range(5):
                tbl[row, col].set_facecolor(bg)
        ax6.set_title("Safety Stock Optimisation Summary",
                      fontsize=12, fontweight="bold", pad=20, color=CHART_TEXT_COLOR)

        save_path = str(self.output_dir / "chart_21_genetic_algorithm.png")
        plt.savefig(save_path, dpi=CHART_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", save_path)
        return save_path

    # ------------------------------------------------------------------
    # Chart 22 — Algorithm overview table
    # ------------------------------------------------------------------

    def plot_algorithm_overview(self, df: pd.DataFrame) -> str:
        """Chart 22: 30-algorithm table — name, category, SC use-case, status, library."""
        logger.info("chart_22: algorithm overview")

        algorithms = [
            ["#1",  "Linear Regression",    "Supervised",    "Predict Inventory Value",        "Applied",  "sklearn"],
            ["#2",  "Logistic Regression",  "Supervised",    "Classify Stock Status",           "Applied",  "sklearn"],
            ["#3",  "Decision Tree",        "Supervised",    "Classify Stock Status",           "Applied",  "sklearn"],
            ["#4",  "Random Forest",        "Supervised",    "Classify + Feature Importance",   "Applied",  "sklearn"],
            ["#5",  "SVM",                  "Supervised",    "Classify Stock Status",           "Applied",  "sklearn"],
            ["#6",  "k-NN",                 "Supervised",    "Classify Stock Status",           "Applied",  "sklearn"],
            ["#7",  "Naive Bayes",          "Supervised",    "Classify Stock Status",           "Applied",  "sklearn"],
            ["#8",  "Gradient Boosting",    "Supervised",    "Classify + Feature Importance",   "Applied",  "sklearn"],
            ["#9",  "AdaBoost",             "Supervised",    "Classify Stock Status",           "Applied",  "sklearn"],
            ["#10", "XGBoost",              "Supervised",    "Classify Stock Status",           "N/A*",     "xgboost"],
            ["#11", "k-Means",              "Unsupervised",  "Product Segmentation",            "Applied",  "sklearn"],
            ["#12", "Hierarchical",         "Unsupervised",  "Product Hierarchy Clustering",    "Applied",  "sklearn"],
            ["#13", "DBSCAN",               "Unsupervised",  "Density-based Outlier Detection", "Applied",  "sklearn"],
            ["#14", "PCA",                  "Unsupervised",  "Dimensionality Reduction",        "Applied",  "sklearn"],
            ["#15", "t-SNE",                "Unsupervised",  "2D/3D Visualisation",             "Applied",  "sklearn"],
            ["#16", "Q-Learning",           "RL",            "Inventory Replenishment Policy",  "N/A**",    "gymnasium"],
            ["#17", "SARSA",                "RL",            "Order Policy Learning",           "N/A**",    "gymnasium"],
            ["#18", "DQN",                  "RL",            "Dynamic Reorder Control",         "N/A**",    "stable-baselines3"],
            ["#19", "Policy Gradient",      "RL",            "Supply Chain Control",            "N/A**",    "stable-baselines3"],
            ["#20", "Actor-Critic",         "RL",            "Multi-objective Optimisation",    "N/A**",    "stable-baselines3"],
            ["#21", "MLP (ANN)",            "Deep Learning", "Classify Stock Status",           "Applied",  "sklearn"],
            ["#22", "CNN",                  "Deep Learning", "Visual Quality Inspection",       "N/A***",   "tensorflow"],
            ["#23", "RNN",                  "Deep Learning", "Demand Forecasting",              "N/A****",  "tensorflow"],
            ["#24", "LSTM",                 "Deep Learning", "Time-series Demand Forecast",     "N/A****",  "tensorflow"],
            ["#25", "Transformer",          "Deep Learning", "Advanced Demand Forecasting",     "N/A****",  "transformers"],
            ["#26", "k-Means++",            "Unsupervised",  "Improved Product Segmentation",   "Applied",  "sklearn"],
            ["#27", "Autoencoder",          "Deep Learning", "Anomaly Detection",               "Applied",  "sklearn (MLP)"],
            ["#28", "Isolation Forest",     "Unsupervised",  "Inventory Anomaly Detection",     "Applied",  "sklearn"],
            ["#29", "MDP",                  "RL Framework",  "Decision Framework",              "N/A**",    "custom"],
            ["#30", "Genetic Algorithm",    "Optimisation",  "Safety Stock Optimisation",       "Applied",  "custom"],
        ]

        cat_row_colors = {
            "Supervised":    "#EBF5FB",
            "Unsupervised":  "#E8F8F5",
            "RL":            "#FEF9E7",
            "RL Framework":  "#FEF9E7",
            "Deep Learning": "#F5EEF8",
            "Optimisation":  "#FDEDEC",
        }

        fig = plt.figure(figsize=(26, 18), facecolor=CHART_BG_COLOR)
        fig.suptitle(
            "30 AI Algorithms — Applicability to Supply Chain Inventory Analysis",
            fontsize=18, fontweight="bold", y=0.99, color=CHART_TEXT_COLOR,
        )
        gs = GridSpec(2, 1, figure=fig, height_ratios=[3.2, 1],
                      hspace=0.22, left=0.03, right=0.97, top=0.94, bottom=0.03)

        # (1) Main algorithm table
        ax1 = fig.add_subplot(gs[0])
        ax1.axis("off")
        ax1.set_facecolor(CHART_BG_COLOR)

        tbl = ax1.table(
            cellText=algorithms,
            colLabels=["#", "Algorithm", "Category", "Supply Chain Use Case", "Status", "Library"],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.18)

        # Header row styling
        for j in range(6):
            tbl[0, j].set_facecolor("#1B2A4A")
            tbl[0, j].set_text_props(color="white", fontweight="bold", fontsize=9)

        # Data row styling
        for row_idx, row_data in enumerate(algorithms, start=1):
            cat = row_data[2]
            bg = cat_row_colors.get(cat, "white")
            status = row_data[4]
            for col_idx in range(6):
                tbl[row_idx, col_idx].set_facecolor(bg)
            if status == "Applied":
                tbl[row_idx, 4].set_text_props(color="#27AE60", fontweight="bold")
            elif status.startswith("N/A"):
                tbl[row_idx, 4].set_text_props(color="#95A5A6")

        # (2) Notes panel
        ax2 = fig.add_subplot(gs[1])
        ax2.axis("off")
        ax2.set_facecolor(CHART_BG_COLOR)

        notes = (
            "Notes on algorithms NOT applied in this analysis:\n\n"
            "*   #10 XGBoost — optional dependency not bundled; identical role to Gradient Boosting (sklearn).\n"
            "**  #16-20, #29 (Reinforcement Learning / MDP) — require a sequential decision environment with state transitions\n"
            "    and reward signals. Dataset is a single cross-sectional snapshot, not a simulation environment.\n"
            "*** #22 CNN — designed for image/spatial data (e.g., visual QC). Not applicable to tabular inventory records.\n"
            "****#23-25 RNN / LSTM / Transformer — require historical time-series demand data.\n"
            "    Current dataset is cross-sectional (single point in time); no multi-period demand sequences available.\n\n"
            "Summary:  20 / 30 algorithms directly applied  |  "
            "10 require different data modalities (time-series, images, or RL simulation environments)."
        )
        ax2.text(0.04, 0.95, notes, transform=ax2.transAxes, fontsize=9.5,
                 va="top", family="monospace", color=CHART_TEXT_COLOR,
                 bbox=dict(boxstyle="round,pad=0.8", fc="#FDFEFE", ec="#BDC3C7", alpha=0.92))

        self.results["algorithm_overview"] = {
            "total_algorithms": 30,
            "applied": 20,
            "not_applicable": 10,
        }

        save_path = str(self.output_dir / "chart_22_algorithm_overview.png")
        plt.savefig(save_path, dpi=CHART_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", save_path)
        return save_path
