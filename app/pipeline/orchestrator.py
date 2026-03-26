"""Pipeline Orchestrator — coordinates ETL -> Stats -> Supply Chain -> ML -> Capacity -> Sensing -> S&OP."""

import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

from app.capacity.models import CapacityPlanner
from app.capacity.visualization import plot_bottleneck_timeline, plot_utilization_timeline
from app.config import CHARTS_DIR, CLEAN_DIR, PipelineStatus
from app.db.models import (
    AnalysisResult,
    InventorySnapshot,
    PipelineRun,
    SessionLocal,
)
from app.pipeline.etl import ETLPipeline
from app.pipeline.ml_engine import MLAnalyzer
from app.pipeline.stats import StatisticalAnalyzer
from app.pipeline.supply_chain import SupplyChainAnalyzer
from app.sensing.signals import SignalProcessor
from app.sensing.visualization import plot_forecast_adjustment, plot_signal_timeline
from app.sop.simulator import SOPSimulator
from app.sop.visualization import plot_demand_supply_balance, plot_scenario_comparison

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Coordinates the full analysis pipeline with progress callbacks."""

    STAGES = ["etl", "stats", "supply_chain", "ml", "capacity", "sensing", "sop"]

    def __init__(self, on_progress: Optional[Callable] = None):
        """
        Args:
            on_progress: Optional callback(stage, status, data) for real-time updates.
        """
        self.on_progress = on_progress or (lambda *a, **kw: None)
        self._current_stage: Optional[str] = None
        # init_db() is already called by main.py at startup; no need to re-init on every run

    def run(self, input_path: str, batch_id: Optional[str] = None) -> dict:
        """Execute the full pipeline.

        Args:
            input_path: Path to raw CSV file.
            batch_id: Optional batch identifier (auto-generated if None).

        Returns:
            Dict with batch_id, all results, and chart paths.
        """
        batch_id = batch_id or f"batch_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        logger.info("Pipeline started — batch_id=%s, input=%s", batch_id, input_path)

        db = SessionLocal()
        run_record = db.query(PipelineRun).filter(PipelineRun.batch_id == batch_id).first()
        if run_record:
            run_record.status = PipelineStatus.RUNNING.value
        else:
            run_record = PipelineRun(
                batch_id=batch_id,
                status=PipelineStatus.RUNNING.value,
                source_file=str(input_path),
            )
            db.add(run_record)
        db.commit()

        all_results = {"batch_id": batch_id, "stages": {}}
        charts_dir = CHARTS_DIR / batch_id
        charts_dir.mkdir(parents=True, exist_ok=True)
        self._current_stage = "etl"

        try:
            # Stage 1: ETL
            self._current_stage = "etl"
            self.on_progress("etl", "running", {})
            etl = ETLPipeline()
            clean_path = CLEAN_DIR / f"{batch_id}_clean.csv"
            df_clean = etl.run(input_path, str(clean_path))
            etl_stats = etl.get_stats()
            all_results["stages"]["etl"] = etl_stats
            self._save_snapshots(db, batch_id, df_clean)
            self._save_analysis(db, batch_id, "etl", etl_stats, [])
            self.on_progress("etl", "completed", etl_stats)

            # Stage 2: Statistical Analysis
            self._current_stage = "stats"
            self.on_progress("stats", "running", {})
            stats_analyzer = StatisticalAnalyzer(output_dir=str(charts_dir))
            stats_results = stats_analyzer.run_all(df_clean)
            all_results["stages"]["stats"] = stats_results
            self._save_analysis(
                db, batch_id, "stats",
                stats_results.get("kpis", {}), stats_results.get("chart_paths", []),
            )
            self.on_progress("stats", "completed", stats_results)

            # Stage 3: Supply Chain Optimization
            self._current_stage = "supply_chain"
            self.on_progress("supply_chain", "running", {})
            sc_analyzer = SupplyChainAnalyzer(output_dir=str(charts_dir))
            sc_results = sc_analyzer.run_all(df_clean)
            all_results["stages"]["supply_chain"] = sc_results
            self._save_analysis(
                db, batch_id, "supply_chain",
                sc_results.get("results", {}), sc_results.get("chart_paths", []),
            )
            self.on_progress("supply_chain", "completed", sc_results)

            # Stage 4: ML Analysis
            self._current_stage = "ml"
            self.on_progress("ml", "running", {})
            ml_analyzer = MLAnalyzer(output_dir=str(charts_dir))
            ml_results = ml_analyzer.run_all(df_clean)
            all_results["stages"]["ml"] = ml_results
            self._save_analysis(db, batch_id, "ml", ml_results.get("results", {}), ml_results.get("chart_paths", []))
            self.on_progress("ml", "completed", ml_results)

            # Stage 5: Capacity Planning
            self._current_stage = "capacity"
            self.on_progress("capacity", "running", {})
            cap_planner = CapacityPlanner()
            # Rows are products/SKUs, not days — use fixed planning horizon
            n_periods = 4  # 4-week planning horizon
            total_weekly_demand = float(df_clean["Daily_Demand_Est"].sum()) * 7
            cap_demand = pd.DataFrame({
                "period": [f"W{i+1}" for i in range(n_periods)],
                "demand": [total_weekly_demand] * n_periods,
            })
            cap_report = cap_planner.check_feasibility(cap_demand)
            cap_adjustments = cap_planner.suggest_adjustments(cap_report.bottlenecks)
            cap_chart_paths = []
            cap_chart_paths.append(plot_utilization_timeline(
                cap_report.period_utilizations, str(charts_dir), cap_planner.utilization_target,
            ))
            cap_chart_paths.append(plot_bottleneck_timeline(
                [{"period": b.period, "demand": b.demand, "capacity": b.capacity} for b in cap_report.bottlenecks],
                str(charts_dir),
            ))
            cap_kpis = {
                "avg_utilization": cap_report.avg_utilization,
                "max_utilization": cap_report.max_utilization,
                "bottleneck_count": cap_report.bottleneck_count,
                "demand_coverage": cap_report.demand_coverage,
                "feasible": cap_report.feasible,
                "adjustments": len(cap_adjustments),
            }
            all_results["stages"]["capacity"] = {"kpis": cap_kpis, "chart_paths": cap_chart_paths}
            self._save_analysis(db, batch_id, "capacity", cap_kpis, cap_chart_paths)
            self.on_progress("capacity", "completed", cap_kpis)

            # Stage 6: Demand Sensing
            self._current_stage = "sensing"
            self.on_progress("sensing", "running", {})
            sensor = SignalProcessor()
            product_ids = df_clean["Product_ID"].unique().tolist() if "Product_ID" in df_clean.columns else []
            signals = sensor.generate_synthetic_signals(product_ids)
            spikes = sensor.detect_spikes(signals)
            sense_base = pd.DataFrame({
                "product_id": product_ids[:10],
                "period": [1] * min(10, len(product_ids)),
                "forecast": [float(df_clean["Daily_Demand_Est"].mean())] * min(10, len(product_ids)),
            }) if product_ids else pd.DataFrame(columns=["product_id", "period", "forecast"])
            sense_adjusted = sensor.compute_sensing_adjustment(sense_base, signals)
            sense_chart_paths = []
            sense_chart_paths.append(plot_signal_timeline(
                [{"source": s.source, "signal_value": s.signal_value} for s in signals[:100]],
                str(charts_dir),
            ))
            adj_records = sense_adjusted.to_dict("records") if not sense_adjusted.empty else []
            sense_chart_paths.append(plot_forecast_adjustment(adj_records, str(charts_dir)))
            avg_adj = float(sense_adjusted["adjustment_pct"].abs().mean()) if not sense_adjusted.empty else 0.0
            sense_kpis = {
                "active_signals": len(signals),
                "spike_count": len(spikes),
                "avg_adjustment_pct": avg_adj,
                "products_sensed": len(sense_adjusted),
            }
            all_results["stages"]["sensing"] = {"kpis": sense_kpis, "chart_paths": sense_chart_paths}
            self._save_analysis(db, batch_id, "sensing", sense_kpis, sense_chart_paths)
            self.on_progress("sensing", "completed", sense_kpis)

            # Stage 7: S&OP Simulation
            self._current_stage = "sop"
            self.on_progress("sop", "running", {})
            sop_sim = SOPSimulator()
            daily_cap = sum(
                p.max_throughput_per_day * p.efficiency_factor
                for p in cap_planner.build_default_profiles()
            )
            sop_report = sop_sim.compare_scenarios(cap_demand, daily_cap)
            best_result = next(
                (r for r in sop_report.results if r.scenario_name == sop_report.best_scenario),
                sop_report.results[0],
            )
            sop_chart_paths = []
            sop_chart_paths.append(plot_demand_supply_balance(best_result.period_details, str(charts_dir)))
            sop_chart_paths.append(plot_scenario_comparison(
                [{"scenario_name": r.scenario_name, "fill_rate": r.fill_rate,
                  "avg_utilization": r.avg_utilization, "total_inventory_cost": r.total_inventory_cost}
                 for r in sop_report.results],
                str(charts_dir),
            ))
            sop_kpis = {
                "fill_rate": best_result.fill_rate,
                "avg_utilization": best_result.avg_utilization,
                "best_scenario": sop_report.best_scenario,
                "balance_pct": best_result.fill_rate,
                "scenarios": [sop_sim.calculate_kpis(r) for r in sop_report.results],
            }
            all_results["stages"]["sop"] = {"kpis": sop_kpis, "chart_paths": sop_chart_paths}
            self._save_analysis(db, batch_id, "sop", sop_kpis, sop_chart_paths)
            self.on_progress("sop", "completed", sop_kpis)

            # Mark pipeline complete
            run_record.status = PipelineStatus.COMPLETED.value
            run_record.completed_at = datetime.now(timezone.utc)
            run_record.etl_stats = etl_stats
            db.commit()

            all_results["status"] = "completed"
            logger.info("Pipeline completed — batch_id=%s", batch_id)

        except Exception as e:
            db.rollback()
            run_record.status = PipelineStatus.FAILED.value
            run_record.error_message = str(e)
            run_record.completed_at = datetime.now(timezone.utc)
            db.commit()
            all_results["status"] = "failed"
            all_results["error"] = str(e)
            failed_stage = self._current_stage or "unknown"
            logger.exception("Pipeline failed at stage=%s — batch_id=%s", failed_stage, batch_id)
            self.on_progress(failed_stage, "failed", {"error": str(e)})
        finally:
            db.close()

        return all_results

    def _save_snapshots(self, db, batch_id: str, df: pd.DataFrame):
        """Save cleaned data as inventory snapshots."""
        now = datetime.now(timezone.utc)
        records = df.to_dict("records")
        snapshots = []
        for rec in records:
            snapshots.append(InventorySnapshot(
                batch_id=batch_id,
                ingested_at=now,
                product_id=rec.get("Product_ID"),
                category=rec.get("Category"),
                unit_cost=rec.get("Unit_Cost"),
                current_stock=rec.get("Current_Stock"),
                daily_demand_est=rec.get("Daily_Demand_Est"),
                safety_stock_target=rec.get("Safety_Stock_Target"),
                vendor_name=rec.get("Vendor_Name"),
                lead_time_days=rec.get("Lead_Time_Days"),
                reorder_point=rec.get("Reorder_Point"),
                stock_status=rec.get("Stock_Status"),
                inventory_value=rec.get("Inventory_Value"),
            ))
        db.bulk_save_objects(snapshots)
        db.commit()
        logger.info("Saved %d inventory snapshots", len(snapshots))

    def _save_analysis(self, db, batch_id: str, analysis_type: str, result_data: dict, chart_paths: list):
        """Save analysis results to DB."""
        # Convert numpy types to Python native for JSON serialization
        clean_data = self._to_serializable(result_data)
        db.add(AnalysisResult(
            batch_id=batch_id,
            analysis_type=analysis_type,
            result_json=clean_data,
            chart_paths=chart_paths,
        ))
        db.commit()

    def _to_serializable(self, obj):
        """Recursively convert numpy types to Python native types."""
        import numpy as np
        if isinstance(obj, dict):
            return {k: self._to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._to_serializable(v) for v in obj]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
