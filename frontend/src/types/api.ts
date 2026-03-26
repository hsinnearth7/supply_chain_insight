export interface PipelineRun {
  batch_id: string;
  status: 'pending' | 'queued' | 'running' | 'completed' | 'failed';
  source_file: string;
  started_at: string | null;
  completed_at: string | null;
  etl_stats?: Record<string, unknown>;
  error?: string | null;
  analyses?: AnalysisEntry[];
}

export interface AnalysisEntry {
  type: string;
  results: Record<string, unknown>;
  chart_paths: string[];
  created_at: string | null;
}

export interface AnalysisResult {
  batch_id: string;
  analysis_type: string;
  kpis: Record<string, unknown>;
  chart_paths: string[];
  created_at: string | null;
}

export interface RunStatus {
  batch_id: string;
  status: 'pending' | 'queued' | 'running' | 'completed' | 'failed';
  progress_pct: number;
  completed_stages: string[];
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
}

export interface KPIData {
  batch_id: string;
  completed_at: string | null;
  kpis: Record<string, unknown>;
}

export interface InventoryRow {
  product_id: string;
  category: string;
  unit_cost: number;
  current_stock: number;
  daily_demand_est: number;
  safety_stock_target: number;
  vendor_name: string;
  lead_time_days: number;
  reorder_point: number;
  stock_status: string;
  inventory_value: number;
}

export interface ChartInfo {
  name: string;
  path: string;
}

export interface IngestResponse {
  batch_id: string;
  status: string;
  message: string;
}
