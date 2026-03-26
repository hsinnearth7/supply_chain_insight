import type { PipelineRun, AnalysisResult, RunStatus, KPIData, InventoryRow, ChartInfo, IngestResponse } from '../types/api';

const BASE = '/api';
const isBrowser = typeof window !== 'undefined';

function getApiKey(): string {
  const envKey = import.meta.env.VITE_API_KEY?.trim();
  if (envKey) return envKey;

  const explicitDevKey = import.meta.env.VITE_DEV_API_KEY?.trim();

  if (isBrowser) {
    const storedKey = window.localStorage.getItem('ci-api-key')?.trim();
    if (storedKey) return storedKey;

    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    if (isLocalhost && explicitDevKey) {
      console.warn('[ChainInsight] Using dev API key — do not use in production');
      return explicitDevKey;
    }
  }

  return '';
}

function buildHeaders(headers?: HeadersInit): Headers {
  const merged = new Headers(headers);
  const apiKey = getApiKey();
  if (apiKey) {
    merged.set('X-API-Key', apiKey);
  }
  return merged;
}

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    ...init,
    headers: buildHeaders(init?.headers),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json();
}

export const api = {
  // Ingest
  async ingest(file: File): Promise<IngestResponse> {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${BASE}/ingest`, {
      method: 'POST',
      body: form,
      headers: buildHeaders(),
    });
    if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
    return res.json();
  },

  async ingestExisting(): Promise<IngestResponse> {
    const res = await fetch(`${BASE}/ingest/existing`, {
      method: 'POST',
      headers: buildHeaders(),
    });
    if (!res.ok) throw new Error(`Failed: ${res.status}`);
    return res.json();
  },

  // Runs
  listRuns: () => fetchJSON<PipelineRun[]>(`${BASE}/runs`),

  getRun: (batchId: string) => fetchJSON<PipelineRun>(`${BASE}/runs/${batchId}`),

  getRunStatus: (batchId: string) => fetchJSON<RunStatus>(`${BASE}/runs/${batchId}/status`),

  // Analysis
  getAnalysis: (batchId: string, type: string) =>
    fetchJSON<AnalysisResult>(`${BASE}/runs/${batchId}/analysis/${type}`),

  // KPIs
  getKPIs: (batchId: string) => fetchJSON<Record<string, unknown>>(`${BASE}/runs/${batchId}/kpis`),
  getLatestKPIs: () => fetchJSON<KPIData>(`${BASE}/latest/kpis`),
  getKPIHistory: (limit = 20) => fetchJSON<KPIData[]>(`${BASE}/history/kpis?limit=${limit}`),

  // Charts
  listCharts: (batchId: string) => fetchJSON<ChartInfo[]>(`${BASE}/runs/${batchId}/charts`),
  getChartURL: (batchId: string, chartName: string) => `${BASE}/runs/${batchId}/charts/${chartName}`,

  // Data
  getInventoryData: (batchId: string) => fetchJSON<InventoryRow[]>(`${BASE}/runs/${batchId}/data`),
};

export const auth = {
  getApiKey,
  buildHeaders,
};
