import { useEffect, useState, useMemo, useCallback } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { api } from '../api/client';
import { useAppStore } from '../stores/appStore';
import ChartImage from '../components/ChartImage';
import LoadingSpinner from '../components/LoadingSpinner';
import { useTranslation } from '../i18n/useTranslation';
import type { InventoryRow } from '../types/api';

const SC_CHARTS = [
  { file: 'chart_09_eoq_analysis.png', label: 'EOQ Analysis' },
  { file: 'chart_10_vendor_radar.png', label: 'Vendor Radar Chart' },
  { file: 'chart_11_inventory_treemap.png', label: 'Inventory Treemap' },
  { file: 'chart_12_monte_carlo_stockout.png', label: 'Monte Carlo Simulation' },
  { file: 'chart_13_reorder_gap_waterfall.png', label: 'Reorder Gap Waterfall' },
  { file: 'chart_14_demand_safety_stock.png', label: 'Demand vs Safety Stock' },
];

const ORDERING_COST = 50;
export default function SupplyChainPage() {
  const [batchId, setBatchId] = useState<string | null>(null);
  const [inventory, setInventory] = useState<InventoryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const latestBatchId = useAppStore((s) => s.latestBatchId);
  const { t } = useTranslation();

  // EOQ calculator state
  const [eoqDemand, setEoqDemand] = useState(1000);
  const [eoqOrderCost, setEoqOrderCost] = useState(ORDERING_COST);
  const [eoqHoldCost, setEoqHoldCost] = useState(5);

  // Monte Carlo state
  const [mcSims, setMcSims] = useState(1000);
  const [mcDemandMean, setMcDemandMean] = useState(50);
  const [mcDemandStd, setMcDemandStd] = useState(15);
  const [mcLeadTime, setMcLeadTime] = useState(7);
  const [mcStock, setMcStock] = useState(400);
  const [mcResult, setMcResult] = useState<{ bins: { range: string; count: number }[]; stockoutPct: number } | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let bid = latestBatchId;
      if (!bid) {
        const kpis = await api.getLatestKPIs();
        bid = kpis.batch_id;
      }
      setBatchId(bid);
      const data = await api.getInventoryData(bid);
      setInventory(data);
      if (data.length > 0) {
        const mean = data.reduce((a, r) => a + r.daily_demand_est, 0) / data.length;
        const variance = data.reduce((a, r) => a + (r.daily_demand_est - mean) ** 2, 0) / data.length;
        setMcDemandMean(Math.round(mean));
        setMcDemandStd(Math.round(Math.sqrt(variance)));
      }
    } catch {
      setError('Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [latestBatchId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  // EOQ calculation
  const eoq = useMemo(() => {
    const q = Math.sqrt((2 * eoqDemand * eoqOrderCost) / eoqHoldCost);
    const points = [];
    for (let qty = Math.max(1, Math.round(q * 0.2)); qty <= q * 3; qty += Math.max(1, Math.round(q * 0.05))) {
      const holding = (qty / 2) * eoqHoldCost;
      const ordering = (eoqDemand / qty) * eoqOrderCost;
      points.push({ qty: Math.round(qty), holding: Math.round(holding), ordering: Math.round(ordering), total: Math.round(holding + ordering) });
    }
    return { optimalQ: Math.round(q), points };
  }, [eoqDemand, eoqOrderCost, eoqHoldCost]);

  // Monte Carlo simulation
  function runMonteCarlo() {
    const stockouts: number[] = [];
    const demands: number[] = [];
    for (let i = 0; i < mcSims; i++) {
      let totalDemand = 0;
      for (let d = 0; d < mcLeadTime; d++) {
        totalDemand += Math.max(0, mcDemandMean + mcDemandStd * boxMuller());
      }
      demands.push(totalDemand);
      if (totalDemand > mcStock) stockouts.push(totalDemand);
    }
    const min = Math.min(...demands);
    const max = Math.max(...demands);
    const bucketSize = (max - min) / 15 || 1;
    const bins = [];
    for (let i = 0; i < 15; i++) {
      const lo = min + i * bucketSize;
      const hi = lo + bucketSize;
      bins.push({
        range: `${Math.round(lo)}-${Math.round(hi)}`,
        count: demands.filter((d) => d >= lo && (i === 14 ? d <= hi : d < hi)).length,
      });
    }
    setMcResult({ bins, stockoutPct: (stockouts.length / mcSims) * 100 });
  }

  if (loading) return <LoadingSpinner text={t('sc.loading')} />;
  if (error) return (
    <div className="text-red-500 p-4 text-center">
      <p>{error}</p>
      <button onClick={loadData} className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
        Retry
      </button>
    </div>
  );
  if (!batchId) return <div className="text-ci-gray text-center py-12">{t('sc.noData')}</div>;

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">{t('sc.title')}</h2>

      {/* EOQ Calculator */}
      <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4">
        <h3 className="text-sm font-medium mb-3">{t('sc.eoqCalc')}</h3>
        <div className="grid grid-cols-3 gap-4 mb-4">
          <label className="text-xs">
            <span className="text-ci-gray">{t('sc.annualDemand')}</span>
            <input type="number" value={eoqDemand} onChange={(e) => setEoqDemand(+e.target.value)} className="mt-1 w-full px-2 py-1 border rounded text-sm dark:bg-gray-800 dark:border-gray-600" />
          </label>
          <label className="text-xs">
            <span className="text-ci-gray">{t('sc.orderCost')}</span>
            <input type="number" value={eoqOrderCost} onChange={(e) => setEoqOrderCost(+e.target.value)} className="mt-1 w-full px-2 py-1 border rounded text-sm dark:bg-gray-800 dark:border-gray-600" />
          </label>
          <label className="text-xs">
            <span className="text-ci-gray">{t('sc.holdingCost')}</span>
            <input type="number" value={eoqHoldCost} onChange={(e) => setEoqHoldCost(+e.target.value)} className="mt-1 w-full px-2 py-1 border rounded text-sm dark:bg-gray-800 dark:border-gray-600" />
          </label>
        </div>
        <p className="text-sm mb-3">
          {t('sc.optimalQty')}: <span className="font-bold text-ci-primary">{eoq.optimalQ} {t('sc.units')}</span>
        </p>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={eoq.points} margin={{ bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="qty" label={{ value: t('common.orderQuantity'), position: 'insideBottom', offset: -10 }} tick={{ fontSize: 11 }} />
            <YAxis tickFormatter={(v) => `$${v}`} />
            <Tooltip formatter={(v: number) => `$${v}`} />
            <Legend verticalAlign="top" align="left" wrapperStyle={{ paddingBottom: 8 }} />
            <Line type="monotone" dataKey="holding" stroke="#F39C12" name={t('sc.holdingCostLabel')} dot={false} />
            <Line type="monotone" dataKey="ordering" stroke="#2E86C1" name={t('sc.orderingCostLabel')} dot={false} />
            <Line type="monotone" dataKey="total" stroke="#E74C3C" name={t('sc.totalCostLabel')} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Monte Carlo Simulator */}
      <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4">
        <h3 className="text-sm font-medium mb-3">{t('sc.monteCarlo')}</h3>
        <div className="grid grid-cols-5 gap-3 mb-3">
          <label className="text-xs">
            <span className="text-ci-gray">{t('sc.simulations')}</span>
            <input type="number" value={mcSims} onChange={(e) => setMcSims(+e.target.value)} className="mt-1 w-full px-2 py-1 border rounded text-sm dark:bg-gray-800 dark:border-gray-600" />
          </label>
          <label className="text-xs">
            <span className="text-ci-gray">{t('sc.dailyDemandMean')}</span>
            <input type="number" value={mcDemandMean} onChange={(e) => setMcDemandMean(+e.target.value)} className="mt-1 w-full px-2 py-1 border rounded text-sm dark:bg-gray-800 dark:border-gray-600" />
          </label>
          <label className="text-xs">
            <span className="text-ci-gray">{t('sc.demandStdDev')}</span>
            <input type="number" value={mcDemandStd} onChange={(e) => setMcDemandStd(+e.target.value)} className="mt-1 w-full px-2 py-1 border rounded text-sm dark:bg-gray-800 dark:border-gray-600" />
          </label>
          <label className="text-xs">
            <span className="text-ci-gray">{t('sc.leadTimeDays')}</span>
            <input type="number" value={mcLeadTime} onChange={(e) => setMcLeadTime(+e.target.value)} className="mt-1 w-full px-2 py-1 border rounded text-sm dark:bg-gray-800 dark:border-gray-600" />
          </label>
          <label className="text-xs">
            <span className="text-ci-gray">{t('sc.currentStock')}</span>
            <input type="number" value={mcStock} onChange={(e) => setMcStock(+e.target.value)} className="mt-1 w-full px-2 py-1 border rounded text-sm dark:bg-gray-800 dark:border-gray-600" />
          </label>
        </div>
        <button onClick={runMonteCarlo} className="px-4 py-1.5 bg-ci-primary text-white text-sm rounded hover:bg-ci-primary/90 transition mb-3">
          {t('sc.runSimulation')}
        </button>
        {mcResult && (
          <>
            <p className="text-sm mb-2">
              {t('sc.stockoutProb')}: <span className={`font-bold ${mcResult.stockoutPct > 10 ? 'text-ci-danger' : 'text-ci-success'}`}>
                {mcResult.stockoutPct.toFixed(1)}%
              </span>
            </p>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={mcResult.bins}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="range" tick={{ fontSize: 9 }} angle={-45} textAnchor="end" height={60} />
                <YAxis />
                <Tooltip />
                <Bar dataKey="count" fill="#2E86C1" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </>
        )}
      </div>

      {/* PNG Charts */}
      <h3 className="text-sm font-medium">{t('sc.pipelineCharts')}</h3>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {SC_CHARTS.map((chart) => (
          <div key={chart.file}>
            <p className="text-xs text-ci-gray mb-1">{chart.label}</p>
            <ChartImage src={api.getChartURL(batchId, chart.file)} alt={chart.label} />
          </div>
        ))}
      </div>
    </div>
  );
}

function boxMuller(): number {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}
