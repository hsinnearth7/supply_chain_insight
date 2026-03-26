import { useEffect, useState, useCallback } from 'react';
import { api } from '../api/client';
import { useAppStore } from '../stores/appStore';
import KPICard from '../components/KPICard';
import ChartImage from '../components/ChartImage';
import DataTable from '../components/DataTable';
import LoadingSpinner from '../components/LoadingSpinner';
import { useTranslation } from '../i18n/useTranslation';
import type { AnalysisResult } from '../types/api';

export default function SOPPage() {
  const [batchId, setBatchId] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const latestBatchId = useAppStore((s) => s.latestBatchId);
  const { t } = useTranslation();

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
      const data = await api.getAnalysis(bid, 'sop');
      setAnalysis(data);
    } catch {
      setError('Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [latestBatchId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  if (loading) return <LoadingSpinner text={t('sop.loading')} />;
  if (error) return (
    <div className="text-red-500 p-4 text-center">
      <p>{error}</p>
      <button onClick={loadData} className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
        Retry
      </button>
    </div>
  );
  if (!analysis) return <div className="text-ci-gray text-center py-12">{t('sop.noData')}</div>;

  const sopData = (analysis.kpis || {}) as Record<string, unknown>;
  const scenarios = (sopData.scenarios || []) as Record<string, unknown>[];

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">{t('sop.title')}</h2>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard
          title={t('sop.fillRate')}
          value={typeof sopData.fill_rate === 'number' ? `${((sopData.fill_rate as number) * 100).toFixed(1)}%` : '--'}
          icon="FIL"
          color="ci-success"
        />
        <KPICard
          title={t('sop.scenarios')}
          value={String(scenarios.length || 3)}
          icon="SCN"
          color="ci-primary"
        />
        <KPICard
          title={t('capacity.utilization')}
          value={typeof sopData.avg_utilization === 'number' ? `${((sopData.avg_utilization as number) * 100).toFixed(1)}%` : '--'}
          icon="UTL"
          color="ci-teal"
        />
        <KPICard
          title={t('sop.demandSupplyBalance')}
          value={typeof sopData.balance_pct === 'number' ? `${((sopData.balance_pct as number) * 100).toFixed(1)}%` : '--'}
          icon="BAL"
          color="ci-purple"
        />
      </div>

      {scenarios.length > 0 && (
        <DataTable
          data={scenarios}
          title={t('sop.comparison')}
          columns={['scenario', 'fill_rate', 'utilization', 'inventory_cost', 'stockout_risk']}
        />
      )}

      {batchId && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ChartImage src={api.getChartURL(batchId, 'chart_sop_balance.png')} alt="Demand-Supply Balance" />
          <ChartImage src={api.getChartURL(batchId, 'chart_sop_scenarios.png')} alt="Scenario Comparison" />
        </div>
      )}
    </div>
  );
}
