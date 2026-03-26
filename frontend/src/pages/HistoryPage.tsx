import { useEffect, useState, useCallback } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { api } from '../api/client';
import { useAppStore } from '../stores/appStore';
import LoadingSpinner from '../components/LoadingSpinner';
import { useTranslation } from '../i18n/useTranslation';
import type { PipelineRun, KPIData } from '../types/api';

const STATUS_STYLES: Record<string, string> = {
  completed: 'bg-ci-success/10 text-ci-success',
  running: 'bg-ci-primary/10 text-ci-primary',
  queued: 'bg-ci-warning/10 text-ci-warning',
  failed: 'bg-ci-danger/10 text-ci-danger',
  pending: 'bg-ci-gray/10 text-ci-gray',
};

export default function HistoryPage() {
  const [runs, setRuns] = useState<PipelineRun[]>([]);
  const [kpiHistory, setKpiHistory] = useState<KPIData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const setLatestBatchId = useAppStore((s) => s.setLatestBatchId);
  const latestBatchId = useAppStore((s) => s.latestBatchId);
  const { t } = useTranslation();

  const loadData = useCallback(async function loadData() {
    setError(null);
    try {
      const [runsData, historyData] = await Promise.all([
        api.listRuns(),
        api.getKPIHistory(20),
      ]);
      setRuns(runsData);
      setKpiHistory(historyData);
      const latestCompletedRun = runsData
        .filter((run) => run.status === 'completed' && run.completed_at)
        .sort((a, b) => new Date(b.completed_at!).getTime() - new Date(a.completed_at!).getTime())[0];
      if (latestCompletedRun) {
        setLatestBatchId(latestCompletedRun.batch_id);
      }
    } catch {
      setError('Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [setLatestBatchId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  if (loading) return <LoadingSpinner text={t('history.loading')} />;
  if (error) return (
    <div className="text-red-500 p-4 text-center">
      <p>{error}</p>
      <button onClick={loadData} className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
        Retry
      </button>
    </div>
  );

  const trendData = kpiHistory
    .slice()
    .reverse()
    .map((entry) => {
      const kpis = entry.kpis as Record<string, number>;
      return {
        date: entry.completed_at?.slice(0, 10) || '--',
        inventory_turnover: kpis.inventory_turnover,
        avg_dsi: kpis.avg_dsi,
        oos_rate: kpis.oos_rate,
        total_value: kpis.total_inventory_value ? kpis.total_inventory_value / 1000 : undefined,
      };
    });

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">{t('history.title')}</h2>

      {trendData.length > 1 && (
        <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium mb-3">{t('history.kpiTrends')}</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trendData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} />
              <YAxis yAxisId="left" />
              <YAxis yAxisId="right" orientation="right" />
              <Tooltip />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="inventory_turnover" stroke="#2E86C1" name="Turnover" dot />
              <Line yAxisId="left" type="monotone" dataKey="avg_dsi" stroke="#27AE60" name="Avg DSI" dot />
              <Line yAxisId="left" type="monotone" dataKey="oos_rate" stroke="#E74C3C" name="OOS Rate %" dot />
              <Line yAxisId="right" type="monotone" dataKey="total_value" stroke="#F39C12" name="Total Value ($k)" dot />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-sm font-medium">{t('history.pipelineRuns')} ({runs.length})</h3>
        </div>
        <div className="overflow-auto max-h-[500px]">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 dark:bg-gray-800 sticky top-0">
              <tr>
                <th className="px-4 py-2 text-left text-xs font-medium text-ci-gray">{t('history.batchId')}</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-ci-gray">{t('history.status')}</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-ci-gray">{t('history.sourceFile')}</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-ci-gray">{t('history.started')}</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-ci-gray">{t('history.completed')}</th>
                <th className="px-4 py-2 text-left text-xs font-medium text-ci-gray">{t('history.duration')}</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
              {runs.map((run) => {
                const duration =
                  run.started_at && run.completed_at
                    ? `${Math.round((new Date(run.completed_at).getTime() - new Date(run.started_at).getTime()) / 1000)}s`
                    : '--';
                return (
                  <tr key={run.batch_id} className="hover:bg-gray-50 dark:hover:bg-gray-700/30">
                    <td className="px-4 py-2 font-mono text-xs">{run.batch_id}</td>
                    <td className="px-4 py-2">
                      <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${STATUS_STYLES[run.status] || ''}`}>
                        {run.status}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-ci-gray text-xs truncate max-w-[200px]">
                      {run.source_file?.split(/[/\\]/).pop() || '--'}
                    </td>
                    <td className="px-4 py-2 text-xs">{run.started_at ? new Date(run.started_at).toLocaleString() : '--'}</td>
                    <td className="px-4 py-2 text-xs">{run.completed_at ? new Date(run.completed_at).toLocaleString() : '--'}</td>
                    <td className="px-4 py-2 text-xs font-mono">{duration}</td>
                  </tr>
                );
              })}
              {runs.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-ci-gray">
                    {t('history.noRuns')}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
