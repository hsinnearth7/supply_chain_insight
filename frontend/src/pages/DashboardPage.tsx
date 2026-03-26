import { useEffect, useState, useMemo, useCallback } from 'react';
import {
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { api } from '../api/client';
import { useAppStore } from '../stores/appStore';
import KPICard from '../components/KPICard';
import DataTable from '../components/DataTable';
import LoadingSpinner from '../components/LoadingSpinner';
import { useTranslation } from '../i18n/useTranslation';
import type { InventoryRow, KPIData } from '../types/api';

const STATUS_COLORS: Record<string, string> = {
  'Normal Stock': '#27AE60',
  'Low Stock': '#F39C12',
  'Out of Stock': '#E74C3C',
};

const CATEGORY_COLORS = ['#2E86C1', '#27AE60', '#E74C3C', '#F39C12', '#8E44AD', '#1ABC9C', '#95A5A6'];

export default function DashboardPage() {
  const [kpiData, setKpiData] = useState<KPIData | null>(null);
  const [inventory, setInventory] = useState<InventoryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const setLatestBatchId = useAppStore((s) => s.setLatestBatchId);
  const latestBatchId = useAppStore((s) => s.latestBatchId);
  const { t } = useTranslation();

  const loadData = useCallback(async function loadData() {
    try {
      const kpis = await api.getLatestKPIs();
      setKpiData(kpis);
      setLatestBatchId(kpis.batch_id);
      const data = await api.getInventoryData(kpis.batch_id);
      setInventory(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to load dashboard');
    } finally {
      setLoading(false);
    }
  }, [setLatestBatchId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const statusCounts = useMemo(() => inventory.reduce<Record<string, number>>((acc, row) => {
    acc[row.stock_status] = (acc[row.stock_status] || 0) + 1;
    return acc;
  }, {}), [inventory]);
  const pieData = useMemo(() => Object.entries(statusCounts).map(([name, value]) => ({ name, value })), [statusCounts]);

  const categoryData = useMemo(() => inventory.reduce<Record<string, number>>((acc, row) => {
    acc[row.category] = (acc[row.category] || 0) + row.inventory_value;
    return acc;
  }, {}), [inventory]);
  const barData = useMemo(() => Object.entries(categoryData)
    .map(([category, value]) => ({ category, value: Math.round(value) }))
    .sort((a, b) => b.value - a.value), [categoryData]);

  const vendorPerf = useMemo(() => inventory.reduce<Record<string, { count: number; totalValue: number; oos: number }>>((acc, row) => {
    if (!acc[row.vendor_name]) acc[row.vendor_name] = { count: 0, totalValue: 0, oos: 0 };
    acc[row.vendor_name].count++;
    acc[row.vendor_name].totalValue += row.inventory_value;
    if (row.stock_status === 'Out of Stock') acc[row.vendor_name].oos++;
    return acc;
  }, {}), [inventory]);
  const vendorRows = useMemo(() => Object.entries(vendorPerf).map(([vendor, data]) => ({
    vendor_name: vendor,
    products: data.count,
    total_value: Math.round(data.totalValue),
    oos_count: data.oos,
    oos_rate: `${((data.oos / data.count) * 100).toFixed(1)}%`,
  })), [vendorPerf]);

  const stockoutAlerts = useMemo(() => inventory
    .filter((r) => {
      const dsi = r.daily_demand_est > 0 ? r.current_stock / r.daily_demand_est : 999;
      return dsi < r.lead_time_days;
    })
    .map((r) => ({
      product_id: r.product_id,
      category: r.category,
      stock: r.current_stock,
      daily_demand: r.daily_demand_est,
      lead_time: r.lead_time_days,
      dsi: r.daily_demand_est > 0 ? (r.current_stock / r.daily_demand_est).toFixed(1) : 'N/A',
      status: r.stock_status,
    })), [inventory]);

  if (loading) return <LoadingSpinner text={t('dashboard.loading')} />;
  if (error) return (
    <div className="text-red-500 p-4 text-center">
      <p>{error}</p>
      <button onClick={loadData} className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600">
        Retry
      </button>
    </div>
  );
  if (!kpiData) return <div className="text-ci-gray text-center py-12">{t('dashboard.noData')}</div>;

  const kpis = kpiData.kpis ?? {};
  const getValue = (key: string): number => {
    const v = (kpis as Record<string, unknown>)[key];
    return typeof v === 'number' ? v : 0;
  };
  const getValueOrNull = (key: string): number | null => {
    const v = (kpis as Record<string, unknown>)[key];
    return typeof v === 'number' ? v : null;
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">{t('dashboard.title')}</h2>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <KPICard
          title={t('dashboard.inventoryTurnover')}
          value={getValueOrNull('inventory_turnover') != null ? getValue('inventory_turnover').toFixed(2) : '--'}
          icon="IT"
          color="ci-primary"
        />
        <KPICard
          title={t('dashboard.avgDSI')}
          value={getValueOrNull('avg_dsi') != null ? getValue('avg_dsi').toFixed(1) : '--'}
          subtitle={t('dashboard.days')}
          icon="DSI"
          color="ci-teal"
        />
        <KPICard
          title={t('dashboard.oosRate')}
          value={getValueOrNull('oos_rate') != null ? `${getValue('oos_rate').toFixed(1)}%` : '--'}
          icon="OOS"
          color="ci-danger"
        />
        <KPICard
          title={t('dashboard.slowMovingValue')}
          value={getValueOrNull('slow_moving_value') != null ? `$${Math.round(getValue('slow_moving_value')).toLocaleString()}` : '--'}
          icon="SM"
          color="ci-warning"
        />
        <KPICard
          title={t('dashboard.totalValue')}
          value={getValueOrNull('total_inventory_value') != null ? `$${Math.round(getValue('total_inventory_value')).toLocaleString()}` : '--'}
          icon="INV"
          color="ci-success"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium mb-3">{t('dashboard.stockStatus')}</h3>
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                outerRadius={100}
                dataKey="value"
                label={({ name, percent }) => `${name} (${(percent * 100).toFixed(0)}%)`}
              >
                {pieData.map((entry) => (
                  <Cell key={entry.name} fill={STATUS_COLORS[entry.name] || '#95A5A6'} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <h3 className="text-sm font-medium mb-3">{t('dashboard.categoryValue')}</h3>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData} layout="vertical" margin={{ left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" tickFormatter={(v) => `$${(v / 1000).toFixed(0)}k`} />
              <YAxis type="category" dataKey="category" width={75} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(v: number) => `$${v.toLocaleString()}`} />
              <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                {barData.map((_, i) => (
                  <Cell key={i} fill={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DataTable
          data={vendorRows}
          title={t('dashboard.vendorPerf')}
          columns={['vendor_name', 'products', 'total_value', 'oos_count', 'oos_rate']}
        />
        <DataTable
          data={stockoutAlerts.slice(0, 20)}
          title={t('dashboard.stockoutAlerts')}
          columns={['product_id', 'category', 'stock', 'dsi', 'lead_time', 'status']}
        />
      </div>
    </div>
  );
}
