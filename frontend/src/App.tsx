import { BrowserRouter, Routes, Route } from 'react-router-dom';
import React, { Suspense, lazy } from 'react';
import Layout from './components/Layout';
import LoadingSpinner from './components/LoadingSpinner';
import { useWebSocket } from './hooks/useWebSocket';
import { useAppStore } from './stores/appStore';
import type { WSMessage } from './types/websocket';

const DashboardPage = lazy(() => import('./pages/DashboardPage'));
const UploadPage = lazy(() => import('./pages/UploadPage'));
const StatsPage = lazy(() => import('./pages/StatsPage'));
const SupplyChainPage = lazy(() => import('./pages/SupplyChainPage'));
const MLPage = lazy(() => import('./pages/MLPage'));
const CapacityPage = lazy(() => import('./pages/CapacityPage'));
const SensingPage = lazy(() => import('./pages/SensingPage'));
const SOPPage = lazy(() => import('./pages/SOPPage'));
const HistoryPage = lazy(() => import('./pages/HistoryPage'));

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error: Error | null }
> {
  state = { hasError: false, error: null as Error | null };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex items-center justify-center h-screen">
          <div className="text-center p-8">
            <h1 className="text-2xl font-bold text-red-500 mb-4">Something went wrong</h1>
            <p className="text-gray-600 mb-4">{this.state.error?.message}</p>
            <button
              onClick={() => {
                this.setState({ hasError: false, error: null });
                window.location.reload();
              }}
              className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            >
              Reload Page
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

function NotFoundPage() {
  return (
    <div className="flex items-center justify-center h-screen">
      <h1 className="text-2xl">404 — Page Not Found</h1>
    </div>
  );
}

export default function App() {
  const { addWatchdogEvent } = useAppStore();

  // Global WS listener for watchdog events
  useWebSocket({
    url: '/ws/global',
    onMessage: (msg: WSMessage) => {
      if (msg.type === 'watchdog:detected') {
        addWatchdogEvent({
          batch_id: msg.batch_id,
          file: (msg.payload?.data as Record<string, string> | undefined)?.file || 'unknown',
          timestamp: msg.timestamp,
        });
      }
    },
  });

  return (
    <ErrorBoundary>
      <BrowserRouter>
        <Suspense fallback={<LoadingSpinner />}>
          <Routes>
            <Route element={<Layout />}>
              <Route index element={<DashboardPage />} />
              <Route path="upload" element={<UploadPage />} />
              <Route path="stats" element={<StatsPage />} />
              <Route path="supply-chain" element={<SupplyChainPage />} />
              <Route path="ml" element={<MLPage />} />
              <Route path="capacity" element={<CapacityPage />} />
              <Route path="sensing" element={<SensingPage />} />
              <Route path="sop" element={<SOPPage />} />
              <Route path="history" element={<HistoryPage />} />
              <Route path="*" element={<NotFoundPage />} />
            </Route>
          </Routes>
        </Suspense>
      </BrowserRouter>
    </ErrorBoundary>
  );
}
