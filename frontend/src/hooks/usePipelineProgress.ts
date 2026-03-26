import { useState, useCallback, useEffect } from 'react';
import { api } from '../api/client';
import type { RunStatus } from '../types/api';
import { useWebSocket } from './useWebSocket';
import type { WSMessage } from '../types/websocket';

export interface StageStatus {
  stage: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress_pct: number;
}

const STAGES = ['etl', 'stats', 'supply_chain', 'ml', 'capacity', 'sensing', 'sop'];

const initialStages: StageStatus[] = STAGES.map((s) => ({
  stage: s,
  status: 'pending',
  progress_pct: 0,
}));

function deriveStagesFromStatus(run: RunStatus): StageStatus[] {
  const completed = new Set(run.completed_stages);
  const failedStage =
    run.status === 'failed'
      ? STAGES.find((stage) => !completed.has(stage)) ?? STAGES[STAGES.length - 1]
      : null;
  const runningStage =
    run.status === 'running'
      ? STAGES.find((stage) => !completed.has(stage)) ?? null
      : null;

  return STAGES.map((stage) => {
    if (completed.has(stage)) {
      return { stage, status: 'completed' as const, progress_pct: 100 };
    }
    if (failedStage === stage) {
      return { stage, status: 'failed' as const, progress_pct: run.progress_pct };
    }
    if (runningStage === stage) {
      return { stage, status: 'running' as const, progress_pct: run.progress_pct };
    }
    return { stage, status: 'pending' as const, progress_pct: 0 };
  });
}

export function usePipelineProgress(batchId: string | null) {
  const [stages, setStages] = useState<StageStatus[]>(initialStages);
  const [overallPct, setOverallPct] = useState(0);
  const [pipelineStatus, setPipelineStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle');

  const handleMessage = useCallback((msg: WSMessage) => {
    if (!msg.payload) return;
    const { stage, status, progress_pct } = msg.payload;

    if (msg.type === 'pipeline:failed') {
      setStages((prev) =>
        prev.map((s) => {
          if (s.stage === stage) {
            return { ...s, status: 'failed', progress_pct };
          }
          return s;
        })
      );
      setOverallPct(progress_pct);
      setPipelineStatus('failed');
      return;
    }

    setOverallPct(progress_pct);

    setStages((prev) =>
      prev.map((s) => {
        if (s.stage === stage) {
          return { ...s, status: status as StageStatus['status'], progress_pct };
        }
        return s;
      })
    );

    // Check if all done
    if (stage === 'sop' && status === 'completed') {
      setPipelineStatus('completed');
      setOverallPct(100);
    } else if (status === 'running') {
      setPipelineStatus((prev) => {
        if (prev !== 'running') return 'running';
        return prev;
      });
    }
  }, []);

  const { connected } = useWebSocket({
    url: batchId ? `/ws/pipeline/${batchId}` : '',
    onMessage: handleMessage,
    enabled: !!batchId,
  });

  useEffect(() => {
    if (!batchId || connected || pipelineStatus === 'completed' || pipelineStatus === 'failed') {
      return;
    }

    let active = true;
    const currentBatchId = batchId;

    async function pollStatus() {
      try {
        const run = await api.getRunStatus(currentBatchId);
        if (!active) {
          return;
        }
        setOverallPct(run.progress_pct);
        setStages(deriveStagesFromStatus(run));
        if (run.status === 'completed' || run.status === 'failed') {
          setPipelineStatus(run.status);
        } else if (run.status === 'running' || run.status === 'queued') {
          setPipelineStatus('running');
        }
      } catch {
        // keep the last known progress state and retry on the next interval
      }
    }

    void pollStatus();
    const interval = setInterval(() => {
      void pollStatus();
    }, 3000);

    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [batchId, connected, pipelineStatus]);

  const reset = useCallback(() => {
    setStages(initialStages.map(s => ({ ...s })));
    setOverallPct(0);
    setPipelineStatus('idle');
  }, []);

  return { stages, overallPct, pipelineStatus, connected, reset };
}
