import { create } from 'zustand';
import type { Language } from '../i18n/translations';

const isBrowser = typeof window !== 'undefined';

function getStoredLanguage(): Language {
  if (!isBrowser) {
    return 'en';
  }
  return (window.localStorage.getItem('ci-lang') as Language) || 'en';
}

interface WatchdogEvent {
  batch_id: string;
  file: string;
  timestamp: string;
}

interface AppState {
  darkMode: boolean;
  toggleDarkMode: () => void;

  language: Language;
  setLanguage: (lang: Language) => void;

  latestBatchId: string | null;
  setLatestBatchId: (id: string) => void;

  activePipelineBatchId: string | null;
  setActivePipelineBatchId: (id: string | null) => void;

  watchdogEvents: WatchdogEvent[];
  addWatchdogEvent: (event: WatchdogEvent) => void;
  clearWatchdogEvents: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  darkMode: (() => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('ci-dark-mode');
      if (saved === 'true') {
        document.documentElement.classList.add('dark');
        return true;
      }
    }
    return false;
  })(),
  toggleDarkMode: () =>
    set((state) => {
      const next = !state.darkMode;
      if (typeof document !== 'undefined') {
        if (next) {
          document.documentElement.classList.add('dark');
        } else {
          document.documentElement.classList.remove('dark');
        }
      }
      if (typeof window !== 'undefined') {
        localStorage.setItem('ci-dark-mode', String(next));
      }
      return { darkMode: next };
    }),

  language: getStoredLanguage(),
  setLanguage: (lang) => {
    if (isBrowser) {
      window.localStorage.setItem('ci-lang', lang);
    }
    set({ language: lang });
  },

  latestBatchId: null,
  setLatestBatchId: (id) => set({ latestBatchId: id }),

  activePipelineBatchId: null,
  setActivePipelineBatchId: (id) => set({ activePipelineBatchId: id }),

  watchdogEvents: [],
  addWatchdogEvent: (event) =>
    set((state) => ({
      watchdogEvents: [event, ...state.watchdogEvents].slice(0, 20),
    })),
  clearWatchdogEvents: () => set({ watchdogEvents: [] }),
}));
