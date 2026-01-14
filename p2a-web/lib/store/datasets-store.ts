import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { api } from '@/lib/api/client'
import { useSessionStore } from './session-store'
import type { DatasetInfo } from '@/lib/types/api'

// Helper to ensure session exists before API calls
async function ensureSession(): Promise<boolean> {
  const sessionStore = useSessionStore.getState()
  if (!sessionStore.isInitialized) {
    await sessionStore.initSession()
  }
  return useSessionStore.getState().isInitialized
}

interface DatasetsState {
  datasets: DatasetInfo[]
  selectedDataset: string | null
  preview: {
    columns: string[]
    rows: unknown[][]
    totalRows: number
  } | null
  isLoading: boolean
  error: string | null

  // Actions
  loadDatasets: () => Promise<void>
  selectDataset: (name: string) => Promise<void>
  loadDatasetFromPath: (path: string, name?: string) => Promise<void>
  deleteDataset: (name: string) => Promise<void>
  clearError: () => void
}

export const useDatasetsStore = create<DatasetsState>()(
  immer((set, get) => ({
    datasets: [],
    selectedDataset: null,
    preview: null,
    isLoading: false,
    error: null,

    loadDatasets: async () => {
      set((state) => {
        state.isLoading = true
        state.error = null
      })

      // Ensure session exists
      const hasSession = await ensureSession()
      if (!hasSession) {
        set((state) => {
          state.error = 'Failed to initialize session'
          state.isLoading = false
        })
        return
      }

      try {
        const response = await api.listDatasets()
        if (response.success && response.data) {
          set((state) => {
            state.datasets = response.data!.datasets
            state.isLoading = false
          })
        } else {
          set((state) => {
            state.error = response.error || 'Failed to load datasets'
            state.isLoading = false
          })
        }
      } catch (err) {
        set((state) => {
          state.error = err instanceof Error ? err.message : 'Network error'
          state.isLoading = false
        })
      }
    },

    selectDataset: async (name: string) => {
      set((state) => {
        state.selectedDataset = name
        state.isLoading = true
        state.error = null
      })

      // Ensure session exists
      const hasSession = await ensureSession()
      if (!hasSession) {
        set((state) => {
          state.error = 'Failed to initialize session'
          state.isLoading = false
        })
        return
      }

      try {
        const response = await api.describeDataset(name)
        if (response.success && response.data) {
          const info = response.data
          set((state) => {
            state.preview = {
              columns: info.columns,
              rows: [], // Would need a separate preview endpoint
              totalRows: info.row_count,
            }
            state.isLoading = false
          })
        } else {
          set((state) => {
            state.error = response.error || 'Failed to get dataset info'
            state.isLoading = false
          })
        }
      } catch (err) {
        set((state) => {
          state.error = err instanceof Error ? err.message : 'Network error'
          state.isLoading = false
        })
      }
    },

    loadDatasetFromPath: async (path: string, name?: string) => {
      set((state) => {
        state.isLoading = true
        state.error = null
      })

      // Ensure session exists
      const hasSession = await ensureSession()
      if (!hasSession) {
        set((state) => {
          state.error = 'Failed to initialize session'
          state.isLoading = false
        })
        return
      }

      // Extract dataset name from path if not provided
      const datasetName = name || path.split('/').pop()?.replace(/\.[^.]+$/, '') || 'dataset'

      try {
        const response = await api.callTool('load_dataset', {
          path,
          name: datasetName,
        })

        if (response.success) {
          // Refresh the dataset list
          await get().loadDatasets()
          set((state) => {
            state.selectedDataset = datasetName
            state.isLoading = false
          })
        } else {
          set((state) => {
            state.error = response.error || 'Failed to load dataset'
            state.isLoading = false
          })
        }
      } catch (err) {
        set((state) => {
          state.error = err instanceof Error ? err.message : 'Load failed'
          state.isLoading = false
        })
      }
    },

    deleteDataset: async (name: string) => {
      set((state) => {
        state.isLoading = true
        state.error = null
      })

      // Ensure session exists
      const hasSession = await ensureSession()
      if (!hasSession) {
        set((state) => {
          state.error = 'Failed to initialize session'
          state.isLoading = false
        })
        return
      }

      try {
        const response = await api.callTool('drop_dataset', { name })

        if (response.success) {
          set((state) => {
            state.datasets = state.datasets.filter((d) => d.name !== name)
            if (state.selectedDataset === name) {
              state.selectedDataset = null
              state.preview = null
            }
            state.isLoading = false
          })
        } else {
          set((state) => {
            state.error = response.error || 'Failed to delete dataset'
            state.isLoading = false
          })
        }
      } catch (err) {
        set((state) => {
          state.error = err instanceof Error ? err.message : 'Delete failed'
          state.isLoading = false
        })
      }
    },

    clearError: () => {
      set((state) => {
        state.error = null
      })
    },
  }))
)
