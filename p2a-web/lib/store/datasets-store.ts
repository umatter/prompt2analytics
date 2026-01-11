import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { api } from '@/lib/api/client'
import type { DatasetInfo } from '@/lib/types/api'

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
  uploadDataset: (file: File, name?: string) => Promise<void>
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

    uploadDataset: async (file: File, name?: string) => {
      set((state) => {
        state.isLoading = true
        state.error = null
      })

      const datasetName = name || file.name.replace(/\.[^.]+$/, '')

      try {
        // Read file content
        const content = await file.text()

        // Determine format from extension
        const extension = file.name.split('.').pop()?.toLowerCase()
        const format = extension === 'json' ? 'json' : extension === 'parquet' ? 'parquet' : 'csv'

        // For now, we'll use the tool call endpoint to load the dataset
        // In a full implementation, we'd have a dedicated upload endpoint
        const response = await api.callTool('load_dataset', {
          path: `data:${format};base64,${btoa(content)}`, // Using data URL for inline content
          name: datasetName,
          format,
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
            state.error = response.error || 'Failed to upload dataset'
            state.isLoading = false
          })
        }
      } catch (err) {
        set((state) => {
          state.error = err instanceof Error ? err.message : 'Upload failed'
          state.isLoading = false
        })
      }
    },

    deleteDataset: async (name: string) => {
      set((state) => {
        state.isLoading = true
        state.error = null
      })

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
