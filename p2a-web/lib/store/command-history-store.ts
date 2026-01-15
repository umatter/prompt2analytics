import { create } from 'zustand'
import { shouldExportTool } from '@/lib/cli/tool-mapping'

export interface CommandRecord {
  id: string
  timestamp: Date
  toolName: string
  arguments: Record<string, unknown>
  success: boolean
}

interface CommandHistoryState {
  commands: CommandRecord[]

  // Actions
  addCommand: (record: Omit<CommandRecord, 'id'>) => void
  clearCommands: () => void
  getExportableCommands: () => CommandRecord[]
}

let commandIdCounter = 0

export const useCommandHistoryStore = create<CommandHistoryState>((set, get) => ({
  commands: [],

  addCommand: (record) => {
    // Only add commands that should be exported (analysis tools, not inspection tools)
    if (!shouldExportTool(record.toolName)) {
      return
    }

    const id = `cmd-${++commandIdCounter}-${Date.now()}`
    set((state) => ({
      commands: [
        ...state.commands,
        {
          id,
          ...record,
        },
      ],
    }))
  },

  clearCommands: () => {
    set({ commands: [] })
  },

  getExportableCommands: () => {
    // Return only successful commands that can be mapped to CLI
    return get().commands.filter((cmd) => cmd.success && shouldExportTool(cmd.toolName))
  },
}))
