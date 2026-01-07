// Tauri API wrappers for prompt2analytics

import { invoke } from '@tauri-apps/api/core';
import type { ToolResult, DatasetInfo, DatasetPreview } from '$lib/types';

/**
 * Invoke an MCP tool by name with arguments.
 */
export async function invokeTool(
	toolName: string,
	args: Record<string, unknown>
): Promise<ToolResult> {
	return invoke<ToolResult>('invoke_tool', {
		toolName,
		arguments: args
	});
}

/**
 * List all available MCP tools.
 */
export async function listTools(): Promise<unknown[]> {
	return invoke<unknown[]>('list_tools');
}

/**
 * List all loaded datasets.
 */
export async function listDatasets(): Promise<DatasetInfo[]> {
	return invoke<DatasetInfo[]>('list_datasets');
}

/**
 * Load a dataset from file.
 */
export async function loadDataset(path: string, name?: string): Promise<DatasetInfo> {
	return invoke<DatasetInfo>('load_dataset', { path, name });
}

/**
 * Get a preview of dataset rows.
 */
export async function getDatasetPreview(
	datasetName: string,
	offset?: number,
	limit?: number
): Promise<DatasetPreview> {
	return invoke<DatasetPreview>('get_dataset_preview', {
		datasetName,
		offset,
		limit
	});
}

/**
 * Describe dataset statistics.
 */
export async function describeDataset(datasetName: string): Promise<string> {
	return invoke<string>('describe_dataset', { datasetName });
}

/**
 * Open a file picker dialog.
 */
export async function pickFile(): Promise<string | null> {
	return invoke<string | null>('pick_file');
}

/**
 * Open a multi-file picker dialog.
 */
export async function pickFiles(): Promise<string[]> {
	return invoke<string[]>('pick_files');
}

/**
 * Open a directory picker dialog.
 */
export async function pickDirectory(): Promise<string | null> {
	return invoke<string | null>('pick_directory');
}

/**
 * Parse a command string into tool name and arguments.
 * Format: tool_name key1=value1 key2=value2
 */
export function parseCommand(input: string): { toolName: string; args: Record<string, unknown> } | null {
	const parts = input.trim().split(/\s+/);
	if (parts.length === 0) return null;

	const toolName = parts[0];
	const args: Record<string, unknown> = {};

	for (let i = 1; i < parts.length; i++) {
		const part = parts[i];
		const eqIdx = part.indexOf('=');
		if (eqIdx > 0) {
			const key = part.slice(0, eqIdx);
			let value: unknown = part.slice(eqIdx + 1);

			// Try to parse as number or boolean
			if (value === 'true') value = true;
			else if (value === 'false') value = false;
			else if (!isNaN(Number(value))) value = Number(value);

			args[key] = value;
		}
	}

	return { toolName, args };
}
