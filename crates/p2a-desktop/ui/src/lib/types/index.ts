// Type definitions for prompt2analytics desktop app

// Re-export LLM types
export * from './llm';

export interface Message {
	id: string;
	role: 'user' | 'assistant' | 'error' | 'system';
	content: string;
	images?: string[];
	timestamp: Date;
}

export interface ToolResult {
	success: boolean;
	content: string;
	images: ImageData[];
	error?: string;
}

export interface ImageData {
	base64: string;
	alt: string;
}

export interface DatasetInfo {
	name: string;
	rows: number;
	columns: number;
	column_names: string[];
	source_path?: string;
}

export interface DatasetPreview {
	name: string;
	columns: string[];
	rows: Record<string, unknown>[];
	total_rows: number;
	offset: number;
	limit: number;
}

export interface AnalysisResult {
	id: string;
	tool: string;
	content: string;
	images: string[];
	timestamp: Date;
}
