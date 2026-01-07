// Results state management using Svelte 5 runes

import type { AnalysisResult } from '$lib/types';

class ResultsState {
	results = $state<AnalysisResult[]>([]);
	expandedResult = $state<string | null>(null);

	addResult(tool: string, content: string, images: string[] = []) {
		this.results.unshift({
			id: crypto.randomUUID(),
			tool,
			content,
			images,
			timestamp: new Date()
		});
		// Auto-expand new result
		this.expandedResult = this.results[0].id;
	}

	toggleExpanded(id: string) {
		this.expandedResult = this.expandedResult === id ? null : id;
	}

	removeResult(id: string) {
		const idx = this.results.findIndex((r) => r.id === id);
		if (idx >= 0) {
			this.results.splice(idx, 1);
		}
		if (this.expandedResult === id) {
			this.expandedResult = null;
		}
	}

	clearResults() {
		this.results = [];
		this.expandedResult = null;
	}
}

export const resultsState = new ResultsState();
