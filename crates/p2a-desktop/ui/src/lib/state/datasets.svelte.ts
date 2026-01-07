// Dataset state management using Svelte 5 runes

import type { DatasetInfo, DatasetPreview } from '$lib/types';

class DatasetsState {
	datasets = $state<DatasetInfo[]>([]);
	activeDataset = $state<string | null>(null);
	preview = $state<DatasetPreview | null>(null);
	isLoading = $state(false);

	// Pagination
	currentPage = $state(0);
	pageSize = $state(50);

	// Sorting
	sortColumn = $state<string | null>(null);
	sortDirection = $state<'asc' | 'desc'>('asc');

	setDatasets(datasets: DatasetInfo[]) {
		this.datasets = datasets;
	}

	addDataset(dataset: DatasetInfo) {
		// Replace if exists, otherwise add
		const idx = this.datasets.findIndex((d) => d.name === dataset.name);
		if (idx >= 0) {
			this.datasets[idx] = dataset;
		} else {
			this.datasets.push(dataset);
		}
	}

	setActiveDataset(name: string | null) {
		this.activeDataset = name;
		this.currentPage = 0;
		this.sortColumn = null;
	}

	setPreview(preview: DatasetPreview | null) {
		this.preview = preview;
	}

	setLoading(value: boolean) {
		this.isLoading = value;
	}

	setPage(page: number) {
		this.currentPage = page;
	}

	toggleSort(column: string) {
		if (this.sortColumn === column) {
			this.sortDirection = this.sortDirection === 'asc' ? 'desc' : 'asc';
		} else {
			this.sortColumn = column;
			this.sortDirection = 'asc';
		}
	}

	get activeDatasetInfo(): DatasetInfo | undefined {
		return this.datasets.find((d) => d.name === this.activeDataset);
	}

	get totalPages(): number {
		const info = this.activeDatasetInfo;
		if (!info) return 0;
		return Math.ceil(info.rows / this.pageSize);
	}
}

export const datasetsState = new DatasetsState();
