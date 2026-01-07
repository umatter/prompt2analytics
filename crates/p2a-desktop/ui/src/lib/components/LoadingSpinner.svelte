<script lang="ts">
	interface Props {
		size?: 'sm' | 'md' | 'lg';
		message?: string;
	}

	let { size = 'md', message = '' }: Props = $props();

	const sizeClasses = {
		sm: 'spinner-sm',
		md: 'spinner-md',
		lg: 'spinner-lg'
	};
</script>

<div class="spinner-container">
	<div class="spinner {sizeClasses[size]}">
		<div class="dot dot-1"></div>
		<div class="dot dot-2"></div>
		<div class="dot dot-3"></div>
	</div>
	{#if message}
		<span class="spinner-message">{message}</span>
	{/if}
</div>

<style>
	.spinner-container {
		display: flex;
		align-items: center;
		gap: var(--spacing-sm);
	}

	.spinner {
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.spinner-sm .dot {
		width: 6px;
		height: 6px;
	}

	.spinner-md .dot {
		width: 8px;
		height: 8px;
	}

	.spinner-lg .dot {
		width: 12px;
		height: 12px;
	}

	.dot {
		border-radius: 50%;
		background-color: var(--color-primary);
		animation: bounce 1.4s infinite ease-in-out both;
	}

	.dot-1 {
		animation-delay: -0.32s;
	}

	.dot-2 {
		animation-delay: -0.16s;
	}

	.dot-3 {
		animation-delay: 0s;
	}

	@keyframes bounce {
		0%,
		80%,
		100% {
			transform: scale(0.6);
			opacity: 0.5;
		}
		40% {
			transform: scale(1);
			opacity: 1;
		}
	}

	.spinner-message {
		color: var(--color-text-secondary);
		font-size: 0.9rem;
	}
</style>
