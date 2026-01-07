import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	// Prevent vite from obscuring Rust errors
	clearScreen: false,
	// Tauri expects a fixed port
	server: {
		port: 5173,
		strictPort: true,
	},
	// Env variables starting with TAURI_ are passed to the client
	envPrefix: ['VITE_', 'TAURI_'],
});
