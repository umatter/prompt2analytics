import { test, expect } from '@playwright/test'

test.describe('App - Basic Navigation', () => {
  test('should load the home page', async ({ page }) => {
    await page.goto('/')

    // Should show either the welcome message, connecting state, or error state
    await expect(
      page.getByText(/Welcome to prompt2analytics|Connecting to analytics server|Connection Error/i)
    ).toBeVisible({ timeout: 15000 })
  })

  test('should load the settings page directly', async ({ page }) => {
    await page.goto('/settings')

    // Settings page should load regardless of backend
    await expect(page.getByRole('heading', { name: 'Settings' })).toBeVisible()
  })
})

test.describe('Settings Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/settings')
    await page.waitForLoadState('networkidle')
  })

  test('should display LLM provider section', async ({ page }) => {
    // Check for provider heading
    await expect(page.getByRole('heading', { name: 'LLM Provider' })).toBeVisible()
  })

  test('should display provider select dropdown', async ({ page }) => {
    // Check provider selection exists
    const providerSelect = page.locator('select').first()
    await expect(providerSelect).toBeVisible()

    // Should have Ollama as the selected value
    await expect(providerSelect).toHaveValue('ollama')
  })

  test('should display temperature slider', async ({ page }) => {
    // Check temperature control exists
    await expect(page.getByText(/Temperature:/)).toBeVisible()
  })

  test('should display theme options', async ({ page }) => {
    // Check theme buttons
    await expect(page.getByRole('button', { name: 'Light' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'Dark' })).toBeVisible()
    await expect(page.getByRole('button', { name: 'System' })).toBeVisible()
  })

  test('should switch theme to dark', async ({ page }) => {
    // Click dark theme
    await page.getByRole('button', { name: 'Dark' }).click()

    // Button should now be selected (blue background)
    const darkButton = page.getByRole('button', { name: 'Dark' })
    await expect(darkButton).toHaveClass(/bg-blue-600/)
  })

  test('should switch theme to light', async ({ page }) => {
    // Click light theme
    await page.getByRole('button', { name: 'Light' }).click()

    // Button should now be selected
    const lightButton = page.getByRole('button', { name: 'Light' })
    await expect(lightButton).toHaveClass(/bg-blue-600/)
  })

  test('should show Ollama settings by default', async ({ page }) => {
    // Ollama is default, should see base URL input
    await expect(page.getByPlaceholder('http://localhost:11434')).toBeVisible()
  })

  test('should switch to Anthropic settings', async ({ page }) => {
    // Select Anthropic provider
    await page.locator('select').first().selectOption('anthropic')

    // Should see API key input
    await expect(page.getByPlaceholder('sk-ant-...')).toBeVisible()
  })

  test('should switch to OpenAI settings', async ({ page }) => {
    // Select OpenAI provider
    await page.locator('select').first().selectOption('openai')

    // Should see API key input
    await expect(page.getByPlaceholder('sk-...')).toBeVisible()
  })

  test('should have model parameters section', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Model Parameters' })).toBeVisible()
    await expect(page.getByText('Max Tokens')).toBeVisible()
  })

  test('should have appearance section', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Appearance' })).toBeVisible()
  })

  test('should have about section', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'About' })).toBeVisible()
    await expect(page.getByText('Version: 0.1.0')).toBeVisible()
  })

  test('should navigate back to home', async ({ page }) => {
    // Click back button (the link with an arrow icon in header)
    await page.locator('header a').first().click()

    // Should be back on home page
    await expect(page).toHaveURL('/')
  })
})

test.describe('Home Page - With Backend Connection Error', () => {
  // These tests run when backend is not available

  test('should show retry button on connection error', async ({ page }) => {
    await page.goto('/')

    // Wait for either success or error state
    await page.waitForTimeout(3000)

    // If showing error, should have retry button
    const errorVisible = await page.getByText('Connection Error').isVisible().catch(() => false)
    if (errorVisible) {
      await expect(page.getByRole('button', { name: 'Retry' })).toBeVisible()
    }
  })

  test('should show server instructions on error', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(3000)

    const errorVisible = await page.getByText('Connection Error').isVisible().catch(() => false)
    if (errorVisible) {
      await expect(page.getByText(/p2a-mcp server.*port 8080/i)).toBeVisible()
    }
  })
})

test.describe('Home Page - With Backend Connected', () => {
  // These tests only make sense when backend is running

  test.skip('should show welcome message', async ({ page }) => {
    await page.goto('/')

    // This only works when backend is running
    await expect(page.getByText('Welcome to prompt2analytics')).toBeVisible({ timeout: 15000 })
  })

  test.skip('should show suggestion cards', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(3000)

    // Only visible when backend is connected and showing welcome
    await expect(page.getByText('Load the sales.csv dataset')).toBeVisible()
  })

  test.skip('should have working chat input', async ({ page }) => {
    await page.goto('/')
    await page.waitForTimeout(3000)

    // Only available when connected
    await expect(page.getByPlaceholder('Ask me to analyze your data...')).toBeVisible()
  })
})
