# Desktop Application User Guide

This guide explains how to use the prompt2analytics desktop application for data analysis with LLM assistance.

## Table of Contents
- [Getting Started](#getting-started)
- [Interface Overview](#interface-overview)
- [Loading Data](#loading-data)
- [Using Commands](#using-commands)
- [LLM Integration](#llm-integration)
- [Conversation Management](#conversation-management)
- [Tips and Best Practices](#tips-and-best-practices)
- [Troubleshooting](#troubleshooting)

---

## Getting Started

### System Requirements

**Linux (Ubuntu/Debian)**:
```bash
sudo apt install libwebkit2gtk-4.1-dev libsoup-3.0-dev libjavascriptcoregtk-4.1-dev
```

**macOS**: No additional requirements.

**Windows**: Coming soon.

### Building the Application

```bash
# Build the analytics engine
cargo build --release -p p2a-mcp

# Install frontend dependencies
cd crates/p2a-desktop/ui
npm install
cd ../../..

# Build the desktop app
cargo build --release -p p2a-desktop
```

### Launching

```bash
./target/release/p2a-desktop
```

The application will:
1. Start the MCP analytics server in the background
2. Open the main application window
3. Display "Ready" status when initialization is complete

---

## Interface Overview

The application has three main panels:

```
┌─────────────────┬────────────────────┬─────────────────┐
│                 │                    │                 │
│   CHAT PANEL    │    DATA PANEL      │  RESULTS PANEL  │
│                 │                    │                 │
│   - Input box   │   - Dataset list   │   - Analysis    │
│   - Messages    │   - Preview table  │     results     │
│   - History     │   - Column info    │   - Charts      │
│                 │                    │   - Tables      │
└─────────────────┴────────────────────┴─────────────────┘
```

### Chat Panel (Left)
- **Message History**: Shows your conversation with the LLM
- **Input Box**: Type commands or natural language queries
- **Conversation List**: Access previous conversations

### Data Panel (Center)
- **Dataset Selector**: Choose from loaded datasets
- **Preview Table**: View the first rows of selected dataset
- **Import Button**: Load new data files

### Results Panel (Right)
- **Analysis Output**: Displays regression tables, statistics
- **Visualizations**: Shows generated charts and plots
- **Collapsible Sections**: Organize multiple results

---

## Loading Data

### Via Import Button

1. Click the **Import** button in the Data Panel
2. Select a file from the file picker
3. Wait for the dataset to load
4. The dataset appears in the dropdown selector

### Via Command

Type in the chat input:
```
/load_dataset path:/path/to/your/data.csv
```

### Supported Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| CSV | .csv | Comma-separated values |
| Parquet | .parquet | Columnar format |
| Excel | .xlsx, .xls | First sheet loaded |
| Stata | .dta | Versions 117-119 |
| SAS | .sas7bdat | SAS data files |

### After Loading

Once loaded, you can:
- Preview data in the Data Panel
- Run analyses using the dataset name
- Ask the LLM questions about your data

---

## Using Commands

### Command Syntax

Commands start with `/` followed by the tool name and parameters:
```
/tool_name param1:value1 param2:value2
```

### Common Commands

**View data**:
```
/head_dataset dataset:mydata n:10
/describe_dataset dataset:mydata
```

**Run regression**:
```
/regression_ols dataset:mydata y:price x:sqft,bedrooms
```

**Create visualizations**:
```
/viz_histogram dataset:mydata column:price
/viz_scatter dataset:mydata x_column:sqft y_column:price
```

**Panel data**:
```
/panel_fixed_effects dataset:panel y:wage x:education entity_col:person_id
```

### Natural Language

You can also use natural language with the LLM:
```
"Show me summary statistics for the housing dataset"
"Run a regression of price on square footage and bedrooms"
"Create a scatter plot of income vs spending"
```

The LLM will translate your request into the appropriate command.

---

## LLM Integration

### Configuring Providers

Click the **Settings** icon (gear) to configure LLM providers.

#### Ollama (Local)
1. Select "Ollama" as provider
2. Enter the base URL (default: `http://localhost:11434`)
3. Click "Refresh Models" to see available models
4. Select a model (e.g., `llama3.2`, `qwen2.5-coder`)

#### Anthropic (Claude)
1. Select "Anthropic" as provider
2. Enter your API key
3. Click "Refresh Models"
4. Select a model (e.g., `claude-sonnet-4-20250514`)

#### OpenAI
1. Select "OpenAI" as provider
2. Enter your API key
3. Click "Refresh Models"
4. Select a model (e.g., `gpt-4o`, `gpt-4o-mini`)

### Testing Connection

Click "Test Connection" to verify your configuration works.

### Using the LLM

Once configured, you can:
- Ask questions in natural language
- Request analysis without knowing exact commands
- Get explanations of results
- Ask follow-up questions

**Example conversation**:
```
You: Load the sales data from /data/sales.csv
LLM: [Loads dataset] I've loaded sales.csv with 1000 rows and 8 columns.

You: What's the average revenue by region?
LLM: [Runs aggregation] Here are the average revenues by region:
     - North: $45,230
     - South: $38,450
     - East: $52,100
     - West: $41,890

You: Is there a significant difference between regions?
LLM: [Runs ANOVA] The ANOVA test shows F=12.4, p<0.001, indicating
     significant differences between regions.
```

---

## Conversation Management

### Starting a New Conversation

Click the **New Chat** button or use the keyboard shortcut.

### Accessing History

Click on previous conversations in the sidebar to resume them.

### Renaming Conversations

1. Right-click on a conversation
2. Select "Rename"
3. Enter a new name

### Exporting Conversations

1. Right-click on a conversation
2. Select "Export"
3. Choose format (JSON or Markdown)
4. Select save location

### Searching Conversations

Use the search box in the sidebar to find conversations by content.

---

## Tips and Best Practices

### Data Preparation

1. **Check your data first**: Run `/describe_dataset` to understand your data
2. **Handle missing values**: The tools handle NaN but results may vary
3. **Use appropriate types**: Ensure numeric columns are numeric

### Efficient Workflow

1. **Load data once**: Datasets persist during the session
2. **Use descriptive names**: When loading multiple datasets
3. **Save important conversations**: Export analyses you want to keep

### Getting Better LLM Results

1. **Be specific**: "Run OLS of price on sqft" is better than "analyze price"
2. **Provide context**: "Using the housing dataset, ..."
3. **Ask follow-ups**: "What does the R-squared mean?" or "Is this coefficient significant?"

### Interpreting Results

1. **Check significance**: Look at p-values (< 0.05 typically significant)
2. **Check R-squared**: Higher means better fit (for regression)
3. **Run diagnostics**: Use `/regression_diagnostics` to check assumptions

---

## Troubleshooting

### Application Won't Start

**Check MCP server**:
```bash
# Verify the binary exists
ls -la target/release/p2a-mcp

# Try running directly
./target/release/p2a-mcp
```

**Check dependencies** (Linux):
```bash
sudo apt install libwebkit2gtk-4.1-dev libsoup-3.0-dev libjavascriptcoregtk-4.1-dev
```

### "Dataset Not Found" Error

- Ensure the dataset is loaded (check Data Panel dropdown)
- Use the exact dataset name (case-sensitive)
- Try `/list_datasets` to see available datasets

### LLM Not Responding

1. Check your API key is correct
2. Click "Test Connection" in Settings
3. Verify internet connection (for cloud providers)
4. For Ollama, ensure the server is running: `ollama serve`

### Charts Not Appearing

- Charts appear in the Results Panel (right side)
- Scroll down if the panel has multiple results
- Charts are PNG images; if blank, the data may be invalid

### Slow Performance

- Large datasets (>100MB) may take time to load
- Complex visualizations take longer to render
- Consider using `/head_dataset` to preview before full analysis

### Regression Errors

- **"Column not found"**: Check column names match exactly
- **"Singular matrix"**: Variables may be perfectly collinear (remove one)
- **"Not enough observations"**: Need more rows than variables

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Enter` | Send message |
| `Shift+Enter` | New line in input |
| `Ctrl+N` | New conversation |
| `Ctrl+L` | Clear current conversation |
| `Escape` | Close dialogs |

---

## Getting Help

- **In-app**: Ask the LLM "How do I...?" questions
- **Documentation**: See `docs/guides/` for detailed guides
- **Tool reference**: See `docs/guides/MCP_TOOL_EXAMPLES.md`
- **Econometrics help**: See `docs/guides/ECONOMETRICS_GUIDE.md`
- **Issues**: Report bugs at the project repository

---

## Quick Start Example

1. **Launch the app**:
   ```bash
   ./target/release/p2a-desktop
   ```

2. **Load sample data** (chat input):
   ```
   /load_dataset path:docs/testing/sample_sales.csv
   ```

3. **Explore the data**:
   ```
   /describe_dataset dataset:sample_sales
   ```

4. **Run a regression**:
   ```
   /regression_ols dataset:sample_sales y:revenue x:units_sold,cost
   ```

5. **Create a visualization**:
   ```
   /viz_scatter dataset:sample_sales x_column:units_sold y_column:revenue
   ```

6. **Ask the LLM** (if configured):
   ```
   "What does this regression tell us about the relationship between units sold and revenue?"
   ```

You're now ready to analyze your data!
