import { w as ensure_array_like, x as attr_class, y as stringify, z as attr } from "../../chunks/index.js";
import "clsx";
import "@tauri-apps/api/core";
import { e as escape_html } from "../../chunks/context.js";
class ChatState {
  messages = [];
  input = "";
  isProcessing = false;
  addMessage(role, content, images) {
    this.messages.push({
      id: crypto.randomUUID(),
      role,
      content,
      images,
      timestamp: /* @__PURE__ */ new Date()
    });
  }
  addUserMessage(content) {
    this.addMessage("user", content);
  }
  addAssistantMessage(content, images) {
    this.addMessage("assistant", content, images);
  }
  addErrorMessage(content) {
    this.addMessage("error", content);
  }
  addSystemMessage(content) {
    this.addMessage("system", content);
  }
  clearInput() {
    this.input = "";
  }
  setInput(value) {
    this.input = value;
  }
  setProcessing(value) {
    this.isProcessing = value;
  }
  clearMessages() {
    this.messages = [];
  }
}
const chatState = new ChatState();
class DatasetsState {
  datasets = [];
  activeDataset = null;
  preview = null;
  isLoading = false;
  // Pagination
  currentPage = 0;
  pageSize = 50;
  // Sorting
  sortColumn = null;
  sortDirection = "asc";
  setDatasets(datasets) {
    this.datasets = datasets;
  }
  addDataset(dataset) {
    const idx = this.datasets.findIndex((d) => d.name === dataset.name);
    if (idx >= 0) {
      this.datasets[idx] = dataset;
    } else {
      this.datasets.push(dataset);
    }
  }
  setActiveDataset(name) {
    this.activeDataset = name;
    this.currentPage = 0;
    this.sortColumn = null;
  }
  setPreview(preview) {
    this.preview = preview;
  }
  setLoading(value) {
    this.isLoading = value;
  }
  setPage(page) {
    this.currentPage = page;
  }
  toggleSort(column) {
    if (this.sortColumn === column) {
      this.sortDirection = this.sortDirection === "asc" ? "desc" : "asc";
    } else {
      this.sortColumn = column;
      this.sortDirection = "asc";
    }
  }
  get activeDatasetInfo() {
    return this.datasets.find((d) => d.name === this.activeDataset);
  }
  get totalPages() {
    const info = this.activeDatasetInfo;
    if (!info) return 0;
    return Math.ceil(info.rows / this.pageSize);
  }
}
const datasetsState = new DatasetsState();
class ResultsState {
  results = [];
  expandedResult = null;
  addResult(tool, content, images = []) {
    this.results.unshift({
      id: crypto.randomUUID(),
      tool,
      content,
      images,
      timestamp: /* @__PURE__ */ new Date()
    });
    this.expandedResult = this.results[0].id;
  }
  toggleExpanded(id) {
    this.expandedResult = this.expandedResult === id ? null : id;
  }
  removeResult(id) {
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
const resultsState = new ResultsState();
function _page($$renderer, $$props) {
  $$renderer.component(($$renderer2) => {
    function formatTime(date) {
      return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }
    $$renderer2.push(`<div class="layout svelte-1uha8ag"><div class="panel chat-panel svelte-1uha8ag"><div class="panel-header"><h2>Chat</h2> <button class="secondary">Import File</button></div> <div class="panel-content messages svelte-1uha8ag"><!--[-->`);
    const each_array = ensure_array_like(chatState.messages);
    for (let $$index_1 = 0, $$length = each_array.length; $$index_1 < $$length; $$index_1++) {
      let message = each_array[$$index_1];
      $$renderer2.push(`<div${attr_class(`message ${stringify(message.role)}`, "svelte-1uha8ag")}><div class="message-header svelte-1uha8ag"><span class="role svelte-1uha8ag">${escape_html(message.role)}</span> <span class="time svelte-1uha8ag">${escape_html(formatTime(message.timestamp))}</span></div> <div class="message-content svelte-1uha8ag"><pre class="svelte-1uha8ag">${escape_html(message.content)}</pre> `);
      if (message.images) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<!--[-->`);
        const each_array_1 = ensure_array_like(message.images);
        for (let $$index = 0, $$length2 = each_array_1.length; $$index < $$length2; $$index++) {
          let img = each_array_1[$$index];
          $$renderer2.push(`<img${attr("src", `data:image/png;base64,${stringify(img)}`)} alt="Chart" class="message-image svelte-1uha8ag"/>`);
        }
        $$renderer2.push(`<!--]-->`);
      } else {
        $$renderer2.push("<!--[!-->");
      }
      $$renderer2.push(`<!--]--></div></div>`);
    }
    $$renderer2.push(`<!--]--></div> <div class="input-area svelte-1uha8ag"><textarea placeholder="Enter command (e.g., load_dataset path=/data/file.csv)"${attr("disabled", chatState.isProcessing, true)} rows="3" class="svelte-1uha8ag">`);
    const $$body = escape_html(chatState.input);
    if ($$body) {
      $$renderer2.push(`${$$body}`);
    }
    $$renderer2.push(`</textarea> <button${attr("disabled", chatState.isProcessing || !chatState.input.trim(), true)} class="svelte-1uha8ag">${escape_html(chatState.isProcessing ? "Processing..." : "Send")}</button></div></div> <div class="panel data-panel svelte-1uha8ag"><div class="panel-header"><h2>Data</h2> `);
    if (datasetsState.datasets.length > 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.select(
        {
          value: datasetsState.activeDataset,
          onchange: (e) => datasetsState.setActiveDataset(e.currentTarget.value),
          class: ""
        },
        ($$renderer3) => {
          $$renderer3.option({ value: "" }, ($$renderer4) => {
            $$renderer4.push(`Select dataset...`);
          });
          $$renderer3.push(`<!--[-->`);
          const each_array_2 = ensure_array_like(datasetsState.datasets);
          for (let $$index_2 = 0, $$length = each_array_2.length; $$index_2 < $$length; $$index_2++) {
            let ds = each_array_2[$$index_2];
            $$renderer3.option({ value: ds.name }, ($$renderer4) => {
              $$renderer4.push(`${escape_html(ds.name)} (${escape_html(ds.rows)} x ${escape_html(ds.columns)})`);
            });
          }
          $$renderer3.push(`<!--]-->`);
        },
        "svelte-1uha8ag"
      );
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div> <div class="panel-content svelte-1uha8ag">`);
    if (datasetsState.datasets.length === 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="empty-state svelte-1uha8ag"><p class="svelte-1uha8ag">No datasets loaded</p> <p class="text-muted svelte-1uha8ag">Use "Import File" or the load_dataset command</p></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
      if (!datasetsState.activeDataset) {
        $$renderer2.push("<!--[-->");
        $$renderer2.push(`<div class="empty-state svelte-1uha8ag"><p class="svelte-1uha8ag">Select a dataset to view</p></div>`);
      } else {
        $$renderer2.push("<!--[!-->");
        if (datasetsState.preview) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<div class="data-table-container svelte-1uha8ag"><table><thead><tr><!--[-->`);
          const each_array_3 = ensure_array_like(datasetsState.preview.columns);
          for (let $$index_3 = 0, $$length = each_array_3.length; $$index_3 < $$length; $$index_3++) {
            let col = each_array_3[$$index_3];
            $$renderer2.push(`<th class="svelte-1uha8ag">${escape_html(col)} `);
            if (datasetsState.sortColumn === col) {
              $$renderer2.push("<!--[-->");
              $$renderer2.push(`<span class="sort-indicator svelte-1uha8ag">${escape_html(datasetsState.sortDirection === "asc" ? "↑" : "↓")}</span>`);
            } else {
              $$renderer2.push("<!--[!-->");
            }
            $$renderer2.push(`<!--]--></th>`);
          }
          $$renderer2.push(`<!--]--></tr></thead><tbody><!--[-->`);
          const each_array_4 = ensure_array_like(datasetsState.preview.rows);
          for (let i = 0, $$length = each_array_4.length; i < $$length; i++) {
            let row = each_array_4[i];
            $$renderer2.push(`<tr${attr_class("", void 0, { "even": i % 2 === 0 })}><!--[-->`);
            const each_array_5 = ensure_array_like(datasetsState.preview.columns);
            for (let $$index_4 = 0, $$length2 = each_array_5.length; $$index_4 < $$length2; $$index_4++) {
              let col = each_array_5[$$index_4];
              $$renderer2.push(`<td>${escape_html(row[col] ?? "")}</td>`);
            }
            $$renderer2.push(`<!--]--></tr>`);
          }
          $$renderer2.push(`<!--]--></tbody></table></div>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]-->`);
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div></div> <div class="panel results-panel svelte-1uha8ag"><div class="panel-header"><h2>Results</h2> `);
    if (resultsState.results.length > 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<button class="secondary">Clear</button>`);
    } else {
      $$renderer2.push("<!--[!-->");
    }
    $$renderer2.push(`<!--]--></div> <div class="panel-content svelte-1uha8ag">`);
    if (resultsState.results.length === 0) {
      $$renderer2.push("<!--[-->");
      $$renderer2.push(`<div class="empty-state svelte-1uha8ag"><p class="svelte-1uha8ag">No results yet</p> <p class="text-muted svelte-1uha8ag">Run analyses to see results here</p></div>`);
    } else {
      $$renderer2.push("<!--[!-->");
      $$renderer2.push(`<!--[-->`);
      const each_array_6 = ensure_array_like(resultsState.results);
      for (let $$index_7 = 0, $$length = each_array_6.length; $$index_7 < $$length; $$index_7++) {
        let result = each_array_6[$$index_7];
        $$renderer2.push(`<div${attr_class("result-item svelte-1uha8ag", void 0, { "expanded": resultsState.expandedResult === result.id })}><button class="result-header svelte-1uha8ag"><span class="tool-name svelte-1uha8ag">${escape_html(result.tool)}</span> <span class="time">${escape_html(formatTime(result.timestamp))}</span></button> `);
        if (resultsState.expandedResult === result.id) {
          $$renderer2.push("<!--[-->");
          $$renderer2.push(`<div class="result-content svelte-1uha8ag"><pre class="svelte-1uha8ag">${escape_html(result.content)}</pre> <!--[-->`);
          const each_array_7 = ensure_array_like(result.images);
          for (let $$index_6 = 0, $$length2 = each_array_7.length; $$index_6 < $$length2; $$index_6++) {
            let img = each_array_7[$$index_6];
            $$renderer2.push(`<img${attr("src", `data:image/png;base64,${stringify(img)}`)}${attr("alt", `${stringify(result.tool)} output`)} class="svelte-1uha8ag"/>`);
          }
          $$renderer2.push(`<!--]--></div>`);
        } else {
          $$renderer2.push("<!--[!-->");
        }
        $$renderer2.push(`<!--]--></div>`);
      }
      $$renderer2.push(`<!--]-->`);
    }
    $$renderer2.push(`<!--]--></div></div></div>`);
  });
}
export {
  _page as default
};
