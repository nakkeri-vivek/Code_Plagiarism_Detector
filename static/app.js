async function api(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  let data = null;
  const text = await res.text();
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }

  if (!res.ok) {
    const msg =
      (data && (data.detail || data.message)) ||
      `Request failed (${res.status})`;
    const err = new Error(msg);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

function $(id) {
  return document.getElementById(id);
}

function setStatus(el, msg, isError = false) {
  el.textContent = msg;
  el.classList.toggle("status--error", !!isError);
}

function readFileAsText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(reader.error || new Error("Failed to read file"));
    reader.readAsText(file);
  });
}

async function getCodeFromInputs(fileInput, textarea) {
  const file = fileInput.files && fileInput.files[0];
  if (file) return await readFileAsText(file);
  return textarea.value || "";
}

function detectLanguageFromFilename(filename) {
  const lower = String(filename || "").toLowerCase();
  const m = lower.match(/\.([a-z0-9]+)$/);
  const ext = m ? m[1] : "";
  const map = {
    py: "python",
    java: "java",
    js: "javascript",
    ts: "typescript",
    c: "c",
    h: "c",
    cpp: "cpp",
    cc: "cpp",
    cxx: "cpp",
    hpp: "cpp",
    cs: "csharp",
    go: "go",
    rs: "rust",
    php: "php",
    rb: "ruby",
  };
  return map[ext] || null;
}

function autoSetLanguageFromFile(fileInput, selectEl) {
  const file = fileInput.files && fileInput.files[0];
  if (!file) return;
  const lang = detectLanguageFromFilename(file.name);
  if (!lang) return;
  // Only set if option exists in select
  const opt = Array.from(selectEl.options).find((o) => o.value === lang);
  if (opt) selectEl.value = lang;
}

function scoreBadge(score) {
  const s = Number(score);
  if (s >= 0.85) return { cls: "badge badge--high", label: "High" };
  if (s >= 0.60) return { cls: "badge badge--mid", label: "Medium" };
  return { cls: "badge badge--low", label: "Low" };
}

function renderResults(result) {
  const empty = $("resultsEmpty");
  const wrap = $("resultsTableWrap");
  const body = $("resultsBody");
  const rawDetails = $("rawDetails");
  const rawJson = $("rawJson");

  body.innerHTML = "";

  const matches = (result && result.matches) || [];
  if (!matches.length) {
    empty.hidden = false;
    wrap.hidden = true;
  } else {
    empty.hidden = true;
    wrap.hidden = false;

    for (const m of matches) {
      const tr = document.createElement("tr");

      const tdId = document.createElement("td");
      tdId.textContent = String(m.submission_id);

      const tdLabel = document.createElement("td");
      tdLabel.textContent = m.label ? String(m.label) : "(no label)";

      const tdScore = document.createElement("td");
      const s = Number(m.similarity);
      const badge = scoreBadge(s);
      tdScore.innerHTML = `<span class="score">${(s * 100).toFixed(
        2
      )}%</span> <span class="${badge.cls}">${badge.label}</span>`;

      tr.appendChild(tdId);
      tr.appendChild(tdLabel);
      tr.appendChild(tdScore);
      body.appendChild(tr);
    }
  }

  rawDetails.hidden = false;
  rawJson.textContent = JSON.stringify(result, null, 2);
}

async function refreshStatus() {
  try {
    const st = await api("/status", { method: "GET" });
    $("baselineCountPill").textContent = `Baselines: ${st.baseline_count}`;
  } catch {
    // ignore
  }
}

async function onAddBaseline() {
  const statusEl = $("baselineStatus");
  setStatus(statusEl, "Adding baseline...");

  try {
    const label = $("baselineLabel").value || null;
    const language = $("baselineLanguage").value || "python";
    const fileInput = $("baselineFile");
    const files = Array.from(fileInput.files || []);

    // If there are files, treat each file as a separate baseline.
    if (files.length > 0) {
      let successCount = 0;
      for (const file of files) {
        const code = await readFileAsText(file);
        if (!code.trim()) continue;

        const autoLabel = file.name;
        const effectiveLabel = label || autoLabel;
        const autoLang = detectLanguageFromFilename(file.name) || language;

        await api("/baseline", {
          method: "POST",
          body: JSON.stringify({
            code,
            language: autoLang,
            label: effectiveLabel,
          }),
        });
        successCount += 1;
      }

      if (successCount === 0) {
        setStatus(
          statusEl,
          "Selected files are empty. Please choose non-empty files.",
          true
        );
      } else {
        setStatus(
          statusEl,
          `Added ${successCount} baseline${successCount > 1 ? "s" : ""} from files.`
        );
        fileInput.value = "";
        await refreshStatus();
      }
      return;
    }

    // Fallback: use the textarea as a single baseline.
    const code = $("baselineCode").value || "";
    if (!code.trim()) {
      setStatus(statusEl, "Please upload files or paste some code.", true);
      return;
    }

    const res = await api("/baseline", {
      method: "POST",
      body: JSON.stringify({ code, language, label }),
    });

    setStatus(
      statusEl,
      `Baseline added. ID: ${res.id}${res.label ? ` (label: ${res.label})` : ""}`
    );
    $("baselineFile").value = "";
    await refreshStatus();
  } catch (e) {
    setStatus(statusEl, e.message || "Failed to add baseline.", true);
  }
}

async function onResetBaselines() {
  const statusEl = $("baselineStatus");
  setStatus(statusEl, "Rebuilding baselines from database...");

  try {
    const res = await api("/baseline/reset", { method: "POST", body: "{}" });
    setStatus(
      statusEl,
      `In-memory models rebuilt from ${res.baseline_count} stored baselines.`
    );
    $("baselineCountPill").textContent = `Baselines: ${res.baseline_count}`;
  } catch (e) {
    setStatus(statusEl, e.message || "Failed to reset baselines.", true);
  }
}

async function onCheck() {
  const statusEl = $("checkStatus");
  setStatus(statusEl, "Checking similarity...");

  try {
    const code = await getCodeFromInputs($("studentFile"), $("studentCode"));
    const student_id = $("studentId").value || null;
    const top_k = Number($("topK").value || 5);
    const language = $("studentLanguage").value || "python";

    if (!code.trim()) {
      setStatus(statusEl, "Please upload a file or paste some code.", true);
      return;
    }

    const res = await api("/submit-code", {
      method: "POST",
      body: JSON.stringify({ code, language, student_id, top_k }),
    });

    setStatus(statusEl, `Done. Query submission ID: ${res.query_submission_id}`);
    $("studentFile").value = "";
    renderResults(res);
  } catch (e) {
    setStatus(statusEl, e.message || "Failed to check similarity.", true);
    renderResults({ matches: [] });
  }
}

function wire() {
  $("addBaselineBtn").addEventListener("click", onAddBaseline);
  $("resetBaselinesBtn").addEventListener("click", onResetBaselines);
  $("checkBtn").addEventListener("click", onCheck);
  $("baselineFile").addEventListener("change", () =>
    autoSetLanguageFromFile($("baselineFile"), $("baselineLanguage"))
  );
  $("studentFile").addEventListener("change", () =>
    autoSetLanguageFromFile($("studentFile"), $("studentLanguage"))
  );
  refreshStatus();
}

document.addEventListener("DOMContentLoaded", wire);

