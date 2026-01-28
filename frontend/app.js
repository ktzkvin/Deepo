const inputText = document.getElementById("inputText");
const outputText = document.getElementById("outputText");
const swapBtn = document.getElementById("swapBtn");
const clearBtn = document.getElementById("clearBtn");
const downloadBtn = document.getElementById("downloadBtn");
const statusEl = document.getElementById("status");
const inputCountEl = document.getElementById("inputCount");

const themeBtn = document.getElementById("themeBtn");
const demoBtn = document.getElementById("demoBtn");

const sourceBtn = document.getElementById("sourceBtn");
const targetBtn = document.getElementById("targetBtn");
const sourceLabel = document.getElementById("sourceLabel");
const targetLabel = document.getElementById("targetLabel");

const copyInputBtn = document.getElementById("copyInputBtn");
const copyOutputBtn = document.getElementById("copyOutputBtn");

const sourceMenu = document.getElementById("sourceMenu");
const targetMenu = document.getElementById("targetMenu");
const sourceSearch = document.getElementById("sourceSearch");
const targetSearch = document.getElementById("targetSearch");
const sourceList = document.getElementById("sourceList");
const targetList = document.getElementById("targetList");

let languages = null;

let sourceLang = "auto";
let targetLang = "fr";

let translateTimer = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function updateCount() {
  inputCountEl.textContent = String(inputText.value.length);
}

function getTheme() {
  return localStorage.getItem("deepo_theme") || "";
}

function isLightTheme() {
  return document.documentElement.getAttribute("data-theme") === "light";
}

function updateThemeIcon() {
  themeBtn.textContent = isLightTheme() ? "☀" : "🌙";
}

function setTheme(next) {
  if (next) document.documentElement.setAttribute("data-theme", next);
  else document.documentElement.removeAttribute("data-theme");
  localStorage.setItem("deepo_theme", next || "");
  updateThemeIcon();
}

function langName(code) {
  const map = {
    auto: "Détection auto",
    fr: "Français",
    en: "Anglais",
    es: "Espagnol",
    de: "Allemand",
    it: "Italien",
    ar: "Arabe",
  };
  return map[code] || code;
}

function refreshLangLabels() {
  sourceLabel.textContent = langName(sourceLang);
  targetLabel.textContent = langName(targetLang);
}

function closeMenus() {
  sourceMenu.classList.remove("open");
  targetMenu.classList.remove("open");
  sourceBtn.setAttribute("aria-expanded", "false");
  targetBtn.setAttribute("aria-expanded", "false");
  sourceMenu.setAttribute("aria-hidden", "true");
  targetMenu.setAttribute("aria-hidden", "true");
}

function openMenu(menuEl, btnEl, searchEl) {
  const isOpen = menuEl.classList.contains("open");
  closeMenus();
  if (isOpen) return;
  menuEl.classList.add("open");
  btnEl.setAttribute("aria-expanded", "true");
  menuEl.setAttribute("aria-hidden", "false");
  searchEl.value = "";
  setTimeout(() => searchEl.focus(), 0);
}

function buildMenu(listEl, items, onPick) {
  listEl.innerHTML = "";
  for (const code of items) {
    const row = document.createElement("div");
    row.className = "menuitem";
    row.setAttribute("role", "option");
    row.dataset.code = code;

    const left = document.createElement("div");
    left.className = "menuname";
    left.textContent = langName(code);

    const right = document.createElement("div");
    right.className = "menucode";
    right.textContent = code;

    row.appendChild(left);
    row.appendChild(right);

    row.addEventListener("click", () => onPick(code));
    listEl.appendChild(row);
  }
}

function filterMenu(mode) {
  if (!languages) return;

  const q = (mode === "source" ? sourceSearch.value : targetSearch.value).trim().toLowerCase();
  const pool = mode === "source" ? languages.source : languages.target;
  const listEl = mode === "source" ? sourceList : targetList;

  if (!q) {
    buildMenu(listEl, pool, (code) => pickLanguage(mode, code));
    return;
  }

  const filtered = pool.filter((c) => langName(c).toLowerCase().includes(q) || c.includes(q));
  buildMenu(listEl, filtered, (code) => pickLanguage(mode, code));
}

function pickLanguage(mode, code) {
  if (mode === "source") sourceLang = code;
  else targetLang = code;
  refreshLangLabels();
  closeMenus();
  scheduleTranslate(250);
}

async function loadLanguages() {
  const res = await fetch("/api/languages");
  if (!res.ok) return null;
  return await res.json();
}

function scheduleTranslate(delayMs = 1000) {
  if (translateTimer) clearTimeout(translateTimer);

  const text = inputText.value.trim();
  if (!text) {
    outputText.value = "";
    setStatus("Prêt");
    return;
  }

  setStatus("En attente...");
  translateTimer = setTimeout(() => {
    translateNow();
  }, delayMs);
}

async function translateNow() {
  const text = inputText.value.trim();
  if (!text) {
    outputText.value = "";
    setStatus("Prêt");
    return;
  }

  setStatus("Traduction...");

  try {
    const res = await fetch("/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        source_lang: sourceLang,
        target_lang: targetLang,
        mode: "lorem",
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    outputText.value = data.translated_text;
    setStatus(`OK ${data.elapsed_ms} ms`);
  } catch (e) {
    setStatus(`Erreur: ${e.message}`);
  }
}

function swap() {
  if (sourceLang === "auto") {
    setStatus("Choisis une langue source pour inverser");
    return;
  }

  const s = sourceLang;
  sourceLang = targetLang;
  targetLang = s;
  refreshLangLabels();

  const tmp = inputText.value;
  inputText.value = outputText.value;
  outputText.value = tmp;

  updateCount();
  scheduleTranslate(250);
}

async function copyText(value) {
  const text = (value || "").trim();
  if (!text) return;
  await navigator.clipboard.writeText(text);
}

function downloadOutput() {
  const text = outputText.value || "";
  const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `deepo_${targetLang}.txt`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

themeBtn.addEventListener("click", () => {
  setTheme(isLightTheme() ? "" : "light");
});

demoBtn.addEventListener("click", () => {
  inputText.value = "Hello world. This is a Deepo interface prototype.";
  updateCount();
  scheduleTranslate(200);
});

sourceBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  openMenu(sourceMenu, sourceBtn, sourceSearch);
  filterMenu("source");
});

targetBtn.addEventListener("click", (e) => {
  e.stopPropagation();
  openMenu(targetMenu, targetBtn, targetSearch);
  filterMenu("target");
});

sourceSearch.addEventListener("input", () => filterMenu("source"));
targetSearch.addEventListener("input", () => filterMenu("target"));

swapBtn.addEventListener("click", swap);

clearBtn.addEventListener("click", () => {
  inputText.value = "";
  outputText.value = "";
  updateCount();
  setStatus("Prêt");
});

downloadBtn.addEventListener("click", downloadOutput);

copyInputBtn.addEventListener("click", async () => {
  await copyText(inputText.value);
  setStatus("Copié");
});

copyOutputBtn.addEventListener("click", async () => {
  await copyText(outputText.value);
  setStatus("Copié");
});

inputText.addEventListener("input", () => {
  updateCount();
  scheduleTranslate(1000);
});

document.addEventListener("click", () => {
  closeMenus();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeMenus();
});

(async () => {
  setTheme(getTheme() || "");
  refreshLangLabels();
  updateCount();
  updateThemeIcon();

  languages = await loadLanguages();
  if (languages) {
    const defaults = languages.defaults || {};
    sourceLang = defaults.source || sourceLang;
    targetLang = defaults.target || targetLang;
    refreshLangLabels();

    buildMenu(sourceList, languages.source, (code) => pickLanguage("source", code));
    buildMenu(targetList, languages.target, (code) => pickLanguage("target", code));
  }

  setStatus("Prêt");
})();