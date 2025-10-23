// ---------- CONFIG ----------
const API = "http://127.0.0.1:8031";   // <-- set to your FastAPI port

// ---------- DOM ----------
const logEl     = document.getElementById("kc-chat-log");
const resultsEl = document.getElementById("kc-results");
const msgEl     = document.getElementById("kc-msg");
const sendBtn   = document.getElementById("kc-send");

// ---------- helpers ----------
function say(who, text) {
  const d = document.createElement("div");
  d.innerHTML = `<b>${who}:</b> ${text}`;
  logEl.appendChild(d);
  logEl.scrollTop = logEl.scrollHeight;
}
function textOrDash(x){ return x ? x : ""; }

// ---------- MAP ----------
const map = L.map("kc-map").setView([52.52, 13.405], 11);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: '&copy; OpenStreetMap'
}).addTo(map);
let markerLayer = L.layerGroup().addTo(map);

// ---------- render ----------
function renderResults(data){
  // list
  resultsEl.innerHTML = "";
  (data.results || []).forEach(r => {
    const d = document.createElement("div");
    d.className = "kc-card";
    d.innerHTML = `
      <div><strong>${textOrDash(r.title)}</strong></div>
      <div>${textOrDash(r.org)} ${r.date ? " · " + r.date : ""}</div>
      <div>${textOrDash(r.address)}</div>
      ${r.link ? `<div><a href="${r.link}" target="_blank" rel="noopener">Open</a></div>` : ""}
    `;
    resultsEl.appendChild(d);
  });

  // pins
  markerLayer.clearLayers();
  const pts = [];
  const inBerlin = (lat, lon) => lat >= 52.2 && lat <= 52.7 && lon >= 13.0 && lon <= 13.8;
  (data.markers || []).forEach(m => {
    const lat = Number(m.lat), lon = Number(m.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) return;
    if (!inBerlin(lat, lon)) return;
    const mk = L.marker([lat, lon]).bindPopup(
      `<strong>${textOrDash(m.title)}</strong><br>${textOrDash(m.org)}<br>${textOrDash(m.address)}<br>${textOrDash(m.date)}${
        m.link ? `<br><a href="${m.link}" target="_blank" rel="noopener">Open</a>` : ""
      }`
    );
    markerLayer.addLayer(mk);
    pts.push([lat, lon]);
  });
  if (pts.length) map.fitBounds(L.latLngBounds(pts).pad(0.2));
  else map.setView([52.52, 13.405], 11);
}

// ---------- chat ----------
async function ask(q){
  say("You", q);
  try {
    const res = await fetch(API + "/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ message: q })
    });
    if (!res.ok) {
      const txt = await res.text().catch(()=>"");
      say("Bot", `Error: ${res.status} ${res.statusText}${txt ? " — " + txt : ""}`);
      return;
    }
    const data = await res.json();
    say("Bot", data.reply || "No reply.");
    renderResults(data);
  } catch (e) {
    say("Bot", `Network error: ${e && e.message ? e.message : e}`);
  }
}

sendBtn.onclick = () => {
  const q = msgEl.value.trim();
  if (!q) return;
  msgEl.value = "";
  ask(q);
};
msgEl.addEventListener("keydown", e => { if (e.key === "Enter") sendBtn.click(); });

say("Bot", "Hello! Ask about tech events or jobs in Berlin (e.g., 'events in mitte', 'AI jobs').");
