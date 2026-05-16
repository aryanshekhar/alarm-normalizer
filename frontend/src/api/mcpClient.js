/**
 * All backend calls in one place.
 * In dev, Vite proxies /tools/* and /ws/* to localhost:8000.
 * In production, nginx does the same (BACKEND_URL env var).
 * VITE_BACKEND_URL is only needed to point at a remote host directly.
 */
const BASE = import.meta.env.VITE_BACKEND_URL ?? '';

// ── Topology ──────────────────────────────────────────────────────────────────

export async function getTopology(domain = null) {
  const url = `${BASE}/tools/get_topology${domain ? `?domain=${domain}` : ''}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`get_topology failed: ${res.status}`);
  return res.json();
}

// ── Training (SSE async generator) ────────────────────────────────────────────

export async function* trainModel(epochs = 5, dataWindow = 30, demoMode = true) {
  const res = await fetch(`${BASE}/tools/train_model`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ epochs, data_window: dataWindow, demo_mode: demoMode }),
  });
  if (!res.ok) throw new Error(`train_model failed: ${res.status}`);

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop(); // keep partial line
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const event = JSON.parse(line.slice(6));
          console.log('[SSE train_model]', event);
          yield event;
          // Yield to the macrotask queue between events from the same chunk
          // so React can flush each state update independently.
          await new Promise((r) => setTimeout(r, 0));
        } catch { /* skip malformed */ }
      }
    }
  }
}

// ── Inference ─────────────────────────────────────────────────────────────────

export async function runInference(kpiWindow = 'anomalous') {
  const res = await fetch(`${BASE}/tools/run_inference`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kpi_window: kpiWindow }),
  });
  if (!res.ok) throw new Error(`run_inference failed: ${res.status}`);
  const data = await res.json();
  console.log('[runInference] raw response:', data);
  return data;
}

// ── Alarm correlation (SSE narrative stream) ──────────────────────────────────

export async function* correlateAlarmsStream(alarmIds = []) {
  const res = await fetch(`${BASE}/tools/correlate_alarms`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ alarm_ids: alarmIds, include_cleared: false }),
  });
  if (!res.ok) throw new Error(`correlate_alarms failed: ${res.status}`);

  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const event = JSON.parse(line.slice(6));
          console.log('[SSE correlate_alarms]', event);
          yield event;
          await new Promise((r) => setTimeout(r, 0));
        } catch { /* skip malformed */ }
      }
    }
  }
}

// ── RCA ───────────────────────────────────────────────────────────────────────

export async function getRca(incidentId = '', anomalyIds = [], alarmIds = []) {
  const res = await fetch(`${BASE}/tools/get_rca`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      incident_id: incidentId,
      anomaly_ids: anomalyIds,
      alarm_ids: alarmIds,
    }),
  });
  if (!res.ok) throw new Error(`get_rca failed: ${res.status}`);
  return res.json();
}

// ── Assistant ─────────────────────────────────────────────────────────────────

export async function askAssistant(question, context = {}) {
  const res = await fetch(`${BASE}/tools/ask_assistant`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, context }),
  });
  if (!res.ok) throw new Error(`ask_assistant failed: ${res.status}`);
  return res.json();
}

// ── WebSocket ─────────────────────────────────────────────────────────────────

export function createMonitorSocket(onMessage, onOpen, onClose) {
  const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws = new WebSocket(`${proto}//${window.location.host}/ws/monitor`);
  ws.onmessage = (e) => {
    try { onMessage(JSON.parse(e.data)); } catch { /* skip malformed */ }
  };
  if (onOpen) ws.onopen = onOpen;
  if (onClose) ws.onclose = onClose;
  ws.onerror = (e) => console.error('ws/monitor error', e);
  return ws;
}
