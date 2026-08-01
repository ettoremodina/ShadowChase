/* Thin wrapper over the server. Every mutating call answers with the whole
   snapshot, so callers never merge partial state. */

async function request(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return response.json();
}

const get = (url) => request(url);

const post = (url, body) =>
  request(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  });

export const api = {
  bootstrap: () => get("/api/bootstrap"),
  state: () => get("/api/state"),
  newGame: (board, detectives, deal) =>
    post("/api/game/new", { board, detectives, deal }),

  clickNode: (node) => post("/api/board/node", { node }),
  togglePhoto: () => post("/api/board/image/toggle"),

  setMode: (mode) => post("/api/setup/mode", { mode }),
  setAgent: (side, agent) => post("/api/setup/agent", { side, agent }),
  setHeuristics: (enabled) => post("/api/setup/heuristics", { enabled }),
  dealStations: () => post("/api/setup/deal"),
  clearSetup: () => post("/api/setup/clear"),
  startGame: () => post("/api/setup/start"),

  chooseTicket: (transport) => post("/api/play/transport", { transport }),
  cancelTicket: () => post("/api/play/transport/cancel"),
  confirmMove: () => post("/api/play/confirm"),
  standStill: () => post("/api/play/skip"),
  playAgent: () => post("/api/play/continue"),
  toggleDouble: () => post("/api/play/double"),
  clearPicks: () => post("/api/play/clear"),

  saves: (query) => get(`/api/saves?query=${encodeURIComponent(query || "")}`),
  openReplay: (path) => post("/api/replay/open", { path }),
  exportVideo: (payload) => post("/api/video/export", payload),
  exportJob: (jobId) => get(`/api/video/job/${jobId}`),
};
