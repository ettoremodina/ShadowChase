/* The play window: board on the left, one panel of decisions on the right. */

import { api } from "./api.js";
import { BoardView } from "./board.js";

const $ = (id) => document.getElementById(id);

const dom = {
  stageKind: $("stage-kind"),
  stageBoard: $("stage-board"),
  ruler: $("ruler"),
  rulerCaption: $("ruler-caption"),
  badgeTurn: $("badge-turn"),
  badgeMrx: $("badge-mrx"),
  legend: $("legend"),
  panel: $("panel"),
  actions: $("actions"),
  toasts: $("toasts"),
};

let snapshot = null;
let boardInfo = null;
let layout = null;
let resultShownFor = null;
let pickedBoard = null;

const board = new BoardView($("board"), {
  onNodeClick: (id) => send(() => api.clickNode(id)),
});

/* ---- plumbing ------------------------------------------------------- */

const escape = (value) =>
  String(value).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

function apply(payload) {
  if (payload.layout) {
    layout = payload.layout;
    board.setLayout(layout);
    renderLegend();
  }
  if (payload.board) boardInfo = payload.board;
  snapshot = payload.state;
  render();
}

async function send(action) {
  try {
    apply(await action());
  } catch (error) {
    toast({ level: "error", title: "Something went wrong", text: error.message });
  }
}

function toast({ level, title, text }) {
  const node = document.createElement("div");
  node.className = "toast";
  node.dataset.level = level || "info";
  node.innerHTML = `<div class="toast-bar"></div><div class="toast-body">
      <b>${escape(title)}</b><span>${escape(text)}</span></div>`;
  dom.toasts.appendChild(node);
  setTimeout(() => {
    node.style.transition = "opacity 220ms";
    node.style.opacity = "0";
    setTimeout(() => node.remove(), 240);
  }, 4200);
}

/* ---- stage ---------------------------------------------------------- */

function renderLegend() {
  if (!layout) return;
  const seen = new Set();
  const items = [];
  for (const edge of layout.edges) {
    if (seen.has(edge.key)) continue;
    seen.add(edge.key);
    const style = layout.transports[String(edge.transport)];
    items.push(
      `<span class="legend-item"><i class="legend-line" style="background:${style.color}"></i>${escape(
        style.name
      )}</span>`
    );
  }
  dom.legend.innerHTML = items.join("");
}

function renderRuler() {
  const game = snapshot.game;
  const reveals = game.revealTurns || [];
  const total = reveals.length ? Math.max(...reveals) : 24;
  const now = game.mrxTurnCount || 0;

  const ticks = [];
  for (let turn = 1; turn <= total; turn += 1) {
    const isReveal = reveals.includes(turn);
    const classes = ["ruler-tick"];
    if (isReveal) classes.push("is-reveal");
    if (turn < now) classes.push("is-past");
    if (turn === now) classes.push("is-now");
    ticks.push(
      `<i class="${classes.join(" ")}" data-reveal="${isReveal}" title="Turn ${turn}${
        isReveal ? " — Mr. X surfaces" : ""
      }"></i>`
    );
  }
  dom.ruler.innerHTML = ticks.join("");

  if (!game.started) {
    dom.rulerCaption.textContent = reveals.join(" · ") || "—";
  } else if (game.mrxVisible) {
    dom.rulerCaption.textContent = `surfaced on ${now}`;
  } else {
    const next = reveals.find((turn) => turn > now);
    dom.rulerCaption.textContent = next
      ? `next: ${next} (${next - now} to go)`
      : "no more reveals";
  }
}

function renderStage() {
  const game = snapshot.game;
  const option = (boardInfo?.options || []).find(
    (item) => item.key === boardInfo.selected
  );
  dom.stageBoard.textContent = option ? option.name : "Board";
  dom.stageKind.textContent = `${snapshot.numDetectives} detectives · ${
    snapshot.kind === "shadow_chase" ? "ticket rules" : "basic rules"
  }`;

  renderRuler();

  dom.badgeTurn.innerHTML = `Turn <b>${game.turnCount || 0}</b>`;

  dom.badgeMrx.className = "badge";
  if (!game.started) {
    dom.badgeMrx.textContent = "Not started";
  } else if (game.gameOver) {
    dom.badgeMrx.textContent =
      game.winner === "detectives" ? "Detectives win" : "Mr. X wins";
  } else if (game.mrxVisible) {
    dom.badgeMrx.classList.add("is-visible-x");
    dom.badgeMrx.innerHTML = `Mr. X at <b>${game.mrxPosition}</b>`;
  } else {
    dom.badgeMrx.classList.add("is-hidden-x");
    dom.badgeMrx.textContent = "Mr. X under cover";
  }

  $("rail-photo").setAttribute("aria-pressed", String(snapshot.showBoardImage));
  $("rail-photo").disabled = !snapshot.hasBoardImage;
  $("rail-suspects").setAttribute(
    "aria-pressed",
    String(snapshot.setup.heuristics)
  );
  $("rail-suspects").disabled = !snapshot.setup.heuristicsAvailable;
}

function renderBoard() {
  const game = snapshot.game;
  const setup = snapshot.setup;

  let detectives = game.detectivePositions;
  let mrx = game.mrxPosition;
  let staged = snapshot.selections.nodes;

  if (snapshot.phase === "setup") {
    detectives = setup.detectives;
    mrx = setup.mrx ?? null;
    staged = [];
  }

  board.render({
    showImage: snapshot.showBoardImage,
    showEdges: !snapshot.showBoardImage,
    detectives,
    mrx,
    mrxVisible: game.mrxVisible || game.gameOver,
    destinations: snapshot.highlight.destinations,
    highlightEdges: snapshot.highlight.edges,
    active: snapshot.highlight.active,
    staged,
    suspects: snapshot.suspects,
    labels: snapshot.showBoardImage ? "notable" : "all",
  });
}

/* ---- panel ---------------------------------------------------------- */

function render() {
  renderStage();
  renderBoard();
  dom.panel.innerHTML =
    snapshot.phase === "setup" ? setupPanel() : playPanel();
  dom.actions.innerHTML =
    snapshot.phase === "setup" ? setupActions() : playActions();
  for (const notice of snapshot.notices) toast(notice);
  maybeShowResult();
}

function setupPanel() {
  const setup = snapshot.setup;
  const needsAgents =
    setup.supportsAgents &&
    ["human_det_vs_ai_mrx", "ai_det_vs_human_mrx", "ai_vs_ai"].includes(
      setup.mode
    );
  const needsMrx = ["human_det_vs_ai_mrx", "ai_vs_ai"].includes(setup.mode);
  const needsDetectives = ["ai_det_vs_human_mrx", "ai_vs_ai"].includes(
    setup.mode
  );

  const modes = setup.modes
    .map(
      (mode) => `
      <label class="choice ${mode.value === setup.mode ? "is-on" : ""}">
        <input type="radio" name="mode" value="${mode.value}" ${
        mode.value === setup.mode ? "checked" : ""
      } ${setup.supportsAgents || mode.value === "human_vs_human" ? "" : "disabled"} />
        <span class="choice-text"><b>${escape(mode.label)}</b><span>${escape(
        mode.detail
      )}</span></span>
      </label>`
    )
    .join("");

  const agentSelect = (side, current, label) => `
    <div class="field">
      <label class="eyebrow" for="agent-${side}">${label}</label>
      <select class="select" id="agent-${side}" data-agent="${side}">
        ${setup.agents
          .map(
            (agent) =>
              `<option value="${agent.value}" ${
                agent.value === current ? "selected" : ""
              }>${escape(agent.label)}</option>`
          )
          .join("")}
      </select>
    </div>`;

  const slots = [];
  for (let index = 0; index < snapshot.numDetectives; index += 1) {
    const station = setup.detectives[index];
    slots.push(`
      <div class="card-slot ${station === undefined ? "is-empty" : ""}" data-side="detectives">
        <span class="who">D${index + 1}</span>
        <span class="slot-station">${station ?? "··"}</span>
      </div>`);
  }
  slots.push(`
    <div class="card-slot ${setup.mrx == null ? "is-empty" : ""}" data-side="mrx">
      <span class="who">Mr. X</span>
      <span class="slot-station">${setup.mrx ?? "··"}</span>
    </div>`);

  return `
    <section class="section">
      <div class="section-head"><span class="eyebrow">Who plays</span></div>
      <div class="choice-list">${modes}</div>
      ${
        needsAgents
          ? `${needsMrx ? agentSelect("mrx", setup.mrxAgent, "Mr. X agent") : ""}
             ${
               needsDetectives
                 ? agentSelect(
                     "detectives",
                     setup.detectiveAgent,
                     "Detective agent"
                   )
                 : ""
             }`
          : ""
      }
    </section>

    <section class="section">
      <div class="section-head">
        <span class="eyebrow">Starting stations</span>
        <span class="section-note">${setup.selected.length} of ${setup.needed}</span>
      </div>
      <div class="hand">${slots.join("")}</div>
      <p class="instruction" style="border-top: none; padding-top: 12px; margin-top: 8px">
        ${escape(setup.hint)}
      </p>
    </section>`;
}

function playPanel() {
  const turn = snapshot.turnPanel;
  const moves = snapshot.moves;
  const picks = snapshot.selections;

  const pips =
    turn.progress && turn.progress.total > 1
      ? `<div class="pips">${Array.from(
          { length: turn.progress.total },
          (_, index) =>
            `<i class="pip ${index < turn.progress.done ? "is-done" : ""}"></i>`
        ).join("")}</div>`
      : "";

  const picker = snapshot.pendingTransport
    ? `<div class="picker">
         <span class="eyebrow">Ticket for station ${snapshot.pendingTransport.node}</span>
         <div class="picker-options">
           ${snapshot.pendingTransport.options
             .map(
               (option) => `
             <button class="ticket-btn" data-ticket="${option.transport}">
               <i class="dot" style="background:${option.color}"></i>${escape(
                 option.name
               )}
             </button>`
             )
             .join("")}
         </div>
       </div>`
    : "";

  const stagedRows = [
    ...picks.detectives.map(
      (pick) => `
      <div class="pick" style="border-left-color:${pick.color}">
        <span class="pick-who">D${pick.index + 1}</span>
        <span class="pick-route">${pick.from} &rarr; ${pick.to}</span>
        <span class="pick-ticket" style="color:${pick.color}">${escape(
        pick.ticket
      )}</span>
      </div>`
    ),
    ...picks.mrx.map(
      (pick) => `
      <div class="pick" style="border-left-color:${pick.color}">
        <span class="pick-who">Mr. X</span>
        <span class="pick-route">&rarr; ${pick.to}</span>
        <span class="pick-ticket" style="color:${pick.color}">${escape(
        pick.ticket
      )}</span>
      </div>`
    ),
  ].join("");

  const routes = moves.empty
    ? `<p class="empty-note">No route out of station ${
        moves.source ?? "—"
      }. This piece stands still.</p>`
    : moves.groups
        .map(
          (group) => `
      <div class="route-group">
        <div class="route-head">
          <i class="route-rule" style="background:${group.color}"></i>
          <span class="route-name">${escape(group.name)}</span>
          <span class="route-count">${group.destinations.length}</span>
        </div>
        <div class="station-chips">
          ${group.destinations
            .map(
              (station) =>
                `<button class="chip ${
                  picks.nodes.includes(station) ? "is-picked" : ""
                }" data-station="${station}">${station}</button>`
            )
            .join("")}
        </div>
      </div>`
        )
        .join("");

  return `
    <section class="section">
      <div class="turn-card" data-actor="${turn.actor}">
        <div class="turn-top">
          <div>
            <div class="turn-name">${escape(turn.label)}</div>
            <div class="turn-sub">${escape(turn.sublabel)}</div>
          </div>
          <span class="controller" data-kind="${turn.controller}">${
    turn.controller === "ai" ? "Agent" : "You"
  }</span>
        </div>
        ${pips}
        <p class="instruction">${escape(turn.instruction)}</p>
        ${picker}
      </div>
    </section>

    ${
      stagedRows
        ? `<section class="section">
             <div class="section-head"><span class="eyebrow">Ready to send</span></div>
             <div class="pick-list">${stagedRows}</div>
           </section>`
        : ""
    }

    <section class="section">
      <div class="section-head">
        <span class="eyebrow">Where you can go</span>
        <span class="section-note">${
          moves.source != null ? `from ${moves.source}` : ""
        }</span>
      </div>
      ${routes}
    </section>

    ${ticketTable()}`;
}

function ticketTable() {
  const tickets = snapshot.tickets;
  if (!tickets.enabled) return "";

  const head = tickets.order
    .map(
      (key) =>
        `<th><i class="th-dot" style="background:${
          tickets.colors[key] || "#6d7a8b"
        }"></i><span>${escape(tickets.labels[key])}</span></th>`
    )
    .join("");

  const activeSide =
    snapshot.turnPanel.actor === "mrx" ? "mrx" : "detectives";
  const activeIndex =
    snapshot.turnPanel.actor === "detectives"
      ? snapshot.selections.detectives.length
      : -1;

  const rows = tickets.rows
    .map((row, index) => {
      const isActive =
        row.side === activeSide &&
        (row.side === "mrx" || index === activeIndex);
      const cells = tickets.order
        .map((key) => {
          const count = row.counts[key];
          if (count === null || count === undefined) {
            return `<td class="count-none">&mdash;</td>`;
          }
          return `<td class="${count === 0 ? "count-zero" : ""}">${count}</td>`;
        })
        .join("");
      return `
        <tr data-side="${row.side}" class="${isActive ? "is-active" : ""}">
          <td>
            <span class="ticket-row-name">
              <b>${escape(row.short)}</b>
              <span class="where">${row.position ?? "··"}</span>
            </span>
          </td>
          ${cells}
        </tr>`;
    })
    .join("");

  return `
    <section class="section">
      <div class="section-head">
        <span class="eyebrow">Tickets</span>
        <span class="section-note">count &middot; station</span>
      </div>
      <table class="ticket-table">
        <thead><tr><th></th>${head}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </section>`;
}

function setupActions() {
  const canStart = snapshot.actions.start;
  const dealable = (boardInfo?.options || []).find(
    (item) => item.key === boardInfo.selected
  )?.dealsCards;

  return `
    <div class="btn-row">
      ${
        dealable
          ? `<button class="btn" data-do="deal">Deal stations</button>`
          : ""
      }
      <button class="btn" data-do="clear">Clear</button>
    </div>
    <button class="btn btn-go" data-do="start" ${canStart ? "" : "disabled"}>
      Start the chase
    </button>`;
}

function playActions() {
  const actions = snapshot.actions;
  const double = snapshot.turnPanel.doubleMove;

  if (snapshot.game.gameOver) {
    return `<button class="btn btn-primary" data-do="newgame">New game</button>`;
  }

  if (actions.aiContinue) {
    return `
      <button class="btn btn-primary" data-do="agent">
        Play agent move<span class="btn-key">Enter</span>
      </button>
      <div class="btn-row">
        <button class="btn btn-ghost" data-do="clear">Back to setup</button>
      </div>`;
  }

  return `
    ${
      double.available || double.requested
        ? `<button class="btn btn-double" data-do="double" aria-pressed="${double.requested}">
             ${double.requested ? "Double move armed" : "Arm double move"}
           </button>`
        : ""
    }
    <button class="btn btn-primary" data-do="confirm" ${
      actions.confirmMove ? "" : "disabled"
    }>
      Confirm move<span class="btn-key">Enter</span>
    </button>
    <div class="btn-row">
      <button class="btn btn-ghost" data-do="still" ${
        actions.skip ? "" : "disabled"
      }>Stand still</button>
      <button class="btn btn-ghost" data-do="undo" ${
        actions.clearSelection ? "" : "disabled"
      }>Undo picks</button>
    </div>`;
}

/* ---- result --------------------------------------------------------- */

function maybeShowResult() {
  const game = snapshot.game;
  if (!game.gameOver) {
    resultShownFor = null;
    return;
  }
  const key = `${game.winner}-${game.turnCount}`;
  if (resultShownFor === key) return;
  resultShownFor = key;

  const detectivesWon = game.winner === "detectives";
  $("result-card").dataset.winner = game.winner || "";
  $("result-eyebrow").textContent = `Game over after ${game.mrxTurnCount} Mr. X turns`;
  $("result-title").textContent = detectivesWon
    ? "Detectives win"
    : "Mr. X escapes";
  $("result-detail").textContent = detectivesWon
    ? `Cornered at station ${game.mrxPosition}.`
    : "The detectives ran out of moves and turns.";
  $("result-saved").textContent = snapshot.savedGameId
    ? `Saved as ${snapshot.savedGameId}`
    : "";
  $("modal-result").hidden = false;
}

/* ---- modals --------------------------------------------------------- */

function openModal(id) {
  $(id).hidden = false;
}

function closeModal(id) {
  $(id).hidden = true;
}

function renderBoardOptions() {
  pickedBoard = pickedBoard || boardInfo.selected;
  const options = boardInfo.options;
  $("board-options").innerHTML = options
    .map(
      (option) => `
      <label class="choice ${option.key === pickedBoard ? "is-on" : ""}">
        <input type="radio" name="board" value="${option.key}" ${
        option.key === pickedBoard ? "checked" : ""
      } />
        <span class="choice-text"><b>${escape(option.name)}</b><span>${escape(
        option.blurb
      )}</span></span>
      </label>`
    )
    .join("");

  const option = options.find((item) => item.key === pickedBoard);
  const select = $("board-detectives");
  select.innerHTML = option.detectives
    .map(
      (count) =>
        `<option value="${count}" ${
          count === option.defaultDetectives ? "selected" : ""
        }>${count} detective${count === 1 ? "" : "s"}</option>`
    )
    .join("");
  select.disabled = option.detectives.length < 2;
}

/* Saved games and export share the same picker. */
const picker = { saves: null, chosen: { saves: null, export: null } };

async function loadSaves(target) {
  const search = $(`${target}-search`).value;
  const rows = $(`${target}-rows`);
  rows.innerHTML = `<tr><td colspan="3">Searching…</td></tr>`;
  try {
    const { saves } = await api.saves(search);
    picker.saves = saves;
    if (!saves.length) {
      rows.innerHTML = `<tr><td colspan="3">No saved games match.</td></tr>`;
      if (target === "saves") $("saves-note").textContent = "";
      return;
    }
    rows.innerHTML = saves
      .map(
        (save) => `
        <tr data-path="${escape(save.path)}" data-id="${escape(save.gameId)}">
          <td class="id">${escape(save.gameId)}</td>
          <td>${escape(save.folder || "—")}</td>
          <td>${escape(save.modifiedLabel)}</td>
        </tr>`
      )
      .join("");
    if (target === "saves") {
      $("saves-note").textContent = `${saves.length} shown, newest first.`;
    }
  } catch (error) {
    rows.innerHTML = `<tr><td colspan="3">${escape(error.message)}</td></tr>`;
  }
}

function wirePicker(target, onPick) {
  const rows = $(`${target}-rows`);
  rows.addEventListener("click", (event) => {
    const row = event.target.closest("tr[data-path]");
    if (!row) return;
    for (const other of rows.querySelectorAll("tr")) {
      other.classList.remove("is-picked");
    }
    row.classList.add("is-picked");
    picker.chosen[target] = { path: row.dataset.path, id: row.dataset.id };
    onPick(picker.chosen[target]);
  });

  let timer = null;
  $(`${target}-search`).addEventListener("input", () => {
    clearTimeout(timer);
    timer = setTimeout(() => loadSaves(target), 220);
  });
}

/* ---- events --------------------------------------------------------- */

function wire() {
  $("rail-board").addEventListener("click", () => {
    pickedBoard = boardInfo.selected;
    renderBoardOptions();
    openModal("modal-board");
  });
  $("rail-replays").addEventListener("click", () => {
    openModal("modal-saves");
    loadSaves("saves");
  });
  $("rail-export").addEventListener("click", () => {
    openModal("modal-export");
    loadSaves("export");
  });
  $("rail-photo").addEventListener("click", () => send(api.togglePhoto));
  $("rail-suspects").addEventListener("click", () =>
    send(() => api.setHeuristics(!snapshot.setup.heuristics))
  );

  $("zoom-in").addEventListener("click", () => board.zoomBy(1.3));
  $("zoom-out").addEventListener("click", () => board.zoomBy(1 / 1.3));
  $("zoom-fit").addEventListener("click", () => board.fit());

  document.addEventListener("click", (event) => {
    const close = event.target.closest("[data-close]");
    if (close) closeModal(close.dataset.close);
  });

  // Panel is re-rendered wholesale, so everything is delegated.
  dom.panel.addEventListener("click", (event) => {
    const ticket = event.target.closest("[data-ticket]");
    if (ticket) {
      send(() => api.chooseTicket(Number(ticket.dataset.ticket)));
      return;
    }
    const chip = event.target.closest("[data-station]");
    if (chip) send(() => api.clickNode(Number(chip.dataset.station)));
  });

  dom.panel.addEventListener("change", (event) => {
    if (event.target.name === "mode") {
      send(() => api.setMode(event.target.value));
      return;
    }
    const agent = event.target.dataset?.agent;
    if (agent) send(() => api.setAgent(agent, event.target.value));
  });

  dom.panel.addEventListener("mouseover", (event) => {
    const chip = event.target.closest("[data-station]");
    board.hint(chip ? Number(chip.dataset.station) : null);
  });
  dom.panel.addEventListener("mouseleave", () => board.hint(null));

  dom.actions.addEventListener("click", (event) => {
    const button = event.target.closest("[data-do]");
    if (!button || button.disabled) return;
    const actions = {
      deal: () => send(api.dealStations),
      clear: () => send(api.clearSetup),
      start: () => send(api.startGame),
      confirm: () => send(api.confirmMove),
      still: () => send(api.standStill),
      undo: () => send(api.clearPicks),
      agent: () => send(api.playAgent),
      double: () => send(api.toggleDouble),
      newgame: () => {
        closeModal("modal-result");
        send(api.clearSetup);
      },
    };
    actions[button.dataset.do]?.();
  });

  // New game
  $("board-options").addEventListener("change", (event) => {
    pickedBoard = event.target.value;
    renderBoardOptions();
  });
  $("board-create").addEventListener("click", async () => {
    closeModal("modal-board");
    closeModal("modal-result");
    await send(() =>
      api.newGame(pickedBoard, Number($("board-detectives").value), true)
    );
  });

  // Saved games
  wirePicker("saves", () => {
    $("saves-open").disabled = false;
  });
  $("saves-open").addEventListener("click", () => {
    const chosen = picker.chosen.saves;
    if (!chosen) return;
    window.open(
      `/replay?path=${encodeURIComponent(chosen.path)}`,
      "_blank",
      "noopener"
    );
    closeModal("modal-saves");
  });

  // Export
  wirePicker("export", (chosen) => {
    $("export-start").disabled = false;
    $("export-name").value = `${chosen.id}.mp4`;
  });
  $("export-start").addEventListener("click", startExport);

  // Result
  $("result-review").addEventListener("click", () => closeModal("modal-result"));
  $("result-new").addEventListener("click", () => {
    closeModal("modal-result");
    pickedBoard = boardInfo.selected;
    renderBoardOptions();
    openModal("modal-board");
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      for (const id of ["modal-board", "modal-saves", "modal-export", "modal-result"]) {
        if (!$(id).hidden) {
          closeModal(id);
          return;
        }
      }
      if (snapshot?.pendingTransport) send(api.cancelTicket);
      return;
    }

    const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(
      document.activeElement?.tagName
    );
    if (typing) return;

    if (event.key === "Enter") {
      if (!$("modal-result").hidden) return;
      if (snapshot?.phase === "setup") {
        if (snapshot.actions.start) send(api.startGame);
      } else if (snapshot?.actions.aiContinue) {
        send(api.playAgent);
      } else if (snapshot?.actions.confirmMove) {
        send(api.confirmMove);
      }
    } else if (event.key === "f") {
      board.fit();
    }
  });
}

async function startExport() {
  const chosen = picker.chosen.export;
  if (!chosen) return;

  $("export-start").disabled = true;
  $("export-progress").hidden = false;
  $("export-message").textContent = "Starting";
  $("export-bar").style.width = "0%";

  try {
    const job = await api.exportVideo({
      path: chosen.path,
      filename: $("export-name").value,
      frameDuration: Number($("export-duration").value) || 1,
      endDelay: Number($("export-delay").value) || 0,
    });
    pollExport(job.jobId);
  } catch (error) {
    $("export-progress").hidden = true;
    $("export-start").disabled = false;
    toast({ level: "error", title: "Export failed", text: error.message });
  }
}

async function pollExport(jobId) {
  try {
    const job = await api.exportJob(jobId);
    const percent = job.total ? Math.round((job.current / job.total) * 100) : 5;
    $("export-bar").style.width = `${percent}%`;
    $("export-message").textContent = job.message;

    if (job.status === "running") {
      setTimeout(() => pollExport(jobId), 600);
      return;
    }

    $("export-start").disabled = false;
    if (job.status === "done") {
      $("export-bar").style.width = "100%";
      toast({
        level: "info",
        title: "Video exported",
        text: `Saved to ${job.output}`,
      });
    } else {
      toast({
        level: "error",
        title: "Export failed",
        text: job.error || "The renderer stopped early.",
      });
    }
  } catch (error) {
    $("export-start").disabled = false;
    toast({ level: "error", title: "Export failed", text: error.message });
  }
}

/* ---- start ---------------------------------------------------------- */

wire();
send(api.bootstrap);
