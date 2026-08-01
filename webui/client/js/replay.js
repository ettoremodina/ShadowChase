/* The replay window: scrub a finished game, one recorded state at a time.
   Everything is loaded up front, so stepping never waits on the server. */

import { api } from "./api.js";
import { BoardView } from "./board.js";

const $ = (id) => document.getElementById(id);
const escape = (value) =>
  String(value).replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);

let data = null;
let step = 0;
let playing = false;
let timer = null;
let showPhoto = true;

const board = new BoardView($("board"));

/* ---- loading -------------------------------------------------------- */

async function load() {
  const path = new URLSearchParams(location.search).get("path");
  if (!path) {
    fail("No game selected", "Open a replay from the saved games list.");
    return;
  }
  try {
    data = await api.openReplay(path);
  } catch (error) {
    fail("Could not open that game", error.message);
    return;
  }

  document.title = `${data.gameId} — Shadow Chase`;
  $("replay-id").textContent = data.gameId;
  board.setLayout(data.board);
  showPhoto = Boolean(data.board.image);
  renderLegend();

  $("scrub").max = String(Math.max(0, data.steps.length - 1));
  step = 0;
  render();
}

function fail(title, text) {
  $("replay-id").textContent = title;
  $("panel").innerHTML = `<div class="empty-note">${escape(text)}</div>`;
}

function renderLegend() {
  const seen = new Set();
  const items = [];
  for (const edge of data.board.edges) {
    if (seen.has(edge.key)) continue;
    seen.add(edge.key);
    const style = data.board.transports[String(edge.transport)];
    items.push(
      `<span class="legend-item"><i class="legend-line" style="background:${style.color}"></i>${escape(
        style.name
      )}</span>`
    );
  }
  $("legend").innerHTML = items.join("");
}

/* ---- rendering ------------------------------------------------------ */

function current() {
  return data.steps[step];
}

function render() {
  const state = current();
  if (!state) return;

  $("scrub").value = String(step);
  $("step-read").textContent = `${step} / ${data.steps.length - 1}`;

  board.render({
    showImage: showPhoto,
    showEdges: !showPhoto,
    detectives: state.detectivePositions,
    mrx: state.mrxPosition,
    mrxVisible: state.mrxVisible,
    destinations: {},
    highlightEdges: [],
    active: [],
    staged: [],
    suspects: [],
    labels: showPhoto ? "notable" : "all",
  });

  renderRuler(state);
  renderBadges(state);
  $("panel").innerHTML = `${stateCard(state)}${moveLog()}${ticketTable(state)}`;

  const log = document.querySelector(".log-turn.is-now");
  log?.scrollIntoView({ block: "nearest" });

  $("go-prev").disabled = step === 0;
  $("go-first").disabled = step === 0;
  $("go-next").disabled = step >= data.steps.length - 1;
  $("go-last").disabled = step >= data.steps.length - 1;
}

function renderRuler(state) {
  const reveals = data.revealTurns || [];
  const total = reveals.length ? Math.max(...reveals) : 24;
  const now = state.mrxTurnCount || 0;

  $("ruler").innerHTML = Array.from({ length: total }, (_, index) => {
    const turn = index + 1;
    const isReveal = reveals.includes(turn);
    const classes = ["ruler-tick"];
    if (isReveal) classes.push("is-reveal");
    if (turn < now) classes.push("is-past");
    if (turn === now) classes.push("is-now");
    return `<i class="${classes.join(" ")}" data-reveal="${isReveal}" title="Turn ${turn}"></i>`;
  }).join("");

  $("ruler-caption").textContent = state.mrxVisible
    ? `surfaced on ${now}`
    : reveals.find((turn) => turn > now)
      ? `next: ${reveals.find((turn) => turn > now)}`
      : "no more reveals";
}

function renderBadges(state) {
  $("badge-turn").innerHTML = `Turn <b>${state.turnCount}</b>`;
  const badge = $("badge-mrx");
  badge.className = "badge";
  if (state.gameOver) {
    badge.textContent =
      state.winner === "detectives" ? "Detectives win" : "Mr. X wins";
  } else if (state.mrxVisible) {
    badge.classList.add("is-visible-x");
    badge.innerHTML = `Mr. X at <b>${state.mrxPosition}</b>`;
  } else {
    badge.classList.add("is-hidden-x");
    badge.innerHTML = `Hidden at <b>${state.mrxPosition}</b>`;
  }
}

function stateCard(state) {
  const actor = state.turn === "MrX" ? "mrx" : "detectives";
  const rows = [
    ["Turn", state.turnCount],
    ["Mr. X turns", state.mrxTurnCount],
    ["Detectives", state.detectivePositions.join(" · ")],
    [
      "Mr. X",
      state.mrxVisible
        ? `${state.mrxPosition} (surfaced)`
        : `${state.mrxPosition} (hidden at the time)`,
    ],
  ];
  if (state.doubleMoveActive) rows.push(["Double move", "in progress"]);
  if (state.gameOver) {
    rows.push([
      "Result",
      state.winner === "detectives" ? "Detectives win" : "Mr. X wins",
    ]);
  }

  return `
    <section class="section">
      <div class="turn-card" data-actor="${actor}">
        <div class="turn-top">
          <div>
            <div class="turn-name">${
              state.turn === "MrX" ? "Mr. X to move" : "Detectives to move"
            }</div>
            <div class="turn-sub">Step ${step} of ${data.steps.length - 1}</div>
          </div>
          <span class="controller">${escape(data.gameId.slice(-6))}</span>
        </div>
      </div>
      <table class="kv-table">
        <tbody>
          ${rows
            .map(
              ([label, value]) =>
                `<tr><td><span class="pick-who">${escape(
                  label
                )}</span></td><td>${escape(value)}</td></tr>`
            )
            .join("")}
        </tbody>
      </table>
    </section>`;
}

/* The move log is the record of what happened, so it reads as a list of
   turns, with everything after the current step held back. */
function moveLog() {
  const shown = data.moves.filter((turn) => turn.index < step);
  if (!shown.length) {
    return `
      <section class="section">
        <div class="section-head"><span class="eyebrow">Moves</span></div>
        <p class="empty-note">The game has not started yet.</p>
      </section>`;
  }

  const cards = shown
    .slice()
    .reverse()
    .map((turn) => {
      const isNow = turn.index === step - 1;
      const entries = turn.entries
        .map(
          (entry) => `
        <div class="log-move">
          <span class="pick-who">${escape(entry.label)}</span>
          <span class="pick-route">${
            entry.stayed
              ? `stayed at ${entry.from}`
              : `${entry.from} &rarr; ${entry.to}`
          }</span>
          <span class="pick-ticket" style="color:${entry.color}">${escape(
            entry.ticket
          )}</span>
        </div>`
        )
        .join("");

      return `
        <div class="log-turn ${isNow ? "is-now" : ""}" data-step="${turn.index + 1}">
          <div class="log-head">
            <span class="eyebrow">Turn ${turn.turnNumber}</span>
            ${turn.doubleMove ? `<span class="log-flag">Double move</span>` : ""}
          </div>
          ${entries}
        </div>`;
    })
    .join("");

  return `
    <section class="section">
      <div class="section-head">
        <span class="eyebrow">Moves</span>
        <span class="section-note">newest first</span>
      </div>
      <div class="log">${cards}</div>
    </section>`;
}

function ticketTable(state) {
  const head = data.ticketOrder
    .map(
      (key) =>
        `<th><i class="th-dot" style="background:${
          data.ticketColors[key] || "#6d7a8b"
        }"></i><span>${escape(data.ticketLabels[key])}</span></th>`
    )
    .join("");

  const rows = state.tickets
    .map((row) => {
      const cells = data.ticketOrder
        .map((key) => {
          const count = row.counts[key];
          if (count === null || count === undefined) {
            return `<td class="count-none">&mdash;</td>`;
          }
          return `<td class="${count === 0 ? "count-zero" : ""}">${count}</td>`;
        })
        .join("");
      return `
        <tr data-side="${row.side}">
          <td><span class="ticket-row-name"><b>${escape(
            row.short
          )}</b><span class="where">${row.position ?? "··"}</span></span></td>
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

/* ---- navigation ----------------------------------------------------- */

function goTo(next) {
  if (!data) return;
  step = Math.max(0, Math.min(data.steps.length - 1, next));
  render();
  if (playing && step >= data.steps.length - 1) stop();
}

function play() {
  if (!data || playing) return;
  if (step >= data.steps.length - 1) step = 0;
  playing = true;
  $("go-play").textContent = "Pause";
  $("go-play").classList.remove("btn-primary");
  timer = setInterval(() => goTo(step + 1), 850);
}

function stop() {
  playing = false;
  clearInterval(timer);
  $("go-play").textContent = "Play";
  $("go-play").classList.add("btn-primary");
}

$("go-play").addEventListener("click", () => (playing ? stop() : play()));
$("go-first").addEventListener("click", () => goTo(0));
$("go-prev").addEventListener("click", () => goTo(step - 1));
$("go-next").addEventListener("click", () => goTo(step + 1));
$("go-last").addEventListener("click", () => goTo(data.steps.length - 1));
$("scrub").addEventListener("input", (event) => {
  stop();
  goTo(Number(event.target.value));
});

$("panel").addEventListener("click", (event) => {
  const turn = event.target.closest("[data-step]");
  if (turn) {
    stop();
    goTo(Number(turn.dataset.step));
  }
});

$("zoom-in").addEventListener("click", () => board.zoomBy(1.3));
$("zoom-out").addEventListener("click", () => board.zoomBy(1 / 1.3));
$("zoom-fit").addEventListener("click", () => board.fit());
$("photo-toggle").addEventListener("click", () => {
  showPhoto = !showPhoto;
  render();
});

document.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) {
    return;
  }
  if (event.key === "ArrowRight") {
    stop();
    goTo(step + 1);
  } else if (event.key === "ArrowLeft") {
    stop();
    goTo(step - 1);
  } else if (event.key === "Home") {
    stop();
    goTo(0);
  } else if (event.key === "End") {
    stop();
    goTo(data.steps.length - 1);
  } else if (event.key === " ") {
    event.preventDefault();
    playing ? stop() : play();
  } else if (event.key === "f") {
    board.fit();
  }
});

load();
