/* The board: an SVG map with pan and zoom, shared by the play and replay
   windows. Stations are drawn as transit roundels — the ring carries the line
   colour, the disc carries whoever is standing on it. */

const SVG_NS = "http://www.w3.org/2000/svg";

function el(name, attrs) {
  const node = document.createElementNS(SVG_NS, name);
  if (attrs) {
    for (const [key, value] of Object.entries(attrs)) {
      node.setAttribute(key, value);
    }
  }
  return node;
}

// Which ticket wins the ring colour when several reach the same station.
const TRANSPORT_RANK = [3, 2, 4, 5, 1];

export class BoardView {
  constructor(svg, options = {}) {
    this.svg = svg;
    this.onNodeClick = options.onNodeClick || null;
    this.onNodeHover = options.onNodeHover || null;

    this.layout = null;
    this.nodeById = new Map();
    this.edgeIndex = new Map();
    this.radius = 12;

    this.scale = 1;
    this.tx = 0;
    this.ty = 0;
    this.minScale = 0.1;
    this.maxScale = 14;

    this.root = el("g", { class: "board-root" });
    this.layers = {};
    for (const name of [
      "image",
      "edges",
      "highlight",
      "suspects",
      "rings",
      "nodes",
    ]) {
      this.layers[name] = el("g", { class: `layer layer-${name}` });
      this.root.appendChild(this.layers[name]);
    }
    this.svg.appendChild(this.root);

    this._bindPointer();
    this._observeSize();
  }

  /* ---- static geometry ------------------------------------------- */

  setLayout(layout) {
    this.layout = layout;
    this.nodeById.clear();
    this.edgeIndex.clear();
    for (const layer of Object.values(this.layers)) {
      layer.replaceChildren();
    }

    this.radius = this._pickRadius(layout);

    if (layout.image) {
      const image = el("image", {
        href: layout.image.url,
        x: 0,
        y: 0,
        width: layout.image.width,
        height: layout.image.height,
        preserveAspectRatio: "none",
      });
      image.setAttribute("opacity", layout.image.alpha ?? 1);
      this.layers.image.appendChild(image);
    }

    for (const edge of layout.edges) {
      const line = el("line", {
        x1: edge.x1,
        y1: edge.y1,
        x2: edge.x2,
        y2: edge.y2,
        stroke: edge.color,
        "stroke-width": edge.width * (this.radius / 12),
        "stroke-linecap": "round",
        class: `edge edge-${edge.key}`,
      });
      this.layers.edges.appendChild(line);

      const key = this._edgeKey(edge.u, edge.v, edge.transport);
      if (!this.edgeIndex.has(key)) this.edgeIndex.set(key, edge);
      const loose = this._edgeKey(edge.u, edge.v, "any");
      if (!this.edgeIndex.has(loose)) this.edgeIndex.set(loose, edge);
    }

    const r = this.radius;
    for (const node of layout.nodes) {
      const group = el("g", {
        class: "station",
        transform: `translate(${node.x} ${node.y})`,
      });
      group.dataset.id = String(node.id);

      const halo = el("circle", { class: "st-halo", r: r * 1.95 });
      const ring = el("circle", { class: "st-ring", r: r * 1.5 });
      const disc = el("circle", { class: "st-disc", r });
      disc.setAttribute("stroke", node.ring);

      const mark = el("text", { class: "st-mark", y: r * 0.36 });
      mark.setAttribute("font-size", r * 1.06);
      const caption = el("text", { class: "st-caption", y: -r * 1.55 });
      caption.setAttribute("font-size", r * 0.92);
      caption.textContent = String(node.id);

      group.append(halo, ring, disc, mark, caption);
      // Generous invisible hit area: stations are small at fit zoom.
      group.appendChild(el("circle", { class: "st-hit", r: r * 1.7 }));

      this.layers.nodes.appendChild(group);
      this.nodeById.set(node.id, { node, group, mark, caption, disc, ring });
    }

    this._hasFitted = false;
    this._userMovedView = false;
    this._waitForSize();
  }

  _pickRadius(layout) {
    // Size stations against how tightly packed the board actually is.
    const points = layout.nodes;
    if (points.length < 2) return 18;
    let closest = Infinity;
    const sample = Math.min(points.length, 90);
    for (let i = 0; i < sample; i += 1) {
      for (let j = i + 1; j < sample; j += 1) {
        const dx = points[i].x - points[j].x;
        const dy = points[i].y - points[j].y;
        const distance = Math.hypot(dx, dy);
        if (distance > 0.5 && distance < closest) closest = distance;
      }
    }
    if (!Number.isFinite(closest)) closest = 80;
    return Math.max(6, Math.min(46, closest * 0.34));
  }

  _edgeKey(u, v, transport) {
    return `${Math.min(u, v)}-${Math.max(u, v)}-${transport}`;
  }

  /* ---- dynamic state ---------------------------------------------- */

  /**
   * @param {object} view
   *  showEdges      draw the route lines
   *  showImage      draw the board photo
   *  detectives     [stationId] in detective order
   *  mrx            stationId or null
   *  mrxVisible     boolean
   *  destinations   {stationId: [transportValue]}
   *  highlightEdges [{u, v, transport}]
   *  active         [stationId] the piece being moved
   *  staged         [stationId] destinations already picked this turn
   *  suspects       [stationId] where Mr. X might be
   *  labels         "all" | "notable"
   */
  render(view) {
    const showImage = Boolean(view.showImage && this.layout?.image);
    this.layers.image.style.display = showImage ? "" : "none";
    this.layers.edges.style.display = view.showEdges ? "" : "none";

    this._renderHighlightEdges(view.highlightEdges || []);
    this._renderSuspects(view.suspects || []);

    const detectives = view.detectives || [];
    const detectiveAt = new Map();
    detectives.forEach((station, index) => {
      if (station !== null && station !== undefined) {
        detectiveAt.set(station, index + 1);
      }
    });

    const destinations = view.destinations || {};
    const staged = new Set(view.staged || []);
    const active = new Set(view.active || []);
    const labelAll = view.labels === "all";

    for (const [id, entry] of this.nodeById) {
      const { group, mark, caption } = entry;
      const detectiveNumber = detectiveAt.get(id);
      const isMrX = view.mrx === id;
      const reachable = Object.prototype.hasOwnProperty.call(
        destinations,
        String(id)
      );

      group.classList.toggle("is-detective", Boolean(detectiveNumber));
      group.classList.toggle("is-mrx", isMrX);
      group.classList.toggle("is-revealed", isMrX && Boolean(view.mrxVisible));
      group.classList.toggle("is-active", active.has(id));
      group.classList.toggle("is-staged", staged.has(id));
      group.classList.toggle("is-reachable", reachable);
      group.classList.toggle("is-dim", showImage && !labelAll);

      if (reachable) {
        const transports = destinations[String(id)] || [];
        const best =
          TRANSPORT_RANK.find((value) => transports.includes(value)) ||
          transports[0];
        const color = this.layout.tickets?.[String(best)]?.color || "#e9e6dc";
        entry.ring.setAttribute("stroke", color);
        group.dataset.tickets = transports.join(",");
      } else {
        group.removeAttribute("data-tickets");
      }

      if (detectiveNumber) {
        mark.textContent = String(detectiveNumber);
      } else if (isMrX) {
        mark.textContent = "X";
      } else if (labelAll) {
        mark.textContent = String(id);
      } else {
        mark.textContent = "";
      }

      // The station number moves to the caption when a piece takes the disc.
      const occupied = Boolean(detectiveNumber) || isMrX;
      caption.style.display =
        occupied || (reachable && !labelAll) ? "" : "none";
    }
  }

  _renderHighlightEdges(edges) {
    const layer = this.layers.highlight;
    layer.replaceChildren();
    const seen = new Set();

    for (const { u, v, transport } of edges) {
      const key = this._edgeKey(u, v, transport);
      if (seen.has(key)) continue;
      seen.add(key);
      // A black ticket rides an ordinary line, so fall back to any edge.
      const edge =
        this.edgeIndex.get(key) || this.edgeIndex.get(this._edgeKey(u, v, "any"));
      if (!edge) continue;
      const color = this.layout.tickets?.[String(transport)]?.color || edge.color;
      layer.appendChild(
        el("line", {
          x1: edge.x1,
          y1: edge.y1,
          x2: edge.x2,
          y2: edge.y2,
          stroke: color,
          "stroke-width": (edge.width + 2.2) * (this.radius / 12),
          "stroke-linecap": "round",
          class: "edge-live",
        })
      );
    }
  }

  _renderSuspects(suspects) {
    const layer = this.layers.suspects;
    layer.replaceChildren();
    for (const id of suspects) {
      const entry = this.nodeById.get(id);
      if (!entry) continue;
      layer.appendChild(
        el("circle", {
          class: "suspect",
          cx: entry.node.x,
          cy: entry.node.y,
          r: this.radius * 1.85,
        })
      );
    }
  }

  /* ---- viewport ---------------------------------------------------- */

  _applyTransform() {
    this.root.setAttribute(
      "transform",
      `translate(${this.tx} ${this.ty}) scale(${this.scale})`
    );
    // Keep strokes and type legible at every zoom level.
    this.svg.style.setProperty("--zoom", String(this.scale));
  }

  fit() {
    if (!this.layout) return;
    const box = this.svg.getBoundingClientRect();
    if (!box.width || !box.height) return;
    const pad = 0.955;
    this.scale = Math.min(
      box.width / this.layout.width,
      box.height / this.layout.height
    ) * pad;
    this.minScale = this.scale * 0.55;
    this.tx = (box.width - this.layout.width * this.scale) / 2;
    this.ty = (box.height - this.layout.height * this.scale) / 2;
    this._userMovedView = false;
    this._applyTransform();
  }

  zoomBy(factor, cx, cy) {
    this._userMovedView = true;
    const box = this.svg.getBoundingClientRect();
    const px = cx ?? box.width / 2;
    const py = cy ?? box.height / 2;
    const next = Math.min(
      this.maxScale,
      Math.max(this.minScale, this.scale * factor)
    );
    const ratio = next / this.scale;
    this.tx = px - (px - this.tx) * ratio;
    this.ty = py - (py - this.ty) * ratio;
    this.scale = next;
    this._applyTransform();
  }

  /** Flag one station without moving the viewport, for panel hover. */
  hint(stationId) {
    if (this._hinted === stationId) return;
    const previous = this.nodeById.get(this._hinted);
    previous?.group.classList.remove("is-hinted");
    const next = this.nodeById.get(stationId);
    next?.group.classList.add("is-hinted");
    this._hinted = stationId;
  }

  centerOn(stationId) {
    const entry = this.nodeById.get(stationId);
    if (!entry) return;
    const box = this.svg.getBoundingClientRect();
    this.tx = box.width / 2 - entry.node.x * this.scale;
    this.ty = box.height / 2 - entry.node.y * this.scale;
    this._applyTransform();
  }

  _observeSize() {
    // With no viewBox, one SVG user unit is one CSS pixel, so resizing the
    // window never distorts the map — it just reveals more or less of it.
    // Re-fit on resize unless the player has taken over the viewport.
    window.addEventListener("resize", () => {
      if (!this.layout) return;
      if (this._userMovedView) return;
      this.fit();
    });
  }

  _waitForSize() {
    // The shell may not be laid out yet on the very first paint.
    if (this._hasFitted || !this.layout) return;
    const box = this.svg.getBoundingClientRect();
    if (box.width && box.height) {
      this._hasFitted = true;
      this.fit();
      return;
    }
    requestAnimationFrame(() => this._waitForSize());
  }

  _bindPointer() {
    const wrap = this.svg.parentElement;
    let dragging = false;
    let moved = 0;
    let lastX = 0;
    let lastY = 0;

    this.svg.addEventListener("pointerdown", (event) => {
      if (event.button !== 0) return;
      dragging = true;
      moved = 0;
      lastX = event.clientX;
      lastY = event.clientY;
      this.svg.setPointerCapture(event.pointerId);
    });

    this.svg.addEventListener("pointermove", (event) => {
      if (!dragging) {
        if (this.onNodeHover) {
          const station = event.target.closest?.(".station");
          this.onNodeHover(station ? Number(station.dataset.id) : null, event);
        }
        return;
      }
      const dx = event.clientX - lastX;
      const dy = event.clientY - lastY;
      moved += Math.abs(dx) + Math.abs(dy);
      lastX = event.clientX;
      lastY = event.clientY;
      this.tx += dx;
      this.ty += dy;
      if (moved > 4) {
        wrap?.classList.add("is-panning");
        this._userMovedView = true;
      }
      this._applyTransform();
    });

    const endDrag = (event) => {
      if (!dragging) return;
      dragging = false;
      wrap?.classList.remove("is-panning");
      try {
        this.svg.releasePointerCapture(event.pointerId);
      } catch {
        /* pointer already released */
      }
      // A drag is not a click.
      if (moved <= 4 && this.onNodeClick) {
        const station = event.target.closest?.(".station");
        if (station) this.onNodeClick(Number(station.dataset.id));
      }
    };

    this.svg.addEventListener("pointerup", endDrag);
    this.svg.addEventListener("pointercancel", endDrag);

    this.svg.addEventListener(
      "wheel",
      (event) => {
        event.preventDefault();
        const box = this.svg.getBoundingClientRect();
        const factor = event.deltaY < 0 ? 1.16 : 1 / 1.16;
        this.zoomBy(factor, event.clientX - box.left, event.clientY - box.top);
      },
      { passive: false }
    );
  }
}
