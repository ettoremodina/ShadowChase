# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pymupdf>=1.25.0",
#   "pymupdf4llm>=0.0.17",
# ]
# ///
"""CLI tool for paper-summarizer-visual skill.

Subcommands:
  preprocess  Convert a PDF to markdown + extract images with caption matching & deduplication
  render      Fill the HTML template with a structured JSON summary

Usage:
  uv run paper_tool.py preprocess --pdf <path.pdf> --output-dir <dir>
  uv run paper_tool.py render --summary <summary.json> --images-dir <dir> \\
      --template <template.html> --output <output.html>
"""

import argparse
import base64
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Preprocess: PDF → Markdown + Images
# ---------------------------------------------------------------------------

def cmd_preprocess(args):
    """Convert a PDF to markdown text and extract embedded images."""
    import pymupdf4llm  # noqa: delay import so --help is fast
    import pymupdf       # noqa: (fitz)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # --- Extract markdown ---
    print(f"Converting PDF to markdown: {pdf_path.name} ...", file=sys.stderr)
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as e:
        print(f"Error converting PDF to markdown: {e}", file=sys.stderr)
        sys.exit(1)

    # Check minimum content threshold
    plain_chars = len(re.sub(r'\s+', '', md_text))
    if plain_chars < 100:
        print(
            "Warning: Extracted text is very short (<100 non-whitespace chars). "
            "This PDF may be scanned/image-only. Please provide guidance.",
            file=sys.stderr,
        )
        sys.exit(1)

    content_path = output_dir / "content.md"
    content_path.write_text(md_text, encoding="utf-8")
    print(f"  Markdown written to: {content_path}", file=sys.stderr)

    # --- Extract images with hash deduplication ---
    print("Extracting images ...", file=sys.stderr)
    doc = pymupdf.open(str(pdf_path))
    seen_hashes = set()
    img_count = 0
    image_info_map = {}

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                continue
            if base_image is None:
                continue
            img_bytes = base_image["image"]
            img_ext = base_image.get("ext", "png")

            # Skip tiny images (likely icons/bullets, <2KB)
            if len(img_bytes) < 2048:
                continue

            # Deduplicate by MD5 hash
            img_hash = hashlib.md5(img_bytes).hexdigest()
            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            img_count += 1
            img_name = f"fig_{img_count:03d}.{img_ext}"
            img_path = images_dir / img_name
            img_path.write_bytes(img_bytes)

            image_info_map[img_name] = {
                "page": page_idx + 1,
                "width": base_image.get("width", 0),
                "height": base_image.get("height", 0),
                "size_bytes": len(img_bytes)
            }

    doc.close()
    print(f"  Extracted {img_count} unique images to: {images_dir}", file=sys.stderr)

    # --- Extract captions from Markdown text heuristics ---
    captions = []
    caption_pattern = re.compile(
        r'^\s*(?:Figure|Fig\.?)\s+(\d+[\.\:\-]?\d*)\s*[\:\.\-]?\s*(.+)$',
        re.IGNORECASE | re.MULTILINE
    )
    for match in caption_pattern.finditer(md_text):
        fig_num, text = match.group(1), match.group(2).strip()
        captions.append({"fig_num": fig_num, "caption": f"Figure {fig_num}: {text[:200]}"})

    # --- Metadata ---
    doc2 = pymupdf.open(str(pdf_path))
    meta = doc2.metadata or {}
    doc2.close()

    word_count = len(md_text.split())
    line_count = md_text.count('\n')
    metadata = {
        "source_pdf": str(pdf_path.resolve()),
        "page_count": len(pymupdf.open(str(pdf_path))),
        "word_count": word_count,
        "line_count": line_count,
        "image_count": img_count,
        "detected_title": meta.get("title", ""),
        "detected_authors": meta.get("author", ""),
        "image_details": image_info_map,
        "extracted_captions": captions
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        f"\nSuccess! Preprocessed '{pdf_path.name}':\n"
        f"  Content:  {content_path}\n"
        f"  Images:   {images_dir} ({img_count} files)\n"
        f"  Metadata: {meta_path}\n"
        f"  Words:    ~{word_count:,}  |  Lines: {line_count:,}"
    )


# ---------------------------------------------------------------------------
# Render: JSON Summary + Template → HTML
# ---------------------------------------------------------------------------

COLOR_THEMES = {
    "purple":  {"start": "#7C3AED", "end": "#A855F7"},
    "blue":    {"start": "#2563EB", "end": "#3B82F6"},
    "green":   {"start": "#059669", "end": "#10B981"},
    "red":     {"start": "#DC2626", "end": "#EF4444"},
    "orange":  {"start": "#EA580C", "end": "#F97316"},
    "gold":    {"start": "#D97706", "end": "#F59E0B"},
}

PAPER_TYPE_LABELS = {
    "textbook":         "📖 Textbook / Guide",
    "research_article": "📄 Research Article",
    "review_survey":    "🔭 Review / Survey",
    "mathematical":     "🧮 Mathematical / Methods",
    "case_study":       "🌍 Case Study / Simulation",
    "technical_report": "📋 Technical Report",
}


def _markdown_to_html(md_text):
    """Minimal markdown-to-HTML converter for section content."""
    if not md_text:
        return ""

    lines = md_text.split('\n')
    html_parts = []
    in_list = None  # 'ul' or 'ol'
    in_blockquote = False

    def inline(text):
        """Convert inline markdown to HTML."""
        # Code (backticks)
        text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
        # Bold
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        # Italic
        text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
        text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
        # Links
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2" style="color:var(--accent)">\1</a>', text)
        return text

    def close_list():
        nonlocal in_list
        if in_list:
            html_parts.append(f'</{in_list}>')
            in_list = None

    def close_blockquote():
        nonlocal in_blockquote
        if in_blockquote:
            html_parts.append('</blockquote>')
            in_blockquote = False

    for line in lines:
        stripped = line.strip()

        # Empty line
        if not stripped:
            close_list()
            close_blockquote()
            continue

        # Horizontal rule
        if re.match(r'^-{3,}$|^\*{3,}$|^_{3,}$', stripped):
            close_list()
            close_blockquote()
            html_parts.append('<hr>')
            continue

        # Headings
        hm = re.match(r'^(#{1,4})\s+(.+)', stripped)
        if hm:
            close_list()
            close_blockquote()
            level = min(len(hm.group(1)) + 2, 6)
            html_parts.append(f'<h{level}>{inline(hm.group(2))}</h{level}>')
            continue

        # Blockquote
        if stripped.startswith('>'):
            close_list()
            if not in_blockquote:
                html_parts.append('<blockquote>')
                in_blockquote = True
            html_parts.append(inline(stripped.lstrip('> ')))
            continue
        else:
            close_blockquote()

        # Unordered list
        ulm = re.match(r'^[-*+]\s+(.+)', stripped)
        if ulm:
            if in_list != 'ul':
                close_list()
                html_parts.append('<ul>')
                in_list = 'ul'
            html_parts.append(f'<li>{inline(ulm.group(1))}</li>')
            continue

        # Ordered list
        olm = re.match(r'^\d+\.\s+(.+)', stripped)
        if olm:
            if in_list != 'ol':
                close_list()
                html_parts.append('<ol>')
                in_list = 'ol'
            html_parts.append(f'<li>{inline(olm.group(1))}</li>')
            continue

        # Normal paragraph
        close_list()
        html_parts.append(f'<p>{inline(stripped)}</p>')

    close_list()
    close_blockquote()
    return '\n'.join(html_parts)


def _slugify(text):
    """Turn heading text into a URL-safe ID."""
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def _embed_image(fig_name, images_dir):
    """Read an image file and return a base64 data-URI string, or None."""
    if not images_dir:
        return None
    fig_path = images_dir / fig_name
    if not fig_path.exists():
        print(f"  Warning: Image not found, skipping: {fig_path}", file=sys.stderr)
        return None
    img_bytes = fig_path.read_bytes()
    ext = fig_path.suffix.lstrip('.').lower()
    mime = 'image/jpeg' if ext in ('jpg', 'jpeg') else f'image/{ext}'
    b64 = base64.b64encode(img_bytes).decode('ascii')
    return f'data:{mime};base64,{b64}'


def _render_section(sec, idx, images_dir):
    """Render a single section dict into HTML, dispatching by `type`."""
    sec_type = sec.get("type", "card")
    icon = sec.get("icon", "📌")
    heading = sec.get("heading", "Section")
    slug = _slugify(heading)
    toc_id = f"sec-{idx}-{slug}"

    header_html = f'''<div class="section-header">
        <div class="section-icon">{icon}</div>
        <h2>{heading}</h2>
      </div>'''

    # ─── MERMAID DIAGRAM ──────────────────────────────────────
    if sec_type == "mermaid":
        code = sec.get("code", "").strip()
        description = _markdown_to_html(sec.get("description", ""))
        desc_html = f'<div class="section-body mb-3">{description}</div>' if description else ""
        return f'''<div class="mermaid-section animate-in" id="{toc_id}" data-toc-id="{toc_id}">
      <div class="section-card">
        {header_html}
        {desc_html}
        <div class="mermaid-container">
          <pre class="mermaid">{code}</pre>
        </div>
      </div>
    </div>''', toc_id, heading

    # ─── FORMULA CARD ─────────────────────────────────────────
    if sec_type == "formula":
        latex = sec.get("latex", "")
        name = sec.get("name", "Key Equation")
        explanation = _markdown_to_html(sec.get("explanation", ""))
        breakdown = sec.get("breakdown", [])
        
        breakdown_items = []
        for b in breakdown:
            sym = b.get("symbol", "")
            desc = b.get("description", "")
            breakdown_items.append(f'<li class="formula-term"><code>{sym}</code><span>{desc}</span></li>')
            
        breakdown_html = f'<ul class="formula-breakdown">{"".join(breakdown_items)}</ul>' if breakdown_items else ""

        return f'''<div class="formula-section animate-in" id="{toc_id}" data-toc-id="{toc_id}">
      <div class="section-card formula-card">
        {header_html}
        <div class="formula-header">{name}</div>
        <div class="formula-display">
          \\[{latex}\\]
          <button class="copy-btn" onclick="navigator.clipboard.writeText(`{latex}`); this.innerText='Copied!'; setTimeout(()=>this.innerText='Copy LaTeX', 2000)">Copy LaTeX</button>
        </div>
        <div class="section-body">{explanation}</div>
        {breakdown_html}
      </div>
    </div>''', toc_id, heading

    # ─── STATS ───────────────────────────────────────────────
    if sec_type == "stats":
        stats = sec.get("stats", [])
        cards = []
        for s in stats:
            cards.append(f'''<div class="stat-card">
            <div class="stat-icon">{s.get("icon", "📊")}</div>
            <div class="stat-value">{s.get("value", "—")}</div>
            <div class="stat-label">{s.get("label", "")}</div>
          </div>''')
        return f'''<div class="stats-section animate-in" id="{toc_id}" data-toc-id="{toc_id}">
      {header_html}
      <div class="stats-grid">{"".join(cards)}</div>
    </div>''', toc_id, heading

    # ─── TIMELINE ────────────────────────────────────────────
    if sec_type == "timeline":
        steps = sec.get("steps", [])
        items = []
        for i, step in enumerate(steps, 1):
            items.append(f'''<div class="timeline-item">
            <div class="timeline-step">Step {i}</div>
            <div class="timeline-title">{step.get("title", "")}</div>
            <div class="timeline-desc">{_markdown_to_html(step.get("description", ""))}</div>
          </div>''')
        return f'''<div class="timeline-section animate-in" id="{toc_id}" data-toc-id="{toc_id}">
      <div class="section-card">
        {header_html}
        <div class="timeline">{"".join(items)}</div>
      </div>
    </div>''', toc_id, heading

    # ─── TABLE ───────────────────────────────────────────────
    if sec_type == "table":
        columns = sec.get("columns", [])
        rows = sec.get("rows", [])
        ths = "".join(f"<th>{c}</th>" for c in columns)
        trs = []
        for row in rows:
            tds = "".join(f"<td>{cell}</td>" for cell in row)
            trs.append(f"<tr>{tds}</tr>")
        return f'''<div class="table-section animate-in" id="{toc_id}" data-toc-id="{toc_id}">
      <div class="section-card">
        {header_html}
        <table class="styled-table">
          <thead><tr>{ths}</tr></thead>
          <tbody>{"".join(trs)}</tbody>
        </table>
      </div>
    </div>''', toc_id, heading

    # ─── QUOTE / CALLOUT ─────────────────────────────────────
    if sec_type == "quote":
        quote_text = sec.get("quote", "")
        attribution = sec.get("attribution", "")
        attr_html = f'<div class="quote-attribution">— {attribution}</div>' if attribution else ""
        return f'''<div class="quote-section animate-in" id="{toc_id}" data-toc-id="{toc_id}">
      <div class="quote-mark">"</div>
      <div class="quote-text">{quote_text}</div>
      {attr_html}
    </div>''', toc_id, heading

    # ─── FLOWCHART ───────────────────────────────────────────
    if sec_type == "flowchart":
        steps = sec.get("steps", [])
        flow_items = []
        for i, step in enumerate(steps):
            if i > 0:
                flow_items.append('<div class="flow-arrow">→</div>')
            flow_items.append(f'''<div class="flow-step">
            <div class="flow-node">
              <div class="flow-num">{i + 1}</div>
              <div class="flow-icon">{step.get("icon", "⚡")}</div>
              <div class="flow-label">{step.get("label", "")}</div>
              <div class="flow-desc">{step.get("description", "")}</div>
            </div>
          </div>''')
        return f'''<div class="flowchart-section animate-in" id="{toc_id}" data-toc-id="{toc_id}">
      <div class="section-card">
        {header_html}
        <div class="flowchart">{"".join(flow_items)}</div>
      </div>
    </div>''', toc_id, heading

    # ─── TWO COLUMN ──────────────────────────────────────────
    if sec_type == "two_column":
        left = sec.get("left", {})
        right = sec.get("right", {})
        left_html = _markdown_to_html(left.get("content", ""))
        right_html = _markdown_to_html(right.get("content", ""))
        return f'''<div class="two-col-section animate-in" id="{toc_id}" data-toc-id="{toc_id}">
      <div class="section-card">
        {header_html}
        <div class="two-col-grid">
          <div class="two-col-panel">
            <div class="two-col-title">{left.get("title", "")}</div>
            <div class="section-body">{left_html}</div>
          </div>
          <div class="two-col-panel">
            <div class="two-col-title">{right.get("title", "")}</div>
            <div class="section-body">{right_html}</div>
          </div>
        </div>
      </div>
    </div>''', toc_id, heading

    # ─── ACCORDION ───────────────────────────────────────────
    if sec_type == "accordion":
        items = sec.get("items", [])
        acc_items = []
        for item in items:
            body_html = _markdown_to_html(item.get("content", ""))
            acc_items.append(f'''<div class="accordion-item">
            <button class="accordion-trigger">
              <span>{item.get("icon", "•")} {item.get("title", "")}</span>
              <span class="accordion-chevron">▼</span>
            </button>
            <div class="accordion-content">
              <div class="accordion-body section-body">{body_html}</div>
            </div>
          </div>''')
        return f'''<div class="accordion-section animate-in" id="{toc_id}" data-toc-id="{toc_id}">
      <div class="section-card">
        {header_html}
        {"".join(acc_items)}
      </div>
    </div>''', toc_id, heading

    # ─── DEFAULT CARD (fallback) ─────────────────────────────
    content_html = _markdown_to_html(sec.get("content", ""))
    return f'''<div class="section-card animate-in" id="{toc_id}" data-toc-id="{toc_id}">
      {header_html}
      <div class="section-body">
        {content_html}
      </div>
    </div>''', toc_id, heading


def cmd_render(args):
    """Fill the HTML template with summary data and produce a self-contained page."""
    summary_path = Path(args.summary)
    template_path = Path(args.template)
    output_path = Path(args.output)
    images_dir = Path(args.images_dir) if args.images_dir else None

    if not summary_path.exists():
        print(f"Error: Summary JSON not found: {summary_path}", file=sys.stderr)
        sys.exit(1)
    if not template_path.exists():
        print(f"Error: Template not found: {template_path}", file=sys.stderr)
        sys.exit(1)

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    # --- Resolve color theme ---
    color_theme = summary.get("color_theme", "blue")
    colors = COLOR_THEMES.get(color_theme, COLOR_THEMES["blue"])

    paper_type = summary.get("paper_type", "research_article")
    type_label = PAPER_TYPE_LABELS.get(paper_type, "📄 Research Article")
    type_icon = type_label.split(" ")[0]
    type_text = " ".join(type_label.split(" ")[1:])

    # --- Build tags HTML ---
    tags = summary.get("tags", [])
    tags_html = "\n".join(f'<span class="tag">{t}</span>' for t in tags)

    # --- Build sections HTML + TOC ---
    sections = summary.get("sections", [])
    sections_html_parts = []
    toc_parts = []
    for idx, sec in enumerate(sections):
        html, toc_id, toc_label = _render_section(sec, idx, images_dir)
        sections_html_parts.append(html)
        toc_parts.append(
            f'<li><a class="toc-item" href="#{toc_id}">{sec.get("icon", "📌")} {toc_label}</a></li>'
        )
    sections_html = "\n".join(sections_html_parts)
    toc_html = "\n".join(toc_parts)

    if summary.get("key_figures"):
        toc_html += '\n<li><a class="toc-item" href="#gallery-section">🖼️ Key Figures</a></li>'
    if summary.get("key_takeaways"):
        toc_html += '\n<li><a class="toc-item" href="#takeaways-section">🎯 Key Takeaways</a></li>'

    # --- Build gallery HTML ---
    key_figures = summary.get("key_figures", [])
    gallery_html = ""
    if key_figures and images_dir and images_dir.exists():
        gallery_items = []
        for fig_item in key_figures:
            if isinstance(fig_item, dict):
                fig_name = fig_item.get("image", "")
                caption = fig_item.get("caption", fig_name)
            else:
                fig_name = str(fig_item)
                caption = fig_name

            data_uri = _embed_image(fig_name, images_dir)
            if not data_uri:
                continue
            gallery_items.append(f'''
        <figure class="gallery-item">
          <img src="{data_uri}" alt="{fig_name}" loading="lazy">
          <figcaption>{caption}</figcaption>
        </figure>''')

        if gallery_items:
            gallery_html = f'''
    <div class="gallery animate-in" id="gallery-section" data-toc-id="gallery-section">
      <h2>🖼️ Key Figures</h2>
      <div class="gallery-grid">
        {"".join(gallery_items)}
      </div>
    </div>'''

    # --- Build takeaways HTML ---
    takeaways = summary.get("key_takeaways", [])
    takeaways_html = ""
    if takeaways:
        items = []
        for i, t in enumerate(takeaways, 1):
            items.append(f'''
        <div class="takeaway-item">
          <div class="takeaway-num">{i}</div>
          <div class="takeaway-text">{t}</div>
        </div>''')
        takeaways_html = f'''
    <div class="takeaways animate-in" id="takeaways-section" data-toc-id="takeaways-section">
      <h2>🎯 Key Takeaways</h2>
      {"".join(items)}
    </div>'''

    # --- Fill template ---
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    replacements = {
        "{{TITLE}}": summary.get("title", "Untitled Paper"),
        "{{AUTHORS}}": summary.get("authors", "Unknown"),
        "{{YEAR}}": summary.get("year", ""),
        "{{GRADIENT_START}}": colors["start"],
        "{{GRADIENT_END}}": colors["end"],
        "{{PAPER_TYPE_ICON}}": type_icon,
        "{{PAPER_TYPE_LABEL}}": type_text,
        "{{TAGS_HTML}}": tags_html,
        "{{TLDR}}": summary.get("tldr", ""),
        "{{TOC_HTML}}": toc_html,
        "{{SECTIONS_HTML}}": sections_html,
        "{{GALLERY_HTML}}": gallery_html,
        "{{TAKEAWAYS_HTML}}": takeaways_html,
        "{{GENERATION_DATE}}": now,
    }

    html = template
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Success! HTML summary written to: {output_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Paper Summarizer Visual — PDF preprocessing and HTML rendering"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- preprocess ---
    p_pre = subparsers.add_parser(
        "preprocess",
        help="Convert a PDF to markdown + extract images",
    )
    p_pre.add_argument(
        "--pdf", required=True,
        help="Path to the input PDF file",
    )
    p_pre.add_argument(
        "--output-dir", required=True,
        help="Directory to write content.md, images/, and metadata.json",
    )

    # --- render ---
    p_render = subparsers.add_parser(
        "render",
        help="Fill HTML template with a structured JSON summary",
    )
    p_render.add_argument(
        "--summary", required=True,
        help="Path to the structured summary JSON file",
    )
    p_render.add_argument(
        "--images-dir", required=False, default=None,
        help="Directory containing extracted images (for base64 embedding)",
    )
    p_render.add_argument(
        "--template", required=True,
        help="Path to the HTML template file",
    )
    p_render.add_argument(
        "--output", required=True,
        help="Path for the output HTML file",
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "render":
        cmd_render(args)
    else:
        print(f"Unknown command: {args.command}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
