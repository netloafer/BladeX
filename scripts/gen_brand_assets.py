#!/usr/bin/env python3
"""Generate the BladeX brand asset set from the single approved symbol geometry.

Source of truth: the two symbol paths below, traced from the approved design
reference (BladeX_Final_Centered.svg, 2026-09-05). Every asset in
``assets/brand/`` is emitted from here so the geometry can never drift between
files -- edit this script, re-run it, never hand-edit the SVGs.

    python3 scripts/gen_brand_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "assets" / "brand"

# --- approved geometry -------------------------------------------------------
# Native coordinate space of the traced paths; bbox measured with svgpathtools.
ARC_OUTER = "M 397.000,82.000 C 378.833,81.333 359.667,81.667 343.000,84.000 C 326.333,86.333 312.500,90.667 297.000,96.000 C 281.500,101.333 266.833,106.500 250.000,116.000 C 233.167,125.500 212.333,138.833 196.000,153.000 C 179.667,167.167 163.333,186.667 152.000,201.000 C 140.667,215.333 136.000,223.167 128.000,239.000 C 120.000,254.833 110.000,275.833 104.000,296.000 C 98.000,316.167 93.833,339.833 92.000,360.000 C 90.167,380.167 90.333,397.000 93.000,417.000 C 95.667,437.000 103.500,464.500 108.000,480.000 C 112.500,495.500 115.333,500.167 120.000,510.000 C 124.667,519.833 128.000,526.833 136.000,539.000 C 144.000,551.167 158.000,571.000 168.000,583.000 C 178.000,595.000 187.000,603.000 196.000,611.000 C 205.000,619.000 209.167,623.000 222.000,631.000 C 234.833,639.000 256.333,651.667 273.000,659.000 C 289.667,666.333 304.333,671.167 322.000,675.000 C 339.667,678.833 368.667,681.000 379.000,682.000 C 389.333,683.000 383.500,682.000 384.000,681.000 C 384.500,680.000 389.167,681.000 382.000,676.000 C 374.833,671.000 351.167,657.833 341.000,651.000 C 330.833,644.167 331.167,645.167 321.000,635.000 C 310.833,624.833 292.833,606.833 280.000,590.000 C 267.167,573.167 252.667,549.333 244.000,534.000 C 235.333,518.667 233.333,514.000 228.000,498.000 C 222.667,482.000 215.167,457.833 212.000,438.000 C 208.833,418.167 208.333,397.000 209.000,379.000 C 209.667,361.000 211.500,347.167 216.000,330.000 C 220.500,312.833 230.167,289.333 236.000,276.000 C 241.833,262.667 244.333,259.833 251.000,250.000 C 257.667,240.167 265.667,228.000 276.000,217.000 C 286.333,206.000 300.000,193.500 313.000,184.000 C 326.000,174.500 338.500,166.667 354.000,160.000 C 369.500,153.333 390.000,147.333 406.000,144.000 C 422.000,140.667 436.167,140.000 450.000,140.000 C 463.833,140.000 477.167,142.000 489.000,144.000 C 500.833,146.000 507.000,146.667 521.000,152.000 C 535.000,157.333 557.667,167.000 573.000,176.000 C 588.333,185.000 605.667,201.500 613.000,206.000 C 620.333,210.500 620.667,209.333 617.000,203.000 C 613.333,196.667 598.000,176.500 591.000,168.000 C 584.000,159.500 586.167,160.667 575.000,152.000 C 563.833,143.333 544.500,126.667 524.000,116.000 C 503.500,105.333 473.167,93.667 452.000,88.000 C 430.833,82.333 415.167,82.667 397.000,82.000 Z"
ARC_INNER = "M 667.000,314.000 C 664.000,303.833 661.667,307.000 661.000,313.000 C 660.333,319.000 663.833,335.833 663.000,350.000 C 662.167,364.167 660.500,380.333 656.000,398.000 C 651.500,415.667 642.000,441.333 636.000,456.000 C 630.000,470.667 626.000,476.500 620.000,486.000 C 614.000,495.500 611.000,501.333 600.000,513.000 C 589.000,524.667 570.333,544.167 554.000,556.000 C 537.667,567.833 516.333,577.500 502.000,584.000 C 487.667,590.500 478.167,592.333 468.000,595.000 C 457.833,597.667 454.000,599.167 441.000,600.000 C 428.000,600.833 402.333,600.667 390.000,600.000 C 377.667,599.333 375.667,598.167 367.000,596.000 C 358.333,593.833 348.333,591.000 338.000,587.000 C 327.667,583.000 310.000,573.167 305.000,572.000 C 300.000,570.833 301.833,572.833 308.000,580.000 C 314.167,587.167 329.833,604.500 342.000,615.000 C 354.167,625.500 368.667,635.667 381.000,643.000 C 393.333,650.333 402.833,654.000 416.000,659.000 C 429.167,664.000 444.500,673.000 460.000,673.000 C 475.500,673.000 492.167,666.167 509.000,659.000 C 525.833,651.833 547.500,638.667 561.000,630.000 C 574.500,621.333 579.000,617.833 590.000,607.000 C 601.000,596.167 618.167,576.167 627.000,565.000 C 635.833,553.833 637.000,551.833 643.000,540.000 C 649.000,528.167 657.667,508.333 663.000,494.000 C 668.333,479.667 672.333,465.833 675.000,454.000 C 677.667,442.167 678.333,436.333 679.000,423.000 C 679.667,409.667 681.000,392.167 679.000,374.000 C 677.000,355.833 670.000,324.167 667.000,314.000 Z"

# svgpathtools bbox of ARC_OUTER u ARC_INNER in native coordinates.
BBOX = (90.78, 62.06, 680.06, 682.41)  # x0, _, x1, y1 -- see SYM_* below
SYM_X0, SYM_Y0, SYM_X1, SYM_Y1 = 90.78, 81.70, 680.06, 682.41
SYM_W = SYM_X1 - SYM_X0          # 589.28
SYM_H = SYM_Y1 - SYM_Y0          # 600.71
SYM_ASPECT = SYM_W / SYM_H       # 0.9810
SYM_CX = (SYM_X0 + SYM_X1) / 2
SYM_CY = (SYM_Y0 + SYM_Y1) / 2

# --- palette (README.txt, 2026-09-05) ---------------------------------------
PRIMARY_BLUE = "#2563FF"
SECONDARY_BLUE = "#4F7CFF"
LIGHT_BLUE = "#8AB4FF"
DEEP_NAVY = "#0B1220"
SLATE_GRAY = "#687280"
DIVIDER = "#D7DCE6"

# --- outlined wordmark ------------------------------------------------------
# Outlined wordmark glyphs, native coordinates, from the designer's
# BladeX_Wordmark_Outlined.svg (archived as
# assets/brand/_source-wordmark-outlined-20260905.svg).
WM_BLADE = [
    ("B", "M513.70 350.00H561.26C589.98 350.00 603.01 335.64 603.01 316.63C603.01 297.62 589.65 286.83 577.28 286.17V284.92C588.74 282.10 598.20 274.21 598.20 258.77C598.20 240.51 585.50 226.32 559.27 226.32H513.70ZM532.62 333.73V294.47H560.10C574.87 294.47 584.25 304.01 584.25 315.72C584.25 326.01 577.11 333.73 559.35 333.73ZM532.62 279.86V242.50H557.94C572.63 242.50 579.60 250.22 579.60 260.35C579.60 272.22 569.98 279.86 557.44 279.86Z"),
    ("l", "M636.19 226.32H617.93V350.00H636.19Z"),
    ("a", "M680.74 352.08C695.77 352.08 704.57 344.52 708.14 337.30H709.05V350.00H726.73V288.41C726.73 261.60 705.40 256.04 690.46 256.04C673.94 256.04 658.33 262.51 652.02 279.11L669.21 283.59C671.78 277.29 678.59 270.89 690.71 270.89C702.33 270.89 708.47 276.87 708.47 287.16V287.66C708.47 294.22 701.41 294.14 685.06 296.04C667.55 298.04 649.37 302.69 649.37 323.77C649.37 342.03 663.06 352.08 680.74 352.08ZM684.81 337.38C674.60 337.38 667.30 332.82 667.30 324.02C667.30 314.47 675.76 310.99 686.06 309.58C691.70 308.83 705.81 307.33 708.55 304.68V316.55C708.55 327.42 699.67 337.38 684.81 337.38Z"),
    ("d", "M779.83 351.83C796.60 351.83 803.32 341.37 806.56 335.47H808.05V350.00H825.82V226.32H807.56V272.30H806.56C803.41 266.58 797.10 256.04 779.83 256.04C757.50 256.04 740.98 273.63 740.98 303.85C740.98 333.81 757.25 351.83 779.83 351.83ZM783.82 336.30C767.96 336.30 759.66 322.11 759.66 303.68C759.66 285.42 767.71 271.56 783.82 271.56C799.50 271.56 807.89 284.42 807.89 303.68C807.89 323.02 799.34 336.30 783.82 336.30Z"),
    ("e", "M884.40 351.91C904.24 351.91 918.43 342.20 922.83 327.75L905.65 323.69C902.41 332.32 894.77 336.80 884.56 336.80C869.37 336.80 859.08 327.09 858.42 309.16H924.24V302.69C924.24 269.23 903.99 256.04 882.82 256.04C856.84 256.04 840.07 275.71 840.07 304.26C840.07 333.07 857.01 351.91 884.40 351.91ZM858.50 295.55C859.50 282.18 868.29 271.14 882.90 271.14C896.93 271.14 904.82 280.94 906.15 295.55Z"),
]
WM_X = ("X", "M1014.57 350.00H1036.31L1056.73 320.45C1062.96 311.40 1065.86 306.67 1069.60 299.53C1073.42 306.84 1076.16 311.48 1082.30 320.45L1102.47 350.00H1124.55L1080.72 286.08L1121.81 226.32H1100.31L1083.63 250.72C1076.82 260.77 1073.75 266.33 1069.77 274.13C1065.78 266.25 1062.88 260.85 1056.07 250.72L1039.72 226.32H1017.72L1058.89 286.67Z")

# Ink metrics of the glyphs above, measured with svgpathtools.
WM_X0 = 513.70          # left edge of "B"
WM_BLADE_X1 = 924.24    # right edge of "e"
WM_CAP_TOP = 226.32     # cap line ("B" top)
WM_BASELINE = 350.00
WM_CAP = WM_BASELINE - WM_CAP_TOP   # 123.68

# The designer's file inherited an absolute x= from the earlier live-text
# version, leaving a 90.33 gap between "e" and "X" where every other letter
# pair sits at 13-15. Pure translate, no glyph geometry touched.
WM_EX_GAP = 13.0
WM_X_SHIFT = 90.33 - WM_EX_GAP
WM_X1 = 1124.55 - WM_X_SHIFT        # right edge of "X" after the shift
WM_W = WM_X1 - WM_X0                # ink width at cap height WM_CAP

FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"

TAGLINE_1 = "Build your private data assets."
TAGLINE_2 = "Connect all your agents and LLMs like a blade."


def gradients(prefix: str) -> str:
    """Gradient defs namespaced by ``prefix`` so several files can be inlined
    into one HTML document without colliding on ``id``."""
    return f"""  <defs>
    <linearGradient id="{prefix}-main" x1="95" y1="650" x2="650" y2="70" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="{PRIMARY_BLUE}"/>
      <stop offset="52%" stop-color="{PRIMARY_BLUE}"/>
      <stop offset="100%" stop-color="{SECONDARY_BLUE}"/>
    </linearGradient>
    <linearGradient id="{prefix}-light" x1="300" y1="510" x2="680" y2="300" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="{LIGHT_BLUE}"/>
      <stop offset="100%" stop-color="{SECONDARY_BLUE}"/>
    </linearGradient>
  </defs>
"""


def symbol(prefix: str | None, *, mono: str | None = None, indent: str = "  ") -> str:
    """The symbol in native coordinates. ``prefix`` selects the gradient fill;
    ``mono`` overrides both arcs with one flat colour."""
    outer = mono or f"url(#{prefix}-main)"
    inner = mono or f"url(#{prefix}-light)"
    return (
        f'{indent}<path d="{ARC_OUTER}" fill="{outer}"/>\n'
        f'{indent}<path d="{ARC_INNER}" fill="{inner}"/>\n'
    )


def placed_symbol(x: float, y: float, height: float, prefix: str | None,
                  *, mono: str | None = None, indent: str = "  ") -> str:
    """Symbol scaled to ``height`` with its bbox top-left at (x, y)."""
    s = height / SYM_H
    tx = x - SYM_X0 * s
    ty = y - SYM_Y0 * s
    return (
        f'{indent}<g transform="translate({tx:.3f} {ty:.3f}) scale({s:.6f})">\n'
        + symbol(prefix, mono=mono, indent=indent + "  ")
        + f"{indent}</g>\n"
    )


def wordmark(x: float, baseline: float, cap: float, blade_fill: str, x_fill: str,
             indent: str = "  ") -> str:
    """Outlined wordmark scaled to cap height ``cap``, ink left edge at ``x``,
    baseline at ``baseline``."""
    sc = cap / WM_CAP
    tx = x - WM_X0 * sc
    ty = baseline - WM_BASELINE * sc
    out = f'{indent}<g transform="translate({tx:.3f} {ty:.3f}) scale({sc:.6f})">\n'
    out += f'{indent}  <g fill="{blade_fill}">\n'
    for ch, d in WM_BLADE:
        out += f'{indent}    <path aria-label="{ch}" d="{d}"/>\n'
    out += f"{indent}  </g>\n"
    out += (f'{indent}  <g fill="{x_fill}" transform="translate({-WM_X_SHIFT} 0)">\n'
            f'{indent}    <path aria-label="X" d="{WM_X[1]}"/>\n'
            f"{indent}  </g>\n")
    out += f"{indent}</g>\n"
    return out


def doc(title: str, view: str, body: str, *, width: str = "", desc: str = "") -> str:
    dim = f' width="{width.split()[0]}" height="{width.split()[1]}"' if width else ""
    d = f"  <desc>{desc}</desc>\n" if desc else ""
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{view}"{dim} role="img" aria-label="{title}">\n'
        f"  <title>{title}</title>\n{d}{body}</svg>\n"
    )


def write(name: str, content: str) -> None:
    (OUT / name).write_text(content, encoding="utf-8")
    print(f"  {name}")


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("writing assets/brand/")

    # 1. symbol, full colour, square canvas with 3.2% optical padding
    side = 640.0
    tx, ty = -(SYM_CX - side / 2), -(SYM_CY - side / 2)
    write("bladex-symbol.svg", doc(
        "BladeX symbol", f"0 0 {side:.0f} {side:.0f}",
        gradients("bx-sym")
        + f'  <g transform="translate({tx:.3f} {ty:.3f})">\n'
        + symbol("bx-sym", indent="    ")
        + "  </g>\n",
        desc="Approved BladeX symbol, full-colour gradient."))

    # 2/3. monochrome symbol, one flat colour each way
    for name, colour, label in (
        ("bladex-symbol-mono-navy.svg", DEEP_NAVY, "dark surfaces on light backgrounds"),
        ("bladex-symbol-mono-white.svg", "#FFFFFF", "light surfaces on dark backgrounds"),
    ):
        write(name, doc(
            "BladeX symbol (monochrome)", f"0 0 {side:.0f} {side:.0f}",
            f'  <g transform="translate({tx:.3f} {ty:.3f})">\n'
            + symbol(None, mono=colour, indent="    ")
            + "  </g>\n",
            desc=f"Single-colour BladeX symbol for {label}."))

    # 4. app icon, 1024 canvas, iOS-style rounded square, symbol at 58%
    write("bladex-app-icon.svg", doc(
        "BladeX app icon", "0 0 1024 1024",
        gradients("bx-app")
        + '  <rect width="1024" height="1024" rx="228" fill="#FFFFFF"/>\n'
        + placed_symbol((1024 - 594 * SYM_ASPECT) / 2, (1024 - 594) / 2, 594, "bx-app"),
        desc="Rounded-square app icon, light background."))

    write("bladex-app-icon-dark.svg", doc(
        "BladeX app icon (dark)", "0 0 1024 1024",
        gradients("bx-appd")
        + f'  <rect width="1024" height="1024" rx="228" fill="{DEEP_NAVY}"/>\n'
        + placed_symbol((1024 - 594 * SYM_ASPECT) / 2, (1024 - 594) / 2, 594, "bx-appd"),
        desc="Rounded-square app icon, deep-navy background."))

    # 5. favicon, transparent, symbol filling 92% of a 32-unit square
    f = 32.0
    write("favicon.svg", doc(
        "BladeX", f"0 0 {f:.0f} {f:.0f}",
        gradients("bx-fav")
        + placed_symbol((f - 29.4 * SYM_ASPECT) / 2, (f - 29.4) / 2, 29.4, "bx-fav"),
        desc="Transparent favicon; renders legibly down to 16 px."))

    # 6. wordmark, outlined -- font-independent
    for name, blade, xf, label in (
        ("bladex-wordmark.svg", DEEP_NAVY, PRIMARY_BLUE, "light backgrounds"),
        ("bladex-wordmark-dark.svg", "#FFFFFF", SECONDARY_BLUE, "dark backgrounds"),
    ):
        cap = 124.0
        pad = 0.16 * cap
        over = 2.08 * cap / WM_CAP     # round-letter overshoot below baseline
        w = WM_W * cap / WM_CAP + pad * 2
        h = cap + over + pad * 2
        write(name, doc(
            "BladeX wordmark", f"0 0 {w:.0f} {h:.0f}",
            wordmark(pad, pad + cap, cap, blade, xf),
            desc=f"Outlined wordmark for {label}."))

    # 7/8. lockups. Proportions carried over from the approved design sheet,
    # expressed in cap-height units so every size stays consistent:
    #   symbol height 1.846 cap | symbol-divider 0.862 cap | divider-word 0.677 cap
    def lockup(prefix, cap, blade_fill, x_fill, *, bg=None, taglines=None,
               tag_fills=("", "")):
        pad = 0.485 * cap
        sym_h = 1.846 * cap
        sym_w = sym_h * SYM_ASPECT
        div_x = pad + sym_w + 0.862 * cap
        word_x = div_x + 0.677 * cap
        word_w = WM_W * cap / WM_CAP
        h = sym_h + pad * 2
        w = word_x + word_w + pad
        if taglines:
            # Live text: no reliable metrics, so estimate the advance and leave
            # 12% slack for the Helvetica/Arial fallback, which runs wider.
            tag_w = max(len(t) * 0.56 * size * cap
                        for t, size in zip(taglines, (0.340, 0.226), strict=False))
            w = max(w, word_x + tag_w * 1.12 + pad)
        baseline = h / 2 + cap / 2 if not taglines else h / 2 - 0.10 * cap
        body = f'  <rect width="{w:.0f}" height="{h:.0f}" fill="{bg}"/>\n' if bg else ""
        body += gradients(prefix)
        body += placed_symbol(pad, (h - sym_h) / 2, sym_h, prefix)
        body += (f'  <line x1="{div_x:.1f}" y1="{(h - 2.75 * cap) / 2:.1f}"'
                 f' x2="{div_x:.1f}" y2="{(h + 2.75 * cap) / 2:.1f}"'
                 f' stroke="{"#243044" if bg else DIVIDER}" stroke-width="2"/>\n')
        body += wordmark(word_x, baseline, cap, blade_fill, x_fill)
        note = ""
        if taglines:
            for i, (t, fill, size, off) in enumerate(
                    zip(taglines, tag_fills, (0.340, 0.226), (0.728, 1.253), strict=False)):
                body += (f'  <text x="{word_x + 0.03 * cap:.1f}"'
                         f' y="{baseline + off * cap:.1f}" font-family="{FONT}"'
                         f' font-size="{size * cap:.1f}"'
                         f' font-weight="{500 if i == 0 else 400}"'
                         f' fill="{fill}">{t}</text>\n')
            note = " Tagline is live text (see BRAND.md); the mark itself is outlined."
        return doc("BladeX", f"0 0 {w:.0f} {h:.0f}", body,
                   desc="BladeX lockup." + note)

    write("bladex-lockup-horizontal.svg",
          lockup("bx-lh", 72.0, DEEP_NAVY, PRIMARY_BLUE))
    write("bladex-lockup-horizontal-dark.svg",
          lockup("bx-lhd", 72.0, "#FFFFFF", SECONDARY_BLUE, bg=DEEP_NAVY))
    write("bladex-lockup-primary.svg",
          lockup("bx-lp", 124.0, DEEP_NAVY, PRIMARY_BLUE,
                 taglines=(TAGLINE_1, TAGLINE_2),
                 tag_fills=(PRIMARY_BLUE, SLATE_GRAY)))
    write("bladex-lockup-primary-dark.svg",
          lockup("bx-lpd", 124.0, "#FFFFFF", SECONDARY_BLUE, bg=DEEP_NAVY,
                 taglines=(TAGLINE_1, TAGLINE_2),
                 tag_fills=(LIGHT_BLUE, "#9AA4B4")))

    # 9. Open Graph card, 1200x630, deep navy
    cap = 96.0
    sym_h = 1.846 * cap
    body = (f'  <rect width="1200" height="630" fill="{DEEP_NAVY}"/>\n'
            + gradients("bx-og")
            + placed_symbol((1200 - sym_h * SYM_ASPECT) / 2, 150, sym_h, "bx-og")
            + wordmark((1200 - WM_W * cap / WM_CAP) / 2, 470, cap,
                       "#FFFFFF", SECONDARY_BLUE)
            + f'  <text x="600" y="530" text-anchor="middle" font-family="{FONT}"'
              f' font-size="30" font-weight="500" fill="{LIGHT_BLUE}">{TAGLINE_1}</text>\n'
            + f'  <text x="600" y="572" text-anchor="middle" font-family="{FONT}"'
              f' font-size="24" font-weight="400" fill="#9AA4B4">{TAGLINE_2}</text>\n')
    write("bladex-og.svg", doc("BladeX", "0 0 1200 630", body,
                               desc="Open Graph / social card, 1200x630."))


def build_png() -> None:
    """Raster exports. Needs cairosvg (``pip install cairosvg``); the generated
    PNGs are committed so consumers never need the dependency."""
    import cairosvg  # noqa: PLC0415 - optional, only for this step

    png = OUT / "png"
    png.mkdir(exist_ok=True)
    jobs = [
        ("favicon.svg", "favicon-16.png", 16),
        ("favicon.svg", "favicon-32.png", 32),
        ("bladex-app-icon.svg", "apple-touch-icon-180.png", 180),
        ("bladex-app-icon.svg", "icon-512.png", 512),
        ("bladex-app-icon-dark.svg", "icon-512-dark.png", 512),
        ("bladex-symbol.svg", "symbol-512.png", 512),
        ("bladex-lockup-horizontal.svg", "lockup-horizontal-1244.png", 1244),
        ("bladex-lockup-horizontal-dark.svg", "lockup-horizontal-dark-1244.png", 1244),
        ("bladex-og.svg", "og-image-1200x630.png", 1200),
    ]
    print("writing assets/brand/png/")
    for src, name, w in jobs:
        cairosvg.svg2png(url=str(OUT / src), write_to=str(png / name),
                         output_width=w, background_color=None)
        print(f"  png/{name}")


if __name__ == "__main__":
    build()
    if "--png" in sys.argv:
        build_png()
