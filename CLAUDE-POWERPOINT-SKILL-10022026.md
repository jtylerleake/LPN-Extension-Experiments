# PowerPoint Build Skill (No PowerPoint / No python-pptx / No LibreOffice)

Reference notes from hand-building a `.pptx` by directly authoring OOXML, on a machine
with **no Python, no python-pptx, no LibreOffice, no PowerPoint automation** — only
PowerShell, Node, and file tools. If any of those tools *are* available in a future
environment, prefer them (`python-pptx`, a headless LibreOffice render, etc.) — this
skill is the fallback for when they aren't.

**Revision 10/02/2026:** later sessions on this machine had Python (Anaconda, with `lxml`
but no `python-pptx`) and **desktop PowerPoint installed**. That enables a much better
workflow: generate the OOXML with a Python script and **render every slide to PNG through
PowerPoint's COM interface** for real visual QA. See §18–§22 (diagrams, native equations,
rendering, Windows encoding traps, user style preferences).

## 0. Check what's actually available first

```powershell
python --version; python -c "import pptx"     # python-pptx would make all of this unnecessary
python -c "import lxml"                         # enough for XML well-formedness checks
node --version                                 # if present, pptxgenjs is an alternative (needs npm registry access)
Test-Path "C:\Program Files\LibreOffice\program\soffice.exe"   # headless render/inspect fallback
Test-Path "C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE"   # COM rendering (§20)
```

If `python-pptx` is available, use it instead of hand-authoring XML — it's far less
error-prone. This skill exists for the case where none of that is true. **If PowerPoint
itself is installed, always use it to render and visually check the result (§20)**:
PowerPoint is the renderer the audience will use, so its output is the ground truth.

## 1. A `.pptx` is just a zip. Treat it that way.

Rename/copy to `.zip` (or use `Expand-Archive` directly — PowerShell doesn't care about
the extension) and explore it like any archive:

```powershell
Copy-Item "formatting_example.pptx" "example.zip" -Force
Expand-Archive -Path "example.zip" -DestinationPath ".\example_pptx_extract" -Force
```

Package layout that matters:

```
[Content_Types].xml              <- declares every part's MIME type; miss one and PowerPoint repairs/rejects the file
_rels/.rels                      <- root relationships (points to ppt/presentation.xml, docProps)
docProps/core.xml, app.xml       <- metadata (title, author, slide count) — optional but cheap to include
ppt/presentation.xml             <- slide size (EMU), slide ID list, master ID list
ppt/_rels/presentation.xml.rels  <- maps rIds in presentation.xml to actual slide/master/theme files
ppt/slideMasters/slideMaster1.xml         <- base placeholder positions/styles + txStyles (title/body defaults)
ppt/slideLayouts/slideLayoutN.xml         <- named layouts (e.g. "Title and Content") that override/add placeholders
ppt/slides/slideN.xml                     <- actual slide content, references its layout via _rels
ppt/slides/_rels/slideN.xml.rels          <- maps each slide's image/layout rIds
ppt/theme/theme1.xml             <- color scheme (accent1-6, dk1/lt1...), font scheme (major/minor = title/body fonts)
ppt/media/imageN.{png,jpg,jpeg}  <- actual image bytes, referenced by rId from slide or layout rels
ppt/presProps.xml, viewProps.xml, tableStyles.xml   <- boilerplate, safe to keep minimal
```

**Every part that has a relationship (a picture, a layout, a theme) is a two-step
reference:** the XML has `r:embed="rId2"` or `r:id="rId2"`, and the sibling
`_rels/<partname>.xml.rels` file maps `rId2` → an actual `Target` path (relative to the
part's own directory, typically `../media/image2.png` style). **Every rId used in a
part must be defined in its `.rels` file, and every file `.rels` points at must exist.**
This is the #1 way hand-authored decks come out corrupted — see the validation
checklist in §9.

## 2. Reading a huge slide XML file

Real slide XML is a single line per file (no pretty-printing) and can be 50-100k+
tokens for slides with lots of shapes/images — too big to `Read` in full. Don't try.
Instead, grep for the specific attributes you need, in small non-greedy chunks:

```
Grep pattern: <p:sp>|<p:pic>|<a:off x="\d+" y="\d+"/>|<a:ext cx="\d+" cy="\d+"/>
Grep pattern: <p:ph[^/]*/>                                  # placeholder type + idx
Grep pattern: idx="13"[\s\S]{0,400}?</p:sp>                 # a specific placeholder's shape, bounded
```

Ripgrep (the underlying tool) does **not** support lookaround (`(?=...)`, `(?<=...)`) —
plan patterns without it. If a match line is enormous, the tool silently omits it; keep
capture groups small and bounded (`.{0,N}`) rather than greedy.

For auditing what a slide or table actually contains — before editing it, or to confirm an edit
landed — write a small Node script that regexes out just the structure you care about (e.g. every
`<a:tr>`/`<a:tc>` pair with its plain `<a:t>` text, or every shape name plus its paragraph text)
rather than reading the raw single-line XML. It's reusable across every slide in the deck, gives
you a compact table/outline view instead of a wall of markup, and is much faster than repeated
bounded greps once you need to check the same shape of information on many slides (e.g. "is every
table cell filled in yet" or "did the font-size fix actually apply everywhere").

To find the design system of a template deck, in this order:
1. `ppt/presentation.xml` → `<p:sldSz cx="..." cy="..."/>` (slide dimensions in EMU).
2. `ppt/theme/theme1.xml` → `<a:clrScheme>` (accent colors) and `<a:fontScheme>`
   (`majorFont` = titles, `minorFont` = body).
3. `ppt/slideMasters/slideMaster1.xml` → default placeholder positions + `<p:txStyles>`
   (default bullet characters, indents, sizes per level).
4. `ppt/slideLayouts/slideLayoutN.xml` → the layout actually used by content slides
   (check `<p:cSld name="...">` — e.g. "Title and Content" vs "Section Divider"). This
   is where a template's *real* visual signature usually lives: kicker/eyebrow tags,
   divider lines, footer placeholders, a logo `<p:pic>` baked directly into the layout
   (not a placeholder — it just renders on every slide using that layout, no per-slide
   work needed).
5. One or two representative `ppt/slides/slideN.xml` → confirms which layout fields
   authors actually filled in vs left as inherited defaults, and reveals the real
   font-size/bullet conventions in practice (templates often define elaborate
   `lstStyle` defaults in the layout that the actual slides override per-run anyway —
   check what slides *do*, not just what the layout declares). **If the target deck
   itself is still blank** (a fresh template with empty placeholder shells — titles and
   content never filled in), its own slides won't show you this. Check for *other*
   `.pptx` files in the same working directory built from the same template lineage
   (same theme colors, same layout names) and dump their filled-in shape text/styles
   instead — often the only way to learn what an ambiguous optional placeholder is
   actually for. For example, one layout had a `Text Placeholder` at both `idx="13"` and
   `idx="14"` above the title, with shape names that gave no hint of intent; a sibling
   deck in the same folder that already had real content showed `idx="13"` used as a
   running "kicker" label repeated across a multi-slide group's topic, and `idx="14"`
   used as an optional one-line subtitle under the title — neither guessable from the
   layout XML alone.
6. Cross-reference `ppt/slideMasters/_rels/slideMaster1.xml.rels` and
   `ppt/slideLayouts/_rels/slideLayoutN.xml.rels` to find which layout the master
   points at and which images (e.g. a logo) a layout embeds.

## 3. Building from scratch: don't just copy the whole template

For a small deck that borrows a template's *look* but needs different content/slide
count, don't try to keep all of the original's slides/layouts/media and delete around
them — dangling relationships are easy to introduce that way. Instead build a **minimal
clean package**:

- Copy over: `theme1.xml` verbatim (colors/fonts), and re-derive **one** slide layout
  (the content layout you actually need) with a new local name (e.g. `slideLayout1.xml`)
  rather than keeping unused layouts (title slide, section divider, etc.) around.
- Renumber `sldLayoutIdLst` in the master down to just what you kept.
- Write `presentation.xml`, `[Content_Types].xml`, all `_rels` files, `docProps/*`,
  `presProps.xml`, `viewProps.xml`, `tableStyles.xml` fresh — these are short and
  formulaic (see §7 for a working skeleton).
- Copy only the media files you'll actually reference (plus anything a *kept* layout
  bakes in, like a logo image).

## 4. Units, placeholders, and text structure

- **EMU (English Metric Units):** 914400 EMU = 1 inch = 72 points. A 13.333in × 7.5in
  widescreen slide is `cx="12192000" cy="6858000"`.
- **Placeholders** (`<p:ph type="..." idx="N"/>`) are matched between a slide and its
  layout/master by `type` + `idx`. A slide only needs to include the placeholder shapes
  it actually populates — non-placeholder decorations defined directly in the layout
  (divider lines, a logo `<p:pic>`, background fills) render automatically on every
  slide using that layout without any per-slide XML.
- **Bullet paragraphs:** a layout's `lstStyle` sets *defaults* (bullet char, indent,
  size) per outline level, but real authors typically override per-paragraph anyway.
  A reliable, explicit pattern:
  ```xml
  <a:p>
    <a:pPr indent="-182880">
      <a:lnSpc><a:spcPct val="100000"/></a:lnSpc>
      <a:buFont typeface="Arial" pitchFamily="34" charset="0"/>
      <a:buChar char="&#8226;"/>
      <a:spcAft><a:spcPts val="500"/></a:spcAft>
    </a:pPr>
    <a:r><a:rPr lang="en-US" sz="1300" dirty="0"/><a:t>Bullet text</a:t></a:r>
  </a:p>
  ```
  Sub-bullets: add `marL="342900" lvl="1"`, swap `buChar` to `"o"` with
  `buFont typeface="Courier New"` (renders as a hollow circle), smaller `sz`.
  Bold section-header lines (no bullet): omit `buChar`/`buFont` from `pPr` (buNone is
  usually the layout default) and set `b="1"` on the run.
- **Always set `<a:lnSpc><a:spcPct val="100000"/></a:lnSpc>` explicitly per paragraph**
  rather than trusting an inherited *fixed-point* line spacing (`spcPts`) from the
  layout. A fixed point value (e.g. "14pt line spacing") sized for 11pt body text will
  visibly overlap if you reuse the same paragraph style at 18pt. Percentage-based
  spacing scales with whatever font size you actually use.

## 5. Pictures: aspect-ratio-safe cropping for a photo grid

`<a:stretch><a:fillRect/></a:stretch>` alone **distorts** an image to exactly fill its
frame — fine for a diagram/screenshot, bad for a real photo whose aspect ratio doesn't
match the target cell. To center-crop a photo into a cell without warping it (like CSS
`object-fit: cover`), compute an `<a:srcRect>` (percentages, in thousandths of a
percent, i.e. `val="18107"` = 18.107%):

```
sourceAspect = imageWidthPx / imageHeightPx
targetAspect = cellWidthEMU / cellHeightEMU

if sourceAspect < targetAspect:      # source relatively taller -> crop top/bottom
    visibleHeightPx = imageWidthPx / targetAspect
    cropEachSide = (imageHeightPx - visibleHeightPx) / 2
    pct = cropEachSide / imageHeightPx * 100000
    -> <a:srcRect t="{pct}" b="{pct}"/>

if sourceAspect > targetAspect:      # source relatively wider -> crop left/right
    visibleWidthPx = imageHeightPx * targetAspect
    cropEachSide = (imageWidthPx - visibleWidthPx) / 2
    pct = cropEachSide / imageWidthPx * 100000
    -> <a:srcRect l="{pct}" r="{pct}"/>
```

Get real pixel dimensions first (don't guess from a resized preview):

```powershell
Add-Type -AssemblyName System.Drawing
$img = [System.Drawing.Image]::FromFile((Resolve-Path "photo.jpg"))
"$($img.Width) x $($img.Height)"; $img.Dispose()
```

For a **transparent-background logo/wordmark**, don't crop-to-fill at all — compute a
box that *preserves* its native aspect ratio, sized modestly, and center it in the
available space instead. Cropping a shield/logo to force it into a grid cell cuts off
the artwork.

## 6. Safety net: `normAutofit`, not `noAutofit`

When you can't render the deck to visually confirm text fits (no PowerPoint/LibreOffice
available), don't rely on hand-estimated line-wrap math alone as the only guard against
overflow. Set the content placeholder's body properties to:

```xml
<a:bodyPr lIns="0"><a:normAutofit/></a:bodyPr>
```

`normAutofit` with no cached `fontScale`/`lnSpcReduction` tells PowerPoint (and
LibreOffice/Keynote) to compute shrink-to-fit live when the file is opened — a safety
margin against text overflowing its box, at the cost of PowerPoint possibly shrinking
text a bit if your estimate ran long. Still do the estimate (approx. char width ≈
0.5 × font-size-in-points; box height in points = EMU / 914400 × 72) to keep the
*unshrunk* size reasonable — autofit is a backstop, not a substitute for sizing content
sensibly in the first place.

## 7. Minimal working skeleton (fill in slide count / content)

Files needed, each is short and mostly boilerplate — write once, reuse the shape every
time:

- `[Content_Types].xml` — `Default` entries for `rels`, `xml`, `png`, `jpg`, `jpeg`;
  `Override` entries for every named part (`presentation.xml`, each `slideN.xml`, each
  `slideLayoutN.xml`, `slideMaster1.xml`, `theme1.xml`, `presProps.xml`, `viewProps.xml`,
  `tableStyles.xml`, `docProps/core.xml`, `docProps/app.xml`).
- `_rels/.rels` — root: officeDocument → `ppt/presentation.xml`, plus core/extended
  properties.
- `docProps/core.xml`, `docProps/app.xml` — title/creator/slide count. Cheap, do it.
- `ppt/presentation.xml` — `sldMasterIdLst` (one entry), `sldIdLst` (one entry per
  slide, arbitrary but unique `id` ≥ 256), `sldSz`, optionally a `defaultTextStyle`.
- `ppt/_rels/presentation.xml.rels` — rId per slide + master + theme + presProps +
  viewProps + tableStyles.
- `ppt/slideMasters/slideMaster1.xml` + its `_rels` (→ one layout + theme).
- `ppt/slideLayouts/slideLayoutN.xml` + its `_rels` (→ master, + any baked-in image).
- `ppt/slides/slideN.xml` + `_rels/slideN.xml.rels` (→ layout + any images that slide
  uses).
- `ppt/theme/theme1.xml`, `ppt/presProps.xml`, `ppt/viewProps.xml`, `ppt/tableStyles.xml`.

## 8. Zipping it up in PowerShell

`Compress-Archive` is unreliable for OOXML in practice (creates a wrapping top-level
folder unless you're careful with the source glob). Build the zip manually instead:

```powershell
$build = "path\to\assembled\package"   # contains [Content_Types].xml at its root
$out   = "path\to\output.pptx"

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
# NOTE: shell state (loaded assemblies) does NOT persist between separate tool calls —
# re-run these Add-Type lines every time, in the same call as the zip logic.

$zip = [System.IO.Compression.ZipFile]::Open($out, [System.IO.Compression.ZipArchiveMode]::Create)
Get-ChildItem -Path $build -Recurse -File | ForEach-Object {
    $relPath = $_.FullName.Substring($build.Length + 1) -replace [regex]::Escape([IO.Path]::DirectorySeparatorChar), "/"
    $entry = $zip.CreateEntry($relPath, [System.IO.Compression.CompressionLevel]::Optimal)
    $es = $entry.Open(); $fs = [System.IO.File]::OpenRead($_.FullName)
    $fs.CopyTo($es); $fs.Close(); $es.Close()
}
$zip.Dispose()
```

`ZipArchiveMode::Create` throws if the target file already exists — it does not
overwrite. Delete first, or write to a new filename. **If the previous `.pptx` is open
in PowerPoint (e.g. the user is reviewing it), the delete will fail with a
locked-file error** — don't fight this; save the revision under a new filename
(`... (v2).pptx`) and let the user close the original before you overwrite in place.

### PowerShell sandbox quirks hit while doing this

- A path containing **`[` or `]`** (e.g. `[Content_Types].xml`) is treated as a wildcard
  by `-Path`-style parameters — use `-LiteralPath` for `Get-Content`, `Test-Path`,
  `Remove-Item`, etc. whenever the filename has brackets.
- The sandbox's command-safety filter can misfire on `Remove-Item` calls that appear
  *anywhere* in a multi-statement script if a later unrelated string literal looks
  path-like (observed: `.Replace('\','/')` and `-replace [...], "/"` both triggered a
  false "Remove-Item on system path blocked" error tied to the wrong token). If a
  `Remove-Item` call errors in a way that doesn't match anything in your actual delete
  logic, isolate it into its own separate tool call with nothing else in the script.

## 9. Validation checklist (do this before calling it done — you can't visually preview)

Run all of these; a hand-authored deck that "looks right" in the XML can still be
silently broken:

1. **Every XML/rels part parses** — load each with `[xml](Get-Content -Raw ...)` in a
   loop and catch parse errors.
2. **Every `r:embed`/`r:id` used in a part exists in that part's `.rels` file** — regex
   both sides (`r:(?:embed|id)="(rId\d+)"` vs `Id="(rId\d+)"`) and diff.
3. **Every `Target` in every `.rels` file resolves to a real file** — join the target
   (which is relative to the *part's* directory, e.g. `ppt/slides/_rels/*.rels` targets
   are relative to `ppt/slides/`) and `Test-Path` it.
4. **`[Content_Types].xml` declares every part** — either via an `Override` for that
   exact part name, or a `Default` covering its extension.
5. **The zip actually opens** — `[System.IO.Compression.ZipFile]::OpenRead()` and list
   entries as a final sanity check.

None of this guarantees the deck *looks* good (there's no way to confirm that without
rendering), but it guarantees PowerPoint won't reject or "repair" the file, which is the
failure mode most worth ruling out mechanically.

## 10. Fitting text into a fixed-size box or cell (when font size AND box size are both locked)

Users will sometimes hand you *both* a fixed font size and a fixed box/cell size ("cells must
be 1.2in tall, 4.3in wide, font must stay at 13pt") and explicitly forbid overflow past the
slide. `normAutofit` is off the table (shrinking the font would break the fixed-size rule), and
`noAutofit` boxes/table rows don't clip — a table row's `h` is a *minimum* PowerPoint silently
grows to fit content, and a `noAutofit` textbox just overflows past its own shape edge into
whatever is below it. With no shrink-to-fit backstop available, the only lever left is *how much
text you write*. Do the arithmetic before writing content, not after:

- Usable width (pt) = `(box width EMU − lIns − rIns) / 12700`, then
  `chars/line ≈ usable width / (0.5 × font size in pt)` — the same char-width heuristic as §6,
  used predictively instead of as a backstop.
- Line height (pt) ≈ `1.2 × font size in pt × (lnSpc spcPct / 100000)`. E.g. 13pt font at 100%
  spacing ≈ 15.6pt/line; at 130% ≈ 20.3pt/line.
- Usable height (pt) = `(box height EMU − tIns − bIns) / 12700` (table cells: tIns/bIns come from
  `marT`/`marB` on `tcPr`, default 45720 EMU each if unset).
- Budget lines = usable height / line height. Sum of `ceil(paragraph length / chars-per-line)`
  across all paragraphs, plus any `spcAft`/`spcBef` gaps converted to points, must stay under
  that budget — leave real margin (20%+), not a hairline fit, since this is all approximation and
  there's no way to render-check the result.

For anything beyond a one-off sanity check, write a tiny Node estimator — box/font/spacing in,
candidate paragraph strings in, lines-needed vs. budget per cell out — and run every candidate
cell through it before committing text to XML. It's far faster and more reliable than
hand-counting characters, and it lets you quickly compare "does bumping line-spacing from 100%
to 130% still fit" without redoing the XML first.

**Check the file's existing geometry before assuming you need to resize anything.** If a user
gives a "must be exactly N inches" spec, convert it to EMU and grep the current file first — it
has a good chance of already matching, especially if a human already laid the deck out and you're
just adding/editing content. If so, the actual bug is almost always that your added text is too
long for the existing box, not that the box itself is wrong.

## 11. Tables in hand-authored decks

**`mc:AlternateContent` around a table usually isn't needed.** PowerPoint often wraps a table
containing an OMML equation (`<a14:m>`) in `<mc:AlternateContent><mc:Choice Requires="a14">`(full
table, real equations)`</mc:Choice><mc:Fallback>`(same table, but any cell with an equation is
flattened to a static image via `<a:blipFill>`)`</mc:Fallback></mc:AlternateContent>`. The
Fallback exists only for pre-2010 PowerPoint, which no realistic viewer runs today. Unless told to
preserve legacy compatibility, strip the wrapper down to just the `Choice` content (move the
`xmlns:a14` declaration onto the surviving `<p:graphicFrame>` tag) — this halves the XML you have
to read/edit and eliminates the "the same text lives in two places, and only one of them still has
the maths" trap that makes surgical edits error-prone.

**A table row's `h` is a minimum, not a cap — and there's no shrink-to-fit for table text.**
Unlike a placeholder body (which can get `normAutofit`), a table cell with content taller than its
row just grows that row; there's no equivalent of "shrink the font until it fits." If a user gives
a fixed row height and a fixed font size, use the estimator technique in §10 — the row height
they gave you is very likely already correct, so the fix is shortening your cell text, not
touching the XML `h` value.

**Uniform cell size across a many-column comparison doesn't scale past ~2 data columns on a
widescreen slide, if you also want a label column.** A label column (~2.5–3.5in) plus data columns
pinned to some fixed width (say 4.3in) adds up fast: 2 data columns fits comfortably
(label + 2×4.3in ≈ 12in of a 13.33in-wide slide); 3 data columns at the same fixed width does not
(label + 3×4.3in ≈ 16in+). When a user wants more items compared than fit at their stated
per-column width, don't silently shrink the columns to cram everything onto one slide — that
breaks the "every cell is the same size" instruction they just gave you. Flag the conflict and
either split the group across multiple identically-shaped slides (2 columns each) or ask which
they'd prefer; don't guess silently on a structural call like this.

## 12. Adding a brand-new slide to an existing deck: every file that has to change

Beyond just writing `ppt/slides/slideN.xml`, a new slide is invisible to PowerPoint unless *all*
of the following are updated together — miss one and the deck silently drops the slide, or
PowerPoint flags it as needing repair:

1. `ppt/slides/slideN.xml` — the slide content itself.
2. `ppt/slides/_rels/slideN.xml.rels` — at minimum, the relationship to its slide layout.
3. `ppt/presentation.xml` — add a `<p:sldId id="..." r:id="rIdN"/>` entry to `sldIdLst` (id must
   be unique and ≥256; follow the deck's existing id-numbering scheme).
4. `ppt/_rels/presentation.xml.rels` — add the `rId → slides/slideN.xml` relationship (pick an
   rId not already used in that file — they're not always contiguous).
5. `[Content_Types].xml` — add an `Override` entry for `/ppt/slides/slideN.xml`.
6. `docProps/app.xml` (optional but cheap, and keeps the file's own metadata honest) — bump
   `<Slides>`, the `Slide Titles` count inside `HeadingPairs`, and add the new title to
   `TitlesOfParts` (bump its `vt:vector size` too).

Run the validation checklist (§9) after wiring all of this in — it catches a missed rel or
Content_Types entry immediately, before anything gets zipped.

## 13. Reading .docx (or other Office) context files the same way

A `.docx` is the same OOXML-zip-of-XML-parts format as a `.pptx` — `word/document.xml` holds the
body text as `<w:p>` paragraphs of `<w:t>` runs. When a user hands you a Word doc for context (a
research writeup, a long-form reference doc), don't rely on a generic binary-file reader;
`Expand-Archive` it like a `.pptx` and pull `<w:t>` runs out of `word/document.xml` the same way
you'd pull `<a:t>` runs from a slide. A tiny Node script
(`matchAll(/<w:t[^>]*>([^<]*)<\/w:t>/g)` per `<w:p>`) gets clean plain text in one pass, and the
same trick (swap the tag names) works for `.xlsx` shared strings if you ever need one of those.

## 14. Writing the actual slide content: style conventions worth defaulting to

- **If a user says "don't be overly formal" or similar, write like you're explaining it to a
  colleague** — contractions, a concrete example before the abstract definition, short sentences.
  Save the dense academic phrasing for when they ask for it.
- **When a request implies math/logic notation but nobody's asked for a rendered equation object,
  write the notation as plain LaTeX source text** (e.g. `$\mathcal{O}(2^{|\varphi|})$`), not OMML
  (`<a14:m>`/`<m:oMath>`). It's far less XML to hand-author, it's trivial for a human to convert to
  a real equation later inside PowerPoint, and it sidesteps the whole
  `mc:AlternateContent`/Fallback-image problem in §11 entirely for anything added fresh.
  **However:** this user has explicitly said that *all* mathematical notation should render as
  real equations ("very clean looking math"), not text. For this user, use native OMML from
  the start. §19 has a tested generator approach that makes this cheap.
- **"Fill the box up more" usually means increase line spacing and/or add more content, not
  enlarge the font** — font size is often independently constrained. Bump `lnSpc`'s `spcPct`
  (100000 → 130000 or 150000 is a reasonable step) and/or `spcAft` between paragraphs, then re-run
  the fit math in §10 with the new line-height to see how much more text that spacing change
  actually affords.
- **When asked to turn flowing paragraph text into bullets, keep every bullet's `pPr` explicit**
  (`marL`/`indent` for the hanging indent, `buFont`/`buChar`, and `lnSpc`/`spcAft` per paragraph)
  rather than relying on inherited layout defaults — many custom layouts default content
  placeholders to `buNone`, so bullets have to be turned on explicitly per paragraph, not assumed.
- **Avoid Unicode combining diacritics** (e.g. a combining overline U+0304 to fake a barred symbol
  like Q̄, or a combining hat for Q̂) inside `<a:t>` runs — font/renderer support for stacking a
  combining mark on an arbitrary base letter is inconsistent, unlike the precomposed Greek/math
  letters (`λ γ α χ φ θ ζ ϱ Σ Δ ◊ □ ≥ ≤ × →`) which are ordinary codepoints every standard font
  renders correctly. When source notation uses a barred/hatted/hollow symbol, either describe it in
  prose ("the extended Q-function") or fall back on the plain-LaTeX-source convention from above:
  write subscripts and superscripts inline as `_x` and `^x` (e.g. `φ_θ`, `G_L^φ`, `Q(χ,a)`) rather
  than attempting to reproduce the exact typeset glyph.

## 15. Writing genuinely dense, thorough content (not just avoiding overflow)

§10 covers the *hard-constrained* case: a user hands you a fixed font size and fixed box and
forbids overflow, so the only lever is how much text you write, and the goal is a safe margin
under the budget. This section covers the opposite ask — "make this denser," "more thorough,"
"less whitespace," "more technical detail" — where box/font aren't fixed by the user, you're
choosing how much to say, and an *under*-filled box is now the failure mode, not overflow.

- **Treat "looks safely under budget" as unfinished, not done, when density was requested.** A
  content placeholder sitting at 60-70% of its usable height with a comfortable margin is exactly
  what §10's overflow-avoidance math produces by default — but it reads as sparse on the slide. If
  the ask is for depth/density, that same box should land around 80-90% of budget: dense but still
  carrying a real (10-15%) buffer against the char-width heuristic being approximate, since there's
  still no way to render-check the result.
- **Build a small Node estimator once per deck and iterate content against it directly**, rather
  than hand-counting characters per bullet. Reuse the geometry/line-height math from §10
  (`usableWidthPt`, `charsPerLine(sz, marL)`, `lineHeightPt(sz, spcPct)`), but drive it from a list
  of paragraph objects (`{text, sz, spcPct, spcAftPt, marL}`) covering an entire slide's content —
  headers, bullets, and sub-bullets together — and have it report total points used vs. the box's
  usable height as a percentage. Adjust wording/add-or-cut bullets and re-run until each slide lands
  in the target band; this turns "is this dense enough" from a guess into a number you can compare
  slide-to-slide across the whole deck.
- **Use a two-level bullet structure to add real depth without just cramming more top-level
  bullets.** Keep level-0 bullets as the conceptual claim (the size already used elsewhere in the
  deck, e.g. 11pt); add indented level-1 sub-bullets — smaller font (e.g. 10pt), doubled `marL`,
  `lvl="1"`, a hollow-circle `buChar="o"` with `buFont typeface="Courier New"` — carrying the
  technical elaboration: a formal definition, a formula, a concrete number, a citation, a worked
  example. This reads as depth (a claim, then its technical backing) rather than clutter (a longer
  flat list), and it's naturally where notation/formulas belong without bloating the top-level line
  a reader skims first.
- **Separate the content DATA from the XML-GENERATION code once a deck needs revision passes.**
  Define slide content as a plain data structure (per slide, an ordered array of
  `{type: 'header'|'bullet'|'subbullet', text}` objects) in one file, and a small builder function
  in another that turns that array into the paragraph XML for a given placeholder, applying the
  house style (bullet char, indents, spacing, font sizes) consistently. When the user asks for a
  revision — "denser," "add a section," "shorten this" — you edit the data array and re-run the
  builder, instead of hand-editing large inline XML strings a second (or third) time. This is
  dramatically less error-prone at density, because the styling boilerplate (which is most of the
  XML by byte count) never has to be retyped or eyeballed for consistency.
- **When swapping in revised paragraph content for a placeholder you've edited before, anchor the
  replacement on a stable structural fragment, not the previous paragraph text.** The previous
  content is exactly what's changing, so matching against it breaks on every revision. Instead,
  regex-match from the placeholder's opening boilerplate through to `</a:p>`-adjacent structural
  markers — e.g. `<p:ph idx="1"/></p:nvPr></p:nvSpPr><p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/>`
  up through `</p:txBody></p:sp>` — and replace everything in between with freshly generated
  paragraphs. This anchor is stable across revisions because it's boilerplate that doesn't change,
  only the paragraphs inside it do. (See §16 for a sharp-edged consequence of this pattern.)

## 16. Gotcha: a content-swap regex can silently eat `<a:bodyPr/>`/`<a:lstStyle/>`

If you build the "stable anchor" regex from §15 carelessly — capturing everything between
`<p:txBody>` and `</p:txBody>` and replacing the *whole* captured span with newly generated
paragraphs — you will also capture and discard `<a:bodyPr/>` and `<a:lstStyle/>` if they appear
(as they normally do) right after `<p:txBody>` and before the first `<a:p>`. The result still looks
fine under `[xml](Get-Content -Raw ...)` (it's well-formed XML, just missing two empty elements) and
still passes the rels/Content-Types checks in §9, so nothing in the standard validation checklist
catches it — but `<a:bodyPr>` is a required child of `CT_TextBody` in the schema, and dropping it is
exactly the kind of subtle structural corruption that "looks right" under every check you'd normally
run and can still cause PowerPoint to flag the file for repair.

- **Anchor the regex to end right *after* `<a:lstStyle/>`**, not right after `<p:txBody>` — i.e.
  match `<p:txBody><a:bodyPr/><a:lstStyle/>` as part of the fixed prefix, and only replace what
  comes after it (the `<a:p>...` paragraphs) through `</p:txBody>`. That keeps the required elements
  outside the replaced span entirely, so they can't be lost regardless of what the replacement text
  is.
- **After any bulk content swap across multiple slides, grep for the literal sequence
  `<p:txBody><a:bodyPr/><a:lstStyle/><a:p>` (or whatever the deck's actual boilerplate is) across
  every edited slide** as a fast regression check — one Grep call listing "files with matches"
  should return every slide you touched. A slide missing from that list lost its `bodyPr`/`lstStyle`
  in the swap and needs a targeted fix before moving on to rezip.

## 17. What this does NOT replace

If `python-pptx`, a headless LibreOffice, or PowerPoint itself becomes available in a
given environment, prefer those over this hand-authoring approach — they eliminate
whole categories of the mistakes above (dangling rIds, content-type mismatches) by
construction, and critically, they let you actually render a preview to check the
result visually. (With desktop PowerPoint installed, hand-authored XML *can* be
render-checked — see §20 — and that combination worked very well for diagram slides.)

---

## 18. Drawing technical diagrams (architecture sketches) as native shapes

Task pattern: "draw a diagram/schema of the architecture into slide N of this deck" on a
template whose slides already hold the title/kicker/subtitle placeholders. What worked:

- **Keep the template slide; inject shapes.** Unzip, read `ppt/slides/slideN.xml`, and insert the
  generated shape XML just before the last `</p:spTree>` (`xml.rindex("</p:spTree>")`). The
  existing placeholders (title, kicker, subtitle, slide number) stay intact. Rezip by copying
  every other entry unchanged (`zipfile` in Python: iterate `zin.infolist()`,
  `zout.writestr(item, data)`). Build to a scratch file first, then copy it over the real
  file once it passes QA.
- **Find the free canvas first.** Read the layout's placeholder geometry
  (`<a:off>`/`<a:ext>` of each `<p:ph>`). For the 4:3 (10" × 7.5") weekly-meeting template:
  subtitle ends ≈1.42", footer rule ≈7.015" → usable diagram area ≈ x 0.5–9.5", y 1.5–6.9".
- **Write a small shape library in Python, working in inches** (`E(v) = round(v*914400)`):
  - `shape(x,y,w,h, paras, geom="roundRect", fill, line, lw, dash, adj, anchor, ins)` →
    `<p:sp>` with `<a:prstGeom>` and a `<p:txBody>` (`wrap="square"`, explicit
    `lIns/tIns/rIns/bIns`, `<a:noAutofit/>`). `adj` controls roundRect corner radius
    (`<a:gd name="adj" fmla="val 9000"/>`); `prstDash val="dash"` makes dashed borders
    (good for "optional" layers and loop containers).
  - `text(...)` = the same shape with no fill/line and zero insets (for labels).
  - `rect(x,y,w,h, fill, line, alpha)` for grid cells. Transparency is
    `<a:srgbClr val="..."><a:alpha val="40000"/></a:srgbClr>` (0–100000). This is how
    heatmaps and "soft"/probabilistic tiles were drawn (opacity = probability).
  - `arrow(points, color, lw, dash)` → **one `custGeom` open path per polyline**
    (`<a:path fill="none">` with `moveTo`/`lnTo`, `<a:tailEnd type="triangle"/>` on the
    `<a:ln>`). One shape gives clean right-angle elbow arrows with a single arrowhead.
    Bounding box = min/max of points; clamp width/height to ≥1 EMU for straight
    vertical/horizontal lines.
  - Give every shape a descriptive `name` (`"Decoder 123"`), so the user can find it in
    PowerPoint's Selection Pane.
- **Diagram vocabulary that read well:** a column per stage (Encoder | dashed
  "Latent Optimization" loop container | Outputs); a circle node for the state being
  optimized (`z_k`); small pill tags for side inputs (`x_i`) with short gray arrows; a
  vertical gray **bus line** collecting several losses into one "Total Objective" box; a
  **dashed red arrow** for the gradient going back to the node; a legend card (Baseline /
  New Module / Loss Term / Gradient). Mini ARC-style grids (black square + a few colored
  cells) are cheap and very effective as concrete input/output examples.
- **Highlight the novel part with color, keep the rest neutral.** Baseline components:
  template navy outline + very light navy fill (`0C234B` / `E9EEF5`). New components:
  template red (`AB0520` / `FBECEE`). Loss terms: gray (`7F7F7F` / `F5F5F5`). Containers:
  dashed `9AA6B8`. This satisfied "professional, not too wild."
- **Make the second ("detail") slide a different layout from the first:** a left-to-right
  forward-pass pipeline across the top (input vector → layer blocks sized by width →
  reshaped logit grid → softmax heatmap → soft tile → hard tile), then three panels below
  (worked example, spec table, how the output is used). Tables were drawn as text boxes
  plus thin rule lines instead of `a:tbl`; that gives full control and nothing auto-grows.

## 19. Native equations (OMML) in hand-authored slides — tested recipe

PowerPoint stores equations as OMML inside the text body, gated by markup compatibility.
This structure renders correctly in PowerPoint 365 (verified by COM render):

```xml
<mc:AlternateContent xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006">
  <mc:Choice xmlns:a14="http://schemas.microsoft.com/office/drawing/2010/main" Requires="a14">
    <p:sp> ... <p:txBody><a:bodyPr .../><a:lstStyle/>
      <a:p><a:pPr algn="l"/>
        <a:r>...plain text run...</a:r>
        <a14:m><m:oMath xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math">
          ...math runs/structures...
        </m:oMath></a14:m>
      </a:p></p:txBody></p:sp>
  </mc:Choice>
  <mc:Fallback>
    <p:sp> ...same shape, same id, math flattened to plain text... </p:sp>
  </mc:Fallback>
</mc:AlternateContent>
```

- **Wrap at the `<p:sp>` level**, not around the paragraph. `<a14:m>` sits directly inside
  `<a:p>`, mixed freely with ordinary `<a:r>` runs (inline math inside a sentence works).
  A plain-text Fallback shape is fine and saves rendering a fallback image.
- **Math run:** `<m:r>[<m:rPr><m:sty m:val="p"/></m:rPr>]<a:rPr lang="en-US" sz="1050" b="0" i="0">
  <a:solidFill>...</a:solidFill><a:latin typeface="Cambria Math" .../></a:rPr><m:t>…</m:t></m:r>`.
  Order matters: `m:rPr`, then `a:rPr`, then `m:t`. Size and color come from the `a:rPr`.
- **Italic variables:** write the Unicode *mathematical italic* code points yourself, the same
  way PowerPoint does: a–z → U+1D44E+, A–Z → U+1D434+, **h → U+210E** (the gap in the block),
  lowercase Greek α–ω → U+1D6FC+. Upright text (`log`, `argmax`, `CE`, subscript labels
  like "dec"/"exec"/"total", ℝ, digits, operators) = normal characters with `m:sty p`.
- **Structures used** (each takes `<m:…Pr><m:ctrlPr>{a:rPr}</m:ctrlPr></m:…Pr>`):
  `m:sSub` (e/sub), `m:sSup` (e/sup), `m:sSubSup`, `m:acc` with
  `<m:chr m:val="&#x302;"/>` for hats (Ŝ, T̂, ŷ: real accents, no combining-char hacks),
  `m:f` with `<m:type m:val="lin"/>` for compact `1/n`, and `m:func` (`m:fName`/`m:e`) for
  log/softmax/argmax. For `argmax_k`, put an `m:sSub` inside `m:fName`.
- **Big-operator gotcha:** `m:nary` (∑ with limits) renders at *display* size even inline. It
  was too tall and cluttered the boxes. **Use an upright `∑` run wrapped in `m:sSub`/`m:sSubSup`
  instead** (compact ∑ᵢ), followed by a thin space U+2009 and the summand.
- **Alignment gotcha:** a paragraph that contains *only* an `m:oMath` is treated as display math
  and **centered regardless of `algn="l"`**. For math-only paragraphs, emit
  `<m:oMathPara><m:oMathParaPr><m:jc m:val="left|center|right"/></m:oMathParaPr><m:oMath>…`
  that matches the paragraph alignment.
- **Script letters:** raw U+1D4A9 (𝒩) rendered as *nothing*. Use plain `N` with
  `<m:rPr><m:scr m:val="script"/><m:sty m:val="p"/></m:rPr>` instead. 𝟙 (U+1D7D9) and ℝ
  (U+211D) rendered fine.
- **Make it cheap: write a tiny LaTeX-subset → OMML converter** (≈200 lines of Python): a
  recursive-descent parser for `x_{..}`, `x^{..}`, `\hat{}`, `\frac{}{}`, `\sum_{}^{}{body}`,
  `\op{name}_{sub}{body}`, `\text{}`, `\mathcal{}`, and symbol macros (`\in \R \mid \nabla
  \odot \sim \times \to \cdot \dots \one \, \quad`), plus a `plain()` flattener for the Fallback.
  In slide text, mark math with `$…$`: the paragraph builder splits on `$` and emits text runs
  and `<a14:m>` alternately. Then every equation on the slide is one readable string, e.g.
  `"$L_{\\text{total}} = L_{\\text{dec}} + λ_{\\text{exec}} L_{\\text{exec}}$"`.
- Use `∣` (U+2223, `\mid`) for conditional bars (p(y ∣ x, z)), not `|`. Use `−` (U+2212) for
  minus: the converter should map `-` automatically.

## 20. Rendering slides with PowerPoint COM for visual QA

```powershell
param($In, $Prefix)
$app = New-Object -ComObject PowerPoint.Application
$p = $app.Presentations.Open($In, $true, $false, $false)   # ReadOnly, Untitled, WithWindow=false
$i=1; foreach ($sl in $p.Slides) { $sl.Export("$Prefix-$i.png", "PNG", 2000, 1500); $i++ }
$p.Close()
```

- Save this as `render.ps1` in the scratchpad and call it from bash with
  `powershell -NoProfile -File render.ps1 -In "<win path>" -Prefix "<win path>"`
  (`cygpath -w` converts paths). Then view the PNGs with the Read tool.
- **PowerShell variables are case-insensitive:** `$S` (scratch dir) and `$s` (loop slide) are the
  *same variable*. Export then failed with "couldn't find System.__ComObject\…png". Use
  distinct names (`$D`, `$sl`).
- Export at 2000×1500 for a 4:3 deck (200 px per inch). That makes pixel → inch conversion trivial
  when measuring where text wraps.
- Loop: build → validate XML (`lxml.etree.fromstring` on every part, `zipfile.testzip()`) → render
  → inspect → patch → repeat. Expect 3–5 iterations for a dense diagram. Each one takes seconds.
- **Text-width reality check:** Aptos at 10pt measured ≈0.063"/char, and lines wrapped earlier than
  the 0.5×font-size heuristic predicted. Lines of ~27 characters wrapped in a 1.8"-wide box
  with 0.06" insets. Leave ≥10% slack, or just render and look. Cambria Math equations are
  wider than the same text in Aptos.

## 21. Windows shell/encoding traps hit while scripting

- **Python reading a bash heredoc on Windows decodes stdin as cp1252.** Non-ASCII characters in
  the patch code (`·`, `θ`, `∑`, `→`) got mangled, so `str.replace` anchors silently failed to
  match. Fix: **write patch scripts to a `.py` file with the Write tool and run them with
  `python -X utf8 patch.py`.** Run the generator itself with `-X utf8` too.
- Make each patch an `assert s.count(old) == 1` replace helper. It fails loudly *before* writing
  anything when an anchor doesn't match, so a half-applied patch can't happen.
- Don't use `sed` to insert `\uXXXX` escapes into Python source: `sed` turns `∑` into
  `u2211`. Use the Edit tool for anything containing backslash escapes.

## 22. This user's presentation preferences (learned 10/02/2026)

- **Technical and thorough, but professional and neat.** Muted colors from the template (navy /
  red / grays), with ARC-palette colors only where they are data (grid cells, color keys).
  "Do not make colors too wild."
- **All math as real equations** (§19). Clean, typeset look, not Unicode-in-text.
- **Proper casing:** Title Case for headings, box titles, and short labels/captions ("Soft Tile",
  "Each Row Sums to 1", "How the Output Is Used"). Sentence case only for full explanatory
  sentences. No ALL-CAPS section headers. US spelling ("color", "optimization", "minimize").
- **Readable from a distance:** on a 10"-wide 4:3 slide, nothing below **10pt**. Body 10.5pt,
  box titles 12pt, equations 10.5–11pt. The first draft's 7–8.5pt text was rejected as too small.
- **De-clutter without deleting information.** When enlarging type, *keep every fact* but:
  reword to shorter phrases, give each box one pattern (title + symbol / one-line equation /
  one-line caption), fold footnotes into the diagram where they belong (training objective
  under the encoder; "all terms minimized" in the objective's title), turn repeated
  per-row formulas into one general rule, and use the whole canvas height so rows breathe.
- Keep the template's own placeholders (kicker, title, subtitle) untouched. Draw only in the
  free area.
- When the source material has a notational inconsistency (here, a log-likelihood "maximized"
  term added to minimized losses), pick a consistent convention on the slide, state it, and tell
  the user in the summary instead of silently copying the inconsistency.
