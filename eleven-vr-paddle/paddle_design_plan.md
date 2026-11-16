# Ergonomic Eleven VR Paddle for Quest 3 + BonWasy Kaiha RH Mount (v3 — Zero-Elbow Central Spine)

**Configuration:** Quest 3 Touch Plus · **Zero-elbow coaxial central spine** (`27.5 mm` wide × `9.5 mm` thick ribbed column straight above the handle) · **Controller facing `71.5°` UP** · **2-piece custom print** (`core_plate.stl` + `handle_AN.stl` or `handle_FL.stl`, plus your `BonWasy-Kaiha-RH.stl` mount) · **`85 mm` deep-mortise epoxy handle** (zero handle screws/inserts) · **PETG · Zero supports**

![Paddle assembly views](/Users/jhonniey/.gemini/antigravity/brain/51cce63b-1bc0-435e-b403-6643d3e68c55/paddle_assembly_views.png)

---

## 1. Why the Zero-Elbow Central Spine Solves Both Flex and Tracking

Because the Quest 3 controller inserts into the tapered `BonWasy-Kaiha-RH.stl` ring from one side (`+Z_stl`), bolting the fork horizontally across a flat top edge (`+t_ax` straight down) forces the controller to point sideways-down (`-28.6°`), while routing a C-arm elbow around the outside creates a cantilevered weak link.

**The Central-Spine Solution:**
1. **No elbow, no cutout window:** Rotating the mount so its fork clamps **horizontally (`+X_p` / `10°` offset) onto a solid central spine** rising straight out of the handle centerline (`X_p = 0`) removes the cantilever arm completely.
2. **Coaxial load path:** Both M3 fork bolts sit directly inline above the handle tang (`X_p = +8.58 mm` and `+14.42 mm`), backed by an integral **`+3.50 mm` dorsal T-rib (`9.50 mm` total spine thickness)** that sweeps continuously into the lower grip throat (`Y_p = 14 mm`). Hard forehand/backhand strokes load straight down the spine into the `85 mm` mortised handle core with zero elbow flex.
3. **Controller faces UP & centered CoG:** The Quest 3 controller sits above the grip platform (`min mount Y_p = 35.95 mm`) tilted **`71.5°` UP** with its tracking faceplate unobstructed at the top (`Y_p = 80–115 mm`) and the combined mount + controller CoG centered near `X_p ≈ 0`.

![Zero-elbow orientation comparison](/Users/jhonniey/.gemini/antigravity/brain/51cce63b-1bc0-435e-b403-6643d3e68c55/zero_elbow_central_spines.png)

---

## 2. Exact Coordinates & FreeCAD Verification

- **Origin:** Shoulder centerline on the `6.000 mm` blade midplane (`Y_p = 0.0`, `Z_p = 0.0`).
- **Fork M3 through-holes (`Ø3.40 mm`):**
  - Upper hole: **`(X_p = +8.581, Y_p = +88.556) mm`**
  - Lower hole: **`(X_p = +14.419, Y_p = +55.444) mm`** (`33.623 mm` spacing)
- **Full 3D collision check:** Verified inside FreeCAD against `BonWasy-Kaiha-RH.stl` across all `16,477` vertices and `5,487` centroids (`Z_p in [-3.10, +7.00] mm`): **`0` vertex hits and `0` centroid hits**.

---

## 3. Generated CAD & Template Files

All files are synced to **`/Volumes/M.2 2TB/code/lfm-panel/eleven-vr-paddle/`**:

| Deliverable | Files in Workspace | Dimensions / Volume | Notes |
|---|---|---|---|
| **Master FreeCAD Assembly** | [paddle_adapter.FCStd](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/paddle_adapter.FCStd) | Full assembly | Parametric script: [generate_paddle_freecad.py](file:///Users/jhonniey/.gemini/antigravity/brain/51cce63b-1bc0-435e-b403-6643d3e68c55/scratch/generate_paddle_freecad.py) |
| **Central-Spine Core Plate** | [core_plate.stl](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/core_plate.stl) · [STEP](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/core_plate.step) | `89.0 × 185.6 × 9.5 mm` (`38.41 cm³`) | `6.000 mm` blade + `3.50 mm` dorsal T-rib + `85 mm` grooved/notched tang + left `Ø8.2 × 5.2 mm` weight pocket |
| **1-Piece Anatomic Handle (AN)** | [handle_AN.stl](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/handle_AN.stl) · [STEP](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/handle_AN.step) | `32.0 × 26.0 × 100.0 mm` (`52.44 cm³`) | `28×22 → 32×26` palm swell; `85.5 mm` internal mortise + `Ø2.2 mm` butt vent |
| **1-Piece Flared Handle (FL)** | [handle_FL.stl](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/handle_FL.stl) · [STEP](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/handle_FL.step) | `33.9 × 25.9 × 100.0 mm` (`47.35 cm³`) | `28×22 → 27×23 → 34×26` flare; `85.5 mm` internal mortise + `Ø2.2 mm` butt vent |
| **1:1 Rubber Templates** | [rubber_cutting_templates_1to1.pdf](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/rubber_cutting_templates_1to1.pdf) · [SVG](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/rubber_cutting_templates_1to1.svg) | `210 × 279 mm` (A4 & Letter) | Topsheet-up & mirrored sponge-up outlines + `50 × 50 mm` calibration square + full core blueprint (PDF page 2) |
| **3D Rubber Jigs** | [rubber_template_fh_thumb.stl](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/rubber_template_fh_thumb.stl) ([STEP](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/rubber_template_fh_thumb.step)) · [rubber_template_bh_index.stl](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/rubber_template_bh_index.stl) ([STEP](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/rubber_template_bh_index.step)) | `38.0 × 34.7 × 5.0 mm` & `84.5 × 34.7 × 5.0 mm` | `2 mm` stencil plates with `3.0 mm` lip hooking onto the `Y_p = 34.5 mm` left wing top edge |
| **Original Mount & Ref** | [BonWasy-Kaiha-RH.stl](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/BonWasy-Kaiha-RH.stl) · [mount_aligned_pos1.stl](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/mount_aligned_pos1.stl) | `17.09 cm³` | Print original `BonWasy-Kaiha-RH.stl` upright on its flat `Z-min` base |

---

## 4. Rubber Cutting Templates

![Rubber cutting templates preview](/Users/jhonniey/.gemini/antigravity/brain/51cce63b-1bc0-435e-b403-6643d3e68c55/rubber_templates_preview.png)

- **FH Thumb Pad (`38.0 × 31.0 mm`, `1,155 mm²`):** Sits flat across `X_p in [-18, +20] mm` cleanly clear of the dorsal T-rib (`X_p >= 22 mm`).
- **BH Index Pad (`84.5 × 31.0 mm`, `2,233 mm²`):** Spans the entire `-Z_p` Backhand throat (`X_p in [-42, +42.5] mm`).
- Print [rubber_cutting_templates_1to1.pdf](file:///Volumes/M.2%202TB/code/lfm-panel/eleven-vr-paddle/cad/rubber_cutting_templates_1to1.pdf) at **100% / Actual Size** and check the `50.0 × 50.0 mm` square before cutting (**1A/2A** topsheet-up, **1B/2B** mirrored sponge-up), or print the 3D jigs flat.

---

## 5. PETG Print Settings & Zero-Support Overhang Audit

**Baseline:** 0.4 mm nozzle · Dry PETG (`65 °C` 4–6 h) · **Nozzle `240–245 °C` / Bed `78 °C`** · Fan **30–40 %** · Textured PEI (or smooth PEI + glue stick) · **Supports: OFF**.

| STL File | Overhang > 46° (Measured) | Layer | Walls | Top/Bot | Infill | Notes |
|---|---|---|---|---|---|---|
| `BonWasy-Kaiha-RH.stl` *(upright on Z-min base)* | `0.7%` (`94 / 13,622 mm²`) | `0.16 mm` | **6** (solid tines) | `6 / 6` | `50%` gyroid | `30 mm/s` outer walls; `5 mm` brim; ream holes `3.4–3.5 mm` if tight |
| `core_plate.stl` *(flat on `Z=0`)* | **`0.00%`** (`0.0 mm²`) | `0.20 mm` | **4** | `5 / 5` | `25%` gyroid | `100%` infill cylinder modifiers (`Ø12 mm`) around the 2 fork bolt holes. Check blade thickness **`5.95–6.05 mm`** |
| `handle_AN.stl` / `handle_FL.stl` *(upright on `Y_p = 0`)* | **`0.16%` / `0.17%`** (`24.0 mm²` at `45°` gable closure peak) | `0.16 mm` | **4** | `5 / 5` | `20%` gyroid | Use a **`6 mm` outer brim** (`0.2 mm` gap); optional fuzzy skin (`0.25 / 0.4 mm`) on grip surface |
| `rubber_template_*.stl` | **`0.00%`** (`0.0 mm²`) | `0.20 mm` | `3` | `4 / 4` | `15%` | PLA works fine for jigs too |

---

## 6. Hardware, Gluing & Weight Estimate

| Qty | Item | Use |
|---|---|---|
| **2** | **M3 × 12 mm** socket/button head + **2 × M3 nyloc** + **4 × wide M3 washers** | Fork → central spine (`2.35 + 6.00 + 2.35 = 10.7 mm` stack) |
| **1** | **M3 × 10 mm** (or **M3 × 12 mm**) socket head cap screw + **1 × M3 hex nut** | Controller ring clamp (`7.8 mm` seat-to-nut-floor + `2.4 mm` nut) |
| **1** | **2-part epoxy** (e.g., 30-min epoxy or J-B Weld Plastic Bonder) | Deep `85 mm` tang-to-handle mortise bond |
| — | Contact cement or TT glue + 240-grit sandpaper | Scuff PETG and bond rubber grip pads |

- **Estimated Weight (PETG):** Mount `~19 g` + Core `~34–38 g` + Handle (`AN` or `FL`) `~36–44 g` + Bolts/Epoxy/Pads `~13 g` = **`~102–114 g`** printed paddle (`~212–240 g` total with Quest 3 controller `110–126 g`).

---

## 7. Build & Eleven Table Tennis Calibration

1. **Dry-fit:** Slide `core_plate.stl` into `handle_AN.stl` / `FL.stl` down to the `Y_p = 0` shoulder step and check the `6.00 mm` slip fit into `BonWasy-Kaiha-RH.stl`.
2. **Epoxy the handle:** Scuff the `85 mm` tang faces (120–240 grit), clean with alcohol, coat the tang grooves, `6` edge notches, and mortise walls with 2-part epoxy, and seat flush to `Y_p = 0` (excess air/epoxy vents out the `Ø2.2 mm` butt pinhole). Cure upright 24 h.
3. **Glue rubber pads & bolt mount:** Scuff and glue pads; clamp Quest 3 controller (battery cover removed) with `M3 × 10–12 mm` screw + nut; bolt fork onto central spine with `2 × M3 × 12 mm` + wide washers + nylocs (`~0.4–0.5 N·m`).
4. **Calibrate in Eleven Table Tennis *(important — rotated orientation)*:** Because the mount is rotated onto the central spine so the controller faces upward, **do not rely on the stock `CYBVRSHAPE` preset**. Open **`PADDLES` → `POSITION` → `NEW POSITION`**, hold the physical Forehand face flat against a real table edge or wall, and align the virtual paddle plane and handle position directly to match.
