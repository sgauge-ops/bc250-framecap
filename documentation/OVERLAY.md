# Overlay specification

`[overlay].renderer` selects `modern` for a graphical dashboard, `helix` for a
flat monospace panel with teal corner brackets and numbered sections (`01 // MODE`,
`02 // SENSORS`, `03 // KEY BINDINGS`), or `ascii` for the original text layout.
The built-in default is `ascii`; the supplied configuration selects `modern`.
All use the same data and controls. Existing
configurations must replace `dashboard` with `modern` and `classic` with `ascii`.
The dashboard displays FPS, 10% low and frametime prominently, followed by mode,
cap, hysteresis, target, hardware readings and key hints. It uses the bundled
Fira Sans font (see `src/layer/fonts/OFL.txt`), with antialiased text and
rounded, shaded panels, keycaps and mode badge. The dashboard has a gray
outline; its background opacity is configurable independently of the outline
and text. Very long values are reduced in size or ellipsized within their
cells. At small swapchain sizes, the dashboard is clipped to the available
area. The full dashboard measures 370 × 230 pixels at 1×. It is rendered at
2× when the swapchain width reaches
`[overlay].scale_2x_min_width_px` (default: 3840 px) and the whole dashboard
fits at that scale.
In the full dashboard, the FPS label follows the displayed value and moves as
its width changes.
The `modern` renderer's RGB palette is defined in `src/layer/renderer/modern.rs`.
Its title, labels, metrics, dividers, mode badge, feedback, background and key
fills can be adjusted independently; the panel and key outlines have their own
colors. The FPS value instead uses `[overlay].fps_color`.
The `helix` panel shows the same data in three numbered sections: mode (`01 // MODE`,
with the active mode in a badge — fully teal for AUTO, half-teal for MANUAL, no
highlight for OFF — followed by CAP, HYS and TARGET), sensors
(`02 // SENSORS`, three columns plus a ten-segment GTT usage bar) and key hints
(`03 // KEY BINDINGS`, outlined key boxes). It uses the bundled Fira Mono font
(same licence file), square corners, a flat background with the configured opacity and
teal corner brackets. The full panel measures 384 × 254 pixels at 1× and the FPS-only
panel 130 × 52; 2× scaling requires the whole offset panel to fit.
The `helix` panel is offset by 2 physical pixels from the top and left edges;
its origin can be changed with `ORIGIN_X` and `ORIGIN_Y` in its renderer.
All three renderers prepare updated text and geometry on background threads;
presentations continue using the last compatible prepared layout without
waiting. The first layout and a change between full and FPS-only modes may take
a few frames to appear. Preparation reacts to metric refreshes and control
changes. The ASCII frametime graph is updated separately on every presentation.
Vulkan commands are recorded before the limiter wait, using metrics last
refreshed after a previous presentation's wait. A newly refreshed metric or
AUTO adjustment is therefore visible no earlier than the next presentation,
in addition to any background preparation delay.
An unknown renderer name rejects the configuration file. The configured
renderer is selected at application startup.

The ascii overlay presents:

```text
         FPS:   60        | 10% LOW: 55 FPS  | FT: 16.67 MS     |
-----------------------------------------------------------------
         MODE: AUTO (JIT) | CAP:     60 FPS  | HYS:   5.0 FPS   | TARGET: 55 FPS
-----------------------------------------------------------------
         GPU:    43 °C    | VRM:   57 °C     | Load:   55 %     | PPT:  32.6 W
         CPU:    70 °C    | VRM:   61 °C     | Load:   35 %     | FAN:  1240 RPM
         GDDR6:  56 °C    | GTT:    512 / 12000 MB              | RAM:  4000 MB
-----------------------------------------------------------------
         [F5] CYCLE MODE  | [F6]  -1 FPS     | [F7]  +1 FPS     | [F8] CYCLE OSD
```

The full modern dashboard groups GDDR6, PPT and FAN on one hardware row,
followed by GTT and RAM, with RAM aligned under FAN. Hardware labels use the
same font size as the main metric labels and share a baseline with their
values; text that must be reduced to fit remains baseline-aligned. Hardware
values are right-aligned in their cells so their units line up vertically.
Both full overlays show
the memory readings; FPS-only mode does not. GTT used and
total come from the selected AMD GPU's `mem_info_gtt_used` and
`mem_info_gtt_total` sysfs attributes. RAM used is `MemTotal - MemAvailable`
from `/proc/meminfo`, excluding swap and accounting for reclaimable memory.
Memory values are displayed as decimal MB (1,000,000 bytes); RAM and GTT may
overlap on unified-memory systems and must not be added together.

On a uniquely selected BC-250 GPU, GDDR6 is the hottest of all eight memory
chips, read from the `bc250_memory` hwmon device's `VRAM Hotspot` channel.
`[framecap].memory_poll_rate_ms` defaults to 5000 ms; values below 2000
disable this reading, without disabling GTT or RAM. The kernel module must be
loaded separately; Framecap does not load it, install its SMU patch, or request
Polkit authorization. The module accesses the SMU and may apply a temporary
patch at load time; follow the kernel's firmware and SMU debug prerequisites.
Its mailbox access does not coordinate with other SMU tools: avoid concurrent
overclocking, patching or other SMU transactions while polling. On an error,
polling stops until the application is restarted; unavailable, invalid or
stale readings display `--` and are logged. Code saturation is shown as a
lower bound with `>=`. Closing the application waits for an ongoing kernel
read to finish. See the [validation guide](VALIDATION.md).

On release, a short press of F8 cycles full overlay → FPS only → off → full
overlay; holding it for one second cycles through `ascii`, `modern` and `helix`
without waiting for release or changing the display mode. The same applies to
a custom `[key-bindings].hotkey_toggle`. Both the display mode and renderer
are saved to the user configuration; switching also works while the overlay is
hidden. FPS-only mode shows just the FPS measurement (or `COLLECT` while
collecting) with its own background. `[overlay].startup` selects the initial mode: `0` off,
`1` FPS only, `2` full overlay (built-in default). These display modes do not
disable frame limiting or automatic regulation.

The FPS value, including `COLLECT` while collecting, is yellow by default in
both visible modes. `[overlay].fps_color` sets its color as three RGB integers
from 0 to 255, for example `[255, 255, 0]`. Other text keeps its usual colors.
An invalid RGB array rejects the configuration file, as with other invalid TOML
types.

Columns are aligned across lines: F6/CAP/10% LOW/VRM, F7/HYS/FT/Load,
and F8/TARGET/PPT/RPM. In the full overlay, FPS is shown as `FPS: value`;
while collecting, both FPS and 10% LOW show `COLLECT`. In AUTO, TARGET shows
the selected algorithm's reference: the displayed 10% low in FPS for JIT,
the configured temperature in °C for VRM, or the configured speed in RPM for
FAN. JIT shows
`TARGET: COLLECT FPS` while its reference is unavailable; MANUAL and OFF show
`TARGET: --`. A cap change highlights `CAP`, a change in active
hysteresis highlights `HYS`, an adjusted VRM/FAN target highlights `TARGET`,
and a key press highlights that key for one second. These highlights are independent. HYS is displayed in FPS for JIT,
°C for VRM, and RPM for FAN; the shortcut hints show the corresponding
adjustment step.

The full ASCII overlay also plots frametime in the empty cell to
the right of `FT`. Each physical pixel represents one interval, with the
newest at the right. The vertical scale is 0–25 ms; larger values are clipped
to the top. The first row is taller to provide a 26-pixel-high graph at 1×.
Each sample is the CPU interval between calls immediately before the downstream
Vulkan presentation.
The graph updates on every presentation, independently of
`[overlay].refresh_rate`, displays each sample on the next presentation, and
clears when the full ASCII overlay is hidden.
It appears after two successful presentations. It does not measure
GPU completion or physical scanout, and drawing the overlay itself can affect
timings. The numeric frametime is the most recent CPU interval in the
refresh-cadenced metrics snapshot, using the same presentation timestamps as
the graph. It is not averaged, can refer to an older interval than the graph, and
shows `COLLECT` until a valid interval is available.
During a frametime capture, the frametime value turns red in all full overlays;
it returns to its normal color after capture stops, before CSV export completes.
FPS-only mode does not show the frametime or capture indicator.
`[overlay].background_opacity` controls the background from 0.0 (transparent)
to 1.0 (opaque); the built-in default is 0.25. For a more opaque dashboard
similar to the mockup, try `background_opacity = 0.8`. The overlay is redrawn
on every presentation. Displayed metrics refresh every `[overlay].refresh_rate`
presentations, built-in default 10. Sensor readings are averaged independently
over `[overlay].gpu_smoothing_window_ms`; an unavailable reading clears that
sensor's history only.
These are overlay values, not the separate rolling values used for AUTO
decisions. Overlay FPS and 10% low restart collection after a cap
change; frametime needs a new valid interval. AUTO histories do not reset.
In particular, JIT's displayed TARGET is the
overlay's 10% low and can differ from AUTO's reference.

Dashboard shaders are checked in as both GLSL and SPIR-V. After changing a
shader, regenerate its binary with `glslangValidator -V`:

```sh
glslangValidator -V src/layer/shaders/dashboard.vert -o src/layer/shaders/dashboard.vert.spv
glslangValidator -V src/layer/shaders/dashboard.frag -o src/layer/shaders/dashboard.frag.spv
```
