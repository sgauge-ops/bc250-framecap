# Configuration

Framecap creates `~/.config/framecap.toml` from its built-in defaults at layer
initialization if it does not exist. It reads configuration in this order:

1. `~/.config/framecap.toml`.
2. The process working directory.
3. `$XDG_CONFIG_HOME/framecap.toml`, if `XDG_CONFIG_HOME` is set.

The first existing file wins. The global file now takes precedence over existing
local or XDG files; copy any settings you want to keep into it before launching.
Manual file edits require restarting the application. Successful shortcut
changes are saved immediately in the global file and restored on the next
launch. Invalid TOML, unknown fields, read or write errors are reported; an
invalid file is never overwritten. Read errors cause a fallback to built-in
settings. Individual invalid values can have their own
validation behavior, documented in the
[example file](../framecap.toml).

**The tracked example is a locally tuned configuration, not the built-in
defaults.** In particular it sets an initial target of 90 FPS, selects specific
hwmon devices and labels, and writes diagnostics to `/tmp/framecap.log`. Review
it before using it on a different machine. See the [installation procedure](INSTALL.md)
for using the example.

## Settings

| Section | Purpose |
| --- | --- |
| `[framecap]` | Metrics window, GPU polling interval and BC-250 memory-temperature polling. |
| `[key-bindings]` | Enable and assign distinct F5-F8 shortcuts. |
| `[sensors]` | Optional hwmon device/label selectors or path overrides for temperatures, power, and fan speed. |
| `[overlay]` | Visibility, refresh, smoothing, text scaling, FPS color, and background. |
| `[auto-mode]` | Algorithm order. |
| `[auto-mode.algorithms.<name>]` | Metric, target, absolute hysteresis, collection/confirmation window, steps, and FPS bounds; `gpu_threshold` also owns the GPU/jitter thresholds. |
| `[manual-mode]` | Startup mode, selected AUTO algorithm, remembered FPS cap, manual FPS step and bounds. |
| `[diagnostics]` | Log path and append behavior. |

The [example file](../framecap.toml) documents individual keys and built-in
defaults. The three automatic algorithms and their complete rules are described
in [automatic regulation](AUTO-MODE.md).

`[framecap].memory_poll_rate_ms` sets the GDDR6 hotspot polling interval
(default 5000 ms). Any value below 2000 ms disables this temperature reading;
zero disables it without a warning. GTT and RAM readings remain enabled.
Polling reads the `bc250_memory` hwmon driver without `pkexec` or Polkit.
See [overlay](OVERLAY.md) for the prerequisites, SMU concurrency warning,
failure behavior, and shutdown implications.

Sensor selectors match the hwmon device's `name` and sensor label rather than
the volatile `hwmonN` index. For example:

```toml
[sensors]
fan_rpm = { device = "nct6686", label = "Pump Fan" }
```

Selectors are supported for `cpu_temp`, `gpu_vrm_temp`, `cpu_vrm_temp`,
`gpu_edge_temp`, `gpu_power`, `cpu_power`, `ppt_power`, and `fan_rpm`.
Names and labels can be inspected with `sensors` or under `/sys/class/hwmon/`.
A selector must match exactly one usable sensor; missing or ambiguous matches
are logged and the reading is unavailable, without falling back to another
sensor. Selected channels use `_input` when available, otherwise `_average`.
The selection is resolved at startup, so restart the application after hardware
or driver changes.

Existing `*_path` overrides remain supported, but hwmon indices may change
after a reboot or hardware changes. A selector and a path for the same sensor
conflict and disable that reading with a warning. Without either override,
Framecap discovers sensors using built-in labels; notably, the default fan
label is `CPU Fan`. An invalid path logs a warning before falling back to that
discovery. Power channel paths can omit `_input`. Missing or stale readings do
not produce synthetic sensor values. The `fan_rpm` algorithm uses the fan
sensor reading and stops adjusting when its AUTO window has no valid data.

Legacy settings moved from `[framecap]` are read for migration without
rewriting the file. In particular, `[framecap].target_fps` is deprecated in
favor of `[manual-mode].startup_fps`; when both are present the latter wins
and a warning is emitted. `startup_fps` defaults to 60 FPS and valid positive values range from 1 to
240 FPS. Without `startup_mode`, zero starts OFF and a positive value starts
MANUAL, preserving existing configurations. `startup_mode` may be `"off"`,
`"manual"`, or `"auto"`; AUTO also requires `startup_algorithm` to name an
enabled algorithm. MANUAL and AUTO require a positive `startup_fps`. An invalid
mode or algorithm falls back to the legacy startup behavior and is logged.
Invalid FPS values disable the cap. The old
single-rule auto-mode keys `jitter_high`, `jitter_low`, and `gpu_saturation`
are obsolete and do not control the current algorithms.

## Automatic mode migration

The current AUTO configuration is not compatible with the earlier
per-algorithm timing and percentage hysteresis keys. In every
`[auto-mode.algorithms.<name>]` section:

1. Remove `hysteresis_percent`, `auto_increase_interval_ms`,
   `auto_decrease_interval_ms`, and `measurement_expiry_ms`.
2. Set `hysteresis` in the algorithm's units and `auto_window_ms` in
   milliseconds. Built-in values are 5 FPS for `gpu_threshold`, 5 °C for
   `vrm_temp`, 50 RPM for `fan_rpm`, and 4000 ms for each window.
3. Keep or retune the existing target, FPS steps, bounds, and GPU/jitter
   thresholds as needed. Percentage and absolute hysteresis are not generally
   equivalent; in particular, the FPS cap can change.

The same window governs collection, direction confirmation, and minimum
spacing between effective adjustments. Removing the old keys is essential:
unknown keys reject the **entire** configuration file, not just that
algorithm, and Framecap falls back to built-in settings. Update the actual
file found by the lookup order above, then restart the application.
See [automatic regulation](AUTO-MODE.md) for timing and decision details.

## Keyboard shortcuts

The defaults are a short F5 press to cycle manual, configured automatic algorithms, OFF,
then manual; F6/F7 decrease/increase the manual cap, JIT hysteresis by 1 FPS,
or the VRM/FAN target by its configured hysteresis;
a short F8 press cycles the overlay through full, FPS only, and off.
Holding F8 for one second cycles through `ascii`, `modern` and `helix` without
waiting for release or changing the display mode. The physical F5 key's long
press is reserved for the frametime capture described below. Its configured
short action happens on release; if F5 is assigned to overlay toggle, the long
renderer-switch action is unavailable on that key. Other keys retain their
configured actions. F5 saves the mode and AUTO algorithm;
F6/F7 save the manual cap or the changed AUTO parameter; F8 saves the overlay
mode or renderer. Automatic cap adjustments themselves are not saved.
Limiter OFF remembers the previous cap. Writes preserve comments and existing
symbolic links and are synchronized between Framecap processes. If saving fails,
the running change remains active but cannot be restored until that setting is
changed and saved successfully again.
All shortcuts are configurable and must be distinct. They require X11/XWayland
and may prevent the application from receiving those keys. Native Wayland has no
global shortcut support in the current backend.
Only F5–F8 are supported. Existing configurations with explicit F9–F12 bindings
must be updated; invalid or duplicate bindings disable all global shortcuts
and produce a diagnostic message.

### Frametime capture

In a stable scene, hold the physical F5 key for one second. This starts one
process-wide capture for 30 seconds without changing the regulation mode.
It requires the X11/XWayland shortcut listener and works with the limiter on or
off. Requests during a capture or its export are ignored. The capture buffers
up to 65536 completed presentations in memory and performs no capture-file
writes until the 30 seconds have elapsed. Presentations still in progress at
the end are excluded. The CSV metadata reports dropped samples, skipped presentation or acquisition
entries and an approximate count of incomplete presentation calls at the boundary. Closing the game
before export completes can lose the capture.
The timestamps and buffer operations also add some CPU overhead to the measured
path; they do not capture physical display scanout.

The CSV is written asynchronously to `$XDG_STATE_HOME/framecap/`, or
`~/.local/state/framecap/` if unset, and its path or an export error is logged
in the normal Framecap diagnostic log. If neither environment variable is set,
the directory is `/tmp/framecap-<uid>/`. Other application and Framecap logging
are not disabled during capture. Avoid sharing the CSV without reviewing it.
While capture is active, the frametime value in the full overlay is red. The
FPS-only and hidden overlays do not display a capture indicator.

Each capture also produces `trace-capture-*.csv` alongside its matching
`capture-*.csv`, with the same filename suffix and capture start time. It
records CPU entry/return intervals for application presentation and image
acquisition, queue submissions (`vkQueueSubmit`, `vkQueueSubmit2` and its KHR
alias), fence and semaphore waits, and queue/device idle waits. Framecap's
own GPU marker and overlay submissions bypass these interceptors. The trace
does not include other Vulkan calls, fence-status polling, operating-system
waits, CPU work or actual display times.

Trace `start_ns` and `end_ns` use the same capture-relative monotonic clock as
the frame CSV. `thread_id` is the Linux thread ID; compare it before interpreting
an interval between calls as time spent on one thread. `object` identifies the
queue for presentations, submissions and queue-idle waits, and the device for
acquisitions, fence/semaphore waits and device-idle waits. `related` is the
swapchain for acquisition, the submission fence, or the first fence/semaphore
for waits; zero means absent. `count` is the number of submitted batches or
waited objects, and `timeout_ns` is the requested timeout when applicable.
`result` is the raw Vulkan result. The first object alone does not describe
multi-object waits or prove which object caused an unblock.

Trace storage is limited to 262144 completed events; `dropped` counts events
lost to buffer contention or capacity, `skipped_entries` counts admission
contention, and `incomplete` approximately counts calls crossing the capture
boundary. Rows may be out of order because calls finish on different threads;
sort by timestamps. These interceptors remain in the dispatch chain outside
capture because applications can cache Vulkan function pointers; no trace
events are recorded then, but dispatch overhead remains. Tracing itself also
adds overhead, so compare capture-on and capture-off behavior.

Each row covers a completed CPU presentation call; queue and first swapchain
identifiers distinguish streams. Timestamp columns ending in `_ns` are
nanoseconds relative to the monotonic capture start; negative values are
possible for previous deadlines and telemetry. Empty fields mean unavailable
or not applicable. The header records the start time in Unix nanoseconds, the
duration, sample count and dropped count. It includes the previous and returned
deadlines and targets, scheduler entry and wake times, work-lock wait,
metrics/AUTO/overlay CPU durations, overlay refresh and AUTO evaluation counts,
GPU load and its sampling time, submission/present timings and Vulkan results.
`initial_present_mode` is the raw Vulkan mode requested when the first
swapchain was created (0=IMMEDIATE, 1=MAILBOX, 2=FIFO, 3=FIFO_RELAXED), not
necessarily a later per-present override. `overlay_mode` is
0=off, 1=FPS only, 2=full; `auto_mode` is -1=off, 0=manual, or a one-based
AUTO algorithm index. A blank deadline means no sleep was requested; compare
the previous deadline and target to the scheduler time and target to distinguish
initialization, target changes, and late presentations. Overlay command
recording, AUTO evaluation, and overlay submission occur before the pre-present
limiter check. Presentation metrics and AUTO histories are updated after a
successful downstream call, using its start timestamp. `auto_ns` includes both
the evaluation and history update; `metrics_ns` records the post-present metrics
update. The pre-present lock waits are measured separately by
`work_lock_start_ns`/`work_lock_acquired_ns` and
`metrics_lock_start_ns`/`metrics_lock_acquired_ns`.
Failed presentations reset the queue's schedule.
Overlay refresh counts refer to the metrics refreshed in that presentation
call, which can be used for command recording starting on the next presentation.
Group rows by queue and
compare consecutive `present_start_ns` or `entry_ns` values; compare wake to
deadline and the work durations to find where a long interval arises.
These are CPU timestamps, not GPU completion or physical scanout times.

The capture also includes `image_index`, `acquire_start_ns`, `acquire_end_ns`
and `acquire_result` for a successful or suboptimal acquisition matched by
device, swapchain and image index to the first swapchain in the present call.
They are empty when no matching acquisition was captured, including acquisitions
started before capture. The metadata counts unsuccessful acquisitions as
`acquisition_failed` and acquisition events lost to contention or capacity as
`acquisition_dropped`; failed acquisitions have no image to associate with a
present row. Acquisitions are intercepted in normal operation as well, so
compare capture-on and capture-off behavior when assessing overhead.
`present_mode_override`, `present_id_khr` and `present_id_google` are recorded
only when the application supplies those structures for the first swapchain.
The override is a requested mode, not proof of actual display behavior. The
IDs alone do not report whether an image was displayed or discarded.

During capture, a GPU timestamp marker is submitted before pacing and waits
on the application's present semaphores. The marker uses a relay semaphore
for the subsequent overlay submission or presentation. Its query is polled
without waiting for the GPU; `gpu_marker_status=ready` provides raw
`gpu_marker_ticks`, `gpu_marker_period_ns` and `gpu_marker_valid_bits`.
The difference between two ready markers on the same queue, adjusted for
timestamp wrap at the reported bit width and multiplied by the period, is
the GPU-clock interval between markers. They indicate when this queue reached
the marker after its waits, not the duration of all game rendering or physical
display time. If the game finished before the marker was submitted, the
timestamp reflects the later marker rather than the earlier completion.
`gpu_timestamp_ns` remains the CPU timestamp of the GPU load sample.

When supported, a separate worker samples calibrated GPU/CPU timestamps every
100 ms while capture is active. It does not explicitly wait for GPU work, and
the presentation thread never waits for calibration. Framecap uses the KHR or EXT calibrated-timestamps
extension if enabled by the application; otherwise it adds a supported extension
when creating the device, provided its instance dependencies and required
`DEVICE` and `CLOCK_MONOTONIC` domains are available. Calibration is disabled
for multi-device groups or when the extension, domains, command or worker are
unavailable. The added extension can affect device creation and the worker adds
some measurement overhead; compare capture-on and capture-off behavior.

`gpu_calibration_status` is `ready`, `waiting`, `busy`, `stale`, `error` or
`unavailable`. Only `ready` rows include calibration values: the GPU clock
sample `gpu_calibration_gpu_ticks`, its paired CPU time
`gpu_calibration_cpu_ns` relative to capture start,
`gpu_calibration_max_deviation_ns` from Vulkan, and the CPU clock bridge
`gpu_calibration_reference_ns` and
`gpu_calibration_reference_uncertainty_ns`. The bridge uncertainty bounds the
local `CLOCK_MONOTONIC`/`Instant` sampling interval, separately from Vulkan's
maximum deviation. Samples over 500 ms old are marked stale; the reference
time also allows their age at the presentation call to be estimated.

For a ready marker and calibration on the same device, an estimated CPU time
for the marker is the calibrated CPU time plus the signed GPU-tick difference
times `gpu_marker_period_ns`. Interpret that difference modulo
`2^gpu_marker_valid_bits`, and only when its wrap count and direction are
unambiguous; otherwise no reliable correlation can be inferred. Account for
both reported uncertainties, calibration age and possible clock drift. The
marker can execute after the calibration sample and even after the CPU present
call; calibration is periodic and does not provide an independent bound on
when the marker actually executed. These values do not prove that a frame was
displayed or measure the completion time of all game rendering.

`pending` means the result was not ready when the CSV was exported; other
status values identify unsupported or skipped markers and errors. Markers
are skipped for multi-device groups, protected queues, multiple swapchains
in one presentation, absent present semaphores, unsupported queue families,
or unavailable capture resources. The marker submission adds CPU and GPU
work only during capture; `gpu_marker_submit_start_ns` and
`gpu_marker_submit_end_ns` bound its CPU preparation and submission.

When an overlay command is submitted, capture also appends a timestamp command
after it in the same submission. The submission's fence is polled without
waiting. `gpu_overlay_marker_status`, `gpu_overlay_marker_ticks`,
`gpu_overlay_marker_period_ns` and `gpu_overlay_marker_valid_bits` describe
this second marker; `gpu_overlay_marker_prepare_start_ns` and
`gpu_overlay_marker_prepare_end_ns` bound its CPU preparation. `no_overlay`
means no overlay command was submitted. `unsupported`, `query_pending`,
`slot_disabled`, `resource_limit`, `error` and other skip statuses mean that
no timestamp is available; `pending` means it was not collected by export.

Both markers use the same device clock. Their difference describes elapsed GPU
time from the pre-overlay marker to the post-overlay marker, including intervening
queue work and scheduling; it is not necessarily the isolated overlay drawing
cost. The post-overlay timestamp can also be correlated with the CPU present
timestamp using the calibration above. It precedes the overlay submission's
semaphore signal and does not measure physical display or guarantee when the
image becomes visible. The added command, query and fence are capture
instrumentation and can perturb timings even though they require no separate
queue submission. Marker resources retired with old swapchains are kept until
device destruction; each marker type is limited to 1024 image slots per device.

## Diagnostics

The built-in log location is `$XDG_STATE_HOME/framecap/framecap.log`, or
`~/.local/state/framecap/framecap.log` when that variable is unset. An
explicit `[diagnostics].log_path`, including the one in the tracked example,
overrides it. If the file cannot be opened, diagnostics go to stderr.

`append_log = 0` truncates the log at startup; `1` appends. Steam/Proton can
load the layer in multiple processes, each of which can truncate the same file.
Use append mode when collecting multi-process traces. Logs include executable
and working-directory paths, selected environment and hardware context,
effective configuration, Vulkan initialization, and errors; review them before
sharing. To diagnose loader discovery, temporarily set
`VK_LOADER_DEBUG=error,warn,layer` for the application.
