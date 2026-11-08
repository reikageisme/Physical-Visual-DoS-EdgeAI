# CPU laptop smoke

This runner measures clean same-input A/A only. It does not run PGD, attack
replay or injected masks. The original smoke used two predetermined profiles.
See `prereg/PREREG_LAPTOP_SEG_SMOKE_20261107.md` and the immutable tagged lock.

Inputs/weights were extracted without changes from the previously verified
confirmation ZIP into a new results directory. PT files stay outside Git;
the package receipt and lock supply byte identities. Runtime differences from
the rented host are recorded rather than treated as bitwise model replay.

Before starting, require user idle declaration and AC power. Tag must resolve
to the reviewed source/lock commit. Do not run other assistant tasks while the
campaign is active. Parent waits on each own worker without progress reads;
output logs are inspected after campaign completion. Startup60s and between-
profile60s rests are part of the locked campaign. No repeat-until-pass loop.

The original smoke is CLOSED. Its code and lock are preserved at tag
`prereg-laptop-seg-smoke-20261107` (9cafdd8). Current source additionally supports
the separate, UNRUN A/A20 protocol; the old lock correctly rejects changed
current source. Do not run the old smoke command using current source.

Original command, for provenance only:

```powershell
python sponge-sim/studies/laptop_seg/run.py --lock sponge-sim/studies/laptop_seg/lock.json --run-dir sponge-sim/results/laptop_seg_20261107/smoke_01 --tag prereg-laptop-seg-smoke-20261107 --commit <tag-commit> --user-idle "user confirms AC and idle"
```

Do not run the example as a shortcut around a pending idle confirmation.
After completion, analyze each profile once with `analyze.py <profile-dir>`;
summary output uses exclusive creation. Aborted/timeout/failed runs remain.
Smoke with two images cannot validate a population CI, full-study MDE, an
attack slowdown or DoS. A positive smoke gate is feasibility only.

## Next controls-only campaign, not started

`PREREG_LAPTOP_AA20_20261107.md` and `lock_aa20.json` specify one P1 session,
20 clean images,22references/max2flags and both forward/total MDE<=1%.
Full control approval and its own tag are pending. Once approved and tagged,
the concrete command is the following, with the actual new tag commit:

```powershell
python sponge-sim/studies/laptop_seg/run.py --lock sponge-sim/studies/laptop_seg/lock_aa20.json --run-dir sponge-sim/results/laptop_seg_20261107/aa20_01 --tag prereg-laptop-aa20-20261107 --commit <new-tag-commit> --user-idle "user confirms AC and idle"
```

This is one fixed session, not repeat-until-pass. Current analysis writes
`controls_summary.json` for this protocol; smoke summaries/raw remain unchanged.

## HighQoS diagnostic, new condition after A/A20 FAIL

AA20 is now CLOSED/FAIL (3flags, MDE2.36%/3.09%); its exact source remains at
tag `prereg-laptop-aa20-20261107` (66f675d). Latest current source supports
`PREREG_LAPTOP_HIGHQOS_20261107.md` and locks `lock_highqos_smoke.json` /
`lock_highqos_full.json`. Do not use old AA20 locks with changed current source.

`windows_qos.py` changes only the current worker's execution-speed policy to
documented HighQoS, checks readback and leaves priority unchanged. There is no
global power or thermal change. New process CPU-time calls outside the wall
timing boundaries are recorded per trial/reference; `analyze_cpu.py` reports
the preregistered descriptive CPU/wall endpoints from raw. It cannot identify
CPU frequency or prove a scheduling cause. Counter overhead is disclosed.

At most one HighQoS smoke and one full20, under a new tag before either runs.
Smoke must pass complete/pin/API/AC/zero-reference-flags. Failure stops. If it
passes, the user's standing instruction authorizes the already locked full20;
report smoke first and proceed without another permission round. Full keeps
max2flags and both MDE<=1%. Failure stops, no retry or relaxed gate.

HighQoS smoke has now FAILED2/4flags; full was cancelled. A separately tagged
PDH diagnostic completed and found several off-CPU dominated stalls. It does
not qualify attack timing despite0/4referenceflags. Current source additionally
supports one own-worker Above-normal priority intervention, with readback each
image/end. Its protocol/locks are `PREREG_LAPTOP_ABOVENORMAL_20261107.md` and
`lock_abovenormal_{smoke,full}.json`. At most one smoke and one full if smoke
passes; no further priority sweep on failure. All older source is preserved by
its tags; old locks correctly reject changed current source.

Power record: configuredACBestPerformance was already selected; reapplying
that SAME selection succeeded but did not change effective-overlay readback.
Powercfg baseBalanced, maximumAC100%, boost/maxfrequency settings are saved.
No thermal limits or protection were disabled. Settings remain unchanged
through the locked smoke; later changes need a separate protocol/amendment.
Microsoft documents the configured mode as a user vote that other signals
may override: [PowerSetUserConfiguredACPowerMode](https://learn.microsoft.com/en-us/windows/win32/api/powrprof/nf-powrprof-powersetuserconfiguredacpowermode).
