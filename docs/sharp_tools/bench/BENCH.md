# Micro-benchmark: update cost and environment step (31 Oct 2026)

`piper_rl.scripts.bench_update_cost`, batch 256, 256×256 MLP, 300 timed updates. PC: Windows 10, 48 threads, Python 3.13.9, torch 2.14.0+cpu, idle (`bench_update_cost_pc.json`). Container: Linux, 2 cores, Python 3.13, torch 2.15.0+cpu (`bench_update_cost_linux.json`).

| | PC, 1 thread | PC, 2 threads | Linux, 1 thread |
|---|---:|---:|---:|
| SB3 SAC ms/update | 27.9 | 21.8 | 16.6 |
| sb3-contrib TQC | 48.1 | 35.3 | 30.1 (earlier run) |
| sb3-contrib CrossQ | 59.1 | 39.1 | 29.5 (earlier run) |
| SBX SAC (JAX) | 16.9 | 21.5 | 11.2 |
| SBX CrossQ | 30.1 | 28.9 | 11.8 (earlier run) |
| env step, 1 env (ms) | 17.2 | | 6.5 |
| env vec step, 4 subproc envs (ms) | 20.1 | | 20.6 (2 cores) |
| env vec step, 8 subproc envs (ms) | 21.0 | | – |

Model that reproduces the 10-minute Stage A rates on the PC: time per vec step ≈ 20 ms (4 envs) + G × ms/update, G = gradient steps per vec step.
SB3 G=4: 20+112=132 ms → 30 steps/s (measured 35). SB3 G=1: 48 ms → 83 (79). SBX G=1: 37 ms → 108 (99). SBX G=4: 88 ms → 45 (42).

Reads:
- SBX SAC is 1.65× SB3 SAC per update on the PC at 1 thread; more threads make SBX slower. Run every JAX learner at 1 thread. [Certain]
- On the PC the environment step is 17 ms, 2.6× the container's 6.5 ms, so at G=1 the environment is about half of the time. [Certain for the measurement; cause not identified — Windows build, Python 3.13, or CPU clock; worth a 10-minute profile, not tonight]
- Lever for Stage B [Guessing until Stage A finishes]: 8 envs with G=2 keeps the update-to-data ratio of 4 envs with G=1 and the model gives 21+34=55 ms per 8 steps ≈ 145 steps/s versus 99.
- sb3-contrib TQC and CrossQ cost 1.7–2.1× SAC per update on torch; they only win if they learn in proportionally fewer steps.
