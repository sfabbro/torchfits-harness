# Harness playbook

- id: subject
  desc: Install a torchfits wheel before benches or parity tests. pixi run dev installs this package with --no-deps. This repo does not compile torchfits or vendor CFITSIO.

- id: local-bench-env
  desc: Local bench tasks run in the default Pixi env. Comparators are already dependencies. Do not recreate a torchfits bench-all env or bench-gpu-install here.

- id: source-probes
  desc: Tests that read torchfits cpp_src, extern/cfitsio, or tests/cpp stay in the torchfits repo.

- id: canfar-wheel
  desc: CANFAR GPU benches clone this repo and pip-install a torchfits wheel. They do not vendor CFITSIO or call bench-install. The per-Python source matrix and the denoise session launcher were removed; the denoise example stays in torchfits.

- id: canfar-verify-subject
  desc: The CANFAR verify lane installs the released wheel and runs all of tests/, so contract tests that poke torchfits internals must track that release. Twenty tests still pinned the pre-1.3 tree (handle_cache_capacity, configure_cache no-ops, positional TableHDU headers, _C.open_and_read_headers, writable raw buffers) and made the lane red; align them with the released contract.
