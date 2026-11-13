# Working on MoEShift

MoEShift is the standalone repository `davide-troiani/moeshift`. Preserve history and existing user changes. Work only in this
repository; instructions from the former parent project do not apply.

Read [README.md](README.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

- Target macOS on Apple Silicon, with SSD expert streaming.
- Scope architecture-specific changes to the affected architectures. Changes to
  shared components require regression checks across other model families.
  Keep docs/supported-architectures.md accurate; distinguish available model
  support from behavior verified with actual models.
- The application starts a localhost HTTP server; inference parallelism is
  always one. Do not add alternate application modes or speculative decoding.
- Runtime tuning belongs in the JSON configuration. Document supported options
  and their practical effects; avoid adding another tuning mechanism.
- Prefer small changes, removal of redundancy and reuse of existing components.
  Ask before adding dependencies. Preserve validation and error handling.
- Match existing code style. Verify affected behavior with focused checks and
  state what was not tested. Do not claim benchmark improvements without data.
- Do not commit models, expert files, binaries, credentials or machine-specific
  configuration. Preserve licenses and copyright notices.
- Commit or push only when explicitly authorized. Use separate focused commits. Do not rewrite history
  or change repository visibility without explicit permission.
- Include documentation and required assets with the corresponding changes.

See [configuration](configs/README.md) and the runtime documents under
[docs/les](docs/les/). Remaining upstream documentation is reference material,
not a promise that those platforms or features are supported by MoEShift.
