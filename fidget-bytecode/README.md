`fidget-bytecode` implements a `u32` bytecode tape for math expressions.

The current packed format has no sidecar section. Tapes containing native
`ShellDistance` operations are rejected with `UnsupportedNativeShell` because
their `ShellTopology` sidecars cannot be represented in this old format.

It is typically used through the [`fidget`](https://crates.io/crates/fidget)
crate, which imports it under the `bytecode` namespace.

[![» Crate](https://badgen.net/crates/v/fidget-bytecode)](https://crates.io/crates/fidget-bytecode)
[![» Docs](https://badgen.net/badge/api/docs.rs/df3600)](https://docs.rs/fidget-bytecode/)
[![» CI](https://badgen.net/github/checks/mkeeter/fidget/main)](https://github.com/mkeeter/fidget/actions/)
[![» MPL-2.0](https://badgen.net/github/license/mkeeter/fidget)](../LICENSE.txt)

