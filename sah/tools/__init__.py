"""sah.tools — the model-callable capabilities, one module each.

A run's built-in tools are declared in this package and only here: a third-party extension
declares its own outside it, and the boot gate (`sah.ext.assemble`) holds both to the same
contract. Presence here is not an offer — what a run is actually given is gated in `sah.cli`,
and a capability may travel no further than a run grants it. The shape every tool must take,
the schema it derives, the declaration it carries and the boundary that judges it, is
`sah.contract`.

The catalog, one module per capability: `sah.tools.calc`, `sah.tools.clock`, `sah.tools.fs`,
`sah.tools.memory`, `sah.tools.recall`, `sah.tools.shell`, `sah.tools.spawn`, `sah.tools.verify`,
`sah.tools.web`.
"""
