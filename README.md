# interact-core

`interact-core` contains the provider-independent Pydantic wire contracts shared by Interact
clients and services. It is a standalone Python package with the `interact_core` import namespace.

The package includes the generated JSON Schemas under `interact_core/schema/`. These files are
release artifacts for consumers that cannot import Python; the Pydantic models in `src/interact_core`
remain their source of truth.

## Development

```bash
uv run pytest
uv build
```

The public `interact` repository can use a sibling checkout of this repository for development.
Released `interact` installations use the versioned `interact-core` dependency instead.

## License

MIT; see [LICENSE](LICENSE).
