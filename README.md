# galaius-core

`galaius-core` contains the provider-independent Pydantic wire contracts shared by Galaius
clients and services. It is a standalone Python package with the `galaius_core` import namespace.

The package includes the generated JSON Schemas under `galaius_core/schema/`. These files are
release artifacts for consumers that cannot import Python; the Pydantic models in `src/galaius_core`
remain their source of truth.

## Development

```bash
uv run pytest
uv build
```

The public `galaius` repository can use a sibling checkout of this repository for development.
Released `galaius` installations use the versioned `galaius-core` dependency instead.

## License

MIT; see [LICENSE](LICENSE).
