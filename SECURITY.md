# Security Policy

## Supported Versions

Security fixes are applied to the latest release on `main`.

## Reporting a Vulnerability

Please **do not** report suspected security vulnerabilities in public GitHub issues.

Report vulnerabilities privately through **GitHub Security Advisories**:

* [Open a private GitHub Security Advisory](https://github.com/Luther22-source/overseer-mcp/security/advisories/new)

Include reproduction steps, the affected component (for example, path jail, symlink/junction handling, rollback, secret redaction, MCP server, or Monitor UI), and potential impact. You will receive an initial acknowledgment as soon as possible.

## API Key & State Storage

Overseer stores OpenRouter API keys and run state **outside** target repositories in your OS user data directory (`<data_dir>`):

* **Windows:** `%LOCALAPPDATA%\overseer\.env`
* **macOS / Linux:** `~/.local/share/overseer/.env` (written with `0600` file permissions on POSIX)
* **Custom location:** override via the `OVERSEER_DATA_DIR` environment variable

Never commit API keys or `<data_dir>/.env` into version control.

## Built-in Security Boundaries

* **No shell access:** Workers only have bounded file inspection and edit tools; shell execution stays with the user/IDE Boss.
* **Folder jail & symlink protection:** Reads are constrained to `project_root`, writes are restricted to each worker's assigned disjoint folder, and symlink/junction escapes are blocked.
* **Sensitive-file deny globs:** Access to `.env*`, SSH/cloud credentials, certificates, `.git`, `.github/workflows`, and IDE config directories is denied.
* **Secret redaction:** Detected bearer tokens, API keys, and credentialed URLs are scrubbed from worker outputs, tool errors, and logs.
