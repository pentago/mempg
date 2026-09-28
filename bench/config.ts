// Shared bench config: topic vocabulary, sentence templates, and helpers for
// generating mempg-shaped synthetic memories.

// 40 infra-flavored topics. Word pools deliberately overlap in flavor (like
// real corpora do - many memories mention clusters or deploys) so the
// generated pairs aren't trivially separable.
export const TOPICS: Array<{ subject: string; words: string[] }> = [
  { subject: "postgres upgrades", words: ["postgres", "pg_upgrade", "replication", "wal", "vacuum", "autovacuum", "primary", "standby", "failover", "checkpoint"] },
  { subject: "kubernetes clusters", words: ["kubernetes", "kubectl", "pod", "node", "kubelet", "deployment", "autoscaler", "eviction", "drain", "cordon"] },
  { subject: "docker builds", words: ["docker", "image", "layer", "buildkit", "registry", "push", "cache", "dockerfile", "containerd", "multi-stage"] },
  { subject: "nginx configs", words: ["nginx", "upstream", "proxy_pass", "keepalive", "worker", "location", "vhost", "rate-limit", "buffer", "reload"] },
  { subject: "systemd units", words: ["systemd", "unit", "journalctl", "daemon-reload", "service", "timer", "socket-activation", "cgroup", "slices", "override"] },
  { subject: "backup jobs", words: ["backup", "restic", "snapshot", "prune", "retention", "restore", "offsite", "borg", "dedup", "verify"] },
  { subject: "git workflows", words: ["git", "rebase", "merge", "cherry-pick", "branch", "hook", "submodule", "bisect", "stash", "reflog"] },
  { subject: "ci pipelines", words: ["pipeline", "runner", "artifact", "cache", "trigger", "matrix", "stage", "pipeline-failure", "concurrency", "webhook"] },
  { subject: "dns and domains", words: ["dns", "zone", "record", "resolver", "ttl", "propagation", "nsupdate", "delegation", "glue", "soa"] },
  { subject: "tls certificates", words: ["tls", "certificate", "acme", "letsencrypt", "renewal", "chain", "ocsp", "sni", "pem", "keystore"] },
  { subject: "monitoring stacks", words: ["prometheus", "exporter", "scrape", "alert", "dashboards", "grafana", "metric", "histogram", "alertmanager", "silence"] },
  { subject: "log shipping", words: ["logs", "vector", "fluentd", "shipper", "index", "retention", "rotation", "syslog", "parse", "pipeline-errors"] },
  { subject: "shell tooling", words: ["shell", "zsh", "bash", "script", "alias", "trap", "quoting", "subshell", "getopts", "shebang"] },
  { subject: "nvim workflows", words: ["neovim", "lsp", "telescope", "keymap", "plugin", "treesitter", "buffer", "lazy-loading", "runtimepath", "lua"] },
  { subject: "linux networking", words: ["networking", "interface", "firewall", "nftables", "route", "masquerade", "bridge", "vlan", "mtu", "ports"] },
  { subject: "wireguard tunnels", words: ["wireguard", "tunnel", "peer", "handshake", "allowed-ips", "endpoint", "mtu-clamp", "keypair", "roaming", "mesh"] },
  { subject: "disk and lvm", words: ["disk", "lvm", "partition", "luks", "filesystem", "mount", "trim", "raid", "zfs-pool", "scrub"] },
  { subject: "boot and firmware", words: ["boot", "systemd-boot", "efi", "initramfs", "kernel-args", "microcode", "tpm", "rollback-entry", "bootloader", "secureboot"] },
  { subject: "yubikey auth", words: ["yubikey", "fido2", "pin", "touch", "challenge", "slot", "gpg-card", "touch-prompt", "pam", "otp"] },
  { subject: "worktrees and repos", words: ["worktree", "repo", "clone", "fetch", "shallow", "bundle", "mirror", "gc", "packed-refs", "alternates"] },
  { subject: "redis caches", words: ["redis", "valkey", "eviction", "ttl", "pubsub", "replica", "sentinel", "persistence", "aof", "latency-spikes"] },
  { subject: "message queues", words: ["queue", "rabbitmq", "consumer", "ack", "dead-letter", "backpressure", "prefetch", "broker", "lag", "routing-key"] },
  { subject: "rust builds", words: ["rust", "cargo", "crate", "workspace", "feature-flags", "cross-compile", "toolchain", "msrv", "lockfile", "bench-suite"] },
  { subject: "python environments", words: ["python", "venv", "pip", "uv", "dependency-resolution", "wheels", "pyproject", "lock", "isolation", "conda"] },
  { subject: "bun and node runtimes", words: ["bun", "node", "runtime", "transpiler", "bunfig", "tsconfig", "module-resolution", "npm", "bundle-size", "startup-time"] },
  { subject: "typescript configs", words: ["typescript", "strict-mode", "decorators", "generics", "type-guard", "narrowing", "declaration", "paths", "project-refs", "eslint-rules"] },
  { subject: "api design", words: ["api", "endpoint", "pagination", "versioning", "idempotency", "rate-limits", "schema-contract", "backcompat", "deprecation", "openapi"] },
  { subject: "auth flows", words: ["oauth", "token", "refresh", "pkce", "session", "jwt", "scope", "redirect", "client-id", "token-expiry"] },
  { subject: "db migrations", words: ["migration", "schema-drift", "backfill", "add-column", "constraint", "lock-timeout", "expand-contract", "rollback-plan", "idempotent-migration", "seed-data"] },
  { subject: "release process", words: ["release", "changelog", "version-bump", "tag", "semver", "publish", "prerelease", "stable-branch", "hotfix", "release-notes"] },
  { subject: "terraform stacks", words: ["terraform", "state", "plan", "apply", "drift-detection", "module", "backend", "locking", "workspace-file", "providers"] },
  { subject: "container hardening", words: ["hardening", "read-only-fs", "cap-drop", "non-root", "seccomp", "localhost-only", "image-scan", "sbom", "pinned-digest", "surface-area"] },
  { subject: "latency debugging", words: ["latency", "flamegraph", "profiler", "hot-path", "p95", "tail", "contention", "syscall", "sampling", "cache-miss"] },
  { subject: "memory leaks", words: ["leak", "heap", "retained", "snapshot-diff", "gc-pressure", "growing", "unbounded", "listener", "closure-capture", "watchdog"] },
  { subject: "ssh and remotes", words: ["ssh", "authorized-keys", "agent", "port-forward", "jump-host", "config-alias", "controlmaster", "known-hosts", "proxyjump", "tunnel-port"] },
  { subject: "cron and timers", words: ["cron", "crontab", "schedule", "anacron", "flock", "stagger", "systemd-timer", "missed-run", "catch-up", "randomized-delay"] },
  { subject: "text processing", words: ["regex", "ripgrep", "awk", "sed", "parse-log", "encoding", "multiline", "delimiter", "unicode", "sort-stability"] },
  { subject: "window managers", words: ["wayland", "niri", "compositor", "swaync", "layer-shell", "keybind", "workspace-rule", "output", "gestures", "backlight"] },
  { subject: "laptop power", words: ["battery", "power-save", "cpufreq", "governor", "suspend", "hibernate", "runtime-pm", "tuned", "thermal", "charge-threshold"] },
  { subject: "gaming setup", words: ["proton", "steam", "wineprefix", "dxvk", "gamescope", "shader-cache", "controller", "vram", "gamemode", "compatdata"] },
];

// Cross-topic filler words - the corpus noise that makes ranking non-trivial.
export const FILLERS = [
  "team", "meeting", "notes", "decision", "follow-up", "context", "remember",
  "temporary", "workaround", "investigation", "observations", "behavior",
  "conclusion", "documented", "yesterday", "recurring", "question", "detail",
  "summary", "reminder", "standing", "constraint", "background", "history",
];

// Sentence templates; {s} = topic word, {f} = filler.
export const TEMPLATES = [
  "{s} must be {f} checked before the {s} window closes.",
  "Decision: the {s} setup stays {f} unchanged until the next {s} review.",
  "Debugging note: {s} failures were {f} traced back to a {s} misconfiguration.",
  "Remember that {s} behavior differs {f} between environments when {s} is involved.",
  "Environment fact: this project's {s} runs {f} on the {s} layer.",
  "Constraint: {s} cannot run {f} without the {s} prerequisites being met.",
  "The {s} pipeline was {f} reworked; {s} now runs before validation.",
  "Workaround: {s} accepts {f} the fallback path when {s} rejects the first.",
];

export function makeSql(db: string): {
  hostname: string;
  port: number;
  username: string;
  password: string;
  database: string;
  ssl: "disable";
  max: number;
} {
  return {
    hostname: process.env.MEMPG_HOST || "localhost",
    port: Number(process.env.MEMPG_PORT) || 5432,
    username: process.env.MEMPG_USER || "mempguser",
    password: process.env.MEMPG_PASSWORD || "",
    database: db,
    ssl: "disable",
    max: 4,
  };
}

// Deterministic PRNG (mulberry32) so runs are reproducible and datasets with
// the same size + seed are comparable across strategies and time.
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const pick = <T>(rand: () => number, arr: readonly T[]): T => arr[Math.floor(rand() * arr.length)];
