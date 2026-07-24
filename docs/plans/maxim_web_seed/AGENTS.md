# AGENTS.md — maxim-web

Durable standards for the Maxim front-door site. Tool-agnostic; Claude Code also reads this.
Kept short on purpose — this repo is a static content site, not a codebase.

## What this repo is

The **public front door** for Maxim: the landing (`pymaxim.bio`) + docs (`docs.pymaxim.bio`).
**Astro + Starlight**, static, deployed on **Cloudflare Pages**. See [README.md](README.md) for
structure, domains, and the ecosystem map (pymaxim = engine, maxim-pulse = product, this = front
door).

## Cardinal rule: presentation + content only

This site **links to** the code and the app — it never reimplements them, and it doesn't need to
duplicate pymaxim's deep technical docs (those can live code-adjacent in `pymaxim/docs`). If a
page starts wanting real logic, it belongs in pymaxim or maxim-pulse, not here.

## Standards

- **Static-first.** Default to static Astro/markdown; reach for a React island only when a page
  genuinely needs interactivity. Keep the output fast and light.
- **Ship bare, then grow.** The first version is homepage + getting-started. Do not over-build a
  marketing site before the product needs it — the same "don't over-invest early" discipline the
  rest of the project follows.
- **Honest voice, no hype.** Maxim's pitch is specific — *cross-session learning without
  fine-tuning*, embodiment, bio-inspired memory. Say it plainly; don't inflate or overclaim.
  (Matches the project's brutal-honesty ethos.)
- **Accessible + responsive + theme-aware.** Light/dark, mobile, semantic HTML, alt text.
- **No secrets, ever.** No API keys, tokens, or private endpoints in the repo or the built site.
  It's a public static site.
- **Content is the product here.** Prose quality > visual gimmicks. Clear > clever.
- **Don't rename the ecosystem terms.** Maxim, pymaxim, maxim-pulse, the bio-system names
  (Hippocampus, NAc, EC, ATL, SCN) are load-bearing — use them consistently.

## Dev

```bash
pnpm install
pnpm dev        # local preview
pnpm build      # → dist/ (what Cloudflare Pages serves)
```

## Deploy

Cloudflare Pages from this repo (build `astro build`, output `dist`), custom domains
`pymaxim.bio` + `docs.pymaxim.bio`. See README § Deploy.

## Versioning / license

Apache-2.0 (match the ecosystem; copy pymaxim's LICENSE). Optionally CC-BY-4.0 for `docs/`
content if you want the writeups openly citable. No package versioning needed — it's a site;
deploys are continuous from `main`.
