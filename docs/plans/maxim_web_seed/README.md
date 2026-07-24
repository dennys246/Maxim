# maxim-web

The **public front door** for Maxim: the landing page (`pymaxim.bio`) + the docs
(`docs.pymaxim.bio`). A static site — **Astro + Starlight**, deployed on **Cloudflare Pages**.

## Where this sits in the ecosystem

| Repo | Concern |
|---|---|
| **pymaxim** | the engine — the Python framework + the Console backend (`maxim serve`) |
| **maxim-pulse** | the product — the React Console app + the Reachy shell |
| **maxim-web** (this) | the front door — landing + docs, *about* all of the above, owned by none |

The site is **presentation/content only.** It links to the code and the app; it does not
reimplement them, and it does not need to duplicate pymaxim's deep technical docs (those can
stay code-adjacent in `pymaxim/docs` — see "Docs source" below).

## Stack

- **Astro** (content-first, fast static output) + **Starlight** (the docs theme).
- Markdown/MDX content. React components can be dropped in where interactivity is wanted
  (consistent with maxim-pulse's stack), but default to static.
- Build → `dist/` → **Cloudflare Pages** (custom domains `pymaxim.bio` + `docs.pymaxim.bio`).

## Proposed structure

```
maxim-web/
├─ astro.config.mjs        # site: https://pymaxim.bio; starlight integration + nav
├─ package.json
├─ src/
│  ├─ pages/index.astro    # the landing (hero + links) at the apex
│  └─ content/docs/        # Starlight docs (getting-started, guides…) → docs.pymaxim.bio
├─ public/                 # favicon, og image, static assets
├─ LICENSE                 # Apache-2.0 (copy from pymaxim); optional CC-BY-4.0 for docs content
└─ README.md
```

## Domains

- `pymaxim.bio` → the landing (`src/pages/index.astro`).
- `docs.pymaxim.bio` → the Starlight docs (`src/content/docs/`).
- Both served by one Cloudflare Pages project (or two — a Pages project can host both with a
  base-path split, or run docs as a Starlight sub-site; decide when wiring Pages).

## Docs source (a decision to defer — don't block the bare version on it)

- **Simplest:** maxim-web owns all docs (homepage + everything).
- **Code-adjacent:** deep API/architecture reference stays in `pymaxim/docs` (updates with
  code); maxim-web is the landing + curated narrative guides that link into it.

For the **bare** first version: ship **homepage + a getting-started page** and link out to the
existing guides. Migrate/curate the real guides as a follow-up once the shell is live.

## Homepage hero (seed copy — refine, don't overclaim; the project values honesty)

- **Name:** Maxim
- **Tagline:** *A bio-inspired cognitive architecture for AI agents — embodied sensation,
  homeostatic drives, and brain-modeled memory that learns across sessions without fine-tuning.*
- **Install:** `pip install pymaxim`
- **Primary links:** GitHub (`github.com/dennys246/Maxim`) · PyPI (`pypi.org/project/pymaxim`)
  · Docs (`docs.pymaxim.bio`) · (later) the Reachy app on Hugging Face.
- **Voice:** honest and specific over hype. Maxim's differentiator is *cross-session learning
  without fine-tuning* + embodiment — say that plainly; don't inflate it.

## Deploy (Cloudflare Pages)

1. `pnpm build` (or `npm run build`) → `dist/`.
2. Cloudflare → Workers & Pages → create Pages project from this repo (build command
   `astro build`, output `dist`).
3. Custom domains → add `pymaxim.bio` (+ `docs.pymaxim.bio`) — Pages auto-creates DNS.

## Standards

See [AGENTS.md](AGENTS.md). In short: content-first, static, fast, accessible; link to the
code/app rather than duplicating; ship bare then grow; no secrets in the repo.
