"""
Paperboy News Scout
Architecture:
  1. Python: RSS fetch → hard filter → keyword cluster
  2. Claude Call 1: merge dupes, split incoherent clusters
  3. Claude Call 2: editorial assessment — Polygon audience, vibe-based
  4. Python: enforce source count rules → post to Slack

Run:   python polygon.py
Debug: python debug.py

Credentials: ~/.claude/.env
  ANTHROPIC_API_KEY=...
  POLYGON_SLACK_WEBHOOK=...
  POLYGON_SLACK_BOT_TOKEN=...
"""

import logging
import os
import re
import json
import time
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from typing import Optional, List, Dict, Tuple, Set
from pathlib import Path

import requests
import feedparser
import yaml
from dotenv import load_dotenv

# ── Bootstrap ──────────────────────────────────────────────────────────────────
load_dotenv(Path.home() / ".claude" / ".env", override=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("paperboy")

_DIR = Path(__file__).parent

# ── Config ─────────────────────────────────────────────────────────────────────
def _load_config() -> dict:
    with open(_DIR / "config" / "agent-config.yaml") as f:
        return yaml.safe_load(f)

def _load_feeds() -> Tuple[dict, dict]:
    with open(_DIR / "config" / "feeds.yaml") as f:
        data = yaml.safe_load(f)
    tier1 = {e["name"]: e["url"] for e in data.get("tier1", [])}
    tier2 = {e["name"]: e["url"] for e in data.get("tier2", [])}
    return tier1, tier2

_CFG            = _load_config()
TIER_1_SOURCES, TIER_2_SOURCES = _load_feeds()
ALL_SOURCES     = {**TIER_1_SOURCES, **TIER_2_SOURCES}
_TIER2_SET      = set(TIER_2_SOURCES.keys())

# Corporate half-weighting: outlets owned by the same parent count as 0.5 toward
# trending thresholds so one company can't manufacture a "Breaking Story" alone.
_HALF_WEIGHT_SOURCE = {
    src: parent
    for parent, srcs in _CFG.get("corporate_half_weight", {}).items()
    for src in srcs
}

_CLAUDE_CFG     = _CFG["claude"]
_SCORING        = _CFG["scoring"]
_SLACK_CFG      = _CFG["slack"]

CLAUDE_MODEL          = _CLAUDE_CFG["model"]
RELEVANCE_MIN         = _SCORING["mw_relevance_min"]
POLYGON_PICK_MIN      = _SCORING["legolas_pick_min_relevance"]
CACHE_HOURS           = _SCORING["article_cache_hours"]
RECENTLY_POSTED_HRS   = _SCORING["recently_posted_hours"]
LOOKBACK_MINS         = _SCORING["lookback_minutes"]
SLACK_CHANNEL_ID      = _SLACK_CFG["channel_id"]
POST_TO_SLACK         = _SLACK_CFG["post_enabled"]

def _require_env(*names: str) -> None:
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise SystemExit(
            f"\n❌ Missing credentials in ~/.claude/.env:\n"
            + "\n".join(f"  {n}=..." for n in missing)
        )

_require_env("ANTHROPIC_API_KEY", "POLYGON_SLACK_WEBHOOK", "POLYGON_SLACK_BOT_TOKEN")

ANTHROPIC_API_KEY  = os.environ["ANTHROPIC_API_KEY"]
SLACK_WEBHOOK      = os.environ["POLYGON_SLACK_WEBHOOK"]
SLACK_BOT_TOKEN    = os.environ["POLYGON_SLACK_BOT_TOKEN"]

LEARNINGS_PATH = _DIR / "data" / "learnings.json"
SEEN_PATH      = _DIR / "data" / "seen_guids.json"

# ── Seen GUID management (file-based) ──────────────────────────────────────────
def load_seen_guids() -> set:
    try:
        if SEEN_PATH.exists():
            data = json.loads(SEEN_PATH.read_text())
            # Prune entries older than 14 days
            cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
            pruned = {guid: ts for guid, ts in data.items() if ts > cutoff}
            return set(pruned.keys())
        return set()
    except Exception as e:
        log.warning(f"Could not load seen GUIDs: {e}")
        return set()

def save_seen_guids(existing: set, new_items: List[dict]):
    try:
        data = {}
        if SEEN_PATH.exists():
            data = json.loads(SEEN_PATH.read_text())
        now = datetime.now(timezone.utc).isoformat()
        for item in new_items:
            data[item["guid"]] = now
        cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
        data = {guid: ts for guid, ts in data.items() if ts > cutoff}
        SEEN_PATH.write_text(json.dumps(data, indent=2))
    except Exception as e:
        log.warning(f"Could not save seen GUIDs: {e}")

# ── Learnings ──────────────────────────────────────────────────────────────────
def load_learnings() -> dict:
    try:
        if LEARNINGS_PATH.exists():
            return json.loads(LEARNINGS_PATH.read_text())
    except Exception as e:
        log.warning(f"Could not load learnings: {e}")
    return {}

def save_learnings(learnings: dict):
    try:
        LEARNINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEARNINGS_PATH.write_text(json.dumps(learnings, indent=2, default=str))
    except Exception as e:
        log.warning(f"Could not save learnings: {e}")

# ── Article cache ───────────────────────────────────────────────────────────────
def load_article_cache(learnings: dict) -> List[dict]:
    return learnings.get("article_cache", [])

def update_article_cache(learnings: dict, items: List[dict]):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=CACHE_HOURS)
    existing = [
        i for i in learnings.get("article_cache", [])
        if datetime.fromisoformat(i["published_dt"]) > cutoff
    ]
    guids = {i["guid"] for i in existing}
    for item in items:
        if item["guid"] not in guids:
            cached = {k: v for k, v in item.items() if k != "published_dt"}
            cached["published_dt"] = item["published_dt"].isoformat()
            existing.append(cached)
    learnings["article_cache"] = existing[-400:]

def restore_cached_items(cached: List[dict]) -> List[dict]:
    restored = []
    cutoff = datetime.now(timezone.utc) - timedelta(hours=CACHE_HOURS)
    for item in cached:
        try:
            dt = datetime.fromisoformat(item["published_dt"])
            if dt > cutoff:
                r = dict(item)
                r["published_dt"]  = dt
                r["_from_cache"]   = True
                restored.append(r)
        except Exception:
            pass
    return restored

# ── Recently posted ────────────────────────────────────────────────────────────
def load_recently_posted(learnings: dict) -> List[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=RECENTLY_POSTED_HRS)
    return [
        s for s in learnings.get("recently_posted", [])
        if datetime.fromisoformat(s["posted_at"]) > cutoff
    ]

def record_posted_story(
    learnings: dict,
    headline: str,
    tag: str,
    urls: Optional[List[str]] = None,
    article_titles: Optional[List[str]] = None,
):
    posted = learnings.setdefault("recently_posted", [])
    posted.append({
        "headline":       headline,
        "tag":            tag.lower().strip(),
        "posted_at":      datetime.now(timezone.utc).isoformat(),
        "urls":           urls or [],
        "article_titles": article_titles or [],
    })
    cutoff = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    learnings["recently_posted"] = [s for s in posted if s["posted_at"] > cutoff]

# ── Claude API ─────────────────────────────────────────────────────────────────
def _call_claude(prompt: str, max_tokens: int = 1000) -> Optional[str]:
    headers = {
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    body = {
        "model":      CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "messages":   [{"role": "user", "content": prompt}],
    }
    for attempt in range(3):
        try:
            r = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers, json=body, timeout=120,
            )
            r.raise_for_status()
            return r.json()["content"][0]["text"]
        except requests.exceptions.Timeout:
            log.warning(f"Claude timeout (attempt {attempt + 1})")
            time.sleep(5 * (attempt + 1))
        except Exception as e:
            log.warning(f"Claude call failed (attempt {attempt + 1}): {e}")
            time.sleep(3)
    return None

# ── RSS fetch ──────────────────────────────────────────────────────────────────
def fetch_rss_items(sources: dict, lookback_mins: int) -> List[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=lookback_mins)
    items  = []
    for name, url in sources.items():
        try:
            # Use a browser UA — some publishers (IGN) block bot UAs with 403/501
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            if "reddit.com" in url:
                headers["User-Agent"] = "PolygonScout/1.0 (news aggregator)"
            r = requests.get(url, timeout=20, headers=headers)
            r.raise_for_status()
            feed = feedparser.parse(r.content)
            tier = 1 if name in TIER_1_SOURCES else 2
            for entry in feed.entries:
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if not pub:
                    continue
                pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
                if pub_dt < cutoff:
                    continue
                guid  = entry.get("id") or entry.get("link") or entry.get("title", "")
                title = entry.get("title", "").strip()
                url_  = entry.get("link", "")
                if not title or not guid:
                    continue
                items.append({
                    "guid":         guid,
                    "title":        title,
                    "url":          url_,
                    "source_name":  name,
                    "source_tier":  tier,
                    "published_dt": pub_dt,
                })
        except Exception as e:
            log.warning(f"Feed error [{name}]: {e}")
    log.info(f"Fetched {len(items)} items from {len(sources)} sources (last {lookback_mins}m)")
    return items

# ── Hard filter ────────────────────────────────────────────────────────────────
_SKIP_TITLE_RE = re.compile(
    r'\b(best\s+\d+|\d+\s+best|top\s+\d+|ranked|ranking|tier\s+list|listicle|'
    r'what\s+to\s+watch|leaving\s+(this|next)\s+month|streaming\s+calendar|'
    r'every\s+game\s+(coming|releasing)|complete\s+guide|all\s+the\s+details|'
    r'everything\s+you\s+need\s+to\s+know|explained|what\s+to\s+expect|'
    r'release\s+date.*details|how\s+to\s+watch|where\s+to\s+watch|'
    r'deals?\s+of\s+the\s+(day|week)|buying\s+guide|review\s+round\s*up|'
    r'where\s+to\s+(pre\s*order|buy)|save\s+\$\d+|(pre\s*order|pre\-order)s?\s+(now|at|available)|'
    r'up\s+for\s+pre\s*order|knocks?\s+\$\d+\s+off|price\s+drop|'
    r'drops?\s+to\s+a\s+(new\s+)?price|just\s+\$\d+\s+at|new\s+price\s+low|'
    r'father.s\s+day\s+gift|subscription\s+service)\b',
    re.IGNORECASE,
)

_SKIP_SOURCE_PATTERNS = re.compile(
    r'\b(reddit\.com|resetera\.com|famiboards\.com)\b', re.IGNORECASE
)

_FORUM_SOURCES = {"Reddit GamingLeaks", "Reddit MarvelStudios"}

def filter_items(items: List[dict]) -> List[dict]:
    out = []
    for item in items:
        title = item["title"]
        # Drop listicles and guide content
        if _SKIP_TITLE_RE.search(title):
            continue
        # Forum sources only pass if title looks like actual news (not a discussion thread)
        if item["source_name"] in _FORUM_SOURCES:
            # Skip if it's clearly a forum discussion, not a news item
            if re.search(r'\b(discussion|weekly|megathread|help|question|\?\s*$)\b', title, re.IGNORECASE):
                continue
        out.append(item)
    return out

# ── Python keyword clustering ──────────────────────────────────────────────────
def _keywords(text: str) -> Set[str]:
    text = text.lower()
    text = re.sub(r"['''\"]", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "that", "this", "these",
        "those", "it", "its", "their", "they", "he", "she", "we", "you",
        "new", "first", "get", "gets", "got", "one", "two", "about", "after",
        "into", "more", "than", "up", "out", "now", "how", "what", "when",
        "report", "says", "say", "said", "set", "making", "made", "according",
    }
    return {w for w in text.split() if len(w) > 3 and w not in stopwords}

def cluster_items(items: List[dict]) -> List[dict]:
    clusters: List[dict] = []
    for item in items:
        kw = _keywords(item["title"])
        best_cluster = None
        best_overlap = 0
        for c in clusters:
            overlap = len(kw & c["_keywords"])
            if overlap >= 3 and overlap > best_overlap:
                best_overlap = overlap
                best_cluster = c
        if best_cluster:
            best_cluster["items"].append(item)
            # Keywords frozen at seed — prevents snowball clustering on generic platform words
            best_cluster["sources"].add(item["source_name"])
            best_cluster["published_dts"].append(item["published_dt"])
            if item["source_tier"] == 1 and item["title"] > best_cluster["headline"]:
                best_cluster["headline"] = item["title"]
        else:
            clusters.append({
                "headline":      item["title"],
                "items":         [item],
                "_keywords":     kw,
                "sources":       {item["source_name"]},
                "published_dts": [item["published_dt"]],
            })

    clusters.sort(key=lambda c: max(c["published_dts"]), reverse=True)
    for i, c in enumerate(clusters, 1):
        c["id"] = i
    log.info(f"Formed {len(clusters)} clusters from {len(items)} items")
    return clusters

def filter_old_single_source_clusters(clusters: List[dict]) -> List[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=2)
    out = []
    for c in clusters:
        if len(c["sources"]) == 1 and max(c["published_dts"]) < cutoff:
            continue
        out.append(c)
    return out

# ── Claude Call 1: merge / dupe / split ────────────────────────────────────────
def claude_find_merges(clusters: list, recently_posted: List[dict]) -> Tuple[dict, set, set]:
    if not clusters:
        return {}, set(), set()

    cluster_lines = []
    for c in clusters:
        local_id = c.get("_local_id", c["id"])
        items_str = "\n".join(
            f"  [{('T1' if it['source_tier'] == 1 else 'T2')}] {it['source_name']}: {it['title']}"
            for it in sorted(c["items"], key=lambda x: x["published_dt"])
        )
        cluster_lines.append(f"[{local_id}] {c['headline']}\n{items_str}")

    posted_section = ""
    if recently_posted:
        lines = []
        for s in recently_posted[-40:]:
            age  = int((datetime.now(timezone.utc) - datetime.fromisoformat(s["posted_at"])).total_seconds() / 60)
            titles = "; ".join(s.get("article_titles", [])[:3])
            lines.append(f"  {age}m ago: \"{s['headline']}\"" + (f"\n    Articles: {titles}" if titles else ""))
        posted_section = "RECENTLY POSTED (last 24h):\n" + "\n".join(lines) + "\n\n"

    prompt = f"""You are reviewing news clusters for Polygon, a gaming and pop culture publication.

{posted_section}CLUSTERS:
{chr(10).join(cluster_lines)}

Tasks:

MERGE: If two or more clusters cover the same product, announcement, or news event — merge them, even if the articles approach it from different angles. A price reveal, a feature confirmation, and a "what it means" reaction piece about the same product are all the SAME story.
Format: MERGE: [id1] + [id2]
You may chain: MERGE: [id1] + [id2], then MERGE: [id1] + [id3] to pull a third cluster in.

DUPE: If a cluster covers the same product/announcement as something in RECENTLY POSTED, mark it as a duplicate. Same product launch = dupe even if the new articles add a new angle or quote.
Format: DUPE: [id]

SPLIT: If a cluster mixes articles from clearly different stories, mark it for splitting.
This is the most common error — look carefully for:
- Same franchise/IP but different news beats (e.g. a game port announcement + a lore controversy about that same franchise)
- One article about a game, another about merchandise/toys/collectibles inspired by that property
- One article about a creative decision, another about a business/sales milestone for the same series
- A gaming article grouped with an entertainment industry article that only shares a franchise name
Format: SPLIT: [id]

NONE: If nothing to do, write NONE.

Merge aggressively when it's the same product or event. Be conservative with dupes — only mark as dupe if the core news beat (not just the franchise) already ran."""

    result = _call_claude(prompt, max_tokens=_CLAUDE_CFG["call1_max_tokens"])
    if not result:
        return {}, set(), set()

    merges: Dict[int, List[int]] = {}
    dupes:  Set[int] = set()
    splits: Set[int] = set()

    for line in result.splitlines():
        line = line.strip()
        m = re.match(r'MERGE:\s*\[?(\d+)\]?\s*\+\s*\[?(\d+)\]?', line)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            merges.setdefault(a, [a]).append(b)
            continue
        m = re.match(r'DUPE:\s*\[?(\d+)\]?', line)
        if m:
            dupes.add(int(m.group(1)))
            continue
        m = re.match(r'SPLIT:\s*\[?(\d+)\]?', line)
        if m:
            splits.add(int(m.group(1)))

    return merges, dupes, splits

def apply_merges(clusters: list, merges: dict) -> list:
    if not merges:
        return clusters

    by_id = {c.get("_local_id", c["id"]): c for c in clusters}
    merged_into: Set[int] = set()

    for primary_id, group_ids in merges.items():
        if primary_id not in by_id:
            continue
        primary = by_id[primary_id]
        for other_id in group_ids:
            if other_id == primary_id or other_id not in by_id:
                continue
            other = by_id[other_id]
            primary["items"]        += other["items"]
            primary["sources"]      |= other["sources"]
            primary["published_dts"] += other["published_dts"]
            primary["_keywords"]    |= other["_keywords"]
            primary.setdefault("_merged_from", []).append(other_id)
            merged_into.add(other_id)

    merged = [c for c in clusters if c.get("_local_id", c["id"]) not in merged_into]
    return merged

# ── Claude Call 2: Polygon editorial assessment ────────────────────────────────
def claude_assess_clusters(clusters: list, learnings: dict) -> List[dict]:
    if not clusters:
        return []

    CHUNK_SIZE = 40
    if len(clusters) > CHUNK_SIZE:
        results = []
        for i in range(0, len(clusters), CHUNK_SIZE):
            chunk = clusters[i:i + CHUNK_SIZE]
            for j, c in enumerate(chunk, 1):
                c["_local_id"] = j
            log.info(f"  Assessing chunk {i // CHUNK_SIZE + 1} ({len(chunk)} clusters)...")
            results.extend(claude_assess_clusters(chunk, learnings))
        return results

    cluster_blocks = []
    for seq, c in enumerate(clusters, 1):
        n_t1 = sum(1 for i in c["items"] if i["source_tier"] == 1)
        n_t2 = sum(1 for i in c["items"] if i["source_tier"] == 2)
        sorted_items = sorted(c["items"], key=lambda x: x["published_dt"])
        articles = []
        for it in sorted_items:
            age  = int((datetime.now(timezone.utc) - it["published_dt"]).total_seconds() / 60)
            tier = "T1" if it["source_tier"] == 1 else "T2"
            cache = " [cached]" if it.get("_from_cache") else ""
            articles.append(f"  [{tier}] {it['source_name']:<22} {age:>3}m ago | {it['title']}{cache}")
        cluster_blocks.append(
            f"CLUSTER {seq} | T1: {n_t1}  T2: {n_t2}  Total: {n_t1 + n_t2}\n"
            f"Sources: {', '.join(c['sources'])}\n"
            + "\n".join(articles)
        )

    # Inject editorial notes from feedback if present
    editorial_notes = learnings.get("editorial_notes", [])
    editorial_section = ""
    if editorial_notes:
        editorial_section = (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "TEAM EDITORIAL PREFERENCES (learned from feedback):\n"
            + "\n".join(f"  • {note}" for note in editorial_notes)
            + "\n"
        )

    prompt = f"""You are the editorial brain of a news scout for Polygon — a gaming and pop culture publication for enthusiasts. Your readers are people who care deeply about video games, anime, comics, tabletop, and the genre side of film and TV.

Your default stance is SKEPTIC. Most clusters should be skipped. You are looking for stories that would make a Polygon reader stop scrolling.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY PRE-CHECK — for every cluster, before any tier except skip, answer YES to ALL:

  1. NAMED? Does the story concern a SPECIFIC named game, franchise, studio, or person?
     → "a new RPG", "an action game", "a streaming show" = NO → skip
     → "The Legend of Zelda", "FromSoftware", "Shigeru Miyamoto" = YES

  2. EVENT? Did a concrete, confirmed development actually happen?
     → "fans love it", "going viral", "trending" = NO → skip
     → "announced", "cancelled", "trailer dropped", "date confirmed", "studio closed" = YES
     → Rumours and leaks from forums (ResetEra, Reddit, Famiboards) are events ONLY if
       multiple sources are picking them up or the forum post has significant credibility signals

  3. NEW? Is this news that broke recently, not evergreen background?
     → Retrospectives, "X years later", anniversary pieces = NO → skip

  4. POLYGON? Would a Polygon reader actually want to know this?
     → Box office results with no gaming/genre hook = NO
     → Reality TV, celebrity personal life without gaming/genre angle = NO
     → "Fans react to" without a confirmed news event behind it = NO
     → Industry/awards news that isn't reader-facing = MAYBE (only if it directly affects games/shows readers care about)

If ANY fail → TIER: skip, RELEVANCE: 1-3.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

TIER rules:

- trending: 3+ T1 sources OR 4+ total sources covering this story. Big confirmed breaking news.

- proven_topic: Real breaking news about a subject Polygon readers follow closely. You must
  identify why this topic has a built-in Polygon audience. Valid: major gaming franchise
  (Zelda, Mario, Pokémon, GTA, FromSoftware, Final Fantasy, CoD, Halo, etc.), popular anime
  (Demon Slayer, Attack on Titan, Dragon Ball, One Piece, etc.), MCU/DC/Star Wars with
  a nerd-culture hook, major platform news (PlayStation, Xbox, Nintendo), D&D or major
  tabletop releases. The event must be concrete news — not a preview, retrospective, or
  "fans are excited."

- polygon_pick: Use VERY SPARINGLY. Stories that fail trending/proven_topic but are genuinely
  compelling to a Polygon reader — an indie game breakthrough, a surprise announcement, a
  cultural moment in gaming. Must clear all 4 pre-check questions. RELEVANCE ≥ {POLYGON_PICK_MIN}.
  If you're not genuinely excited about it, skip.

- skip: Everything else. When in doubt, skip.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Polygon CARES: video games (all platforms), gaming hardware, indie games, game adaptations to
film/TV, anime, comics/manga, MCU/DC/Star Wars (nerd-culture angle), D&D and major tabletop,
gaming industry news (studio closures, layoffs, acquisitions), speedrunning/gaming culture,
sci-fi and fantasy film/TV, Nintendo Direct / PlayStation State of Play / Xbox events.

Polygon does NOT care: box office (unless gaming-adjacent), reality TV, celebrity personal
life without gaming/genre angle, music tours, fashion, sports, financial/earnings reports
(unless gaming company), politics, cooking/lifestyle, pure entertainment news with no nerd
culture hook.

ALWAYS SKIP regardless of source count:
- "Best games of", tier lists, ranked lists, "top N"
- "What to play", guides, walkthroughs, explainers (not news)
- Stories about a game being "popular" or "beloved" without a specific new development
- Forum rumours with only one forum source and no corroboration
- "Fans react to", "players love", "community celebrates" without a confirmed event
- Minor DLC or cosmetic updates for live service games (skins, battle pass) unless it's major
- Celebrity gaming content unless it's genuinely significant (not just "streamed for 2 hours")
- Industry awards shows coverage (Game Awards predictions, BAFTA nominations list) unless a
  major surprise announcement happened at the event

POLYGON_RELEVANCE:
10 = Every Polygon reader needs this now (Nintendo Direct with major announcements, new mainline Zelda/Mario/Pokémon, GTA VI release date, major studio closure)
8-9 = Big story (major game announced, FromSoftware reveal, PlayStation/Xbox exclusive confirmed, popular anime new season confirmed)
6-7 = Solid Polygon story (DLC announced, game delay, casting for major adaptation, anime cancellation)
4-5 = Niche or minor — skip
1-3 = Pre-check failed or wrong audience

{editorial_section}{chr(10).join(cluster_blocks)}

For EACH cluster respond with EXACTLY this format:

CLUSTER [N]
TIER: trending/proven_topic/polygon_pick/skip
RELEVANCE: [1-10]
HEADLINE: [punchy headline — must name the specific game/show/franchise]
ANGLE: [one sentence — what specifically happened?]
TOPIC: [specific game/franchise/series name, or blank if skip]
NOTE: [one sentence — which pre-check passed/failed, or why posting/skipping]

Post threshold: RELEVANCE >= {RELEVANCE_MIN} and tier is not skip.
polygon_pick requires RELEVANCE >= {POLYGON_PICK_MIN}.
"""

    result = _call_claude(prompt, max_tokens=_CLAUDE_CFG["call2_max_tokens"])
    if not result:
        log.warning("Claude assessment returned nothing — using fallback")
        return [_fallback_assessment(c) for c in clusters]

    return _parse_assessments(result, len(clusters), clusters)


_FIELD_LABELS = {"TIER", "RELEVANCE", "HEADLINE", "ANGLE", "TOPIC", "NOTE", "CLUSTER"}

def _parse_assessments(text: str, expected: int, clusters: list) -> List[dict]:
    def get_field(block, field):
        m = re.search(rf'^{field}:\s*(.+)$', block, re.MULTILINE | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def clean(value: str) -> str:
        if not value:
            return ""
        first_token = value.split(":")[0].strip().upper()
        return "" if first_token in _FIELD_LABELS else value

    parts = re.split(r'\n?CLUSTER\s+(\d+)\n', text)
    cluster_texts = {}
    i = 1
    while i < len(parts) - 1:
        try:
            cluster_texts[int(parts[i])] = parts[i + 1]
        except (ValueError, IndexError):
            pass
        i += 2

    results = []
    for n in range(1, expected + 1):
        block = cluster_texts.get(n, "")
        if not block:
            log.warning(f"No assessment for cluster {n} — using fallback")
            results.append(_fallback_assessment(clusters[n - 1]))
            continue
        try:
            mw = int(re.search(r'\d+', get_field(block, "RELEVANCE") or "0").group())
        except (AttributeError, ValueError):
            mw = 0
        tier = get_field(block, "TIER").lower()
        if tier not in ("trending", "proven_topic", "polygon_pick"):
            tier = "skip"

        headline = clean(get_field(block, "HEADLINE")) or clusters[n - 1]["headline"]
        angle    = clean(get_field(block, "ANGLE"))
        topic    = clean(get_field(block, "TOPIC"))

        if not clean(get_field(block, "HEADLINE")) and tier != "skip":
            log.warning(f"Cluster {n}: blank/malformed HEADLINE — using raw cluster headline")

        results.append({
            "tier":       tier,
            "relevance":  mw,
            "headline":   headline,
            "angle":      angle,
            "topic":      topic,
            "note":       get_field(block, "NOTE"),
        })

    return results

def _fallback_assessment(cluster: dict) -> dict:
    n_pubs = len({i["source_name"] for i in cluster["items"]})
    return {
        "tier":      "trending" if n_pubs >= 3 else "skip",
        "relevance": 7 if n_pubs >= 3 else 3,
        "headline":  cluster["headline"],
        "angle":     "",
        "topic":     "",
        "note":      f"fallback — Claude unavailable ({n_pubs} publishers)",
    }

# Entertainment trade outlets — corroborate each other on non-gaming stories
_TRADE_SOURCES = {"Deadline", "Variety", "Hollywood Reporter"}

# Gaming press sources — get a relevance boost
_GAMING_SOURCES = {"IGN", "VGC", "Eurogamer", "Kotaku", "PC Gamer", "Ars Technica Games",
                   "GamesRadar", "GameSpot", "Nintendo Life", "Push Square",
                   "Siliconera", "Time Extension", "Gematsu", "Automaton"}

# ── Tier enforcement ───────────────────────────────────────────────────────────
def enforce_tier(story: dict, cluster: dict) -> str:
    tier      = story["tier"]
    relevance = story["relevance"]

    # Gaming press boost: +1 relevance if any gaming source is in the cluster
    if any(s in _GAMING_SOURCES for s in cluster["sources"]):
        relevance = min(relevance + 1, 10)
        log.info(f"Gaming boost applied → relevance={relevance}: {story['headline'][:60]}")

    t1_pubs        = set()
    weighted_t1    = 0.0
    weighted_total = 0.0
    for s in cluster["sources"]:
        w = 0.5 if s in _HALF_WEIGHT_SOURCE else 1.0
        weighted_total += w
        if s not in _TIER2_SET:
            t1_pubs.add(s)
            weighted_t1 += w

    # Validate trending: needs 2+ weighted T1 sources or 3+ weighted total.
    # Corporate half-weighting means two PMC mastheads = 1.0 weighted T1, not 2 —
    # so a pair of trades alone can't trigger trending without independent coverage.
    if tier == "trending" and weighted_t1 < 2 and weighted_total < 3:
        log.info(f"Demote trending→polygon_pick (T1={weighted_t1:g} wt, total={weighted_total:g} wt): {story['headline'][:60]}")
        tier = "polygon_pick"

    # Validate polygon_pick: needs at least 1 T1 source or 2+ total
    if tier == "polygon_pick":
        if relevance < POLYGON_PICK_MIN:
            log.info(f"Demote polygon_pick→skip (relevance={relevance}): {story['headline'][:60]}")
            tier = "skip"
        elif len(t1_pubs) == 0 and total < 2:
            log.info(f"Demote polygon_pick→skip (single T2-only source): {story['headline'][:60]}")
            tier = "skip"

    # Single-source proven_topic: T2-only sources need corroboration; T1 sources can post solo
    if tier == "proven_topic" and total == 1 and len(t1_pubs) == 0 and relevance < POLYGON_PICK_MIN:
        log.info(f"Demote single-source proven_topic→skip (T2-only, relevance={relevance}): {story['headline'][:60]}")
        tier = "skip"

    # Trades-only clusters: Deadline/Variety/Hollywood Reporter move in lockstep.
    # Without a gaming/nerd-culture source in the mix, require very high confidence.
    if all(s in _TRADE_SOURCES for s in cluster["sources"]) and relevance < 8:
        log.info(f"Demote trades-only cluster (rel={relevance}): {story['headline'][:60]}")
        tier = "skip"

    # Forum-only clusters: require 2+ sources unless Claude is highly confident
    forum_sources = {"Reddit GamingLeaks", "Reddit MarvelStudios"}
    if all(s in forum_sources for s in cluster["sources"]) and total < 2:
        log.info(f"Demote forum-only cluster (only {total} forum sources): {story['headline'][:60]}")
        tier = "skip"
    elif all(s in forum_sources for s in cluster["sources"]) and total < 3 and relevance < POLYGON_PICK_MIN:
        log.info(f"Demote forum-only cluster (low confidence, {total} sources, rel={relevance}): {story['headline'][:60]}")
        tier = "skip"

    # Global relevance gate
    if relevance < RELEVANCE_MIN:
        tier = "skip"

    return tier

# ── Feedback system ────────────────────────────────────────────────────────────
def _log_learning(learnings: dict, entry: dict):
    log_entries = learnings.setdefault("learnings_log", [])
    log_entries.append({**entry, "logged_at": datetime.now(timezone.utc).isoformat()})
    learnings["learnings_log"] = log_entries[-500:]

def _extract_tier_from_message(text: str) -> str:
    if "Breaking Story" in text or "📈" in text:
        return "trending"
    if "Polygon Topic" in text or "🎮" in text:
        return "proven_topic"
    if "Polygon Pick" in text or "⭐" in text:
        return "polygon_pick"
    return ""

def _interpret_reply_with_claude(reply_text: str, story_headline: str, story_topic: str) -> Optional[dict]:
    prompt = f"""You are reviewing editorial feedback on a news story posted by a Polygon news scout.

Story headline: {story_headline}
Story topic: {story_topic}
Editor reply: {reply_text}

Return a JSON object:
{{
  "action": "boost" | "penalize" | "note" | "ignore",
  "topic": "{story_topic}",
  "delta": 0.0,
  "reason": "one sentence",
  "story_type_note": "if feedback is about the TYPE of story, note it here, else null"
}}

- "boost": great pick, more like this (+0.5 to +1.5)
- "penalize": shouldn't have posted (-0.5 to -1.5)
- "note": about process, not quality (delta 0)
- "ignore": not relevant

Respond with valid JSON only."""

    result = _call_claude(prompt, max_tokens=200)
    if not result:
        return None
    try:
        clean = re.sub(r'```json|```', '', result).strip()
        return json.loads(clean)
    except Exception:
        return None

def process_feedback(learnings: dict):
    try:
        headers = {"Authorization": f"Bearer {SLACK_BOT_TOKEN}"}
        resp    = requests.get(
            "https://slack.com/api/conversations.history",
            headers=headers,
            params={"channel": SLACK_CHANNEL_ID, "limit": 50},
            timeout=15,
        )
        msgs         = resp.json().get("messages", [])
        processed_r  = set(learnings.get("processed_reactions", []))
        processed_re = set(learnings.get("processed_replies", []))

        for msg in msgs:
            ts       = msg.get("ts", "")
            text     = msg.get("text", "")
            msg_tier = _extract_tier_from_message(text)

            for reaction in msg.get("reactions", []):
                reaction_id = f"{ts}:{reaction['name']}"
                if reaction_id in processed_r:
                    continue
                if reaction["name"] in ("thumbsup", "+1"):
                    _log_learning(learnings, {"type": "emoji_reaction", "reaction": "👍", "story": text[:100], "tier": msg_tier, "delta": +1.0})
                elif reaction["name"] in ("thumbsdown", "-1"):
                    _log_learning(learnings, {"type": "emoji_reaction", "reaction": "👎", "story": text[:100], "tier": msg_tier, "delta": -1.0})
                processed_r.add(reaction_id)

            if msg.get("reply_count", 0) > 0 and ts not in processed_re:
                try:
                    replies_resp = requests.get(
                        "https://slack.com/api/conversations.replies",
                        headers=headers,
                        params={"channel": SLACK_CHANNEL_ID, "ts": ts},
                        timeout=15,
                    ).json()

                    story_headline = ""
                    story_topic    = ""
                    headline_match = re.search(r'\*(.*?)\*', text)
                    topic_match    = re.search(r'Topic:\s*([^\n_(]+)', text)
                    if headline_match:
                        story_headline = headline_match.group(1).strip()
                    if topic_match:
                        story_topic = topic_match.group(1).strip()

                    for reply in replies_resp.get("messages", [])[1:]:
                        reply_id  = f"{ts}:{reply.get('ts','')}"
                        if reply_id in processed_re:
                            continue
                        reply_text = reply.get("text", "").strip()
                        if not reply_text or len(reply_text) < 3:
                            continue
                        action = _interpret_reply_with_claude(reply_text, story_headline, story_topic)
                        if action and action.get("action") != "ignore":
                            _log_learning(learnings, {
                                "type":               "reply_comment",
                                "reply_text":         reply_text[:200],
                                "story_headline":     story_headline,
                                "story_topic":        story_topic,
                                "tier":               msg_tier,
                                "interpreted_action": action.get("action"),
                                "reason":             action.get("reason", ""),
                                "story_type_note":    action.get("story_type_note"),
                            })
                        processed_re.add(reply_id)
                    processed_re.add(ts)
                except Exception as e:
                    log.warning(f"Could not process replies for {ts}: {e}")

        learnings["processed_reactions"] = list(processed_r)
        learnings["processed_replies"]   = list(processed_re)
        log.info("Processed Slack feedback")
    except Exception as e:
        log.warning(f"Feedback processing failed: {e}")

def synthesize_editorial_notes(learnings: dict) -> None:
    log_entries = learnings.get("learnings_log", [])
    if not log_entries:
        return

    last_count   = learnings.get("_editorial_notes_log_count", 0)
    last_updated = learnings.get("editorial_notes_updated_at")

    if len(log_entries) == last_count:
        return

    if last_updated:
        age_hours = (datetime.now(timezone.utc) - datetime.fromisoformat(last_updated)).total_seconds() / 3600
        if age_hours < 6:
            return

    feedback_lines = []
    for entry in log_entries[-60:]:
        etype = entry.get("type", "")
        if etype == "emoji_reaction":
            tier_hint = entry.get("tier", "")
            label     = f"[{tier_hint}] " if tier_hint else ""
            feedback_lines.append(f"REACTION {entry.get('reaction','')} on {label}{entry.get('story','')[:120]}")
        elif etype == "reply_comment":
            tier_hint = entry.get("tier", "")
            label     = f"[{tier_hint}] " if tier_hint else ""
            note      = entry.get("story_type_note") or ""
            feedback_lines.append(
                f"REPLY ({entry.get('interpreted_action','')}) on {label}'{entry.get('story_headline','')}': \"{entry.get('reply_text','')[:150]}\""
                + (f" [type: {note}]" if note else "")
            )

    if not feedback_lines:
        return

    recently_posted = learnings.get("recently_posted", [])[-20:]
    posted_section  = ("Recent stories posted:\n" + "\n".join(f"  - {s['headline']}" for s in recently_posted) + "\n\n") if recently_posted else ""

    prompt = f"""You are the editorial director of a Polygon news scout.

{posted_section}FEEDBACK LOG:
{chr(10).join(feedback_lines)}

Synthesize 3–8 editorial notes for future story selection. Each note should:
- Be one specific sentence
- Distinguish "we don't want this topic at all" from "we want this topic but not this story type"
- Note what IS interesting vs what is not
- Only write where feedback is clear enough to act on

Output notes only — no bullets, no numbers, no preamble."""

    result = _call_claude(prompt, max_tokens=400)
    if not result:
        return

    notes = [line.strip() for line in result.strip().splitlines() if line.strip()]
    if notes:
        learnings["editorial_notes"]            = notes
        learnings["editorial_notes_updated_at"] = datetime.now(timezone.utc).isoformat()
        learnings["_editorial_notes_log_count"] = len(log_entries)
        log.info(f"Synthesized {len(notes)} editorial notes")

# ── Slack posting ──────────────────────────────────────────────────────────────
def post_to_slack(cluster: dict, assessment: dict, tier: str) -> bool:
    tier_labels = {
        "trending":     "📈 Breaking Story",
        "proven_topic": "🎮 Polygon Topic",
        "polygon_pick": "⭐ Polygon Pick",
    }
    tier_label = tier_labels.get(tier, "📈 Breaking Story")

    items         = cluster["items"]
    published_dts = cluster["published_dts"]
    oldest        = min(published_dts)
    mins_ago      = int((datetime.now(timezone.utc) - oldest).total_seconds() / 60)
    span          = int((max(published_dts) - oldest).total_seconds() / 60)

    sources_str = " · ".join(sorted(cluster["sources"]))
    span_str    = f"  |  Coverage window: {span} mins" if span > 5 else ""

    lines = [
        f"{tier_label} — {assessment['headline']}",
        f"Sources: {sources_str}",
        f"First Seen: {mins_ago} mins ago{span_str}",
    ]
    if assessment.get("angle"):
        lines.append(f"Angle: {assessment['angle']}")
    if assessment.get("topic"):
        lines.append(f"Topic: {assessment['topic']}")

    t1_items = sorted(
        [it for it in items if it["source_tier"] == 1 and not it.get("_from_cache")],
        key=lambda x: x["published_dt"],
    )
    display_items = t1_items or sorted(
        [it for it in items if not it.get("_from_cache")],
        key=lambda x: x["published_dt"],
    )
    article_lines = []
    for it in display_items[:4]:
        article_lines.append(f"  › [{it['source_name']}] {it['title'][:90]}")
    if article_lines:
        lines.append("In this cluster:")
        lines.extend(article_lines)

    best_url = ""
    for it in display_items:
        if it.get("url"):
            best_url = it["url"]
            break
    if best_url:
        lines.append(best_url)
    lines.append("—" * 44)

    try:
        r = requests.post(SLACK_WEBHOOK, json={"text": "\n".join(lines)}, timeout=15)
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning(f"Slack post failed: {e}")
        return False

# ── Main run ───────────────────────────────────────────────────────────────────
def run():
    log.info("=" * 60)
    log.info("Paperboy — starting run")
    log.info("=" * 60)

    learnings = load_learnings()

    # 1. Process feedback
    process_feedback(learnings)
    synthesize_editorial_notes(learnings)
    save_learnings(learnings)

    # 2. Load seen GUIDs
    seen_guids = load_seen_guids()

    # 3. Load recently posted for dupe suppression
    recently_posted = load_recently_posted(learnings)

    # 4. Fetch RSS + filter
    all_items = fetch_rss_items(ALL_SOURCES, LOOKBACK_MINS)
    new_items = [i for i in all_items if i["guid"] not in seen_guids]
    log.info(f"{len(new_items)} new items after seen filter")
    new_items = filter_items(new_items)

    cached_items    = load_article_cache(learnings)
    cached_restored = restore_cached_items(cached_items)
    update_article_cache(learnings, new_items)

    # 5. Cluster
    all_for_clustering = new_items + cached_restored
    clusters = cluster_items(all_for_clustering)
    clusters = [c for c in clusters if any(not i.get("_from_cache") for i in c["items"])]
    log.info(f"{len(clusters)} clusters with new content")
    clusters = filter_old_single_source_clusters(clusters)

    # 6a. Claude Call 1: merge / dupe / split
    log.info(f"Claude call 1: {len(clusters)} clusters, {len(recently_posted)} recently posted...")
    for i, c in enumerate(clusters, 1):
        c["_local_id"] = i
    merges, dupes, splits = claude_find_merges(clusters, recently_posted)
    clusters = apply_merges(clusters, merges)
    suppress = dupes | splits
    clusters_to_assess = [c for c in clusters if c.get("_local_id", c["id"]) not in suppress]
    if len(clusters) != len(clusters_to_assess):
        log.info(f"Suppressed {len(clusters) - len(clusters_to_assess)} clusters → {len(clusters_to_assess)} to assess")

    # 6b. Hard dupe gate — Python-level cross-run check before spending Claude tokens
    def _cross_run_dupe(cluster: dict) -> bool:
        cluster_urls  = {it["url"] for it in cluster["items"] if it.get("url")}
        cluster_words = set()
        for w in cluster["headline"].lower().split():
            if len(w) > 4: cluster_words.add(w)
        for it in cluster["items"]:
            for w in it["title"].lower().split():
                if len(w) > 4: cluster_words.add(w)
        for prev in recently_posted:
            if cluster_urls & set(prev.get("urls", [])):
                log.info(f"Cross-run dupe (URL match): {cluster['headline'][:60]}")
                return True
            prev_words = set()
            for w in prev["headline"].lower().split():
                if len(w) > 4: prev_words.add(w)
            for title in prev.get("article_titles", []):
                for w in title.lower().split():
                    if len(w) > 4: prev_words.add(w)
            if len(cluster_words & prev_words) >= 3:
                log.info(f"Cross-run dupe (word overlap): {cluster['headline'][:60]}")
                return True
        return False

    pre_dupe = len(clusters_to_assess)
    clusters_to_assess = [c for c in clusters_to_assess if not _cross_run_dupe(c)]
    if len(clusters_to_assess) < pre_dupe:
        log.info(f"Cross-run dupe gate: {pre_dupe - len(clusters_to_assess)} suppressed → {len(clusters_to_assess)} remain")

    # 6c. Claude Call 2: editorial assessment
    log.info(f"Claude call 2: assessing {len(clusters_to_assess)} clusters...")
    assessments = claude_assess_clusters(clusters_to_assess, learnings)

    # 7. Tier enforcement + post
    posted           = 0
    new_seen_items:  List[dict] = []
    posted_this_run: List[str]  = []

    def _is_mid_run_dupe(headline: str) -> bool:
        words = set(w for w in headline.lower().split() if len(w) > 4)
        for prev in posted_this_run:
            if len(words & set(w for w in prev.lower().split() if len(w) > 4)) >= 3:
                log.info(f"Mid-run dupe suppressed: '{headline[:60]}'")
                return True
        return False

    for cluster, assessment in zip(clusters_to_assess, assessments):
        tier = enforce_tier(assessment, cluster)

        if tier == "skip":
            log.info(f"⏭ Skip (rel={assessment['relevance']}, claude_tier={assessment['tier']}): {assessment['headline'][:60]}")
            continue

        if _is_mid_run_dupe(assessment.get("headline", "")):
            continue

        posted_this_run.append(assessment.get("headline", ""))
        for item in cluster["items"]:
            if not item.get("_from_cache"):
                new_seen_items.append(item)

        cluster_urls   = [it["url"]   for it in cluster["items"] if it.get("url")]
        cluster_titles = [it["title"] for it in cluster["items"] if it.get("title")]

        if POST_TO_SLACK:
            if post_to_slack(cluster, assessment, tier):
                posted += 1
                record_posted_story(learnings, assessment["headline"], assessment.get("topic", ""), cluster_urls, cluster_titles)
                log.info(f"✅ Posted [{tier}] rel={assessment['relevance']}: {assessment['headline'][:60]}")
        else:
            log.info(f"[DRY RUN] {tier} rel={assessment['relevance']}: {assessment['headline'][:70]}")
            posted += 1
            record_posted_story(learnings, assessment["headline"], assessment.get("topic", ""), cluster_urls, cluster_titles)

    # 8. Save seen GUIDs + learnings
    save_seen_guids(seen_guids, new_seen_items)
    save_learnings(learnings)

    log.info(f"Run complete — {posted} stories posted")
    return {
        "clusters_total":    len(clusters),
        "clusters_assessed": len(assessments),
        "clusters_posted":   posted,
        "new_items":         len(new_items),
    }


if __name__ == "__main__":
    output = run()
    print(f"\nClusters found:    {output['clusters_total']}")
    print(f"Clusters assessed: {output['clusters_assessed']}")
    print(f"Clusters posted:   {output['clusters_posted']}")
    print(f"New items seen:    {output['new_items']}")
