"""
Polygon Scout — Debug Runner
Run: python debug.py

Fetches and clusters articles, runs both Claude calls, shows what would post.
Does NOT write to seen GUIDs or learnings by default.
Set POST_TO_SLACK_DBG = True to actually post.
"""

import re
from datetime import datetime, timezone

from paperboy import (
    ALL_SOURCES, TIER_1_SOURCES, TIER_2_SOURCES,
    RELEVANCE_MIN, POLYGON_PICK_MIN,
    load_learnings, save_learnings,
    load_seen_guids,
    fetch_rss_items, filter_items,
    load_article_cache, restore_cached_items,
    cluster_items, load_recently_posted,
    claude_find_merges, apply_merges,
    claude_assess_clusters,
    enforce_tier, post_to_slack, record_posted_story,
)

# ── Debug settings ──────────────────────────────────────────────────────────────
IGNORE_SEEN       = True    # True = reprocess all items regardless of seen file
IGNORE_RECENT     = False   # True = skip dupe suppression (repush without blocking past posts)
LOOKBACK_MINS_DBG = 120
POST_TO_SLACK_DBG = True


def probe_feeds():
    import requests, feedparser
    print(f"\n{'═' * 60}")
    print("FEED PROBE")
    print("═" * 60)
    ok, broken = [], []
    for name, url in ALL_SOURCES.items():
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            if "reddit.com" in url:
                headers["User-Agent"] = "PolygonScout/1.0 (news aggregator)"
            r = requests.get(url, timeout=15, headers=headers)
            r.raise_for_status()
            feed  = feedparser.parse(r.content)
            count = len(feed.entries)
            first = feed.entries[0].title[:50] if feed.entries else "(empty)"
            tier  = "T1" if name in TIER_1_SOURCES else "T2"
            ok.append(name)
            print(f"  ✅ [{tier}] {name:<30} {count:>3} entries  │ '{first}'")
        except Exception as e:
            broken.append((name, url, str(e)[:60]))
            tier = "T1" if name in TIER_1_SOURCES else "T2"
            print(f"  ❌ [{tier}] {name:<30} {str(e)[:60]}")
    print(f"\n  {len(ok)} working, {len(broken)} broken")
    if broken:
        print("\n  ── Broken sources ──")
        for name, url, err in broken:
            print(f"  {name:<30} {url}")


def run_debug():
    print("█" * 60)
    print("  PAPERBOY DEBUG RUN")
    print("█" * 60)

    learnings = load_learnings()

    # ── Feed probe ─────────────────────────────────────────────────
    probe_feeds()

    # ── Seen ───────────────────────────────────────────────────────
    seen_guids = load_seen_guids()
    print(f"\n{'═' * 60}")
    print("SEEN")
    print("═" * 60)
    print(f"  {len(seen_guids)} GUIDs in seen file")
    if IGNORE_SEEN:
        print("  ⚠️  IGNORE_SEEN=True — seen file is read-only this run")

    # ── Fetch ──────────────────────────────────────────────────────
    print(f"\n{'═' * 60}")
    print(f"FETCHING RSS (last {LOOKBACK_MINS_DBG} mins)")
    print("═" * 60)

    all_items = fetch_rss_items(ALL_SOURCES, LOOKBACK_MINS_DBG)
    if IGNORE_SEEN:
        new_items = all_items
        print(f"\n  ⚠️  IGNORE_SEEN=True — reprocessing all {len(new_items)} items")
    else:
        new_items = [i for i in all_items if i["guid"] not in seen_guids]
        print(f"\n  {len(new_items)} new items (after seen filter)")

    new_items = filter_items(new_items)
    print(f"  {len(new_items)} items after hard filter")

    cached_items    = load_article_cache(learnings)
    cached_restored = restore_cached_items(cached_items)
    if cached_restored:
        print(f"  {len(cached_restored)} cached articles added for clustering")

    if not new_items:
        print("\n⚠️  No new items to process.")
        return

    # ── Cluster ────────────────────────────────────────────────────
    all_for_clustering = new_items + cached_restored
    clusters = cluster_items(all_for_clustering)
    clusters = [c for c in clusters if any(not i.get("_from_cache") for i in c["items"])]

    print(f"\n{'═' * 60}")
    print(f"CLUSTERS ({len(clusters)} clusters from {len(new_items)} items)")
    print("═" * 60)
    for i, c in enumerate(clusters):
        n_t1 = sum(1 for it in c["items"] if it["source_tier"] == 1)
        n_t2 = sum(1 for it in c["items"] if it["source_tier"] == 2)
        print(f"\n  [{i + 1}] {c['headline'][:75]}")
        print(f"       Sources: {', '.join(c['sources'])}")
        print(f"       T1: {n_t1}  T2: {n_t2}  Total: {n_t1 + n_t2}")
        for it in sorted(c["items"], key=lambda x: x["published_dt"]):
            age   = int((datetime.now(timezone.utc) - it["published_dt"]).total_seconds() / 60)
            tier  = "T1" if it["source_tier"] == 1 else "T2"
            cache = " [cached]" if it.get("_from_cache") else ""
            print(f"         [{tier}] {it['source_name']:<22} {age:>3}m ago | {it['title'][:60]}{cache}")

    # ── Claude Call 1 ──────────────────────────────────────────────
    recently_posted = [] if IGNORE_RECENT else load_recently_posted(learnings)
    if IGNORE_RECENT:
        print(f"\n  ⚠️  IGNORE_RECENT=True — dupe suppression disabled")
    print(f"\n  Claude call 1: {len(clusters)} clusters, {len(recently_posted)} recently posted...")
    for i, c in enumerate(clusters, 1):
        c["_local_id"] = i
    merges, dupes, splits = claude_find_merges(clusters, recently_posted)
    clusters = apply_merges(clusters, merges)

    if dupes:
        print(f"  ⛔ Suppressed {len(dupes)} dupe clusters: {dupes}")
    if splits:
        print(f"  ✂️  Suppressed {len(splits)} split clusters: {splits}")

    suppress = dupes | splits
    clusters_to_assess = [c for c in clusters if c.get("_local_id", c["id"]) not in suppress]
    print(f"  {len(clusters_to_assess)} clusters to assess")

    # ── Claude Call 2 ──────────────────────────────────────────────
    print(f"\n  Claude call 2: assessing {len(clusters_to_assess)} clusters...")
    assessments = claude_assess_clusters(clusters_to_assess, learnings)

    print(f"\n{'═' * 60}")
    print(f"CLAUDE ASSESSMENT ({len(assessments)} clusters)")
    print("═" * 60)

    will_post: list[tuple] = []
    posted_this_run_dbg: list[str] = []

    def _is_mid_run_dupe_dbg(headline: str) -> bool:
        words = set(w for w in headline.lower().split() if len(w) > 4)
        for prev in posted_this_run_dbg:
            if len(words & set(w for w in prev.lower().split() if len(w) > 4)) >= 3:
                print(f"  🚫 Mid-run dupe suppressed: '{headline[:60]}'")
                return True
        return False

    for cluster, assessment in zip(clusters_to_assess, assessments):
        tier          = enforce_tier(assessment, cluster)
        rel           = assessment["relevance"]
        original_tier = assessment["tier"]
        demoted       = tier != original_tier

        tier_icons = {
            "trending":     "📈 Breaking Story",
            "proven_topic": "🎮 Polygon Topic",
            "polygon_pick": "⭐ Polygon Pick",
        }

        if tier == "skip" or rel < RELEVANCE_MIN:
            suffix = f" ⬇️ demoted from {original_tier}" if demoted else ""
            print(f"\n  ⏭  SKIP (rel={rel}, claude_tier={original_tier}){suffix}")
        elif _is_mid_run_dupe_dbg(assessment.get("headline", "")):
            print(f"\n  ⏭  SKIP (mid-run dupe)")
        else:
            print(f"\n  ✅ POST — {tier_icons.get(tier, tier)}")
            will_post.append((cluster, assessment, tier))
            posted_this_run_dbg.append(assessment.get("headline", ""))

        print(f"       headline : {assessment['headline'][:70]}")
        print(f"       relevance: {rel}/10  |  claude: {original_tier}  |  final: {tier}")
        if assessment.get("topic"):
            print(f"       topic    : {assessment['topic']}")
        if cluster.get("_merged_from"):
            print(f"       merged   : clusters {cluster['_merged_from']}")
        if assessment.get("angle"):
            print(f"       angle    : {assessment['angle'][:80]}")
        if assessment.get("note"):
            print(f"       note     : {assessment['note'][:80]}")
        print(f"       sources  : {', '.join(cluster['sources'])}")

    print(f"\n  After assessment: {len(will_post)}/{len(assessments)} stories would post")

    if recently_posted:
        print(f"\n  Recently posted (last 24h):")
        for s in recently_posted[-10:]:
            age = int((datetime.now(timezone.utc) - datetime.fromisoformat(s["posted_at"])).total_seconds() / 60)
            print(f"    {age:>4}m ago | {s['headline'][:70]}")

    # ── Post (if enabled) ──────────────────────────────────────────
    if POST_TO_SLACK_DBG and will_post:
        print("\n  Posting to Slack...")
        posted = 0
        for cluster, assessment, tier in will_post:
            if post_to_slack(cluster, assessment, tier):
                posted += 1
                cluster_urls   = [it["url"]   for it in cluster["items"] if it.get("url")]
                cluster_titles = [it["title"] for it in cluster["items"] if it.get("title")]
                record_posted_story(learnings, assessment["headline"], assessment.get("topic", ""), cluster_urls, cluster_titles)
        save_learnings(learnings)
        print(f"  Posted {posted} messages.")
    else:
        print(f"\n  ℹ️  POST_TO_SLACK_DBG=False — nothing sent to Slack.")
        print("     Set POST_TO_SLACK_DBG = True at the top of debug.py when ready.")


if __name__ == "__main__":
    run_debug()
