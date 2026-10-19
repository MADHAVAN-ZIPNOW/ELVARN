from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from app import analytics

P = "/api/jarvis/memory"
IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 1, 7, 15, 0, tzinfo=IST)   # Wednesday


def seed(add):
    add("a", source="voice", created_at=datetime(2026, 1, 7, 9, 30, tzinfo=IST), tags=["work", "plan"], people=["Ravi"], projects=["ELVARN"], sentiment="positive")
    add("b", source="email", created_at=datetime(2026, 1, 7, 9, 45, tzinfo=IST), tags=["work"], people=["Ravi"], tasks=[{"task": "x", "status": "done"}, {"task": "y", "status": "pending"}])
    add("c", source="manual", created_at=datetime(2026, 1, 6, 20, 0, tzinfo=IST), tags=["home"], sentiment="negative")
    add("d", source="voice", created_at=datetime(2026, 1, 5, 11, 0, tzinfo=IST))
    add("old", source="manual", created_at=datetime(2025, 11, 1, 12, 0, tzinfo=IST))     # outside week and month


def test_week_series_sources_tasks_and_topics(client, add):
    seed(add)
    out = analytics.build(client.app_state.service, "week", NOW)
    assert out["bucket"] == "day" and out["labels"][0] == "2026-01-01" and out["labels"][-1] == "2026-01-07" and len(out["labels"]) == 7
    assert out["series"]["voice"][6] == 1 and out["series"]["email"][6] == 1 and out["series"]["manual"][5] == 1 and out["series"]["voice"][4] == 1
    assert sum(sum(v) for v in out["series"].values()) == 4                       # the old one is excluded
    assert out["tasks"]["open"] == 1 and out["tasks"]["created"][6] == 2
    assert out["top"]["tags"][0] == {"name": "work", "count": 2} and out["top"]["people"][0] == {"name": "ravi", "count": 2}
    assert out["sentiment"] == {"positive": 1, "negative": 1, "unknown": 2}
    assert out["totals"]["memories"] == 4


def test_day_view_is_hourly_and_heat_map_finds_the_busiest_slot(client, add):
    seed(add)
    day = analytics.build(client.app_state.service, "day", NOW)
    assert day["bucket"] == "hour" and len(day["labels"]) == 24 and day["series"]["voice"][9] == 1 and day["series"]["email"][9] == 1
    heat = analytics.build(client.app_state.service, "month", NOW)
    assert heat["heat"][2][9] == 2                                               # Wednesday 09:00 has two memories
    assert heat["totals"]["busiest_day"] == "Wed" and heat["totals"]["busiest_hour"] == 9


def test_streak_counts_consecutive_days_ending_today(client, add):
    seed(add)
    assert analytics.build(client.app_state.service, "week", NOW)["streak"] == 3      # 5th, 6th, 7th
    assert analytics.build(client.app_state.service, "week", NOW + timedelta(days=2))["streak"] == 0


def test_analytics_endpoint_validates_range_and_needs_the_key(client, make_client):
    assert client.get(f"{P}/analytics", params={"range": "week"}).status_code == 200
    assert client.get(f"{P}/analytics", params={"range": "decade"}).status_code == 422
    assert make_client().get(f"{P}/analytics", headers={"X-Owner-Key": ""}).status_code == 401
