"""Teams meeting notes saved as memories ("Meeting: ...") can be listed and read back by the chat. No database: plain dicts stand in for memories."""
from app import meetings


def part(title, day, start, end, lines):
    head = f"Meeting: {title} - {day}, {start} to {end} (notes taken by JARVIS on Madhavan's laptop)\nSpeakers: Madhavan (microphone); Others (everyone else).\n"
    return {"title": f"Meeting: {title}", "content": head + "\n".join(lines), "created_at": f"2026-11-0{day.split()[1][-1]}T{end}:00"}


STANDUP = [part("Stand up", "Tue 10 Nov 2026", "09:36", "09:50", ["[09:40] Others: Ravi will finish the API on Friday.", "[09:41] Madhavan: I am blocked on database access."])]
IDEAS = [part("Innovative ideas", "Wed 04 Nov 2026", "11:31", "11:41", ["[11:35] Others: Let us try a voice demo."]),
         part("Innovative ideas", "Wed 04 Nov 2026", "11:41", "11:52", ["[11:45] Others: Priya owns the demo."])]
NEWEST_FIRST = STANDUP + IDEAS[::-1] + [{"title": "A shopping note", "content": "Buy milk", "created_at": "2026-11-01T10:00:00"}]


def test_lists_meetings_newest_first_and_ignores_other_memories():
    out = meetings.digest(NEWEST_FIRST + [])
    assert out.splitlines()[0] == "Meetings with notes (2):"
    assert out.index("Stand up") < out.index("Innovative ideas") and "milk" not in out
    assert "Innovative ideas - Wed 04 Nov 2026, 11:31 to 11:52 (2 lines of notes)" in out       # two saved parts are one meeting


def test_a_query_brings_in_what_was_said_in_order():
    out = meetings.digest(NEWEST_FIRST, "innovative ideas meeting")
    assert "What was said in Innovative ideas" in out and "Stand up" not in out
    assert out.index("voice demo") < out.index("Priya owns the demo")                         # the earlier part first
    assert "Speakers:" not in out                                                                # the header lines are not repeated


def test_nothing_saved_or_nothing_matching_says_so():
    assert "No meeting notes found" in meetings.digest([{"title": "x", "content": "y"}])
    assert "No meeting notes found for 'budget'" in meetings.digest(NEWEST_FIRST, "budget")
