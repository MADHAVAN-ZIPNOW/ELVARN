"""The colourful "your new chat is ready" email: a subject, a plain-text version and an HTML version (tables and inline styles only, so Gmail,
Outlook and phones show it the same way)."""
from html import escape

GRADIENT = "linear-gradient(100deg,#1d15ab 0%,#065fc3 22%,#03b9dc 44%,#511c83 64%,#a52976 82%,#d27b47 100%)"
SOLID = "#065fc3"                      # shown where gradients are not supported

TIPS = [("#2f7dff", "&#128269;", "Tell me about the DVAC in Tamil Nadu"),
        ("#22b35a", "&#128176;", "What was the amount in the DVAC raid on the Pollution Control Board in December 2018?"),
        ("#ff9f1c", "&#9878;", "What is the Prevention of Corruption Act?"),
        ("#ff4f97", "&#127908;", "Tap the headphones and just talk to me")]


def part_of_day(hour):
    if 5 <= hour < 12:
        return "Good morning", "&#9728;&#65039;"
    if 12 <= hour < 17:
        return "Good afternoon", "&#127780;&#65039;"
    if 17 <= hour < 21:
        return "Good evening", "&#127751;"
    return "Hello, night owl", "&#127769;"


def new_chat_email(name, hour, site_url, minute=0):
    """-> (subject, text, html). `hour` is the visitor's local hour (0-23)."""
    who = escape((name or "there").strip()[:40]) or "there"
    greet, icon = part_of_day(int(hour) % 24)
    url = escape(site_url, quote=True)
    subject = f"{greet}, {who}! Your new ELVARN chat is ready"
    text = (f"{greet}, {who}!\n\nYour new ELVARN chat is ready: {site_url}\n\nTry asking:\n" +
            "\n".join(f"  - {t}" for _, _, t in TIPS[:3]) +
            "\n\nELVARN is a small AI model and can be wrong. Allegations are not proof of guilt: always check the sources it shows.\n"
            "You get this mail because you switched on new-chat emails in the Model lab; switch it off there any time.\n")
    chips = "".join(
        f'<tr><td style="padding:6px 0"><table role="presentation" cellspacing="0" cellpadding="0" width="100%"><tr>'
        f'<td width="44" style="background:{c};border-radius:14px 0 0 14px;text-align:center;font-size:20px;padding:12px 0;color:#ffffff">{e}</td>'
        f'<td style="background:#f3f6fd;border-radius:0 14px 14px 0;padding:12px 16px;font:600 14px/1.4 Segoe UI,Arial,sans-serif;color:#1f2a44">{escape(t)}</td>'
        f'</tr></table></td></tr>' for c, e, t in TIPS)
    html = f"""<!doctype html>
<html><body style="margin:0;padding:0;background:#e6eaf3">
<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:#e6eaf3"><tr><td align="center" style="padding:24px 12px">
<table role="presentation" width="600" cellspacing="0" cellpadding="0" style="max-width:600px;width:100%;background:#ffffff;border-radius:24px;overflow:hidden;font-family:Segoe UI,Arial,sans-serif">
  <tr><td style="background:{SOLID};background-image:{GRADIENT};padding:34px 28px 28px;text-align:center">
    <div style="font:900 30px/1 Segoe UI,Arial,sans-serif;letter-spacing:6px;color:#ffffff">ELVARN</div>
    <div style="font-size:46px;line-height:1.3;padding-top:10px">{icon}</div>
  </td></tr>
  <tr><td style="padding:30px 30px 6px;text-align:center">
    <div style="font:800 26px/1.25 Segoe UI,Arial,sans-serif;color:#1f2a44">{greet}, <span style="color:#a52976">{who}</span>! &#128075;</div>
    <div style="font:400 16px/1.6 Segoe UI,Arial,sans-serif;color:#55617f;padding-top:10px">Your new chat is ready. Ask about Tamil Nadu corruption-case records and I will show you the sources.</div>
  </td></tr>
  <tr><td align="center" style="padding:22px 30px 8px">
    <a href="{url}" style="display:inline-block;background:{SOLID};background-image:{GRADIENT};color:#ffffff;text-decoration:none;font:800 15px/1 Segoe UI,Arial,sans-serif;letter-spacing:2px;padding:16px 38px;border-radius:999px">OPEN ELVARN &#128640;</a>
  </td></tr>
  <tr><td style="padding:18px 30px 6px">
    <div style="font:800 12px/1 Segoe UI,Arial,sans-serif;letter-spacing:2px;color:#8a94ad;padding-bottom:6px">TRY ASKING</div>
    <table role="presentation" width="100%" cellspacing="0" cellpadding="0">{chips}</table>
  </td></tr>
  <tr><td style="padding:22px 30px 30px;text-align:center;font:400 12px/1.6 Segoe UI,Arial,sans-serif;color:#8a94ad">
    ELVARN is a small AI model and can be wrong. Allegations are not proof of guilt: always check the sources it shows.<br>
    You get this mail because you switched on new-chat emails in the Model lab. Switch it off there any time.
  </td></tr>
</table></td></tr></table></body></html>"""
    return subject, text, html
