// ELVARN mail relay: paste this into a new Google Apps Script project (script.google.com) in the Google account that should send the mail,
// change SECRET to a long random text, then  Deploy > New deployment > Web app > Execute as: Me, Who has access: Anyone.
// Copy the web app URL. ELVARN sends it the recipient, subject and text over HTTPS; this script sends the mail from your Gmail
// (limit: about 100 mails a day for a normal Gmail account). Only requests that carry the right SECRET are accepted.
const SECRET = 'PASTE_A_LONG_RANDOM_TEXT_HERE';

function doPost(e) {
  let ok = false;
  try {
    const d = JSON.parse(e.postData.contents);
    if (d.secret === SECRET && /^[^\s@,;<>]+@[^\s@,;<>]+\.[A-Za-z]{2,}$/.test(d.to)) {
      MailApp.sendEmail({ to: d.to, subject: String(d.subject).slice(0, 120), body: String(d.body).slice(0, 5000), name: 'ELVARN' });
      ok = true;
    }
  } catch (err) { ok = false; }
  return ContentService.createTextOutput(JSON.stringify({ ok: ok })).setMimeType(ContentService.MimeType.JSON);
}
