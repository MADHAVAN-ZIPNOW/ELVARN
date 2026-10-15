// ELVARN mail relay: paste this into your Google Apps Script project (script.google.com) in the Google account that should send the mail.
// Keep your own SECRET text below (the same one as MAIL_WEBHOOK_SECRET in your .env), then
//   Deploy > Manage deployments > (pencil) Edit > Version: New version > Deploy      (the web app URL stays the same)
// or, for a first time:  Deploy > New deployment > Web app > Execute as: Me, Who has access: Anyone.
// ELVARN sends this script the recipient, subject, text and (optionally) a colourful HTML version over HTTPS; the script sends the mail from
// your Gmail (limit: about 100 mails a day for a normal Gmail account). Only requests that carry the right SECRET are accepted.
const SECRET = 'PASTE_YOUR_SECRET_TEXT_HERE';

function doPost(e) {
  let ok = false;
  try {
    const d = JSON.parse(e.postData.contents);
    if (d.secret === SECRET && /^[^\s@,;<>]+@[^\s@,;<>]+\.[A-Za-z]{2,}$/.test(d.to)) {
      const mail = { to: d.to, subject: String(d.subject).slice(0, 120), body: String(d.body).slice(0, 5000), name: 'ELVARN' };
      if (d.html) mail.htmlBody = String(d.html).slice(0, 60000);
      MailApp.sendEmail(mail);
      ok = true;
    }
  } catch (err) { ok = false; }
  return ContentService.createTextOutput(JSON.stringify({ ok: ok })).setMimeType(ContentService.MimeType.JSON);
}
