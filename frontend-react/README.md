# ELVARN frontend (React + Vite)

The chat UI, built with React. It looks and behaves like the earlier single-file page in `../frontend/index.html`
(the stylesheets were moved over unchanged, in the same order), but the code is split into small files.

```
src/
  main.jsx            starts the app, loads the stylesheets in order
  App.jsx             state (chats, settings, the answer being streamed) and wiring
  appHelpers.js       first state, stored message -> answer row, the rocket effect
  api.js  storage.js  files.js  export.js  constants.js  utils.js  icons.jsx  UiContext.js
  hooks/
    useTilt.js          cards tilt toward the cursor
    useSpeech.js        read answers aloud
    useVoiceInput.js    speak your message
    useAttachment.js    attach / upload / drag-and-drop a file
  components/
    Chrome.jsx          sidebar, top bar, banner
    Messages.jsx        welcome page, your message, the model's answer
    RichText.jsx        paragraphs, lists, bold, code, links, [n] marks (React elements only, never HTML strings)
    AnswerInfo.jsx      "Answered from ...", stored facts
    ActionBar.jsx       sources button, Copy, Like, Dislike, Share, Download, Listen, Details
    DownloadMenu.jsx    PDF / Word / Text menu
    SourcesPanel.jsx    right-side drawer with one card per source
    ProcessCard.jsx     "How this reply was made"
    SessionsView.jsx    sessions page: search, filters, pin, rename, download, delete
    Composer.jsx        message box, attached file bar, mic, Send
    FileParts.jsx       file chip, converted-file card, file viewer
    LabPanel.jsx        model lab and settings
    Logo.jsx            the animated canvas logo and tab icon
  styles/               the CSS, split by area (order matters, see main.jsx)
  assets/               the two logo layers
public/                 logo files copied as they are
```

## Commands

```powershell
cd frontend-react
npm install
npm run dev      # http://localhost:5173 (talks to the backend on port 8000)
npm run build    # writes dist/
```

## How it is served

The Python server serves `frontend-react/dist/` when it exists (index at `/`, files under `/static`).
`dist/` is **committed**, because the Render build is Python only and has no Node step. After changing anything in `src/`, run
`npm run build` and commit the new `dist/`.

If `dist/` is missing, or `ELVARN_LEGACY_UI=1` is set, the server falls back to the earlier single-file page in `../frontend/`.
