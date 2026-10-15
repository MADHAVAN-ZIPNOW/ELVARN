import { createRoot } from "react-dom/client";
import App from "./App";
import { startAnimatedTabIcon } from "./components/Logo";
import { captureOwnerKey } from "./ownerKey";

captureOwnerKey();                       // ?owner=KEY in the address: remember it once

// The stylesheets are loaded in this order on purpose: later files override earlier ones, exactly as in the single-file version.
import "./styles/base.css";
import "./styles/sidebar-main.css";
import "./styles/messages.css";
import "./styles/process.css";
import "./styles/composer-files.css";
import "./styles/layout-logo-voice.css";
import "./styles/panels.css";
import "./styles/sidebar-3d.css";
import "./styles/sessions.css";
import "./styles/voice.css";
import "./styles/lead.css";
import "./styles/responsive.css";

createRoot(document.getElementById("root")).render(<App />);
startAnimatedTabIcon();
