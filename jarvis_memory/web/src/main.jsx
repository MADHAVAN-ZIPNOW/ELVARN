import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./tokens.css";
import "./styles.css";
import SceneBackground from "./SceneBackground";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <SceneBackground />
    <App />
  </React.StrictMode>,
);
