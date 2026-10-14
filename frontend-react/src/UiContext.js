// Things many small components need: announce a message to screen readers, open the file viewer.
import { createContext, useContext } from "react";

export const UiContext = createContext({
  announce: () => {},
  openFilePanel: () => {},
});

export const useUi = () => useContext(UiContext);
