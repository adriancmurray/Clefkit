import type { Backend, BackendConfig } from "../types.ts";
import { httpBackend } from "./http.ts";
import { workersAiBackend } from "./workers-ai.ts";

/** Builds the Backend for one configured entry, dispatching on its `kind`. */
export function makeBackend(cfg: BackendConfig): Backend {
  switch (cfg.kind) {
    case "http":
      return httpBackend(cfg);
    case "workers-ai":
      return workersAiBackend(cfg);
  }
}
