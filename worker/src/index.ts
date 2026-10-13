/**
 * SystemOne over HTTP, answered by Cloudflare's Clef models through the Workers AI binding:
 * POST /v1/systemone {model?, state, questions, images?} and GET /health. For `wrangler dev` on this machine only.
 */
const MODELS: Record<string, string> = { "clef": "@cf/cloudflare/clef", "clef-flash": "@cf/cloudflare/clef-flash" };

export default {
  async fetch(request: Request, env: { AI: { run(model: string, input: unknown): Promise<unknown> } }): Promise<Response> {
    const { pathname } = new URL(request.url);
    if (request.method === "GET" && pathname === "/health") return Response.json({ ok: true, backend: "workers-ai" });
    if (request.method !== "POST" || pathname !== "/v1/systemone") return Response.json({ error: "not found" }, { status: 404 });
    let body: { model?: string; state?: unknown; questions?: unknown; images?: unknown };
    try {
      body = await request.json();
    } catch {
      return Response.json({ error: "invalid JSON" }, { status: 400 });
    }
    const short = MODELS[body.model ?? ""] ? (body.model as string) : "clef-flash";
    try {
      const result = await env.AI.run(MODELS[short], { ...body, model: short });
      return Response.json(result);
    } catch (err) {
      return Response.json({ error: err instanceof Error ? err.message : String(err) }, { status: 502 });
    }
  },
};
