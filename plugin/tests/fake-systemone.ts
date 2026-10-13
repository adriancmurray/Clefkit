import { createServer } from "node:http";
import type { AddressInfo } from "node:net";

type QuestionShape = { type: string; criteria?: unknown };

/** A well-formed answer to one question, shaped the way a SystemOne endpoint would return it. */
function answerFor(question: QuestionShape): Record<string, unknown> {
  if (question.type === "noul") return { type: "noul", value: true, probability: 0.9, confidence: 0.8 };
  if (question.type === "choice") {
    const options = Object.keys(question.criteria as Record<string, string>);
    const probabilities = Object.fromEntries(options.map((option, i) => [option, i === 0 ? 1 : 0]));
    return { type: "choice", choice: options[0], confidence: 0.8, probabilities };
  }
  const levels = (question.criteria as string[]).length;
  const probabilities = Array.from({ length: levels }, (_, i) => (i === 1 ? 1 : 0));
  return { type: "score", score: 1, level: 1, confidence: 0.8, probabilities };
}

/**
 * Starts a fake SystemOne endpoint on 127.0.0.1 with a random port. It answers every question in the request and
 * records each decoded request body in `requests`.
 */
export async function startFakeSystemOne(): Promise<{
  url: string;
  requests: { model?: string; questions?: Record<string, QuestionShape> }[];
  close: () => Promise<void>;
}> {
  const requests: { model?: string; questions?: Record<string, QuestionShape> }[] = [];
  const server = createServer((req, res) => {
    if (req.method === "GET" && req.url === "/health") {
      res.writeHead(200).end("ok");
      return;
    }
    if (req.method !== "POST" || req.url !== "/v1/systemone") {
      res.writeHead(404).end();
      return;
    }
    let raw = "";
    req.setEncoding("utf8");
    req.on("data", (chunk: string) => (raw += chunk));
    req.on("end", () => {
      let body: { model?: string; questions?: Record<string, QuestionShape> };
      try {
        body = JSON.parse(raw);
      } catch {
        res.writeHead(400).end();
        return;
      }
      requests.push(body);
      const answers = Object.fromEntries(
        Object.entries(body.questions ?? {}).map(([id, question]) => [id, answerFor(question)]),
      );
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ model: body.model ?? "fake-model", answers }));
    });
  });
  await new Promise<void>((ready) => server.listen(0, "127.0.0.1", ready));
  const { port } = server.address() as AddressInfo;
  return {
    url: `http://127.0.0.1:${port}`,
    requests,
    close: () => new Promise<void>((done) => server.close(() => done())),
  };
}
