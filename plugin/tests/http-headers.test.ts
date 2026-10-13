import assert from "node:assert/strict";
import { test } from "node:test";
import { expandHeaders } from "../src/backends/http.ts";

test("header values take ${NAME} from the environment", () => {
  process.env.CLEFKIT_TEST_TOKEN = "abc";
  assert.deepEqual(expandHeaders({ Authorization: "Bearer ${CLEFKIT_TEST_TOKEN}" }), { Authorization: "Bearer abc" });
  assert.equal(expandHeaders(undefined), undefined);
});

test("a missing variable is a clear error", () => {
  delete process.env.CLEFKIT_TEST_MISSING;
  assert.throws(() => expandHeaders({ Authorization: "${CLEFKIT_TEST_MISSING}" }), /CLEFKIT_TEST_MISSING/);
});
