/**
 * Validate JSON trace payloads with the eval platform's OWN request schema
 * (src/lib/trace-input.ts). Used by tests/test_tracing_contract.py when
 * EVAL_PLATFORM_REPO points at a checkout of the platform with node_modules.
 *
 *   cd $EVAL_PLATFORM_REPO && npx tsx <this file> <payloads.json>
 *
 * Prints one JSON line: [{ok, error?, costUsd?}] in input order.
 */
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { join } from "node:path";

async function main() {
  const repo = process.cwd();
  const mod = await import(pathToFileURL(join(repo, "src", "lib", "trace-input.ts")).href);
  const payloads: unknown[] = JSON.parse(readFileSync(process.argv[2], "utf8"));
  const out = payloads.map((p) => {
    const r = mod.TraceInput.safeParse(p);
    return r.success
      ? { ok: true, costUsd: mod.traceCostUsd(r.data) }
      : { ok: false, error: r.error.flatten() };
  });
  console.log(JSON.stringify(out));
}

main();
