/**
 * Vercel "Other" preset defaults Output Directory to `public` when that folder exists.
 * Some dashboards still expect `public` after `npm run build`. Always ensure it exists.
 */
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.join(__dirname, "..");
const pub = path.join(root, "public");
fs.mkdirSync(pub, { recursive: true });
const index = path.join(pub, "index.html");
if (!fs.existsSync(index)) {
  fs.writeFileSync(
    index,
    `<!DOCTYPE html><html><head><meta charset="utf-8"><title>Gong API</title></head><body><p>See <code>/api/health</code>.</p></body></html>\n`
  );
}
