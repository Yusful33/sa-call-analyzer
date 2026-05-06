import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __dirname = dirname(fileURLToPath(import.meta.url));

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

/** Flat config wrapping Next.js ESLint presets (non-interactive `next lint`). */
export default [...compat.extends("next/core-web-vitals", "next/typescript")];
