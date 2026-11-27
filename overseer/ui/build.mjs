import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import * as esbuild from "esbuild";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const srcDir = path.join(__dirname, "src");
const distDir = path.join(__dirname, "dist");
const outHtmlPath = path.join(distDir, "app.html");

const MAX_BUNDLE_BYTES = 500 * 1024;

const [jsResult, cssResult] = await Promise.all([
  esbuild.build({
    entryPoints: [path.join(srcDir, "app.ts")],
    bundle: true,
    minify: true,
    format: "iife",
    platform: "browser",
    target: ["es2022"],
    write: false,
    legalComments: "none",
  }),
  esbuild.build({
    entryPoints: [path.join(srcDir, "styles.css")],
    bundle: true,
    minify: true,
    write: false,
  }),
]);

const jsCode = jsResult.outputFiles[0].text.replace(/<\/script/gi, "<\\/script");
const cssCode = cssResult.outputFiles[0].text.replace(/<\/style/gi, "<\\/style");

const html = [
  "<!DOCTYPE html>",
  '<html lang="en">',
  "<head>",
  '  <meta charset="utf-8" />',
  '  <meta name="viewport" content="width=device-width, initial-scale=1.0" />',
  '  <meta name="color-scheme" content="light dark" />',
  "  <title>Overseer Monitor</title>",
  `  <style>${cssCode}</style>`,
  "</head>",
  "<body>",
  '  <div id="app"></div>',
  `  <script>${jsCode}</script>`,
  "</body>",
  "</html>",
  "",
].join("\n");

fs.mkdirSync(distDir, { recursive: true });
fs.writeFileSync(outHtmlPath, html, "utf8");

const byteLength = Buffer.byteLength(html, "utf8");
if (byteLength === 0 || byteLength >= MAX_BUNDLE_BYTES) {
  throw new Error(
    `Bundle size ${byteLength} bytes violates (0, ${MAX_BUNDLE_BYTES}) byte limit.`
  );
}

console.log(`Built ${outHtmlPath} (${byteLength} bytes, ${(byteLength / 1024).toFixed(1)} KB)`);
