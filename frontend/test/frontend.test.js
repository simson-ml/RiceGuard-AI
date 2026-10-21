import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const frontendDir = path.resolve(__dirname, "..");
const distDir = path.resolve(frontendDir, "dist");

function formatLabel(label = "") {
  return label.replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

function isValidImageType(type = "", name = "") {
  const validTypes = ["image/jpeg", "image/png", "image/webp"];
  return validTypes.includes(type.toLowerCase()) || /\.(jpe?g|png|webp)$/i.test(name);
}

function isValidFileSize(sizeBytes, maxLimitBytes = 10 * 1024 * 1024) {
  return typeof sizeBytes === "number" && sizeBytes > 0 && sizeBytes <= maxLimitBytes;
}

test("formatLabel transforms snake_case to Title Case", () => {
  assert.equal(formatLabel("bacterial_leaf_blight"), "Bacterial Leaf Blight");
  assert.equal(formatLabel("bacterial_leaf_streak"), "Bacterial Leaf Streak");
  assert.equal(formatLabel("blast"), "Blast");
  assert.equal(formatLabel("normal"), "Normal");
  assert.equal(formatLabel(""), "");
});

test("Image validation accepts supported types and rejects others", () => {
  assert.equal(isValidImageType("image/jpeg", "sample.jpg"), true);
  assert.equal(isValidImageType("image/png", "sample.png"), true);
  assert.equal(isValidImageType("image/webp", "sample.webp"), true);
  assert.equal(isValidImageType("image/pjpeg", "sample.jpg"), true);

  assert.equal(isValidImageType("application/pdf", "document.pdf"), false);
  assert.equal(isValidImageType("text/plain", "notes.txt"), false);
  assert.equal(isValidImageType("image/gif", "anim.gif"), false);
});

test("Image size validation respects 10 MB limit", () => {
  assert.equal(isValidFileSize(500 * 1024), true); // 500 KB
  assert.equal(isValidFileSize(10 * 1024 * 1024), true); // exactly 10 MB
  assert.equal(isValidFileSize(10 * 1024 * 1024 + 1), false); // oversized
  assert.equal(isValidFileSize(0), false); // empty
});

test("FormData encodes with exact field name 'file'", () => {
  const formData = new FormData();
  const fakeBlob = new Blob(["fake-image-bytes"], { type: "image/jpeg" });
  formData.append("file", fakeBlob, "rice_leaf.jpg");

  assert.equal(formData.has("file"), true);
  assert.equal(formData.get("file").name, "rice_leaf.jpg");
});

test("Production build output integrity", () => {
  assert.equal(fs.existsSync(distDir), true, "dist directory must exist");

  const indexHtmlPath = path.join(distDir, "index.html");
  assert.equal(fs.existsSync(indexHtmlPath), true, "dist/index.html must exist");

  const htmlContent = fs.readFileSync(indexHtmlPath, "utf-8");
  assert.match(htmlContent, /<div id="root"><\/div>/);
  assert.match(htmlContent, /<link rel="icon"/);

  const assetsDir = path.join(distDir, "assets");
  assert.equal(fs.existsSync(assetsDir), true, "dist/assets directory must exist");

  const assetFiles = fs.readdirSync(assetsDir);
  const jsFiles = assetFiles.filter((f) => f.endsWith(".js"));
  const cssFiles = assetFiles.filter((f) => f.endsWith(".css"));

  assert.ok(jsFiles.length > 0, "At least one bundle JS file must be generated");
  assert.ok(cssFiles.length > 0, "At least one bundle CSS file must be generated");
});
