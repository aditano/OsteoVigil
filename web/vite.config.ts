import { cpSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig, type Plugin } from "vite";

const ENGINE_FILES = [
  "cpt_predictor/__init__.py",
  "cpt_predictor/browser_api.py",
  "cpt_predictor/comparison.py",
  "cpt_predictor/dicom_codecs.py",
  "cpt_predictor/errors.py",
  "cpt_predictor/materials.py",
  "cpt_predictor/modality.py",
  "cpt_predictor/models.py",
  "cpt_predictor/mri_geometry.py",
  "cpt_predictor/radiograph.py",
  "cpt_predictor/reference_leg.py",
  "cpt_predictor/segmentation.py",
  "cpt_predictor/strength_io.py",
  "cpt_predictor/strength_pipeline.py",
  "cpt_predictor/voxel_fea.py",
];

function copyEngine(): Plugin {
  const webRoot = dirname(fileURLToPath(import.meta.url));
  const sourceRoot = resolve(webRoot, "../src");
  const targetRoot = resolve(webRoot, "public/engine");
  const copy = () => {
    rmSync(targetRoot, { recursive: true, force: true });
    for (const relative of ENGINE_FILES) {
      const destination = resolve(targetRoot, relative);
      mkdirSync(dirname(destination), { recursive: true });
      cpSync(resolve(sourceRoot, relative), destination);
    }
    writeFileSync(resolve(targetRoot, "manifest.json"), JSON.stringify(ENGINE_FILES));
  };
  return {
    name: "copy-strength-engine",
    buildStart: copy,
    configureServer: copy,
  };
}

export default defineConfig({
  root: ".",
  base: process.env.VITE_BASE || "/",
  plugins: [copyEngine()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
