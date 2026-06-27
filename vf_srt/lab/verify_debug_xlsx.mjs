import fs from "node:fs/promises";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { FileBlob, SpreadsheetFile } = require("@oai/artifact-tool");

const [input, previewDir] = process.argv.slice(2);
const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(input));
const sheets = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 3000 });
const errors = await workbook.inspect({
  kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 }, summary: "formula error scan", maxChars: 3000,
});
const segments = await workbook.inspect({
  kind: "table", range: "Segments!A1:K8", include: "values,formulas",
  tableMaxRows: 8, tableMaxCols: 11, maxChars: 5000,
});
await fs.mkdir(previewDir, { recursive: true });
const previewRanges = { Summary: "A1:E9", Segments: "A1:K30", Candidates: "A1:M30" };
for (const name of ["Summary", "Segments", "Candidates"]) {
  const preview = await workbook.render({ sheetName: name, range: previewRanges[name], scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${name}.png`, new Uint8Array(await preview.arrayBuffer()));
}
console.log(sheets.ndjson);
console.log(errors.ndjson);
console.log(segments.ndjson);
