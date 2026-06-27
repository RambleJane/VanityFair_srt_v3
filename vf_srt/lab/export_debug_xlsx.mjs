import fs from "node:fs/promises";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const { SpreadsheetFile, Workbook } = require("@oai/artifact-tool");

const [episode, labDir, reportsDir, output] = process.argv.slice(2);
if (!episode || !labDir || !reportsDir || !output) throw new Error("Expected episode, labDir, reportsDir, output");

const parseCsv = (text) => {
  const rows = []; let row = []; let field = ""; let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted && ch === '"' && text[i + 1] === '"') { field += '"'; i++; }
    else if (ch === '"') quoted = !quoted;
    else if (!quoted && ch === ',') { row.push(field); field = ""; }
    else if (!quoted && (ch === '\n' || ch === '\r')) {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(field); field = ""; if (row.some((v) => v !== "")) rows.push(row); row = [];
    } else field += ch;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  if (rows[0]?.[0]?.charCodeAt(0) === 0xFEFF) rows[0][0] = rows[0][0].slice(1);
  return rows;
};

const workbook = Workbook.create();
const addCsvSheet = async (name, path, widths) => {
  const rawRows = parseCsv(await fs.readFile(path, "utf8"));
  const rows = rawRows.map((row, rowIndex) => row.map((value) => {
    if (rowIndex > 0 && /^-?\d+(?:\.\d+)?$/.test(value)) return Number(value);
    return value === "" ? null : value;
  }));
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  if (rows.length) sheet.getRangeByIndexes(0, 0, rows.length, rows[0].length).values = rows;
  const header = sheet.getRangeByIndexes(0, 0, 1, rows[0].length);
  header.format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF" }, wrapText: true };
  header.format.rowHeight = 28;
  sheet.freezePanes.freezeRows(1);
  widths.forEach((width, col) => { sheet.getRangeByIndexes(0, col, Math.max(rows.length, 1), 1).format.columnWidth = width; });
  if (rows.length > 1) sheet.getRangeByIndexes(1, 0, rows.length - 1, rows[0].length).format.borders = { preset: "inside", style: "thin", color: "#E2E8F0" };
  return sheet;
};

const segments = await addCsvSheet("Segments", `${labDir}/${episode}_segments_preview.csv`, [8, 12, 12, 11, 8, 9, 46, 25, 18, 11, 52]);
const segmentRows = segments.getUsedRange();
segments.getRange(`G1:G${segmentRows.rowCount}`).format.wrapText = true;
segments.getRange(`H1:H${segmentRows.rowCount}`).format.wrapText = true;
const candidates = await addCsvSheet("Candidates", `${labDir}/${episode}_cut_candidates.csv`, [10, 10, 12, 12, 12, 12, 12, 12, 12, 14, 14, 10, 50]);
const candidateRows = candidates.getUsedRange();
candidates.getRange(`M1:M${candidateRows.rowCount}`).format.wrapText = true;

const report = JSON.parse(await fs.readFile(`${reportsDir}/${episode}_segmentation_report.json`, "utf8"));
const summary = workbook.worksheets.add("Summary"); summary.showGridLines = false;
summary.getRange("A1:D1").merge(); summary.getRange("A1").values = [[`Episode ${episode} segmentation`]];
summary.getRange("A1:D1").format = { fill: "#17365D", font: { bold: true, color: "#FFFFFF", fontSize: 16 } };
summary.getRange("A3:B9").values = [
  ["Metric", "Value"], ["Utterances", report.total_utterances], ["Words", report.total_words],
  ["Speech islands", report.total_islands], ["Subtitle segments", report.total_segments],
  ["Average chars", report.averages.chars], ["Average duration", report.averages.duration],
];
summary.getRange("A3:B3").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };
summary.getRange("D3:E7").values = [
  ["Gap", "Seconds"], ["weak", report.gap_profile.weak_gap], ["soft", report.gap_profile.soft_gap],
  ["strong", report.gap_profile.strong_gap], ["p99", report.gap_profile.p99],
];
summary.getRange("D3:E3").format = { fill: "#D9EAF7", font: { bold: true, color: "#17365D" } };
summary.getRange("A1:A9").format.columnWidth = 22; summary.getRange("B1:B9").format.columnWidth = 14;
summary.getRange("D1:D9").format.columnWidth = 18; summary.getRange("E1:E9").format.columnWidth = 14;

const blob = await SpreadsheetFile.exportXlsx(workbook);
await blob.save(output);
