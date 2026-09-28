const fs = require("fs");
const path = require("path");

const content = fs.readFileSync(path.join(__dirname, "../tests/mitumbes-item-creation-comparison.test.ts"), "utf8");
const lines = content.split("\n");
const toolsBlock = lines.slice(20, 696).join("\n");
fs.mkdirSync(path.join(__dirname, "../tests/fixtures"), { recursive: true });
fs.writeFileSync(path.join(__dirname, "../tests/fixtures/mitumbes-production-tools.ts"), toolsBlock);
console.log("Fixtures created successfully, lines:", lines.slice(20, 696).length);
