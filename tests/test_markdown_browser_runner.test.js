"use strict";

const { test, before, after } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const http = require("node:http");

const runner = require("./run_markdown_browser_tests");

const HOST = "127.0.0.1";
let temporaryRoot = null;
let siteRoot = null;
let staticServer = null;

function requestWithPort(port, requestPath, method) {
  return new Promise(function (resolve, reject) {
    const req = http.request({
      host: HOST,
      port: port,
      path: requestPath,
      method: method || "GET"
    }, function (response) {
      const chunks = [];
      response.on("data", function (chunk) { chunks.push(chunk); });
      response.on("end", function () {
        resolve({
          statusCode: response.statusCode,
          headers: response.headers,
          body: Buffer.concat(chunks).toString("utf8")
        });
      });
    });
    req.on("error", reject);
    req.setTimeout(5000, function () {
      req.destroy(new Error("request timed out: " + requestPath));
    });
    req.end();
  });
}

function requestFile(requestPath, method) {
  return requestWithPort(staticServer.port, requestPath, method);
}

before(async function () {
  temporaryRoot = fs.mkdtempSync(
    path.join(os.tmpdir(), "teachable-markdown-server-limit-")
  );
  siteRoot = path.join(temporaryRoot, "site");
  fs.mkdirSync(siteRoot);
  fs.mkdirSync(path.join(siteRoot, "assets"));
  fs.mkdirSync(path.join(siteRoot, "backend"));

  fs.writeFileSync(
    path.join(siteRoot, "allowed.txt"), "allowed-content"
  );
  fs.writeFileSync(
    path.join(siteRoot, "assets", "hello.js"), "window.hello = true;"
  );
  fs.writeFileSync(
    path.join(siteRoot, ".env"), "SYNTHETIC_PLACEHOLDER_SECRET"
  );
  fs.writeFileSync(
    path.join(siteRoot, "data.sqlite"), "SYNTHETIC_PLACEHOLDER_DATABASE"
  );
  fs.writeFileSync(
    path.join(siteRoot, "backend", "secret.py"), "SYNTHETIC_PLACEHOLDER_BACKEND"
  );

  const outsideRoot = path.join(temporaryRoot, "outside");
  fs.mkdirSync(outsideRoot);
  const outsideFile = path.join(outsideRoot, "outside.txt");
  fs.writeFileSync(outsideFile, "OUTSIDE_PLACEHOLDER");
  fs.symlinkSync(outsideFile, path.join(siteRoot, "escape.txt"));

  staticServer = await runner.startStaticServer({
    rootDir: siteRoot,
    allowlist: {
      "/allowed.txt": "allowed.txt",
      "/assets/hello.js": "assets/hello.js",
      "/escape.txt": "escape.txt"
    }
  });
});

after(async function () {
  if (staticServer !== null) {
    await new Promise(function (resolve) {
      staticServer.server.close(function () { resolve(); });
    });
  }
  if (temporaryRoot !== null) {
    fs.rmSync(temporaryRoot, { recursive: true, force: true });
  }
});

test("static test server binds only to 127.0.0.1", function () {
  assert.equal(staticServer.host, HOST);
  assert.equal(staticServer.server.address().address, HOST);
  assert.equal(typeof staticServer.port, "number");
});

test("serves only exact allowlisted files", async function () {
  const text = await requestFile("/allowed.txt");
  assert.equal(text.statusCode, 200);
  assert.equal(text.body, "allowed-content");
  assert.equal(text.headers["x-content-type-options"], "nosniff");

  const script = await requestFile("/assets/hello.js");
  assert.equal(script.statusCode, 200);
  assert.equal(script.body, "window.hello = true;");
  assert.match(script.headers["content-type"], /^text\/javascript/);
});

test("does not read or serve .env, database, or backend files", async function () {
  const blocked = [
    "/.env",
    "/data.sqlite",
    "/backend/secret.py"
  ];
  for (let i = 0; i < blocked.length; i += 1) {
    const response = await requestFile(blocked[i]);
    assert.equal(response.statusCode, 404);
    assert.equal(
      response.body.indexOf("SYNTHETIC_PLACEHOLDER") === -1,
      true,
      "blocked response leaked placeholder content for " + blocked[i]
    );
  }
});

test("rejects path traversal attempts with synthetic paths", async function () {
  const attempts = [
    "/../.env",
    "/assets/../../.env",
    "/assets/../.env",
    "/%2e%2e/.env",
    "/assets%2f..%2f.env"
  ];
  for (let i = 0; i < attempts.length; i += 1) {
    const response = await requestFile(attempts[i]);
    assert.equal(
      response.statusCode === 200,
      false,
      "traversal was served: " + attempts[i]
    );
    assert.equal(
      response.body.indexOf("SYNTHETIC_PLACEHOLDER") === -1,
      true,
      "traversal leaked placeholder content for " + attempts[i]
    );
  }
});

test("rejects allowlisted symlinks that resolve outside the root", async function () {
  const response = await requestFile("/escape.txt");
  assert.equal(response.statusCode, 404);
  assert.equal(response.body.indexOf("OUTSIDE_PLACEHOLDER") === -1, true);
});

test("rejects unsupported HTTP methods", async function () {
  const response = await requestFile("/allowed.txt", "POST");
  assert.equal(response.statusCode, 405);
});

test("three browser fixture pages declare an HTML5 doctype", function () {
  const bang = String.fromCharCode(33);
  const expected = "<" + bang + "DOCTYPE html>";
  const pages = [
    path.join(__dirname, "browser", "markdown-renderer.integration.html"),
    path.join(__dirname, "browser", "markdown-renderer.app.html"),
    path.join(__dirname, "browser", "markdown-renderer.visual.html")
  ];
  for (let i = 0; i < pages.length; i += 1) {
    const firstLine = fs.readFileSync(pages[i], "utf8").split(/\r?\n/)[0];
    assert.equal(firstLine, expected, pages[i] + " first line is not HTML5");
  }
});

test("runner does not retry with disabled browser sandbox", function () {
  const source = fs.readFileSync(
    path.join(__dirname, "run_markdown_browser_tests.js"),
    "utf8"
  );
  assert.equal(source.indexOf("--no-sandbox") === -1, true);
});

test("default allowlist omits secrets, git data, and backend files", function () {
  const allowed = Object.keys(runner.ALLOWED_PATHS).join("\n");
  assert.equal(allowed.indexOf(".env") === -1, true);
  assert.equal(allowed.indexOf("/.git") === -1, true);
  assert.equal(allowed.indexOf("/data") === -1, true);
  assert.equal(allowed.indexOf("/backend") === -1, true);
});

test("default repository allowlist serves every declared resource", async function () {
  const defaultServer = await runner.startStaticServer();
  try {
    const declared = Object.keys(runner.ALLOWED_PATHS);
    assert.equal(declared.length >= 19, true, "allowlist is unexpectedly small");
    for (let i = 0; i < declared.length; i += 1) {
      const response = await requestWithPort(
        defaultServer.port, declared[i]
      );
      assert.equal(response.statusCode, 200, declared[i] + " was not served");
      assert.equal(response.body.length > 0, true, declared[i] + " is empty");
    }
    const blocked = await requestWithPort(defaultServer.port, "/README.md");
    assert.equal(blocked.statusCode, 404);
  } finally {
    await new Promise(function (resolve) {
      defaultServer.server.close(function () { resolve(); });
    });
  }
});
