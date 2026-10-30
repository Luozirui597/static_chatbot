"use strict";

/*
 * Markdown browser test runner.
 *
 * Starts a temporary static server on 127.0.0.1 with a strict file
 * allowlist, drives real Chrome through the DevTools protocol, waits
 * for explicit JSON reports, validates execution counts, and exits
 * non-zero on any failure.  It never retries with browser security
 * features disabled: an environment that cannot run the normal sandbox
 * fails explicitly.
 */

const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const net = require("node:net");
const { spawn } = require("node:child_process");

const ROOT = path.resolve(__dirname, "..");
const HOST = "127.0.0.1";
const CHROME = process.env.MARKDOWN_TEST_CHROME ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const SCREENSHOT_DIR = path.join(__dirname, "browser", "screenshots");
const DEFAULT_CDP_TIMEOUT_MS = 10000;
const TARGET_WAIT_TIMEOUT_MS = 20000;
const BROWSER_START_TIMEOUT_MS = 20000;

const MIME = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".md": "text/markdown; charset=utf-8",
  ".png": "image/png"
};

/*
 * Exact URL to file allowlist.  The server never walks a directory and
 * never resolves a request path outside this map.
 */
const ALLOWED_PATHS = Object.freeze({
  "/frontend/style.css": "frontend/style.css",
  "/frontend/index.html": "frontend/index.html",
  "/frontend/app.js": "frontend/app.js",
  "/frontend/network-recovery.js": "frontend/network-recovery.js",
  "/frontend/clipboard.js": "frontend/clipboard.js",
  "/frontend/copy-controller.js": "frontend/copy-controller.js",
  "/frontend/interaction-mode.js": "frontend/interaction-mode.js",
  "/frontend/model-selection.js": "frontend/model-selection.js",
  "/frontend/session-profile-switch.js": "frontend/session-profile-switch.js",
  "/frontend/history-review.js": "frontend/history-review.js",
  "/frontend/markdown-renderer.js": "frontend/markdown-renderer.js",
  "/frontend/vendor/marked.umd.js": "frontend/vendor/marked.umd.js",
  "/frontend/vendor/purify.min.js": "frontend/vendor/purify.min.js",
  "/tests/browser/markdown-renderer.integration.html":
    "tests/browser/markdown-renderer.integration.html",
  "/tests/browser/markdown-renderer.integration.tests.js":
    "tests/browser/markdown-renderer.integration.tests.js",
  "/tests/browser/markdown-renderer.app.html":
    "tests/browser/markdown-renderer.app.html",
  "/tests/browser/markdown-renderer.app.tests.js":
    "tests/browser/markdown-renderer.app.tests.js",
  "/tests/browser/markdown-renderer.visual.html":
    "tests/browser/markdown-renderer.visual.html",
  "/tests/browser/markdown-renderer.visual.js":
    "tests/browser/markdown-renderer.visual.js"
});

function sleep(ms) {
  return new Promise(function (resolve) { setTimeout(resolve, ms); });
}

function withTimeout(promise, timeoutMs, label) {
  return new Promise(function (resolve, reject) {
    let settled = false;
    const timer = setTimeout(function () {
      if (settled) return;
      settled = true;
      reject(new Error(label + " timed out after " + timeoutMs + "ms"));
    }, timeoutMs);

    Promise.resolve(promise).then(function (value) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve(value);
    }, function (error) {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(error);
    });
  });
}

function getFreePort() {
  return withTimeout(new Promise(function (resolve, reject) {
    const probe = net.createServer();
    probe.on("error", reject);
    probe.listen(0, HOST, function () {
      const port = probe.address().port;
      probe.close(function () { resolve(port); });
    });
  }), 5000, "Free port probe");
}

function deny(response, statusCode) {
  const body = statusCode === 404 ? "not found\n" : "forbidden\n";
  response.writeHead(statusCode, {
    "Content-Type": "text/plain; charset=utf-8",
    "X-Content-Type-Options": "nosniff"
  });
  response.end(body);
}

function startStaticServer(options) {
  const opts = options || {};
  const rootDir = path.resolve(opts.rootDir || ROOT);
  const rootReal = fs.realpathSync(rootDir);
  const allowlist = opts.allowlist || ALLOWED_PATHS;
  const rootPrefix = rootReal.endsWith(path.sep)
    ? rootReal
    : rootReal + path.sep;

  return withTimeout(new Promise(function (resolve, reject) {
    const server = http.createServer(function (request, response) {
      try {
        if (request.method === "GET" || request.method === "HEAD") {
          /* allowed below */
        } else {
          deny(response, 405);
          return;
        }

        const parsed = new URL(request.url, "http://" + HOST);
        const pathname = decodeURIComponent(parsed.pathname);
        if (pathname.indexOf("\0") >= 0 ||
            pathname.indexOf("\\") >= 0 ||
            pathname.charAt(0) !== "/") {
          deny(response, 400);
          return;
        }

        const segments = pathname.split("/");
        for (let i = 0; i < segments.length; i += 1) {
          if (segments[i] === "." || segments[i] === "..") {
            deny(response, 403);
            return;
          }
        }

        const relative = allowlist[pathname];
        if (typeof relative !== "string") {
          deny(response, 404);
          return;
        }

        if (path.isAbsolute(relative) === true) {
          deny(response, 403);
          return;
        }

        const relativeSegments = relative.split(path.sep);
        for (let j = 0; j < relativeSegments.length; j += 1) {
          if (relativeSegments[j] === "." || relativeSegments[j] === "..") {
            deny(response, 403);
            return;
          }
        }

        const target = path.resolve(rootReal, relative);
        const realTarget = fs.realpathSync(target);
        if (realTarget !== target ||
            realTarget.startsWith(rootPrefix) === false) {
          deny(response, 404);
          return;
        }

        const stat = fs.statSync(realTarget);
        if (stat.isFile() === false) {
          deny(response, 404);
          return;
        }

        fs.readFile(realTarget, function (error, data) {
          if (error) {
            deny(response, 404);
            return;
          }
          const ext = path.extname(realTarget).toLowerCase();
          const headers = {
            "Content-Type": MIME[ext] || "application/octet-stream",
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff"
          };
          response.writeHead(200, headers);
          if (request.method === "HEAD") {
            response.end();
          } else {
            response.end(data);
          }
        });
      } catch (_) {
        deny(response, 400);
      }
    });

    server.on("error", reject);
    server.listen(0, HOST, function () {
      resolve({
        server: server,
        host: HOST,
        port: server.address().port,
        rootReal: rootReal
      });
    });
  }), 10000, "Static test server listen");
}

function stopStaticServer(staticServer) {
  return new Promise(function (resolve) {
    try {
      if (typeof staticServer.server.closeAllConnections === "function") {
        staticServer.server.closeAllConnections();
      }
    } catch (_) {
      /* close below */
    }
    staticServer.server.close(function () { resolve(); });
  });
}

function launchChrome(url) {
  return getFreePort().then(function (port) {
    return new Promise(function (resolve, reject) {
      const userDataDir = fs.mkdtempSync(
        path.join(os.tmpdir(), "teachable-markdown-chrome-")
      );
      const args = [
        "--headless=new",
        "--disable-gpu",
        "--remote-debugging-port=" + port,
        "--remote-allow-origins=*",
        "--user-data-dir=" + userDataDir,
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        url
      ];
      const child = spawn(CHROME, args, {
        stdio: ["ignore", "ignore", "pipe"]
      });
      let stderrText = "";
      let settled = false;

      child.stderr.on("data", function (chunk) {
        if (stderrText.length < 16000) {
          stderrText += chunk.toString("utf8");
        }
      });

      function cleanup() {
        try {
          fs.rmSync(userDataDir, { recursive: true, force: true });
        } catch (_) {
          /* best effort */
        }
      }

      const timer = setTimeout(function () {
        if (settled) return;
        settled = true;
        cleanup();
        try { child.kill("SIGKILL"); } catch (_) { /* already gone */ }
        reject(new Error(
          "Chrome did not become spawnable within " +
          BROWSER_START_TIMEOUT_MS + "ms"
        ));
      }, BROWSER_START_TIMEOUT_MS);

      child.once("spawn", function () {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        resolve({
          child: child,
          port: port,
          userDataDir: userDataDir,
          stderr: function () { return stderrText.trim(); }
        });
      });

      child.once("error", function (error) {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        cleanup();
        reject(new Error(
          "Failed to launch Chrome at " + CHROME + ": " +
          (error && error.message ? error.message : String(error))
        ));
      });
    });
  });
}

function stopChrome(browser) {
  return new Promise(function (resolve) {
    let settled = false;
    let killTimer = null;
    let hardKillTimer = null;

    function finish() {
      if (settled) return;
      settled = true;
      if (killTimer !== null) clearTimeout(killTimer);
      if (hardKillTimer !== null) clearTimeout(hardKillTimer);
      try {
        fs.rmSync(browser.userDataDir, { recursive: true, force: true });
      } catch (_) {
        /* best effort */
      }
      resolve();
    }

    if (browser.child.exitCode !== null ||
        browser.child.signalCode !== null) {
      finish();
      return;
    }

    browser.child.once("exit", finish);
    try { browser.child.kill("SIGTERM"); } catch (_) { /* already gone */ }

    killTimer = setTimeout(function () {
      try { browser.child.kill("SIGKILL"); } catch (_) { /* already gone */ }
      hardKillTimer = setTimeout(finish, 1000);
    }, 3000);
  });
}

async function waitForTarget(browser, expectedUrl, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (browser.child.exitCode !== null ||
        browser.child.signalCode !== null) {
      throw new Error(
        "Chrome exited before the DevTools target appeared (exit code " +
        browser.child.exitCode + ", signal " + browser.child.signalCode +
        "): " + browser.stderr()
      );
    }

    try {
      const response = await fetch(
        "http://" + HOST + ":" + browser.port + "/json/list",
        { signal: AbortSignal.timeout(2000) }
      );
      const targets = await response.json();
      const match = targets.find(function (target) {
        return target.type === "page" &&
          typeof target.url === "string" &&
          target.url.startsWith(expectedUrl);
      });
      if (match && match.webSocketDebuggerUrl) return match;
    } catch (_) {
      /* Chrome is still starting; retry until the deadline. */
    }
    await sleep(100);
  }
  throw new Error(
    "Chrome DevTools target did not appear for " + expectedUrl +
    " within " + timeoutMs + "ms; stderr: " + browser.stderr()
  );
}

function connectCdp(webSocketUrl) {
  return new Promise(function (resolve, reject) {
    const socket = new WebSocket(webSocketUrl);
    const pending = new Map();
    let nextId = 1;
    let opened = false;
    let finished = false;

    function rejectPending(error) {
      for (const entry of pending.values()) {
        clearTimeout(entry.timer);
        entry.reject(error);
      }
      pending.clear();
    }

    function failBeforeOpen(error) {
      if (finished) return;
      finished = true;
      clearTimeout(connectTimer);
      rejectPending(error);
      try { socket.close(); } catch (_) { /* already closed */ }
      reject(error);
    }

    const connectTimer = setTimeout(function () {
      failBeforeOpen(new Error(
        "Timed out opening the DevTools websocket after " +
        DEFAULT_CDP_TIMEOUT_MS + "ms"
      ));
    }, DEFAULT_CDP_TIMEOUT_MS);

    socket.onopen = function () {
      if (finished) return;
      opened = true;
      clearTimeout(connectTimer);
      resolve({
        send: function (method, params, timeoutMs) {
          return new Promise(function (resolveCommand, rejectCommand) {
            const id = nextId;
            nextId += 1;
            const commandTimeout = typeof timeoutMs === "number"
              ? timeoutMs
              : DEFAULT_CDP_TIMEOUT_MS;
            const timer = setTimeout(function () {
              pending.delete(id);
              rejectCommand(new Error(
                "CDP command timed out after " + commandTimeout +
                "ms: " + method
              ));
            }, commandTimeout);

            pending.set(id, {
              method: method,
              timer: timer,
              resolve: function (value) {
                clearTimeout(timer);
                resolveCommand(value);
              },
              reject: function (error) {
                clearTimeout(timer);
                rejectCommand(error);
              }
            });

            try {
              socket.send(JSON.stringify({
                id: id,
                method: method,
                params: params || {}
              }));
            } catch (error) {
              clearTimeout(timer);
              pending.delete(id);
              rejectCommand(error);
            }
          });
        },
        close: function () {
          try { socket.close(); } catch (_) { /* already closed */ }
        }
      });
    };

    socket.onmessage = function (event) {
      let text = event.data;
      if (typeof text !== "string") {
        try {
          text = Buffer.from(text).toString("utf8");
        } catch (_) {
          return;
        }
      }
      let message;
      try { message = JSON.parse(text); } catch (_) { return; }
      if (message.id === undefined || pending.has(message.id) === false) {
        return;
      }
      const entry = pending.get(message.id);
      pending.delete(message.id);
      clearTimeout(entry.timer);
      if (message.error) {
        entry.reject(new Error(
          "CDP " + entry.method + " failed: " +
          JSON.stringify(message.error)
        ));
      } else {
        entry.resolve(message.result || {});
      }
    };

    socket.onerror = function () {
      const error = new Error("DevTools websocket error");
      if (opened) rejectPending(error);
      else failBeforeOpen(error);
    };

    socket.onclose = function () {
      const error = new Error("DevTools websocket closed");
      if (opened) rejectPending(error);
      else failBeforeOpen(error);
    };
  });
}

async function cdpEvaluate(cdp, expression, awaitPromise, timeoutMs) {
  const result = await cdp.send("Runtime.evaluate", {
    expression: expression,
    awaitPromise: awaitPromise === true,
    returnByValue: true,
    userGesture: true
  }, timeoutMs);

  if (result.exceptionDetails) {
    throw new Error(
      "Runtime.evaluate exception: " + JSON.stringify(result.exceptionDetails)
    );
  }
  if (result.result === undefined) return undefined;
  return result.result.value;
}

async function waitForReport(cdp, timeoutMs) {
  const expression =
    "(async function () {" +
    "  var deadline = Date.now() + " + timeoutMs + ";" +
    "  while (Date.now() < deadline) {" +
    "    var report = window.__markdownTestReport;" +
    "    if (report && report.done === true) return report;" +
    "    await new Promise(function (resolve) { setTimeout(resolve, 50); });" +
    "  }" +
    "  return { done: false, timedOut: true, href: location.href };" +
    "})()";
  return cdpEvaluate(cdp, expression, true, timeoutMs + 5000);
}

function annotateChromeError(browser, error) {
  if (browser && typeof browser.stderr === "function") {
    let stderr = browser.stderr();
    if (stderr.length > 4000) {
      stderr = stderr.slice(0, 2500) +
        "\n[chrome stderr truncated]\n" +
        stderr.slice(-1000);
    }
    if (stderr && error && typeof error.message === "string" &&
        error.message.indexOf(stderr) === -1) {
      error.message = error.message + "\nChrome stderr excerpt:\n" + stderr;
      if (typeof error.stack === "string") {
        error.stack = error.stack + "\nChrome stderr excerpt:\n" + stderr;
      }
    }
  }
  return error;
}

async function runPageReport(port, pagePath, timeoutMs) {
  return runPageReportWithBrowser(port, pagePath, timeoutMs);
}

async function runPageReportWithBrowser(port, pagePath, timeoutMs) {
  const url = "http://" + HOST + ":" + port + pagePath;
  const browser = await launchChrome(url);
  try {
    const target = await waitForTarget(browser, url, TARGET_WAIT_TIMEOUT_MS);
    const cdp = await connectCdp(target.webSocketDebuggerUrl);
    try {
      await cdp.send("Runtime.enable", {}, 8000);
      return await waitForReport(cdp, timeoutMs);
    } finally {
      cdp.close();
    }
  } catch (error) {
    throw annotateChromeError(browser, error);
  } finally {
    await stopChrome(browser);
  }
}

async function waitForExpression(cdp, expression, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  const wrapped = "Boolean(" + expression + ")";
  while (Date.now() < deadline) {
    const value = await cdpEvaluate(cdp, wrapped, false, 5000);
    if (value === true) return true;
    await sleep(100);
  }
  throw new Error("Timed out waiting for expression: " + expression);
}

async function captureVisualViewport(port, width, height, outputPath) {
  const browser = await launchChrome("about:blank");
  try {
    const target = await waitForTarget(
      browser, "about:blank", TARGET_WAIT_TIMEOUT_MS
    );
    const cdp = await connectCdp(target.webSocketDebuggerUrl);
    try {
      await cdp.send("Page.enable", {}, 8000);
      await cdp.send("Runtime.enable", {}, 8000);
      await cdp.send("Emulation.setDeviceMetricsOverride", {
        width: width,
        height: height,
        deviceScaleFactor: 1,
        mobile: false
      }, 8000);

      const url = "http://" + HOST + ":" + port +
        "/tests/browser/markdown-renderer.visual.html";
      await cdp.send("Page.navigate", { url: url }, 8000);
      await waitForExpression(
        cdp,
        "document.readyState === 'complete' && " +
        "window.__visualReady === true && " +
        "Array.from(document.styleSheets).some(function (sheet) {" +
        "  return typeof sheet.href === 'string' && " +
        "    sheet.href.indexOf('/frontend/style.css') >= 0;" +
        "})",
        20000
      );

      const report = await cdpEvaluate(
        cdp, "window.__runMarkdownVisualChecks()", false, 10000
      );
      await cdpEvaluate(
        cdp,
        "(function () { var list = document.getElementById('visual-messages');" +
        " if (list) list.scrollTop = 0; return true; })()",
        false, 5000
      );
      const shot = await cdp.send("Page.captureScreenshot", {
        format: "png",
        fromSurface: true
      }, 10000);
      fs.mkdirSync(path.dirname(outputPath), { recursive: true });
      fs.writeFileSync(outputPath, Buffer.from(shot.data, "base64"));
      process.stdout.write("screenshot: " + outputPath + "\n");
      return report;
    } finally {
      cdp.close();
    }
  } catch (error) {
    throw annotateChromeError(browser, error);
  } finally {
    await stopChrome(browser);
  }
}

function validateReport(label, report, minimumTotal) {
  const problems = [];

  if (report === null ||
      (typeof report === "object") === false ||
      Array.isArray(report) === true) {
    problems.push(label + ": no report object was returned");
    return problems;
  }
  if (report.done === true) {
    /* completed */
  } else {
    problems.push(label + ": test suite did not finish (timed out or threw)");
  }
  if ((typeof report.executed === "number") === false ||
      report.executed < minimumTotal) {
    problems.push(label + ": executed count is below the expected minimum (" +
      minimumTotal + ")");
  }
  if ((typeof report.total === "number") === false ||
      report.total < minimumTotal) {
    problems.push(label + ": total count is below the expected minimum (" +
      minimumTotal + ")");
  }
  if ((typeof report.passed === "number") === false || report.passed < 1) {
    problems.push(label + ": no tests passed");
  }
  if ((typeof report.failed === "number") === false || report.failed === 0) {
    if ((typeof report.failed === "number") === false) {
      problems.push(label + ": failure count is missing");
    }
  } else {
    problems.push(label + ": failure count is nonzero");
  }
  if ((typeof report.total === "number") &&
      (typeof report.executed === "number") &&
      report.total === report.executed) {
    /* consistent */
  } else if ((typeof report.total === "number") &&
      (typeof report.executed === "number")) {
    problems.push(label + ": total does not equal executed");
  }
  if ((typeof report.total === "number") &&
      (typeof report.passed === "number") &&
      (typeof report.failed === "number") &&
      report.total === report.passed + report.failed) {
    /* consistent */
  } else if ((typeof report.total === "number") &&
      (typeof report.passed === "number") &&
      (typeof report.failed === "number")) {
    problems.push(label + ": total does not equal passed + failed");
  }
  if (Array.isArray(report.failures) === false) {
    problems.push(label + ": failures array is missing");
  } else if (report.failures.length > 0) {
    problems.push(label + ": " + JSON.stringify(report.failures, null, 2));
  }

  return problems;
}

function printReport(label, report) {
  const total = report && typeof report.total === "number"
    ? report.total : "?";
  const passed = report && typeof report.passed === "number"
    ? report.passed : "?";
  const failed = report && typeof report.failed === "number"
    ? report.failed : "?";
  process.stdout.write(
    label + ": executed=" + total +
    " passed=" + passed +
    " failed=" + failed + "\n"
  );
  if (report && Array.isArray(report.failures) &&
      report.failures.length > 0) {
    process.stdout.write(JSON.stringify(report.failures, null, 2) + "\n");
  }
}

async function main() {
  if (fs.existsSync(CHROME) === false) {
    throw new Error("Chrome not found at " + CHROME);
  }

  const staticServer = await startStaticServer();
  const port = staticServer.port;
  let failure = null;

  try {
    const integration = await runPageReport(
      port,
      "/tests/browser/markdown-renderer.integration.html",
      25000
    );
    printReport("integration", integration);

    const app = await runPageReport(
      port,
      "/tests/browser/markdown-renderer.app.html",
      30000
    );
    printReport("app", app);

    const desktopLayout = await captureVisualViewport(
      port, 1280, 900,
      path.join(SCREENSHOT_DIR, "markdown-desktop.png")
    );
    printReport("layout-desktop", desktopLayout);

    const narrowLayout = await captureVisualViewport(
      port, 375, 812,
      path.join(SCREENSHOT_DIR, "markdown-narrow.png")
    );
    printReport("layout-narrow", narrowLayout);

    const problems = [];
    problems.push.apply(problems, validateReport(
      "integration", integration, 27
    ));
    problems.push.apply(problems, validateReport("app", app, 11));
    problems.push.apply(problems, validateReport(
      "layout-desktop", desktopLayout, 10
    ));
    problems.push.apply(problems, validateReport(
      "layout-narrow", narrowLayout, 10
    ));

    if (problems.length > 0) {
      failure = new Error(
        "Markdown browser tests failed:\n" +
        problems.map(function (problem) { return " - " + problem; }).join("\n")
      );
    }
  } finally {
    try {
      await withTimeout(
        stopStaticServer(staticServer), 5000,
        "Static test server close"
      );
    } catch (error) {
      if (failure === null) failure = error;
    }
  }

  if (failure !== null) throw failure;
  process.stdout.write("Markdown browser tests passed.\n");
}

if (require.main === module) {
  main().catch(function (error) {
    const detail = error && error.stack ? error.stack : String(error);
    if (detail.indexOf(
      "sandbox initialization failed: Operation not permitted"
    ) >= 0) {
      process.stderr.write(
        "Browser sandbox initialization failed in this environment; " +
        "the runner does not retry with a disabled sandbox.\n"
      );
    }
    process.stderr.write("Markdown browser test runner failed: " +
      detail + "\n");
    process.exitCode = 1;
  });
}

module.exports = {
  ALLOWED_PATHS: ALLOWED_PATHS,
  startStaticServer: startStaticServer
};
