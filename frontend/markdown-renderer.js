/* ------------------------------------------------------------------ */
/* Assistant Markdown renderer                                        */
/*                                                                    */
/* marked parses Markdown; DOMPurify sanitizes the generated HTML.    */
/* Both dependencies are vendored local files and are never fetched   */
/* from a CDN at runtime.                                             */
/*                                                                    */
/* This module deliberately does not call window.marked.use(),        */
/* window.marked.setOptions(), DOMPurify.setConfig() or any DOMPurify  */
/* hook.  It owns one private Marked instance; the public render/apply */
/* functions lazily bind that instance to the browser-global marked    */
/* and DOMPurify objects.                                              */
/* ------------------------------------------------------------------ */
(function (root, factory) {
  "use strict";

  var api = factory(root);

  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.TeachableMarkdownRenderer = api;
  }
})(
  typeof globalThis !== "undefined" ? globalThis : this,
  function (root) {
    "use strict";

    var MARKDOWN_CLASS = "message-content--markdown";
    var EXPECTED_PURIFY_VERSION = "3.4.16";

    var HTML_ESCAPE_MAP = {
      "&": "&amp;",
      "<": "&lt;",
      ">": "&gt;",
      '"': "&quot;",
      "'": "&#39;"
    };

    var HTML_ESCAPE_RE = /[&<>"']/g;

    /* Matches the escaping strategy used by Marked for text/attribute
       values: encode when the ampersand does not already start a valid
       character reference.  This avoids double-encoding entities that
       Marked left in the input. */
    var ENTITY_ESCAPE_RE =
      /[<>"']|&(?!(?:#\d{1,7}|#[xX][0-9a-fA-F]{1,6}|\w+);)/g;

    function escapeHtml(value) {
      return String(value).replace(HTML_ESCAPE_RE, function (ch) {
        return HTML_ESCAPE_MAP[ch];
      });
    }

    function escapeEntities(value) {
      return String(value).replace(ENTITY_ESCAPE_RE, function (ch) {
        return HTML_ESCAPE_MAP[ch];
      });
    }

    function escapeRawHtmlText(value) {
      return escapeHtml(value).replace(/\r\n|\r|\n/g, "<br>\n");
    }

    function stringValue(value, fallback) {
      return typeof value === "string" ? value : (fallback || "");
    }

    function buildSanitizeConfig() {
      return {
        ALLOWED_TAGS: [
          "a", "blockquote", "br", "code", "del", "em",
          "h1", "h2", "h3", "h4", "h5", "h6", "hr",
          "li", "ol", "p", "pre", "strong",
          "table", "tbody", "td", "th", "thead", "tr", "ul"
        ],
        ALLOWED_ATTR: ["href", "title", "target", "rel", "align", "start"],
        /* target/rel/align/start are inert, fixed attributes produced by
           our own renderer.  Mark them as URI-safe so the strict
           ALLOWED_URI_REGEXP does not reject values such as "_blank". */
        ADD_URI_SAFE_ATTR: ["target", "rel", "align", "start"],
        ALLOWED_URI_REGEXP: /^(?:(?:https?|mailto):)/i,
        ALLOWED_NAMESPACES: ["http://www.w3.org/1999/xhtml"],
        ALLOW_DATA_ATTR: false,
        ALLOW_ARIA_ATTR: false,
        ALLOW_UNKNOWN_PROTOCOLS: false,
        RETURN_DOM: false,
        RETURN_DOM_FRAGMENT: false,
        RETURN_TRUSTED_TYPE: false,
        KEEP_CONTENT: true
      };
    }

    function normalizeHref(rawHref) {
      if (typeof rawHref !== "string") return null;

      var value = rawHref.trim();
      if (!/^(?:https?|mailto):/i.test(value)) return null;

      try {
        var URLClass = root && root.URL ? root.URL : null;
        if (typeof URLClass !== "function") return null;

        var url = new URLClass(value);
        var protocol = String(url.protocol || "").toLowerCase();
        if (protocol !== "http:" && protocol !== "https:" &&
            protocol !== "mailto:") {
          return null;
        }
        return url.href;
      } catch (_) {
        return null;
      }
    }

    function buildRendererExtension() {
      return {
        async: false,
        gfm: true,
        breaks: true,
        pedantic: false,
        renderer: {
          /**
           * Raw HTML must never execute.  Return escaped literal text.
           * Block-level raw HTML is wrapped in a paragraph so it keeps a
           * stable block position; line breaks remain visible as <br>.
           */
          html: function (token) {
            var text = stringValue(token && token.text, "");
            var body = escapeRawHtmlText(text);
            return token && token.block
              ? "<p>" + body + "</p>\n"
              : body;
          },

          /**
           * Images never become <img>.  Flatten the alt text through
           * Marked's TextRenderer, escape it, and return it as text.
           */
          image: function (token) {
            var alt = stringValue(token && token.text, "");

            try {
              if (token && token.tokens && this && this.parser &&
                  this.parser.textRenderer) {
                var flattened = this.parser.parseInline(
                  token.tokens, this.parser.textRenderer
                );
                if (typeof flattened === "string") alt = flattened;
              }
            } catch (_) {
              /* Fall back to token.text. */
            }

            return escapeEntities(alt);
          },

          /**
           * Validate the link scheme before emitting an anchor.  Any
           * dangerous, relative or fragment-only destination degrades
           * to the visible label text.  Safe links always get fixed
           * target/rel attributes.
           */
          link: function (token) {
            var label;

            if (token && token.autolink) {
              label = escapeHtml(stringValue(
                token.text || token.href, ""
              ));
            } else {
              try {
                label = this.parser.parseInline(
                  (token && token.tokens) || []
                );
              } catch (_) {
                label = escapeEntities(stringValue(token && token.text, ""));
              }
            }

            if (typeof label !== "string") {
              label = escapeEntities(stringValue(token && token.text, ""));
            }

            var safeHref = normalizeHref(token && token.href);
            if (safeHref === null) return label;

            var out = '<a href="' + escapeEntities(safeHref) + '"';
            if (token && typeof token.title === "string" && token.title) {
              out += ' title="' + escapeEntities(token.title) + '"';
            }
            out += ' target="_blank" rel="noopener noreferrer">' +
              label + "</a>";
            return out;
          }
        }
      };
    }

    function createRenderer(markedNamespace, purify, katexNamespace) {
      var MATH_NS = "http://www.w3.org/1998/Math/MathML";
      var XHTML_NS = "http://www.w3.org/1999/xhtml";
      var MATH_TAGS = [
        "math", "semantics", "annotation", "mrow", "mi", "mn", "mo",
        "mtext", "mspace", "msub", "msup", "msubsup", "mfrac", "msqrt",
        "mroot", "mover", "munder", "munderover"
      ];
      var MATH_ATTRS = [
        "xmlns", "encoding", "display", "accent", "accentunder",
        "mathvariant", "stretchy", "largeop", "movablelimits", "fence",
        "separator", "displaystyle", "scriptlevel", "linethickness",
        "lspace", "rspace", "width", "height", "depth"
      ];
      var MATH_MAX_SCAN = 8000;
      var MATH_MAX_EXPR = 4000;
      var MATH_MAX_COUNT = 20;
      var MATH_MAX_TOTAL = 20000;
      var MATH_MAX_PLACEHOLDERS = 128;
      var MATH_MAX_CANDIDATES = 512;
      var doc = root ? root.document : null;

      var canConfigure = !!(markedNamespace &&
        typeof markedNamespace.Marked === "function" &&
        purify &&
        purify.isSupported === true &&
        typeof purify.sanitize === "function");
      var sanitizeConfig = buildSanitizeConfig();
      var mathSanitizeConfig = {
        ALLOWED_TAGS: MATH_TAGS.slice(),
        ALLOWED_ATTR: MATH_ATTRS.slice(),
        ALLOWED_NAMESPACES: [XHTML_NS, MATH_NS],
        ALLOW_DATA_ATTR: false,
        ALLOW_ARIA_ATTR: false,
        ALLOW_UNKNOWN_PROTOCOLS: false,
        RETURN_DOM: false,
        RETURN_DOM_FRAGMENT: false,
        RETURN_TRUSTED_TYPE: false,
        KEEP_CONTENT: true
      };

      function template(html) {
        var t = doc.createElement("template");
        t.innerHTML = html;
        return t;
      }

      function signatureValue(node) {
        if (node.nodeType === 3) return ["text", node.nodeValue];
        if (node.nodeType !== 1) return ["other", node.nodeType];
        var attrs = [];
        for (var i = 0; i < node.attributes.length; i += 1) {
          var attr = node.attributes[i];
          if (attr.name === "xmlns") continue;
          attrs.push([attr.namespaceURI || "", attr.name, attr.value]);
        }
        attrs.sort(function (a, b) {
          return JSON.stringify(a).localeCompare(JSON.stringify(b));
        });
        var children = [];
        for (var j = 0; j < node.childNodes.length; j += 1) {
          children.push(signatureValue(node.childNodes[j]));
        }
        return [node.namespaceURI, node.localName, attrs, children];
      }

      function signature(node) {
        return JSON.stringify(signatureValue(node));
      }

      function validateMath(math) {
        if (!math || math.localName !== "math" ||
            math.namespaceURI !== MATH_NS) return false;
        var all = [math].concat(Array.prototype.slice.call(
          math.querySelectorAll("*")
        ));
        var arity = {
          mfrac: 2, msub: 2, msup: 2, msubsup: 3,
          mover: 2, munder: 2, munderover: 3, mroot: 2
        };
        if (math.children.length !== 1 ||
            math.firstElementChild.localName !== "semantics") return false;
        var semantics = math.firstElementChild;
        if (semantics.children.length !== 2 ||
            semantics.lastElementChild.localName !== "annotation" ||
            semantics.lastElementChild.getAttribute("encoding") !==
              "application/x-tex") return false;
        return all.every(function (el) {
          if (el.namespaceURI !== MATH_NS ||
              MATH_TAGS.indexOf(el.localName) < 0) return false;
          if (arity[el.localName] !== undefined &&
              el.children.length !== arity[el.localName]) return false;
          if (el.localName === "msqrt" && el.children.length < 1) return false;
          for (var i = 0; i < el.attributes.length; i += 1) {
            if (MATH_ATTRS.indexOf(el.attributes[i].name) < 0) return false;
          }
          return true;
        });
      }

      function renderMath(item) {
        if (!katexNamespace ||
            typeof katexNamespace.renderToString !== "function") {
          return { reason: "missing-katex" };
        }
        if (!doc) return { reason: "missing-document" };
        var blocked = [];
        var out;
        try {
          out = katexNamespace.renderToString(item.tex, {
            output: "mathml",
            displayMode: item.display,
            throwOnError: true,
            strict: "ignore",
            maxSize: 10,
            maxExpand: 1000,
            globalGroup: false,
            macros: {},
            trust: function (context) {
              blocked.push(context.command);
              return false;
            }
          });
        } catch (err) {
          return { reason: "parse-error", errorType: err && err.name };
        }
        if (blocked.length) {
          return { reason: "trust-rejected", blocked: blocked };
        }
        var before = template(out).content;
        var maths = before.querySelectorAll("math");
        if (maths.length !== 1 || !validateMath(maths[0])) {
          return { reason: "unsupported-structure" };
        }
        var expected = signature(maths[0]);
        var clean = purify.sanitize(maths[0].outerHTML, mathSanitizeConfig);
        if (typeof clean !== "string") {
          return { reason: "math-cleaning-failed" };
        }
        var fragment = template(clean).content;
        var roots = Array.prototype.slice.call(fragment.childNodes).filter(
          function (n) {
            return n.nodeType !== 3 || String(n.nodeValue).trim() !== "";
          }
        );
        if (roots.length !== 1 || !validateMath(roots[0]) ||
            signature(roots[0]) !== expected) {
          return { reason: "math-cleaning-changed-structure" };
        }
        var wrapper = doc.createElement("span");
        wrapper.className = item.display ? "math-display" : "math-inline";
        wrapper.appendChild(roots[0]);
        return { node: wrapper };
      }

      function fallbackNode(item) {
        var code = doc.createElement("code");
        code.textContent = item.raw;
        if (!item.display) return code;
        var pre = doc.createElement("pre");
        pre.className = "math-fallback";
        pre.appendChild(code);
        return pre;
      }

      function createMathContext(source) {
        var nonce;
        do {
          nonce = "MATHPH" +
            Math.random().toString(36).slice(2).toUpperCase() + "Q";
        } while (source.indexOf(nonce) >= 0);
        return {
          nonce: nonce,
          items: [],
          count: 0,
          total: 0,
          scanCount: 0,
          scannedChars: 0,
          maxSearch: 0,
          placeholderLimit: false,
          lastBlockSource: null,
          lastInlineSource: null,
          inlineScopes: []
        };
      }

      function boundedClose(src, close, ctx, allowHtmlBoundary, openLength) {
        var limit = openLength + MATH_MAX_SCAN + close.length;
        var windowText = src.slice(openLength, limit);
        ctx.scanCount += 1;
        ctx.scannedChars += windowText.length;
        ctx.maxSearch = Math.max(ctx.maxSearch, windowText.length);
        for (var i = 0; i + close.length <= windowText.length; i += 1) {
          if (allowHtmlBoundary && windowText.charAt(i) === "<" &&
              windowText.charAt(i + 1) === "/") {
            return { end: -1, reason: "html-boundary", boundary: i };
          }
          if (windowText.slice(i, i + close.length) !== close) continue;
          var escapes = 0;
          var j = i - 1;
          while (j >= 0 && windowText.charAt(j) === "\\") {
            escapes += 1;
            j -= 1;
          }
          if (escapes % 2 !== 0) continue;
          if (i <= MATH_MAX_SCAN) {
            return { end: openLength + i + close.length };
          }
          return { end: -1, reason: "scan-limit" };
        }
        return {
          end: -1,
          reason: src.length > limit ? "scan-limit" : "unclosed"
        };
      }

      function addMathItem(ctx, type, raw, tex, display, reason, unresolved) {
        if (ctx.items.length >= MATH_MAX_CANDIDATES) {
          throw new Error("candidate-limit");
        }
        var item = {
          raw: raw,
          tex: tex,
          display: display,
          reason: reason,
          id: ctx.nonce + ctx.items.length + "Z",
          unresolved: unresolved === true,
          excluded: false,
          disabled: false
        };
        ctx.items.push(item);
        return { type: type, raw: raw, item: item };
      }

      function scanMath(src, display, ctx, allowHtmlBoundary, open, close) {
        if (src.indexOf(open) !== 0) return;
        var found = boundedClose(
          src, close, ctx, allowHtmlBoundary, open.length
        );
        if (found.end < 0) {
          var failedRaw = found.reason === "html-boundary"
            ? src.slice(0, open.length + found.boundary)
            : open;
          return addMathItem(
            ctx, display ? "mathBlock" : "mathInline",
            failedRaw, "", display, found.reason, true
          );
        }
        var raw = src.slice(0, found.end);
        var tex = src.slice(open.length, found.end - close.length);
        var reason = null;
        if (!tex.trim()) reason = "empty";
        else if (tex.length > MATH_MAX_EXPR) reason = "expr-limit";
        return addMathItem(
          ctx, display ? "mathBlock" : "mathInline",
          raw, tex, display, reason, false
        );
      }

      function validDollarInlineContent(tex) {
        return Boolean(tex.trim()) &&
          !/^\s|\s$/.test(tex) &&
          tex.indexOf("\n") < 0;
      }

      function findDollarInlineStart(src) {
        var at = findUnescaped(src, "$", 0);
        while (at >= 0) {
          if (src.charAt(at + 1) === "$") {
            at = findUnescaped(src, "$", at + 2);
            continue;
          }
          var close = findUnescaped(src, "$", at + 1);
          if (close < 0) return -1;
          var tex = src.slice(at + 1, close);
          if (!validDollarInlineContent(tex)) {
            at = findUnescaped(src, "$", close + 1);
            continue;
          }
          return at;
        }
        return -1;
      }

      function scanDollarInline(src, ctx, allowHtmlBoundary) {
        if (src.charAt(0) !== "$" || src.charAt(1) === "$") return;
        var found = boundedClose(src, "$", ctx, allowHtmlBoundary, 1);
        if (found.end < 0) {
          if (found.reason === "scan-limit" ||
              found.reason === "html-boundary") {
            return addMathItem(
              ctx, "mathInline", "$", "", false, found.reason, true
            );
          }
          return;
        }
        var tex = src.slice(1, found.end - 1);
        if (!validDollarInlineContent(tex)) return;
        var reason = tex.length > MATH_MAX_EXPR ? "expr-limit" : null;
        return addMathItem(
          ctx, "mathInline", src.slice(0, found.end), tex, false, reason, false
        );
      }

      function scanDollarBlock(src, ctx) {
        if (src.indexOf("$$") !== 0) return;
        var found = boundedClose(src, "$$", ctx, false, 2);
        if (found.end < 0) {
          var failedRaw = found.reason === "html-boundary"
            ? src.slice(0, 2 + found.boundary)
            : "$$";
          return addMathItem(
            ctx, "mathBlock", failedRaw, "", true, found.reason, true
          );
        }
        var tex = src.slice(2, found.end - 2);
        var reason = null;
        if (!tex.trim()) reason = "empty";
        else if (tex.length > MATH_MAX_EXPR) reason = "expr-limit";
        return addMathItem(
          ctx, "mathBlock", src.slice(0, found.end), tex, true, reason, false
        );
      }

      function updateHtmlStack(raw, stack) {
        var match = /^<\s*(\/?)\s*([a-zA-Z][\w:-]*)/.exec(String(raw || ""));
        if (!match) return;
        var tag = match[2].toLowerCase();
        if (match[1]) {
          var at = stack.lastIndexOf(tag);
          if (at >= 0) stack.length = at;
          return;
        }
        var voidTags = ["br", "img", "hr", "input", "meta", "link", "area",
          "base", "col", "embed", "param", "source", "track", "wbr"];
        if (!/\/\s*>$/.test(raw) && voidTags.indexOf(tag) < 0) {
          stack.push(tag);
        }
      }

      function guardInlineMath(tokens, stack, disabled) {
        (tokens || []).forEach(function (token) {
          if (token.type === "html") {
            updateHtmlStack(token.raw, stack);
          } else if (token.item) {
            if (disabled) {
              token.item.excluded = true;
              token.item.disabled = true;
            } else if (stack.length) {
              token.item.excluded = true;
              token.item.reason = "raw-html";
            }
          } else if (token.tokens) {
            var excludedChild = token.type === "link" ||
              token.type === "image";
            guardInlineMath(
              token.tokens,
              excludedChild ? stack.slice() : stack,
              disabled || excludedChild
            );
          }
        });
      }

      function guardBlockMath(tokens) {
        (tokens || []).forEach(function (token) {
          if (token.type === "list") {
            (token.items || []).forEach(function (item) {
              guardBlockMath(item.tokens);
            });
          } else if (token.type === "table") {
            (token.header || []).forEach(function (cell) {
              guardInlineMath(cell.tokens, [], false);
            });
            (token.rows || []).forEach(function (row) {
              row.forEach(function (cell) {
                guardInlineMath(cell.tokens, [], false);
              });
            });
          } else if (token.type === "blockquote") {
            guardBlockMath(token.tokens);
          } else if (token.type !== "html" && token.type !== "code" &&
                     token.type !== "mathInline" &&
                     token.type !== "mathBlock" && token.tokens) {
            guardInlineMath(token.tokens, [], false);
          }
        });
      }

      function processScopeHtml(scope) {
        for (var i = scope.seen; i < scope.tokens.length; i += 1) {
          var token = scope.tokens[i];
          if (token.type === "html") {
            updateHtmlStack(token.raw, scope.stack);
          }
        }
        scope.seen = scope.tokens.length;
      }

      function inlineHtmlContext(ctx, src, tokens, inLink) {
        var scopes = ctx.inlineScopes;
        var at = -1;
        for (var i = scopes.length - 1; i >= 0; i -= 1) {
          if (scopes[i].tokens === tokens) {
            at = i;
            break;
          }
        }
        var scope;
        if (at >= 0) {
          scopes.length = at + 1;
          scope = scopes[at];
          scope.source = src;
        } else {
          var parent = scopes[scopes.length - 1];
          var nested = false;
          if (parent && typeof parent.source === "string") {
            var offset = parent.source.indexOf(src);
            nested = offset === 1 || offset === 2;
          }
          var stack = nested ? parent.stack : [];
          if (inLink) stack = stack.slice();
          scope = {
            tokens: tokens || [],
            source: src,
            seen: 0,
            stack: stack,
            inLink: inLink === true
          };
          if (nested) {
            scopes.push(scope);
          } else {
            scopes.length = 0;
            scopes.push(scope);
          }
        }
        processScopeHtml(scope);
        return {
          depth: scope.stack.length,
          allow: !scope.inLink && scope.stack.length > 0
        };
      }

      function applyMathBudget(ctx) {
        var active = ctx.items.filter(function (item) {
          return !item.excluded;
        });
        if (active.length > MATH_MAX_PLACEHOLDERS) {
          ctx.placeholderLimit = true;
          return;
        }
        var count = 0;
        var total = 0;
        for (var i = 0; i < active.length; i += 1) {
          var item = active[i];
          if (item.reason || item.unresolved) continue;
          if (count >= MATH_MAX_COUNT) {
            item.reason = "count-limit";
            continue;
          }
          if (total + item.tex.length > MATH_MAX_TOTAL) {
            item.reason = "total-limit";
            continue;
          }
          count += 1;
          total += item.tex.length;
        }
        ctx.count = count;
        ctx.total = total;
      }

      function findUnescaped(src, opener, start) {
        var at = src.indexOf(opener, start || 0);
        while (at >= 0) {
          var escapes = 0;
          var before = at - 1;
          while (before >= 0 && src.charAt(before) === "\\") {
            escapes += 1;
            before -= 1;
          }
          if (escapes % 2 === 0) return at;
          at = src.indexOf(opener, at + opener.length);
        }
        return -1;
      }

      function findStartIndex(slice, fullSource, opener) {
        var source = slice;
        if (fullSource && fullSource.slice(1) === slice) {
          source = fullSource;
        }
        var at = findUnescaped(source, opener);
        if (at < 0) return -1;
        if (source !== slice) {
          return at === 0 ? -1 : at - 1;
        }
        return at;
      }

      function mathTokenHtml(token) {
        if (token.item.excluded || token.item.disabled) {
          return escapeHtml(token.item.raw);
        }
        return token.item.id;
      }

      function createMarkedInstance(ctx) {
        var md = new markedNamespace.Marked();
        md.use(buildRendererExtension());
        md.use({
          extensions: [
            {
              name: "mathInline",
              level: "inline",
              start: function (src) {
                var at = findStartIndex(
                  src, ctx.lastInlineSource, "\\("
                );
                return at < 0 ? undefined : at;
              },
              tokenizer: function (src, tokens) {
                ctx.lastInlineSource = src;
                var html = inlineHtmlContext(
                  ctx, src, tokens,
                  this.lexer && this.lexer.state &&
                    this.lexer.state.inLink === true
                );
                return scanMath(src, false, ctx, html.allow, "\\(", "\\)");
              },
              renderer: mathTokenHtml
            },
            {
              name: "mathInlineDollar",
              level: "inline",
              start: function (src) {
                var at = findDollarInlineStart(src);
                return at < 0 ? undefined : at;
              },
              tokenizer: function (src, tokens) {
                var html = inlineHtmlContext(
                  ctx, src, tokens,
                  this.lexer && this.lexer.state &&
                    this.lexer.state.inLink === true
                );
                return scanDollarInline(src, ctx, html.allow);
              },
              renderer: mathTokenHtml
            },
            {
              name: "mathBlockDollar",
              level: "block",
              start: function (src) {
                var at = findUnescaped(src, "$$", 0);
                return at < 0 ? undefined : at;
              },
              tokenizer: function (src) {
                return scanDollarBlock(src, ctx);
              },
              renderer: mathTokenHtml
            },
            {
              name: "mathBlock",
              level: "block",
              start: function (src) {
                var at = findStartIndex(
                  src, ctx.lastBlockSource, "\\["
                );
                return at < 0 ? undefined : at;
              },
              tokenizer: function (src) {
                ctx.lastBlockSource = src;
                return scanMath(src, true, ctx, false, "\\[", "\\]");
              },
              renderer: mathTokenHtml
            }
          ],
          hooks: {
            processAllTokens: function (tokens) {
              guardBlockMath(tokens);
              applyMathBudget(ctx);
              return tokens;
            }
          }
        });
        return md;
      }

      function countOccurrences(text, needle) {
        var count = 0;
        var at = 0;
        while ((at = text.indexOf(needle, at)) >= 0) {
          count += 1;
          at += needle.length;
        }
        return count;
      }

      function collectPlaceholders(rootNode, items) {
        var nodes = [];
        var walker = doc.createTreeWalker(rootNode, 4);
        var node;
        while ((node = walker.nextNode())) nodes.push(node);
        var all = [rootNode].concat(Array.prototype.slice.call(
          rootNode.querySelectorAll("*")
        ));
        for (var a = 0; a < all.length; a += 1) {
          if (all[a].nodeType !== 1) continue;
          for (var b = 0; b < all[a].attributes.length; b += 1) {
            for (var c = 0; c < items.length; c += 1) {
              if (all[a].attributes[b].value.indexOf(items[c].id) >= 0) {
                return { ok: false, reason: "placeholder-in-attribute" };
              }
            }
          }
        }
        for (var i = 0; i < items.length; i += 1) {
          var hits = 0;
          var parent = null;
          for (var n = 0; n < nodes.length; n += 1) {
            var occurrences = countOccurrences(nodes[n].nodeValue, items[i].id);
            if (occurrences > 0) {
              hits += occurrences;
              parent = nodes[n].parentElement;
            }
          }
          if (hits !== 1) return { ok: false, reason: "placeholder-count" };
          if (parent && parent.closest &&
              parent.closest("pre,code,a")) {
            return { ok: false, reason: "placeholder-context" };
          }
        }
        return { ok: true };
      }

      function replacePlaceholder(rootNode, item, replacement) {
        var walker = doc.createTreeWalker(rootNode, 4);
        var node;
        while ((node = walker.nextNode())) {
          var at = node.nodeValue.indexOf(item.id);
          if (at < 0) continue;
          var fragment = doc.createDocumentFragment();
          fragment.appendChild(doc.createTextNode(
            node.nodeValue.slice(0, at)
          ));
          fragment.appendChild(replacement);
          fragment.appendChild(doc.createTextNode(
            node.nodeValue.slice(at + item.id.length)
          ));
          node.parentNode.replaceChild(fragment, node);
          return true;
        }
        return false;
      }

      function render(markdown) {
        if (!canConfigure || typeof markdown !== "string") {
          return { ok: false };
        }
        var ctx = createMathContext(markdown);
        try {
          var md = createMarkedInstance(ctx);
          var parsed = md.parse(markdown);
          if (typeof parsed !== "string") return { ok: false };
          if (ctx.placeholderLimit) return { ok: false };

          /* Do not let an unresolved delimiter in an excluded region
             (code/link/raw HTML/image) fail the whole message.  The guard
             has already run in processAllTokens by this point. */
          var unresolved = ctx.items.some(function (item) {
            return item.unresolved && !item.excluded;
          });
          if (unresolved) return { ok: false };

          var clean = purify.sanitize(parsed, sanitizeConfig);
          if (typeof clean !== "string") return { ok: false };

          var active = ctx.items.filter(function (item) {
            return !item.excluded;
          });
          if (active.length === 0) return { ok: true, html: clean };

          var parsedTemplate = template(clean);
          var refs = collectPlaceholders(parsedTemplate.content, active);
          if (!refs.ok) return { ok: false };

          var acceptedMath = [];
          for (var i = 0; i < active.length; i += 1) {
            var item = active[i];
            var replacement = null;
            if (!item.reason) {
              var mathResult;
              try {
                mathResult = renderMath(item);
              } catch (_) {
                mathResult = { reason: "math-render-error" };
              }
              if (mathResult.node) {
                replacement = mathResult.node;
                acceptedMath.push(replacement.firstElementChild);
              } else {
                replacement = fallbackNode(item);
              }
            } else {
              replacement = fallbackNode(item);
            }
            if (!replacePlaceholder(parsedTemplate.content, item, replacement)) {
              return { ok: false };
            }
          }

          if (parsedTemplate.content.textContent.indexOf(ctx.nonce) >= 0) {
            return { ok: false };
          }
          var holder = doc.createElement("div");
          holder.appendChild(parsedTemplate.content.cloneNode(true));
          var html = holder.innerHTML;
          if (html.indexOf(ctx.nonce) >= 0) return { ok: false };

          var reparsed = template(html);
          var finalMath = Array.prototype.slice.call(
            holder.querySelectorAll("math")
          );
          var reparsedMath = Array.prototype.slice.call(
            reparsed.content.querySelectorAll("math")
          );
          if (finalMath.length !== acceptedMath.length ||
              reparsedMath.length !== finalMath.length) {
            return { ok: false };
          }
          for (var m = 0; m < finalMath.length; m += 1) {
            if (!validateMath(finalMath[m])) return { ok: false };
          }
          for (var r = 0; r < reparsedMath.length; r += 1) {
            if (!validateMath(reparsedMath[r])) return { ok: false };
          }
          var expectedSignatures = acceptedMath.map(signature).sort();
          var finalSignatures = finalMath.map(signature).sort();
          var reparsedSignatures = reparsedMath.map(signature).sort();
          if (JSON.stringify(expectedSignatures) !==
              JSON.stringify(finalSignatures) ||
              JSON.stringify(expectedSignatures) !==
              JSON.stringify(reparsedSignatures)) {
            return { ok: false };
          }
          return { ok: true, html: html };
        } catch (_) {
          return { ok: false };
        } finally {
          ctx = null;
        }
      }

      function fallbackToText(contentEl, markdown) {
        try {
          if (contentEl && contentEl.classList &&
              typeof contentEl.classList.remove === "function") {
            contentEl.classList.remove(MARKDOWN_CLASS);
          }
        } catch (_) {
          /* Best-effort class cleanup. */
        }

        try {
          if (contentEl) contentEl.textContent = markdown;
        } catch (_) {
          /* Nothing else can be done. */
        }

        return false;
      }

      function apply(contentEl, markdown) {
        if (!contentEl || typeof contentEl !== "object") return false;

        var result = render(markdown);
        if (!result.ok) return fallbackToText(contentEl, markdown);

        try {
          if (!contentEl.classList ||
              typeof contentEl.classList.add !== "function") {
            return fallbackToText(contentEl, markdown);
          }

          contentEl.classList.add(MARKDOWN_CLASS);
          contentEl.innerHTML = result.html;
          return true;
        } catch (_) {
          return fallbackToText(contentEl, markdown);
        }
      }

      return {
        configured: canConfigure,
        purifyVersion: purify && purify.version ? purify.version : null,
        render: render,
        apply: apply
      };
    }


    var globalRenderer = null;

    function getGlobalRenderer() {
      if (globalRenderer === null) {
        globalRenderer = createRenderer(
          root ? root.marked : null,
          root ? root.DOMPurify : null,
          root ? root.katex : null
        );
      }
      return globalRenderer;
    }

    function render(markdown) {
      try {
        return getGlobalRenderer().render(markdown);
      } catch (_) {
        return { ok: false };
      }
    }

    function apply(contentEl, markdown) {
      try {
        return getGlobalRenderer().apply(contentEl, markdown);
      } catch (_) {
        try {
          if (contentEl && contentEl.classList &&
              typeof contentEl.classList.remove === "function") {
            contentEl.classList.remove(MARKDOWN_CLASS);
          }
          if (contentEl) contentEl.textContent = markdown;
        } catch (_) {
          /* Best-effort fallback. */
        }
        return false;
      }
    }

    return {
      MARKDOWN_CLASS: MARKDOWN_CLASS,
      PURIFY_VERSION: EXPECTED_PURIFY_VERSION,
      createRenderer: createRenderer,
      render: render,
      apply: apply
    };
  }
);
