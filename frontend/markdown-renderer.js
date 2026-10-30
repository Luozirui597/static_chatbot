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

    function createRenderer(markedNamespace, purify) {
      var md = null;

      if (markedNamespace &&
          typeof markedNamespace.Marked === "function" &&
          purify &&
          purify.isSupported === true &&
          typeof purify.sanitize === "function") {
        try {
          md = new markedNamespace.Marked();
          md.use(buildRendererExtension());
        } catch (_) {
          md = null;
        }
      }

      var sanitizeConfig = buildSanitizeConfig();

      function render(markdown) {
        if (md === null || typeof markdown !== "string") {
          return { ok: false };
        }

        try {
          var parsed = md.parse(markdown);
          if (typeof parsed !== "string") return { ok: false };

          var clean = purify.sanitize(parsed, sanitizeConfig);
          if (typeof clean !== "string") return { ok: false };

          return { ok: true, html: clean };
        } catch (_) {
          return { ok: false };
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
        configured: md !== null,
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
          root ? root.DOMPurify : null
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
