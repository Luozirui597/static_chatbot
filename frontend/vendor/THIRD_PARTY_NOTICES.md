# Third-party notices

These files are vendored into the frontend so the application does not
load them from a CDN at runtime.

## marked 18.0.14

- Source: `https://registry.npmjs.org/marked/-/marked-18.0.14.tgz`
- Repository: `https://github.com/markedjs/marked` (tag `v18.0.14`)
- Browser build: `package/lib/marked.umd.js`
- Vendored as: `frontend/vendor/marked.umd.js`
- npm `dist.integrity`:
  `sha512-mBHK6FBHuBAlhgRe88w9F0O1AbwwXJUcQibUbC/QcdTbVGAD7aWza+xt3N6oT/jCZx3/OMeS+8rnuiHZcQ9s7A==`
- npm `dist.shasum` (SHA-1): `90c0ed46b3f9f1e99d8d96cc8b06937884adedc5`
- Tarball SHA-256:
  `f74bd78902b6e83c4c488d6ea979c15619a11dbd0d8e21f4246bdd357c19c091`
- Vendored browser file SHA-256:
  `21568877a938d2c4e7d74e27f18e60da96bb73a68809610ca39216e1efebae62`
- License: MIT. Full text: `frontend/vendor/LICENSE-marked.txt`
- Copyright banner retained in the generated UMD file.

## DOMPurify 3.4.16

- Source: `https://registry.npmjs.org/dompurify/-/dompurify-3.4.16.tgz`
- Repository: `https://github.com/cure53/DOMPurify` (tag `3.4.16`)
- Browser build: `package/dist/purify.min.js`
- Vendored as: `frontend/vendor/purify.min.js`
- npm `dist.integrity`:
  `sha512-sqo+pNp3qRhCIpbgRi1y8Tgk27Bo2Ry7w0dC1NBeNTdZChWjz9Xb/KOoZbRP/R6pQZ80Qw8YhXw13hWWBbMRnQ==`
- npm `dist.shasum` (SHA-1): `c51b3709ea15ee9abb07e804f2ef30ea9b953128`
- Tarball SHA-256:
  `d288ced25e6bda3e5fc76d1fc9bca899260cf886e005e84f232e686ada57f1b4`
- Vendored browser file SHA-256:
  `2c90a9b46d6463f26038a29b686e82bc91de01fdac9d5229e7cfe3b360134ea2`
- License: MPL-2.0 OR Apache-2.0. This project uses the Apache-2.0
  option. Full texts: `frontend/vendor/LICENSE-DOMPurify-Apache-2.0.txt`
  and `frontend/vendor/LICENSE-DOMPurify-MPL-2.0.txt`.
- `@license` banner retained in the minified browser build.

## Verification

The downloaded tarballs were checked against the npm registry
`dist.integrity` values above.  The vendored browser files were copied
byte-for-byte from those tarballs and their SHA-256 values are recorded
here.
