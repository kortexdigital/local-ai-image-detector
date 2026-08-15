import test from 'node:test';
import assert from 'node:assert/strict';
import zlib from 'node:zlib';

import { readGenerationSignals } from '../../extension/shared/metadata.js';

/* ---------- byte fixtures ---------- */

function crc32(buf) {
  let c;
  const table = [];
  for (let n = 0; n < 256; n += 1) {
    c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  let crc = 0xffffffff;
  for (const byte of buf) crc = table[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

function pngChunk(type, data) {
  const typeBytes = Buffer.from(type, 'latin1');
  const body = Buffer.concat([typeBytes, Buffer.from(data)]);
  const length = Buffer.alloc(4);
  length.writeUInt32BE(data.length, 0);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body), 0);
  return Buffer.concat([length, body, crc]);
}

function buildPng(chunks = []) {
  const signature = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
  const ihdrData = Buffer.alloc(13);
  ihdrData.writeUInt32BE(1, 0);
  ihdrData.writeUInt32BE(1, 4);
  ihdrData[8] = 8;
  ihdrData[9] = 2;
  const idat = pngChunk('IDAT', zlib.deflateSync(Buffer.from([0, 0, 0, 0])));
  return Buffer.concat([
    signature,
    pngChunk('IHDR', ihdrData),
    ...chunks,
    idat,
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
}

function textChunk(keyword, value) {
  return pngChunk('tEXt', Buffer.concat([
    Buffer.from(keyword, 'latin1'),
    Buffer.from([0]),
    Buffer.from(value, 'latin1'),
  ]));
}

/** Minimal JPEG carrying an APP1/Exif block with the given IFD0 tags. */
function buildJpegWithExif(tags) {
  const entries = [];
  const extra = [];
  let extraOffset = 8 + 2 + tags.length * 12 + 4;

  for (const [tag, type, value] of tags) {
    const entry = Buffer.alloc(12);
    entry.writeUInt16BE(tag, 0);
    entry.writeUInt16BE(type, 2);
    if (type === 2) {
      const str = Buffer.from(`${value}\0`, 'latin1');
      entry.writeUInt32BE(str.length, 4);
      if (str.length <= 4) {
        str.copy(entry, 8);
      } else {
        entry.writeUInt32BE(extraOffset, 8);
        extra.push(str);
        extraOffset += str.length;
      }
    } else {
      // RATIONAL, stored out of line as two longs.
      entry.writeUInt32BE(1, 4);
      entry.writeUInt32BE(extraOffset, 8);
      const rational = Buffer.alloc(8);
      rational.writeUInt32BE(1, 0);
      rational.writeUInt32BE(125, 4);
      extra.push(rational);
      extraOffset += 8;
    }
    entries.push(entry);
  }

  const header = Buffer.from('Exif\0\0', 'latin1');
  const tiff = Buffer.alloc(8);
  tiff.write('MM', 0, 'latin1');
  tiff.writeUInt16BE(42, 2);
  tiff.writeUInt32BE(8, 4);
  const count = Buffer.alloc(2);
  count.writeUInt16BE(tags.length, 0);
  const nextIfd = Buffer.alloc(4);

  const payload = Buffer.concat([header, tiff, count, ...entries, nextIfd, ...extra]);
  const app1Length = Buffer.alloc(2);
  app1Length.writeUInt16BE(payload.length + 2, 0);

  return Buffer.concat([
    Buffer.from([0xff, 0xd8]),
    Buffer.from([0xff, 0xe1]),
    app1Length,
    payload,
    Buffer.from([0xff, 0xd9]),
  ]);
}

const MAKE = 0x010f;
const MODEL = 0x0110;
const SOFTWARE = 0x0131;
const EXPOSURE_TIME = 0x829a;
const ASCII = 2;
const RATIONAL = 5;

/* ---------- tests ---------- */

test('a PNG parameters chunk is a generation tag', () => {
  const png = buildPng([textChunk('parameters', 'a photo of a cat, Steps: 20, Sampler: Euler')]);
  const signals = readGenerationSignals(png.buffer.slice(png.byteOffset, png.byteOffset + png.length));
  assert.equal(signals.generatorTag, true);
});

test('a ComfyUI workflow chunk is a generation tag', () => {
  const png = buildPng([textChunk('workflow', '{"nodes": []}')]);
  const signals = readGenerationSignals(png.buffer.slice(png.byteOffset, png.byteOffset + png.length));
  assert.equal(signals.generatorTag, true);
});

test('a plain PNG carries no generation tag', () => {
  const png = buildPng([textChunk('Comment', 'holiday photo')]);
  const signals = readGenerationSignals(png.buffer.slice(png.byteOffset, png.byteOffset + png.length));
  assert.equal(signals.generatorTag, false);
  assert.equal(signals.cameraExif, false);
});

test('EXIF Software naming a generator is a generation tag', () => {
  const jpeg = buildJpegWithExif([[SOFTWARE, ASCII, 'Midjourney v6']]);
  const signals = readGenerationSignals(jpeg.buffer.slice(jpeg.byteOffset, jpeg.byteOffset + jpeg.length));
  assert.equal(signals.generatorTag, true);
});

test('EXIF Software naming an ordinary editor is not a generation tag', () => {
  const jpeg = buildJpegWithExif([[SOFTWARE, ASCII, 'GIMP 2.10']]);
  const signals = readGenerationSignals(jpeg.buffer.slice(jpeg.byteOffset, jpeg.byteOffset + jpeg.length));
  assert.equal(signals.generatorTag, false);
});

test('a full camera EXIF triple is a camera signal', () => {
  const jpeg = buildJpegWithExif([
    [MAKE, ASCII, 'Canon'],
    [MODEL, ASCII, 'Canon EOS 5D Mark IV'],
    [EXPOSURE_TIME, RATIONAL, 0],
  ]);
  const signals = readGenerationSignals(jpeg.buffer.slice(jpeg.byteOffset, jpeg.byteOffset + jpeg.length));
  assert.equal(signals.cameraExif, true);
});

test('Make alone is not a camera signal', () => {
  const jpeg = buildJpegWithExif([[MAKE, ASCII, 'Canon']]);
  const signals = readGenerationSignals(jpeg.buffer.slice(jpeg.byteOffset, jpeg.byteOffset + jpeg.length));
  assert.equal(
    signals.cameraExif,
    false,
    'a lone Make tag is trivially added by re-saving and must not count',
  );
});

test('garbage bytes return all false and never throw', () => {
  const signals = readGenerationSignals(Uint8Array.from([1, 2, 3, 4, 5, 6, 7, 8]).buffer);
  assert.equal(signals.generatorTag, false);
  assert.equal(signals.cameraExif, false);
});

test('empty input returns all false and never throws', () => {
  const signals = readGenerationSignals(new ArrayBuffer(0));
  assert.equal(signals.generatorTag, false);
  assert.equal(signals.cameraExif, false);
});

test('a truncated PNG chunk length does not hang or throw', () => {
  const png = buildPng([textChunk('parameters', 'x')]);
  const truncated = png.subarray(0, png.length - 20);
  const signals = readGenerationSignals(
    truncated.buffer.slice(truncated.byteOffset, truncated.byteOffset + truncated.length),
  );
  assert.equal(typeof signals.generatorTag, 'boolean');
});
