/**
 * Reading what an image file says about its own origin.
 *
 * This is cheap, exact where it fires, and completely local. Generative tools
 * routinely leave their fingerprints in the container: Automatic1111 writes
 * the full prompt into a PNG `parameters` text chunk, ComfyUI embeds its
 * workflow JSON, several products write a C2PA manifest, and many name
 * themselves in EXIF Software.
 *
 * Everything here is defensive. These parsers run on arbitrary bytes from
 * arbitrary websites, so a malformed file must return "no signal" rather than
 * throw or loop.
 */

/** PNG text keywords that mean the file carries generation parameters. */
const GENERATION_KEYWORDS = new Set([
  'parameters',
  'prompt',
  'workflow',
  'sd-metadata',
  'invokeai_metadata',
  'dream',
  'comfyui',
  'generation_data',
]);

/** Substrings in EXIF/XMP Software that identify a generator. */
const GENERATOR_SOFTWARE = [
  'midjourney',
  'stable diffusion',
  'stablediffusion',
  'automatic1111',
  'comfyui',
  'dall-e',
  'dalle',
  'openai',
  'firefly',
  'adobe firefly',
  'imagen',
  'ideogram',
  'leonardo.ai',
  'novelai',
  'flux',
  'gemini',
  'nano banana',
  'grok',
  'invokeai',
  'niji',
];

const EMPTY = Object.freeze({ generatorTag: false, cameraExif: false, details: '' });

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];

function isPng(view) {
  if (view.byteLength < 8) return false;
  return PNG_SIGNATURE.every((byte, index) => view.getUint8(index) === byte);
}

function isJpeg(view) {
  return view.byteLength >= 2 && view.getUint8(0) === 0xff && view.getUint8(1) === 0xd8;
}

function latin1(bytes) {
  let out = '';
  for (const byte of bytes) out += String.fromCharCode(byte);
  return out;
}

function namesAGenerator(text) {
  const lowered = text.toLowerCase();
  return GENERATOR_SOFTWARE.some((needle) => lowered.includes(needle));
}

/* ---------------------------- PNG ---------------------------- */

function readPngSignals(bytes, view) {
  let offset = 8;
  let generatorTag = false;
  let details = '';

  // A corrupt length field could otherwise walk backwards forever.
  while (offset + 8 <= view.byteLength) {
    const length = view.getUint32(offset);
    const type = latin1(bytes.subarray(offset + 4, offset + 8));
    const dataStart = offset + 8;
    const dataEnd = dataStart + length;
    if (length < 0 || dataEnd > view.byteLength) break;

    if (type === 'tEXt' || type === 'iTXt' || type === 'zTXt') {
      const chunk = bytes.subarray(dataStart, dataEnd);
      const nul = chunk.indexOf(0);
      const keyword = latin1(chunk.subarray(0, nul === -1 ? chunk.length : nul));
      if (GENERATION_KEYWORDS.has(keyword.toLowerCase().trim())) {
        generatorTag = true;
        details = `PNG ${type} chunk "${keyword}"`;
      } else if (keyword.toLowerCase() === 'software') {
        const value = latin1(chunk.subarray(nul + 1, Math.min(dataEnd - dataStart, nul + 200)));
        if (namesAGenerator(value)) {
          generatorTag = true;
          details = `PNG Software "${value.trim()}"`;
        }
      }
    } else if (type === 'caBX') {
      // JUMBF box: how C2PA content credentials ride inside a PNG.
      generatorTag = true;
      details = 'C2PA manifest';
    }

    if (type === 'IEND') break;
    offset = dataEnd + 4; // skip the CRC
  }

  return { generatorTag, cameraExif: false, details };
}

/* ---------------------------- EXIF ---------------------------- */

const TAG_MAKE = 0x010f;
const TAG_MODEL = 0x0110;
const TAG_SOFTWARE = 0x0131;
const TAG_EXIF_IFD = 0x8769;
const TAG_EXPOSURE_TIME = 0x829a;
const TAG_FNUMBER = 0x829d;
const TAG_ISO = 0x8827;

function readAscii(bytes, view, tiffStart, valueOffset, count, big) {
  let offset;
  if (count <= 4) {
    // Values of four bytes or fewer sit inside the entry itself.
    offset = valueOffset;
  } else {
    // Anything larger is stored out of line, at an offset measured from the
    // start of the TIFF header rather than from the start of the file.
    offset = tiffStart + view.getUint32(valueOffset, !big);
  }
  if (offset < 0 || offset + count > view.byteLength) return '';
  const slice = bytes.subarray(offset, offset + count);
  const nul = slice.indexOf(0);
  return latin1(slice.subarray(0, nul === -1 ? slice.length : nul));
}

function walkIfd(bytes, view, tiffStart, ifdOffset, big, found, depth = 0) {
  if (depth > 2) return;
  const base = tiffStart + ifdOffset;
  if (base + 2 > view.byteLength) return;
  const count = view.getUint16(base, !big);
  if (count > 512) return; // implausible; treat as corrupt

  for (let i = 0; i < count; i += 1) {
    const entry = base + 2 + i * 12;
    if (entry + 12 > view.byteLength) return;
    const tag = view.getUint16(entry, !big);
    const type = view.getUint16(entry + 2, !big);
    const valueCount = view.getUint32(entry + 4, !big);
    const valueField = entry + 8;

    if (tag === TAG_MAKE && type === 2) {
      found.make = readAscii(bytes, view, tiffStart, valueField, valueCount, big);
    } else if (tag === TAG_MODEL && type === 2) {
      found.model = readAscii(bytes, view, tiffStart, valueField, valueCount, big);
    } else if (tag === TAG_SOFTWARE && type === 2) {
      found.software = readAscii(bytes, view, tiffStart, valueField, valueCount, big);
    } else if (tag === TAG_EXPOSURE_TIME || tag === TAG_FNUMBER || tag === TAG_ISO) {
      found.exposure = true;
    } else if (tag === TAG_EXIF_IFD && valueCount === 1) {
      const sub = view.getUint32(valueField, !big);
      walkIfd(bytes, view, tiffStart, sub, big, found, depth + 1);
    }
  }
}

function readExifSignals(bytes, view, tiffStart) {
  const found = { make: '', model: '', software: '', exposure: false };
  if (tiffStart + 8 > view.byteLength) return EMPTY;

  const byteOrder = latin1(bytes.subarray(tiffStart, tiffStart + 2));
  const big = byteOrder === 'MM';
  if (!big && byteOrder !== 'II') return EMPTY;
  if (view.getUint16(tiffStart + 2, !big) !== 42) return EMPTY;

  walkIfd(bytes, view, tiffStart, view.getUint32(tiffStart + 4, !big), big, found);

  const generatorTag = namesAGenerator(found.software);
  // Make and Model alone are trivially added by re-saving pipelines. Requiring
  // an exposure field as well makes this evidence of an actual capture.
  const cameraExif = Boolean(found.make && found.model && found.exposure);

  return {
    generatorTag,
    cameraExif,
    details: generatorTag
      ? `EXIF Software "${found.software.trim()}"`
      : cameraExif
        ? `EXIF camera ${found.make.trim()} ${found.model.trim()}`
        : '',
  };
}

function readJpegSignals(bytes, view) {
  let offset = 2;
  let result = { ...EMPTY };

  while (offset + 4 <= view.byteLength) {
    if (view.getUint8(offset) !== 0xff) break;
    const marker = view.getUint8(offset + 1);
    if (marker === 0xd8 || marker === 0xd9) {
      offset += 2;
      continue;
    }
    if (marker === 0xda) break; // start of scan: no more metadata segments
    const length = view.getUint16(offset + 2);
    if (length < 2 || offset + 2 + length > view.byteLength) break;
    const segStart = offset + 4;
    const segEnd = offset + 2 + length;

    if (marker === 0xe1) {
      const header = latin1(bytes.subarray(segStart, Math.min(segStart + 6, segEnd)));
      if (header.startsWith('Exif')) {
        const exif = readExifSignals(bytes, view, segStart + 6);
        result = {
          generatorTag: result.generatorTag || exif.generatorTag,
          cameraExif: result.cameraExif || exif.cameraExif,
          details: result.details || exif.details,
        };
      } else if (header.startsWith('http')) {
        // XMP packet.
        const xmp = latin1(bytes.subarray(segStart, Math.min(segEnd, segStart + 4096)));
        if (namesAGenerator(xmp) || xmp.includes('c2pa')) {
          result = { ...result, generatorTag: true, details: result.details || 'XMP generator tag' };
        }
      }
    } else if (marker === 0xeb) {
      // APP11 carries JUMBF, which is how C2PA content credentials travel.
      result = { ...result, generatorTag: true, details: result.details || 'C2PA manifest' };
    }

    offset = segEnd;
  }

  return result;
}

/**
 * Inspect image bytes for what they say about their own origin.
 * @param {ArrayBuffer|Uint8Array} input
 * @returns {{generatorTag: boolean, cameraExif: boolean, details: string}}
 */
export function readGenerationSignals(input) {
  try {
    const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
    if (bytes.byteLength < 8) return { ...EMPTY };
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

    if (isPng(view)) return readPngSignals(bytes, view);
    if (isJpeg(view)) return readJpegSignals(bytes, view);
    return { ...EMPTY };
  } catch {
    // Arbitrary bytes from arbitrary sites. No metadata is a fine answer;
    // taking down the service worker is not.
    return { ...EMPTY };
  }
}
