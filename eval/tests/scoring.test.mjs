import test from 'node:test';
import assert from 'node:assert/strict';

import {
  applyCalibration,
  classify,
  fuseMetadata,
  l2Normalize,
} from '../../extension/shared/scoring.js';

test('l2Normalize gives a unit vector', () => {
  const out = l2Normalize(Float32Array.from([3, 4]));
  assert.ok(Math.abs(Math.hypot(out[0], out[1]) - 1) < 1e-6);
});

test('l2Normalize leaves a zero vector finite', () => {
  const out = l2Normalize(Float32Array.from([0, 0]));
  assert.ok(out.every(Number.isFinite));
});

test('calibration maps the fitted threshold onto the decision confidence', () => {
  // Values as exported by the training pipeline.
  const cal = { a: 0.22343252720203058, b: 1.1595814950877068 };
  const tStar = -2.4192640769475693;
  assert.ok(Math.abs(applyCalibration(cal, tStar) - 0.65) < 1e-6);
});

test('calibration is monotonic and stays inside the unit interval', () => {
  const cal = { a: 0.5, b: 0.1 };
  let previous = -1;
  for (let s = -200; s <= 200; s += 0.5) {
    const c = applyCalibration(cal, s);
    assert.ok(c >= previous - 1e-12, 'calibration must never decrease');
    assert.ok(c >= 0 && c <= 1);
    previous = c;
  }
});

test('classify flags at or above the threshold', () => {
  assert.equal(classify(0.65, 0.65), 'ai');
  assert.equal(classify(0.6499, 0.65), 'real');
  assert.equal(classify(0.99, 0.65), 'ai');
});

test('an embedded generation tag overrides toward ai', () => {
  const { confidence, reason } = fuseMetadata(0.1, {
    generatorTag: true,
    cameraExif: false,
  });
  assert.ok(confidence >= 0.95);
  assert.match(reason, /generat/i);
});

test('camera exif only nudges and can never decide alone', () => {
  const { confidence } = fuseMetadata(0.9, {
    generatorTag: false,
    cameraExif: true,
  });
  assert.ok(
    confidence < 0.9 && confidence > 0.65,
    'exif must not be able to flip a confident ai verdict to real',
  );
});

test('a generation tag wins over camera exif', () => {
  const { confidence } = fuseMetadata(0.2, {
    generatorTag: true,
    cameraExif: true,
  });
  assert.ok(confidence >= 0.95);
});

test('no metadata leaves the model score untouched', () => {
  const { confidence, reason } = fuseMetadata(0.42, {
    generatorTag: false,
    cameraExif: false,
  });
  assert.equal(confidence, 0.42);
  assert.equal(reason, '');
});

test('fused confidence always stays inside the unit interval', () => {
  for (const base of [0, 0.01, 0.5, 0.99, 1]) {
    for (const tag of [true, false]) {
      for (const exif of [true, false]) {
        const { confidence } = fuseMetadata(base, {
          generatorTag: tag,
          cameraExif: exif,
        });
        assert.ok(confidence >= 0 && confidence <= 1);
      }
    }
  }
});
