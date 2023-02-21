import * as tf from '@tensorflow/tfjs-core';

function scalePoint(point, size) {
    point.x *= size.width;
    point.y *= size.height;
    return point;
}

export function scaledPointList(pts, size) {
    return pts.map(p => scalePoint(p, size));
}

export function normalize(x) {
    return tf.tidy(() => tf.mul(tf.sub(x, tf.scalar(127.5)), tf.scalar(0.0078125)));
}

export function getPathSize(pts) {
    let minx = pts[0].x;
    let maxx = pts[0].x;
    let miny = pts[0].y;
    let maxy = pts[0].y;

    for (let index = 1; index < pts.length; index++) {
        if (pts[index].x < minx) minx = pts[index].x;
        if (pts[index].x > maxx) maxx = pts[index].x;
        if (pts[index].y < miny) miny = pts[index].y;
        if (pts[index].y > maxy) maxy = pts[index].y;
    }

    const width = maxx - minx;
    const height = maxy - miny;
    const max = Math.max(width, height);
    const min = Math.min(width, height);

    return {
        minx: minx,
        maxx: maxx,
        miny: miny,
        maxy: maxy,
        width: width,
        height: height,
        max: max,
        min: min,
    };
}
