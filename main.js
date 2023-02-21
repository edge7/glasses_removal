import "./style.css";

import * as tflite from '@tensorflow/tfjs-tflite';
import * as tf from "@tensorflow/tfjs";
import {FaceMesh} from "@mediapipe/face_mesh";
import {Camera} from "@mediapipe/camera_utils/camera_utils.js";
// Utility
import Stats from "./libs/stats/stats.module.js";
import {getPathSize, scaledPointList} from "./libs/utils/FaceMeshUtils.js";

const maxSize = 3; // maximum size of the array
const tensorArray = new Array(maxSize);
var inserted = 0;
// Call this function.
tflite.setWasmPath(
   'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/'
);

// Variables
const modelName = "pixtopix32_128_20feb_shrinked_retr_unet_best_graph_model_tfjs";
let renderOptions = {
  showCrop: true,
  cropPadding: 35,
};

tf.enableProdMode()

// Canvases
const videoElement = document.getElementById("video");
videoElement.style.display = "none";

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d", { willReadFrequently: true });

const canvas_crop = document.getElementById("canvas_crop");
const ctx_crop = canvas_crop.getContext("2d", { willReadFrequently: true });

const canvas_output = document.getElementById("canvas_output");
const ctx_output = canvas_output.getContext("2d", { willReadFrequently: true });

// FPS Stats
const stats = new Stats();
stats.dom.style.top = "26px";
stats.dom.style.left = "23px";

const body = document.querySelector("body");
body.appendChild(stats.dom);

let model = null;

async function main() {
  // Initialize MediaPipe FaceMesh
  const faceMesh = new FaceMesh({
    locateFile: (file) => {
      return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
    },
  });

  const faceMeshOptions = {
    cameraNear: 1,
    cameraFar: 2000,
    cameraVerticalFovDegrees: 66,
    enableFaceGeometry: false,
    selfieMode: true,
    maxNumFaces: 1,
    refineLandmarks: false,
    minDetectionConfidence: 0.1,
    minTrackingConfidence: 0.3,
    staticImage: false
  };

  faceMesh.setOptions(faceMeshOptions);

  await faceMesh.initialize();

  // Load model
  let modelUrl = `assets/${modelName}/model.json`;
  console.log("model", modelUrl);

  model = await tf.loadGraphModel(modelUrl);
  tflite.setWasmPath(
   'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/'
);
  //model = await tflite.loadTFLiteModel('assets/pix_pix_256_32_tf_lite_training_none.tflite')

  // Resize
  let width, height;

  var C = 0
  // Construct camera input
  const camera = new Camera(videoElement, {
    onFrame: async () => {
      if (
        width !== camera.video.videoWidth ||
        height !== camera.video.videoHeight
      ) {
        width = camera.video.videoWidth;
        height = camera.video.videoHeight;
      }
      C+=1
      //if( C % 2 === 0)
      await faceMesh.send({ image: videoElement });
    },
    width: canvas.width,
    height: canvas.height,
  });
  camera.start();

  // Set callback on FaceMesh output result
  faceMesh.onResults(async function (faceMeshResults) {
    await render(faceMeshResults);
    stats.update();
  });
}

function insertElement(element, index, array, maxSize) {
  array[index % maxSize] = element;
}

function computeAverage(array) {
  const sum = array.reduce((acc, tensor) => acc.add(tensor));
  return sum.div(array.length);
}
async function render(faceMeshResult) {
  if (faceMeshResult.multiFaceLandmarks[0]) {
    const scrSize = { width: canvas.width, height: canvas.height };
    const facePoints = scaledPointList(
      faceMeshResult.multiFaceLandmarks[0],
      scrSize
    );
    const faceCoords = getPathSize(facePoints);

    // Calc face contour coordinates
    const bottom_right = { x: faceCoords.maxx, y: faceCoords.maxy };
    const top_left = { x: faceCoords.minx, y: faceCoords.miny };
    const top_right = {
      x: faceCoords.minx + faceCoords.width,
      y: faceCoords.miny,
    };
    const bottom_left = {
      x: faceCoords.maxx - faceCoords.width,
      y: faceCoords.maxy,
    };

    const facePadding = renderOptions.cropPadding;

    // Draw rect around face
    if (renderOptions.showCrop == false) {
      ctx.beginPath();
      ctx.rect(
        top_left.x - facePadding, // sx
        top_left.y - facePadding, // sy
        Math.abs(top_left.x - top_right.x) + facePadding * 2, // sw
        Math.abs(top_left.y - bottom_left.y) + facePadding * 2 // sh
      );
      ctx.stroke();
      ctx.closePath();
    }

    // Face Crop 256x256
    ctx_crop.drawImage(
      faceMeshResult.image,
      top_left.x - facePadding, // sx
      top_left.y - facePadding, // sy
      Math.abs(top_left.x - top_right.x) + facePadding * 2, // sw
      Math.abs(top_left.y - bottom_left.y) + facePadding * 2, // sh
      0, // (scrSize.width / 2.) - 128, // dx
      0, // (scrSize.height / 2.) - 128, // dy
      128, // baseWidth * zoomRate, // dWidth
      128 // baseHeight * zoomRate, // dHeight
    );

    // Get cropImageData
    const cropImageData = ctx_crop.getImageData(0, 0, 128, 128);

    const img = tf.browser.fromPixels(cropImageData);
    const input = tf.sub(tf.div(img.toFloat(), 127.5), 1);
    const expanded = input.expandDims(0);

    const prediction = await model.predict(expanded)
    //let prediction = model.predict(expanded)
    const reduction = tf.add(tf.mul(prediction, 0.5), 0.5);
    let squeeze = reduction.squeeze();

    insertElement(squeeze, inserted, tensorArray, maxSize)
    inserted +=1
    if (inserted >= maxSize){
      squeeze = computeAverage(tensorArray)
    }

    await tf.browser.toPixels(squeeze, canvas_output);

  ctx.drawImage(
    faceMeshResult.image,
    0, // sx
    0, // sy
    canvas.width, // dWidth
    canvas.height // dHeight
  );

    ctx.drawImage(
      canvas_output,
      top_left.x - facePadding, // dx
      top_left.y - facePadding, // dy
      Math.abs(top_left.x - top_right.x) + facePadding * 2, // dWidth
      Math.abs(top_left.y - bottom_left.y) + facePadding * 2 // dHeight
    );
  }
}

// Launch main when page is loaded
window.addEventListener("load", main);
