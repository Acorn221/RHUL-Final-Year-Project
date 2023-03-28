/* eslint-disable no-param-reassign */
import * as tf from '@tensorflow/tfjs';
import { Crop } from 'react-image-crop';

export type PredictionType = {
  prediction: boolean; // true = benign, false = malignant
  confidence: number;
};

/**
 * Preprocesses the image to be used in the model
 * @param canvas The canvas that the image is drawn to
 * @returns The preprocessed image
 */
const preprocess = async (canvas: any) => {
  const tensor = tf.browser.fromPixels(canvas as HTMLCanvasElement);
  const resized = tf.image.resizeBilinear(tensor, [224, 224]);
  const batched = resized.expandDims(0);
  return batched;
};

/**
 * Returns the raw prediction from the model
 */
const getPrediction = async (model: tf.LayersModel, canvas: any) => {
  const preprocessedData = await preprocess(canvas as HTMLCanvasElement);
  const prediction = model.predict(preprocessedData);
  return Array.isArray(prediction) ? prediction[0] : prediction;
};

/**
 * @param model The TF model that is to be used to predict the image
 * @param offscreen The offscreen canvas that the image is drawn to
 * @returns The prediction and the confidence of the prediction
 */
const predictImage = async (model: tf.LayersModel, offscreen: any): Promise<PredictionType> => {
  const prediction = await getPrediction(model, offscreen);
  console.log(prediction);
  const result = prediction.dataSync();
  // Index 0 is benign, index 1 is malignant
  console.log(result);
  console.log(result[0] > result[1] ? 'Benign' : 'Malignant');
  console.log(`Confidence: ${Math.max(result[0], result[1]) * 100}%`);

  return { prediction: result[0] > result[1], confidence: Math.max(result[0], result[1]) * 100 };
};

/**
 * This function draws the image to the canvas, it also crops the image so the user can select the area they want to analyse
 *
 * @param image The image element that is to be drawn to the canvas
 * @param canvas The canvas that the image is to be drawn to
 * @param crop The crop information, this is used to crop the image to the correct size
 */
const setCanvas = (
  image: HTMLImageElement,
  canvas: OffscreenCanvas,
  crop: Crop,
) => {
  // Get the context of the canvas, this is used to modify the canvas
  const ctx = canvas.getContext('2d');

  if (!ctx) {
    // throw the error if the context is not 2d, mostly for typescript
    throw new Error('No 2d context');
  }

  const scaleX = image.naturalWidth / image.width;
  const scaleY = image.naturalHeight / image.height;

  // Get the pixel ratio of the device, this is to make sure that the image is not blurry on high pixel density devices
  const pixelRatio = window.devicePixelRatio;

  // Set the canvas size to the crop size, multiplied by the pixel ratio
  canvas.width = Math.floor(crop.width * scaleX * pixelRatio);
  canvas.height = Math.floor(crop.height * scaleY * pixelRatio);

  // Set the canvas to the correct size, multiplied by the pixel ratio
  ctx.scale(pixelRatio, pixelRatio);
  ctx.imageSmoothingQuality = 'high';

  // Get the crop coordinates, multiplied by the scale
  const cropX = crop.x * scaleX;
  const cropY = crop.y * scaleY;

  // Get the center of the image
  const centerX = image.naturalWidth / 2;
  const centerY = image.naturalHeight / 2;

  // Save the current state of the canvas
  ctx.save();

  // Translate the canvas to the center of the image
  ctx.translate(-cropX, -cropY);
  ctx.translate(centerX, centerY);
  ctx.translate(-centerX, -centerY);

  // Draw the image to the canvas
  ctx.drawImage(
    image,
    0,
    0,
    image.naturalWidth,
    image.naturalHeight,
    0,
    0,
    image.naturalWidth,
    image.naturalHeight,
  );

  // Restore the canvas to the previous state
  ctx.restore();
};

export {
  predictImage,
  setCanvas,
};
