import { useState, useEffect, useRef } from 'react';
import * as tf from '@tensorflow/tfjs';
import ImageUploading, { ImageType } from 'react-images-uploading';
import Button from '@mui/material/Button';
import ReactCrop, { Crop } from 'react-image-crop';
import 'react-image-crop/dist/ReactCrop.css';
import { FaFileUpload } from 'react-icons/fa';
import { AiOutlineCloseCircle } from 'react-icons/ai';
import { GrDocumentUpdate } from 'react-icons/gr';
import { RiBodyScanFill } from 'react-icons/ri';
import { IoChevronBackCircleSharp } from 'react-icons/io5';
import { Link } from 'react-router-dom';
import {
  setCanvas, predictImage, PredictionType, getLocation,
} from '@/utils';

// this is the url of the model that is to be loaded
const modelURL = `${getLocation()}/skin-cancer-model/model.json`;

/**
 * This is the page to predict skin cancer, it allows the user to upload an image, then processes it locally
 * to give the user a prediction of whether or not they have skin cancer
 *
 * This would be a client facing page, so the client could see if they need to speak to a doctor about their skin
 *
 * @returns The skin cancer predicter page
 */
const SkinCancer = () => {
  // This stores the image that the user uploads
  const [image, setImage] = useState<ImageType[]>([]);
  // This stores the prediction that the model gives
  const [prediction, setPrediction] = useState<PredictionType | null>(null);
  // This stores the model that is loaded from the server
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  // This is to give the user feedback that the model is loading (as it might take a while)
  const [loading, setLoading] = useState(false);

  const imgRef = useRef<HTMLImageElement>(null);

  const [offscreen, setOffscreen] = useState<OffscreenCanvas>(
    new OffscreenCanvas(224, 224),
  );

  // This saves the crop information
  const [crop, setCrop] = useState<Crop>({
    x: 0,
    y: 0,
    width: 100,
    height: 100,
    unit: '%',
  });

  // This useEffect runs when the page loads, it loads the model from the server
  useEffect(() => {
    async function loadModel() {
      setLoading(true);
      const loadedModel = await tf.loadLayersModel(modelURL);
      setModel(loadedModel);
      setLoading(false);
    }
    loadModel();
  }, []);

  /**
   * This function runs whenever the user changes the crop, it updates the offscreen canvas which is a 224x224 canvas
   * This is done so that the UI is not restricted to a 224x224 canvas, and the user can crop the image to their liking
   * The image needs to be 224x224 for the model to work
   */
  const updateCanvas = () => {
    if (crop) {
      const ctx = offscreen.getContext('2d');
      if (ctx && imgRef.current) {
        setCanvas(imgRef.current, offscreen, crop);
        // console log the image blob to see what it looks like
        console.log(`X: ${crop.x}, Y: ${crop.y}, Width: ${crop.width}, Height: ${crop.height}`);
        // Console log the blob URL to see what it looks like
        offscreen.convertToBlob().then((blob) => URL.createObjectURL(blob)).then((b) => console.log(b));
      }
    }
  };

  const onImageChange = (imageList: ImageType[]) => {
    setImage(imageList);
    console.log('Image Changed');

    // on image load, run updateCanvas();
    imgRef.current?.addEventListener('load', updateCanvas);
  };

  const predict = () => {
    if (!loading && model && imgRef.current) {
      setCanvas(imgRef.current, offscreen, crop);
      setLoading(true);
      window.requestAnimationFrame(async () => {
        const pred = predictImage(model, offscreen);
        setLoading(false);
        setPrediction(await pred);
      });
    }
  };

  return (
    <div className="h-full flex">
      <div className="justify-center align-middle bg-white p-5 m-auto text-center">
        <div className="flex justify-between mt-3 mb-3 gap-4">
          <Link to="/">
            <div className="bg-slate-300 justify-center align-middle flex rounded-md p-2 h-full hover:bg-slate-400">
              <IoChevronBackCircleSharp className="h-10 w-10 m-auto" />
            </div>
          </Link>
          <div className="bg-slate-300 p-3 text-2xl rounded-md flex-1">
            Skin Cancer Predicter
          </div>
        </div>
        {loading && <p>Loading Model...</p>}
        {!loading && (
          <ImageUploading
            value={image}
            onChange={onImageChange}
            dataURLKey="data_url"
          >
            {({
              imageList,
              onImageUpload,
              onImageUpdate,
              onImageRemove,
              isDragging,
              dragProps,
            }) => (
              <div className="flex-col w-full justify-center align-middle gap-4">
                {imageList.length === 0 && (
                  <div className="flex gap-3 mt-4">
                    <button
                      className={`flex-1 h-20 d bg-slate-300 rounded-md p-5 flex align-middle justify-center gap-4 hover:shadow-xl ${
                        isDragging && 'bg-slate-400'
                      }`}
                      onClick={onImageUpload}
                      {...dragProps}
                    >
                      <div className="flex-1 m-auto">Click or Drag and Drop here</div>
                      {' '}
                      <FaFileUpload className="h-full m-auto" />
                    </button>
                  </div>
                )}

                {imageList.length > 0 && (
                  <div className="flex flex-col gap-5">
                    <div className="image-item">
                      <ReactCrop
                        crop={crop}
                        onChange={(c) => setCrop(c)}
                        aspect={1}
                      >
                        <img
                          src={imageList[0].data_url}
                          ref={imgRef}
                          alt=""
                          className="m-auto mb-4 max-w-[50vw] max-h-[60vh]"
                        />
                      </ReactCrop>
                      <div className="flex w-full m-auto gap-3">
                        <Button
                          variant="contained"
                          size="large"
                          className="flex-1 h-20 flex justify-center"
                          onClick={() => onImageUpdate(0)}
                        >
                          <div className="m-auto flex-1">Update</div>
                          <GrDocumentUpdate className="m-auto invert h-10 flex-auto" />
                        </Button>
                        <Button
                          variant="contained"
                          size="large"
                          className="flex-1 h-20 flex justify-center"
                          onClick={() => onImageRemove(0)}
                        >
                          <div className="m-auto flex-1">Remove</div>
                          <AiOutlineCloseCircle className="m-auto h-10 flex-auto" />
                        </Button>
                      </div>
                    </div>
                    <Button
                      variant="contained"
                      size="large"
                      className="flex-1 h-20 flex justify-center"
                      onClick={() => predict()}
                    >
                      <div className="m-auto flex-1">Check</div>
                      <RiBodyScanFill className="m-auto flex-auto h-10" />
                    </Button>
                  </div>
                )}
              </div>
            )}
          </ImageUploading>
        )}
        {(prediction != null && !loading && image.length > 0) && (
          <div className="mt-3 mb-3 bg-slate-300 p-3 text-2xl rounded-md">
            {prediction.prediction && (
              <p>
                You are unlikely to have skin cancer
              </p>
            )}
            {!prediction.prediction && (
              <p>
                You may have skin cancer
              </p>
            )}

            <p>
              Confidence:
              {' '}
              {prediction.confidence}
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default SkinCancer;
