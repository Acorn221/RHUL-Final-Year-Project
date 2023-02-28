import React, { useEffect, useState } from 'react';
import { XyzTransition, XyzTransitionGroup } from '@animxyz/react';
import '@/index.css';

/**
 * The different states that the app can be in,
 * waitingForFile: The user has not selected an image to upload
 * waitingForPrediction: The user has selected an image to upload, and the app is waiting for the server to respond
 * predictionReceived: The server has responded with a prediction and the app is displaying it to the user
 */
enum states {
  waitingForFile,
  waitingForPrediction,
  predictionReceived,
}

type predictionType = {
  isPositive: number;
};

const App = () => {
  const [currentState, setCurrentState] = useState<states>(states.waitingForFile);
  // blob or undefined
  const [image, setImage] = useState<Blob | undefined>(undefined);
  const [prediction, setPrediction] = useState<predictionType | undefined>(undefined);

  /**
   * Called when the user selects an image to upload
   * @param event The event that triggered this function
   * @returns void
   */
  const imageUploaded = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files === null) return;
    setImage(event.target.files[0]);
    setCurrentState(states.waitingForPrediction);
  };

  /**
   * Called when the current state changes, this function will send the image to the server
   * then wait for the response, and then set the prediction state
   */
  useEffect(() => {
    if (currentState === states.waitingForPrediction) {
      const formData = new FormData();
      if (image === undefined) return;
      formData.append('file', image);
      fetch('http://localhost:3001/predict', {
        method: 'POST',
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          console.log(data);
          setPrediction(data);
          setCurrentState(states.predictionReceived);
        });
    }
  }, [currentState]);

  return (
    <div className="flex justify-center align-middle h-screen">
      <div className="bg-white m-auto p-10 rounded-xl w-3/4 md:w-1/2 text-center">
        <div className="underline text-5xl">Alzheimer&apos;s Disease Predictor</div>
        <div className="m-5 text-left">
          <div className="text-2xl p-20 text-center bg-slate-300 hover:bg-slate-400 rounded-2xl select-none">
            {currentState === states.waitingForFile && (
              <div>
                <input type="file" name="file" onChange={imageUploaded} className="" />
                <br />
                Drag and drop your MRI image here, or click to select the file
              </div>
            )}
            <XyzTransitionGroup xyz="fade delay-1" appearVisible={currentState === states.waitingForPrediction}>
              {currentState === states.waitingForPrediction && (
              <div>
                <div className="text-2xl">Processing image...</div>
                <div className="text-xl">Please wait...</div>
              </div>
              )}
            </XyzTransitionGroup>
            <XyzTransitionGroup xyz="fade duration-1" appearVisible={currentState === states.waitingForPrediction}>
              {states.predictionReceived === currentState && (
              <div>
                <div className="text-2xl">Prediction received!</div>
                <div className="text-xl">
                  The model predicts that the patient has a probability of &nbsp;
                  {prediction?.isPositive}
                  % of having Alzheimer&apos;s Disease
                </div>
              </div>
              )}
            </XyzTransitionGroup>
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
