/* eslint-disable jsx-a11y/label-has-associated-control */
import React, { useEffect, useState } from 'react';
import { XyzTransitionGroup } from '@animxyz/react';
import '@/index.css';

/**
 * Importing the different material ui components
 */
import Slider from '@mui/material/Slider';
import Radio from '@mui/material/Radio';
import RadioGroup from '@mui/material/RadioGroup';
import FormControlLabel from '@mui/material/FormControlLabel';
import FormControl from '@mui/material/FormControl';
import FormLabel from '@mui/material/FormLabel';
import Button from '@mui/material/Button';
import { Link } from 'react-router-dom';
import { IoChevronBackCircleSharp } from 'react-icons/io5';

/**
 * The different states that the app can be in,
 * waitingForFile: The user has not selected an image to upload
 * waitingForPrediction: The user has selected an image to upload, and the app is waiting for the server to respond
 * predictionReceived: The server has responded with a prediction and the app is displaying it to the user
 */
enum states {
  waitingForFile,
  waitingForData,
  waitingForPrediction,
  predictionReceived,
}

// The type for the prediction, this is the same as the type in the server (0 = non AD, 3 = moderate AD)
type predictionType = {
  0: number;
  1: number;
  2: number;
  3: number;
};

/**
 * This is the page to predict Alzheimers, it allows the user to upload an image, then processes it on the server
 * This would not be a client facing page, as it sends the sensitive info of the user to the server and
 * it's more likely that a doctor would have the MRI files of the patient, aside from the fact that the doctor
 * should take this into account when diagnosing the patient
 *
 * @returns The Alzheimers predicter page
 */
const Alzheimers = () => {
  // This stores the current state of the app
  const [currentState, setCurrentState] = useState<states>(states.waitingForFile);
  // blob or undefined
  const [image, setImage] = useState<Blob | undefined>(undefined);
  // predictionType or undefined, this is the prediction that the server returns
  const [prediction, setPrediction] = useState<predictionType | undefined>(undefined);

  // These are the values that the user can change to increase the accuracy of the prediction
  const [age, setAge] = useState<number>(50);
  const [sex, setSex] = useState<string>('male');
  const [mmse, setMMSE] = useState<number>(15);

  /**
   * Called when the user selects an image to upload
   * @param event The event that triggered this function
   * @returns void
   */
  const imageUploaded = (event: React.ChangeEvent<HTMLInputElement>) => {
    if (event.target.files === null) return;
    setImage(event.target.files[0]);
    setCurrentState(states.waitingForData);
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
      formData.append('age', age.toString());
      formData.append('sex', sex);
      formData.append('mmse', mmse.toString());
      fetch('http://localhost:3002/predict', {
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
        <div className="text-2xl p-20 text-center bg-slate-300 hover:shadow-2xl rounded-2xl select-none">
          {currentState === states.waitingForFile && (
          <div>
            <input type="file" name="file" onChange={imageUploaded} className="" />
            <br />
            Drag and drop your MRI image here, or click to select the file
          </div>
          )}
          <XyzTransitionGroup xyz="fade" appearVisible={currentState === states.waitingForPrediction}>
            {currentState === states.waitingForData && (
              <div className="flex flex-col space-y-5">
                <div className="text-2xl">Please enter the following details:</div>
                <div className="text-2xl">
                  <div className="flex justify-center m-3">
                    <div className="flex flex-col w-1/2">
                      <label htmlFor="age">
                        Age
                        {' '}
                        {age}
                        {' '}
                        years old
                      </label>
                      <Slider
                        className="m-5"
                        value={age}
                        onChange={(event, newValue) => {
                          setAge(newValue as number);
                        }}
                        aria-label="Default"
                        valueLabelDisplay="off"
                        min={0}
                        max={120}
                      />
                    </div>
                  </div>
                  <div className="flex justify-center m-3">
                    <FormControl>
                      <FormLabel id="demo-radio-buttons-group-label">Sex</FormLabel>
                      <RadioGroup
                        aria-labelledby="demo-radio-buttons-group-label"
                        defaultValue=""
                        name="radio-buttons-group"
                        value={sex}
                        onChange={(evt, val) => setSex(val)}
                      >
                        <FormControlLabel value="male" control={<Radio />} label="Male" />
                        <FormControlLabel value="female" control={<Radio />} label="Female" />
                      </RadioGroup>
                    </FormControl>
                  </div>
                </div>
                <div className="flex justify-center m-3">
                  <div className="flex flex-col w-1/2 text-2xl">
                    <label htmlFor="mmse">
                      MMSE Score:
                      {' '}
                      {mmse}
                    </label>
                    <Slider
                      className="m-5"
                      value={mmse}
                      onChange={(evt, val) => {
                        setMMSE(val as number);
                      }}
                      aria-label="Default"
                      valueLabelDisplay="off"
                      min={0}
                      max={30}
                    />
                  </div>
                </div>
                <div className="w-full">
                  <Button variant="contained" size="large" className="w-1/2 h-20" onClick={() => setCurrentState(states.waitingForPrediction)}>Submit</Button>
                </div>
              </div>
            )}
          </XyzTransitionGroup>
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
                  {JSON.stringify(prediction, null, 2)}
                  % of having Alzheimer&apos;s Disease
                </div>
              </div>
            )}
          </XyzTransitionGroup>
        </div>
      </div>
    </div>
  );
};

export default Alzheimers;
