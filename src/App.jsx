import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  // useStates for updating components
  const [start, setStart] = useState(false);
  const [faceRisks, setFaceRisks] = useState([]);
  const [imageUrl, setImageUrl] = useState('');
  const [sec, setSec] = useState(0);
  const [min, setMin] = useState(0);
  const [frame, setFrame] = useState(0)

  // update frame state
  const updateFrame = 5;

  // fethc data from backend
  const fetchFaceRisks = async () => {
    try {
      const response = await axios.get('http://localhost:5000/get_face_risks');
      setFaceRisks(response.data.face_risks);
      setImageUrl(`http://localhost:5000/image?timestamp=${new Date().getTime()}`);
    } catch (error) {
      console.error('Error fetching face risks:', error);
    }
  };

  //function for time
  function startTime() {
    setSec(sec + 1);
  }

  // Start or stop the interval based on `start` state
  useEffect(() => {
    let interval;
    let timer;
    if (start) {
      fetchFaceRisks();
      startTime();
      interval = setInterval(fetchFaceRisks, updateFrame * 1000);
      timer = setInterval(() => setSec(prevTime => prevTime + 1), 1000)
    } else if (!start && interval) {
      clearInterval(interval);
      clearInterval(timer);
    }
    return () => {
      clearInterval(interval);
      clearInterval(timer);
    };
  }, [start]);

  // Update frame every 5 seconds
  // Update min and sec every minute
  useEffect(() => {
    if (sec >= 60) {
      setMin(min + 1);
      setSec(0);
    }
    if (sec % updateFrame == 0) {
      setFrame(prevTime => prevTime + updateFrame);
    }
  }, [sec]);

  // handle start button when clicked
  const handleStartButtonClick = () => {
    setStart(!start);
    setFaceRisks([]);
    setImageUrl('');
    setSec(0);
    setMin(0);
    setFrame(0);
  };

  return (
    <>
      <h1 className='frame-num'>Frame of {frame} seconds</h1>

      <div className='frame-image'>
        {imageUrl && <img src={imageUrl} alt="Face" />}
      </div>

      <div className='controllers'>
        <p>Number of faces: {faceRisks.length}</p>
        <p>timer: {min < 10 && '0'}{min}:{sec < 10 && '0'}{sec}</p>
        <button
          onClick={handleStartButtonClick}
          className={`start-button ${start ? 'stop' : 'start'}`}
        >
          {start ? 'Stop' : 'Start'}
        </button>
      </div>

      <div className='faces-container'>
        {faceRisks.map((face, index) => (
          <div key={index}>
            <p className='face'>face_{face.face_num}:
              <span className={`face ${face.risk == 'No Risk' ? 'green' : face.risk == 'Moderate Risk' ? 'orange' : 'red'}`}> {face.risk}</span>
            </p>
          </div>
        ))}
      </div>
    </>
  );
}

export default App;
