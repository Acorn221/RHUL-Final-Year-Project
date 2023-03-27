/* eslint-disable jsx-a11y/label-has-associated-control */
import React, { useEffect, useState } from 'react';
import { XyzTransitionGroup } from '@animxyz/react';
import '@/index.css';
import { createRoot } from 'react-dom/client';
import {
  BrowserRouter as Router,
  Routes,
  Route,
} from 'react-router-dom';

import HomePage from '@/misc/HomePage';
import Alzheimers from '@/Alzheimers';
import Layout from '@/misc/Layout';
import SkinCancer from '@/SkinCancer';

const App = () => (
  <Router>
    <Routes>
      <Route element={<Layout />}>
        <Route path="/AD-Predicter" element={<Alzheimers />} />
        <Route path="/Skin-Cancer-Predicter" element={<SkinCancer />} />
        <Route path="/" element={<HomePage />} />
        <Route path="*">404 Not Found</Route>
      </Route>
    </Routes>
  </Router>
);

export default App;
