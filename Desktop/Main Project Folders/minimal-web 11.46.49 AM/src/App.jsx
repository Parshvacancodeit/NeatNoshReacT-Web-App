
import './App.css'
import Navbar from './components/Navbar';
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.min.js';
import LandPage from './components/LandPage';
import Card from './components/Card';
import NewLaunchPage from './components/NewLaunchPage';
import SalePage from './components/SalePage';
import Login from './components/Login';
import AutoSlider from './components/AutoSlider';
import Catagories from './components/Catagories';
import CatCardLayout from './components/CatCardLayout';
import Combinercat from './components/Combinercat';
import Footer from './components/Footer';
import Home from './components/Home';
import Settings from './components/Settings';
import {BrowserRouter,Routes,Route} from "react-router-dom"
import Logout from './components/Logout';
import ProductView from './components/ProductView';
import Register from './components/Register';
import Cart from './components/Cart';
import NewCardInfo from './NewCardInfo';
import Productview2 from './components/Productview2';

function App() {
  

  return (
    <>
      <Navbar />  
     <Routes>
     <Route path="" element={<Home />} />
     <Route path="/settings" element={<Settings />}/>
     <Route path="/login" element={<Login />}/>
     <Route path="/logout" element={<Logout />}/>
     <Route path="/register" element={<Register />}/>
     <Route path="/cart" element={<Cart/>}/>
     <Route path="/productview/:id" element={<ProductView/>} exact />
     <Route path="/productviewcat/:id" element={<Productview2 />} exact />

     </Routes>
  
  
    </>
  )
}

export default App
