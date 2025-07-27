
import 'bootstrap/dist/css/bootstrap.min.css';
import 'bootstrap/dist/js/bootstrap.min.js';
import { RxSketchLogo } from "react-icons/rx";
import { IoIosHome } from "react-icons/io";
import { FaCartShopping } from "react-icons/fa6";
import { MdAccountCircle } from "react-icons/md";
import { CiLogin } from "react-icons/ci";
import { CiLogout } from "react-icons/ci";
import { CiSettings } from "react-icons/ci";
import React from 'react'
import { Link } from 'react-router-dom';
import "../App.css";


function Navbar(){
  return(
     <>
     <nav className="navbar navbar-expand-lg bg-body-tertiary ">
      <div className="container-fluid">
    <Link className="navbar-brand" to="/"><RxSketchLogo /></Link>
    <button className="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarScroll" aria-controls="navbarScroll" aria-expanded="false" aria-label="Toggle navigation">
      <span className="navbar-toggler-icon"></span>
    </button>
    <div className="collapse navbar-collapse" id="navbarScroll">
      <ul className="navbar-nav me-auto my-2 my-lg-0 navbar-nav-scroll" styles="--bs-scroll-height: 100px;">
        <li className="nav-item">
          <Link className="nav-link active" aria-current="page" to="/">Home</Link>
        </li>
        <li className="nav-item">
          <Link className="nav-link" to="/cart">Cart</Link>
        </li>
        <li className="nav-item dropdown">
          <Link className="nav-link dropdown-toggle" to="#" role="button" data-bs-toggle="dropdown" aria-expanded="false">
           Account <MdAccountCircle />
          </Link>
          <ul className="dropdown-menu">
            <li><Link className="dropdown-item" to="/login">Log-in <CiLogin /></Link></li>
            <li><Link className="dropdown-item" to="/logout">Log-out <CiLogout /> </Link></li>
            <li><hr className="dropdown-divider"></hr></li>
            <li><Link className="dropdown-item" to="/register">Register</Link></li>
          </ul>
        </li>
        <li className="nav-item">
          <Link className="nav-link" to='/settings'>Settings <CiSettings /></Link>
        </li>
      </ul>
      <form className="d-flex" role="search">
        <input className="form-control me-2" type="search" placeholder="Search" aria-label="Search" />
        <button className="btn" type="submit">Search</button>
      </form>
    </div>
  </div>
</nav>

     </>
  );
}

export default Navbar;