import React from 'react'
import { Link } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';
import { useState } from 'react';
import { RiAccountPinCircleLine } from "react-icons/ri";


const Login =() =>{
 const[name,setName]= useState("");
 

    return <div className='Loginpage'>
        <form className='Loginform shadow'>
        <div className='Accountpic'>
             <RiAccountPinCircleLine />
             </div>
             <div className='LoginCard'>
             
                <input onChange={""} className="inptBtn" type="text" placeholder='@username'/>
        
                <input type="password" placeholder='password' />
               
                <button id="btna2z">Login</button>
                <a href="https://shopify.dev/assets/themes/templates/customer-reset-password.png">forgot password?</a>
             </div>
             <div className="NoAccount">
            <h6> New on Neat Nosh ? <br /> To create new account <br /></h6><button> <Link to="/register">Register Now!!  </Link></button>
             </div>
        </form>
    </div>
};
export default Login;