import { ImGoogle3 } from "react-icons/im";
import { Link } from "react-router-dom";
import { FaFacebook } from "react-icons/fa";
import { FaXTwitter } from "react-icons/fa6";
const Register = () =>{
      return(
        <div className="Mainrgster">
        <div className="rgstrform">
          <div className="Toprgstr">
            <h3>Create Account</h3>
            <label>Email-id/Phone Number</label>
            <input className="inptBtn"type="text" placeholder=''/>
            <a className="veracct">Verify email-id</a>
            <label>Create Password</label>
            <input className="inptBtn"type="text" placeholder=''/>
            <label>Confirm Password</label>
            <input className="inptBtn"type="text" placeholder=''/>
           </div>
           <div className="Alrdyacct">
           <h5><a>Already have an account??</a></h5><br/>
          <button><Link to="/login">Login Now!!  </Link></button>
           </div>
           <div className="Fromotracct">
            <label>Create account using</label>
            <div className="Iconsofotr">
            <a href="https://www.google.com/account/about/"><ImGoogle3 /></a>
            <a href="https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwi7s5zAnNKDAxXwYUgAHbaXCIAYABAAGgJjZQ&ase=2&gclid=CjwKCAiA-vOsBhAAEiwAIWR0TdhL7ZNmH9ua-K6ocm5yOEEhx_yiXzanBNYCWovADgEJsP8DQ7dIsxoCOx8QAvD_BwE&ei=HD2eZYfzOKn14-EP5MCjuAU&ohost=www.google.com&cid=CAESVeD2GAhgZuJlTw8bPtSQG9i-dYF6YZs4wqDHhDvLkPX1arnqqJJGu1RkjVbVPzh4J0QGilmIhtc5YNoasFZmj64qbBpRFpGJ1gAdiwyqmrYPTmouImA&sig=AOD64_35wq1XEwwFjAklaiWZ_ck452aJsQ&q&sqi=2&nis=4&adurl&ved=2ahUKEwiH8YrAnNKDAxWp-jgGHWTgCFcQ0Qx6BAgJEAE"><FaFacebook /></a>
            <a href="https://twitter.com/settings/account?lang=en"><FaXTwitter /></a>
             </div>
           </div>
           </div>
        </div>
      )
};

export default Register;